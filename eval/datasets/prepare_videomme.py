"""Download a small, cache-friendly subset of Video-MME for pmcache benchmarks.

Video-MME (lmms-lab/Video-MME) is the CVPR-2025 video-LLM benchmark:
900 human-annotated videos, 2,700 multiple-choice questions across
6 domains × 30 subcategories × 12 task types.

Videos are stored as YouTube URLs, not files — we use `yt-dlp` to
fetch them and rescale to a small resolution to keep the demo light.

## What we filter for (the pmcache use case)

`pmcache` wins on **low cross-frame visual change** (talking heads,
slideshows, presentations, OCR-heavy scenes). We bias the subset:

  - duration  == "short"        (< 2 min, keeps benchmark < 10 min)
  - domain    favors Knowledge / Humanity & History / Literature & Art
                                (lecture-style content, mostly static)
  - task_type favors Information Synopsis, OCR Problems, Object
                Recognition, Attribute Perception
                (questions that don't require dense temporal reasoning)

The defaults are overridable on the command line.

## Output

For each downloaded video the script appends entries to qa.jsonl in
the pmcache schema, deriving `gold_answer` from the correct MCQ
option text (so the eval harness's substring match works without
modification). The original MCQ letter + options are preserved
under `mcq_answer` / `mcq_options` for later LLM-judge eval.

## Usage

    pip install -e '.[videomme]'

    # Default: 10 short, cache-friendly videos.
    python -m eval.datasets.prepare_videomme --output_dir workspace/videos

    # Custom selection.
    python -m eval.datasets.prepare_videomme --output_dir workspace/videos \\
        --n 20 --domains Knowledge --task_types "Information Synopsis"

    # Or call as a function (returns the qa.jsonl path).
    from eval.datasets.prepare_videomme import main
    main("workspace/videos", n=10)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from pmcache._lazy import require


# --- selection defaults -----------------------------------------------------
#
# These are biased toward content where adjacent frames are visually
# near-identical (the pmcache sweet spot). They are NOT authoritative
# Video-MME categories — adjust based on what your local subset shows.

DEFAULT_DOMAINS_PREFERRED = {
    "Knowledge",
    "Humanity & History",
    "Literature & Art",
}

DEFAULT_TASK_TYPES_PREFERRED = {
    "Information Synopsis",
    "OCR Problems",
    "Object Recognition",
    "Attribute Perception",
}


# --- HF metadata load -------------------------------------------------------


def _load_metadata(hf_token: str | None = None) -> Any:
    """Return the Video-MME test split as a HuggingFace `Dataset`."""
    datasets = require("datasets", "videomme")
    return datasets.load_dataset(
        "lmms-lab/Video-MME",
        split="test",
        token=hf_token,
    )


def _row_score(row: dict[str, Any], preferred_domains: set[str], preferred_tasks: set[str]) -> int:
    """Higher score = more cache-friendly. Used to rank candidates."""
    score = 0
    if row.get("domain") in preferred_domains:
        score += 2
    if row.get("task_type") in preferred_tasks:
        score += 1
    return score


def select_videos(
    metadata: Iterable[dict[str, Any]],
    n: int = 10,
    duration: str = "short",
    domains: set[str] | None = None,
    task_types: set[str] | None = None,
    max_questions_per_video: int = 3,
) -> list[dict[str, Any]]:
    """Pick `n` videos matching the duration + domain/task filters.

    Pure function — accepts any iterable of {video_id, duration, domain,
    task_type, ...} dicts so it's trivially testable without a real
    HF download.

    Returns a list of {video_id, videoID, url, questions: [...]} dicts.
    """
    preferred_domains = domains or DEFAULT_DOMAINS_PREFERRED
    preferred_tasks = task_types or DEFAULT_TASK_TYPES_PREFERRED

    # First, group rows by video_id (Video-MME has ~3 questions per video).
    per_video: dict[str, dict[str, Any]] = {}
    for row in metadata:
        if duration and row.get("duration") != duration:
            continue
        vid = row["video_id"]
        if vid not in per_video:
            per_video[vid] = {
                "video_id": vid,
                "videoID": row.get("videoID", vid),
                "url": row.get("url"),
                "domain": row.get("domain"),
                "sub_category": row.get("sub_category"),
                "task_type": row.get("task_type"),
                "questions": [],
            }
        # Domain/task filtering applies at the question level.
        if domains is not None and row.get("domain") not in domains:
            continue
        if task_types is not None and row.get("task_type") not in task_types:
            continue
        if len(per_video[vid]["questions"]) >= max_questions_per_video:
            continue
        per_video[vid]["questions"].append({
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "options": list(row.get("options", [])),
            "answer": row.get("answer"),
            "task_type": row.get("task_type"),
        })

    # Drop videos that ended up with no surviving questions.
    candidates = [v for v in per_video.values() if v["questions"]]

    # Rank by cache-friendliness score (preferred domain + preferred task),
    # tie-break by question count desc, then video_id for stability.
    candidates.sort(
        key=lambda v: (
            -_row_score(v, preferred_domains, preferred_tasks),
            -len(v["questions"]),
            v["video_id"],
        )
    )
    return candidates[:n]


# --- yt-dlp download --------------------------------------------------------


def _download_video(url: str, dest: Path, max_height: int = 360) -> bool:
    """Download a YouTube `url` to `dest` via yt-dlp. Returns True on success."""
    yt_dlp = require("yt_dlp", "videomme")
    opts = {
        "outtmpl": str(dest.with_suffix("")) + ".%(ext)s",
        "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 2,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  ! download failed for {url}: {e}", file=sys.stderr)
        return False
    # yt-dlp may emit .mp4 or .webm — find what landed.
    for candidate in dest.parent.glob(dest.stem + ".*"):
        if candidate.suffix in {".mp4", ".webm", ".mkv"}:
            if candidate != dest:
                candidate.replace(dest if dest.suffix == candidate.suffix
                                  else dest.with_suffix(candidate.suffix))
            return True
    return False


# --- qa.jsonl construction --------------------------------------------------


def _mcq_to_gold_answer(question_entry: dict[str, Any]) -> str:
    """Convert MCQ {options, answer='A'} to a substring-matchable gold string.

    Falls back to the letter itself if options aren't well-formed.
    """
    options = question_entry.get("options") or []
    letter = (question_entry.get("answer") or "").strip().upper()
    if letter in {"A", "B", "C", "D"} and len(options) >= 4:
        opt = options[ord(letter) - ord("A")]
        # Options often look like "A. Some text" — strip the prefix so
        # substring matching works against a model's free-form answer.
        for prefix in (f"{letter}.", f"{letter})", f"{letter}:"):
            if opt.startswith(prefix):
                opt = opt[len(prefix):].strip()
                break
        return opt
    return letter or ""


def write_qa_jsonl(
    selected: list[dict[str, Any]],
    successfully_downloaded: set[str],
    qa_path: Path,
) -> int:
    """Write the QA file in pmcache schema for downloaded videos."""
    written = 0
    with open(qa_path, "w", encoding="utf-8") as f:
        for video in selected:
            video_filename = f"{video['video_id']}.mp4"
            if video_filename not in successfully_downloaded:
                continue
            for q in video["questions"]:
                entry = {
                    "video": video_filename,
                    "question": q["question"],
                    "gold_answer": _mcq_to_gold_answer(q),
                    # Preserve the original MCQ for LLM-judge eval later.
                    "mcq_answer": q["answer"],
                    "mcq_options": q["options"],
                    "task_type": q["task_type"],
                    "video_meta": {
                        "domain": video["domain"],
                        "sub_category": video["sub_category"],
                        "url": video["url"],
                    },
                }
                f.write(json.dumps(entry) + "\n")
                written += 1
    return written


# --- public entry point -----------------------------------------------------


def main(
    output_dir: str | Path,
    n: int = 10,
    duration: str = "short",
    domains: set[str] | None = None,
    task_types: set[str] | None = None,
    max_height: int = 360,
    hf_token: str | None = None,
    skip_existing: bool = True,
) -> Path:
    """Download Video-MME subset + write qa.jsonl. Returns the qa.jsonl path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Video-MME metadata from HuggingFace...")
    ds = _load_metadata(hf_token=hf_token)
    print(f"  got {len(ds)} rows")

    selected = select_videos(
        ds, n=n, duration=duration, domains=domains, task_types=task_types
    )
    print(f"selected {len(selected)} videos (duration={duration})")

    downloaded: set[str] = set()
    for i, video in enumerate(selected, start=1):
        filename = f"{video['video_id']}.mp4"
        dest = output_dir / filename
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            print(f"  [{i}/{len(selected)}] skip (exists): {filename}")
            downloaded.add(filename)
            continue
        if not video.get("url"):
            print(f"  [{i}/{len(selected)}] skip (no url): {filename}")
            continue
        print(f"  [{i}/{len(selected)}] downloading {filename} <- {video['url']}")
        if _download_video(video["url"], dest, max_height=max_height):
            downloaded.add(filename)

    qa_path = output_dir / "qa.jsonl"
    written = write_qa_jsonl(selected, downloaded, qa_path)
    print(f"wrote {written} QA entries from {len(downloaded)} videos -> {qa_path}")
    return qa_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Video-MME subset for pmcache.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, default=10, help="Number of videos")
    parser.add_argument("--duration", default="short", choices=("short", "medium", "long"))
    parser.add_argument("--domains", nargs="*", help="Filter to these domains (default: cache-friendly set)")
    parser.add_argument("--task_types", nargs="*", help="Filter to these task types")
    parser.add_argument("--max_height", type=int, default=360, help="Max video height for yt-dlp")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (if dataset is gated)")
    args = parser.parse_args()
    main(
        args.output_dir,
        n=args.n,
        duration=args.duration,
        domains=set(args.domains) if args.domains else None,
        task_types=set(args.task_types) if args.task_types else None,
        max_height=args.max_height,
        hf_token=args.hf_token,
    )
