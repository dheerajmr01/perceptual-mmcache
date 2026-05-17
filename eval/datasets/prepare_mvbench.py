"""Prepare an MVBench subset for pmcache benchmarks.

MVBench (`OpenGVLab/MVBench`) is a 20-task multiple-choice video QA
benchmark hosted on HuggingFace. Each task config has ~200 entries; each
entry is one (video, question, candidates, answer) tuple. Unlike
Video-MME (3 questions per video), MVBench questions are mostly 1:1
with their video clip — but the clips are short (often <30s), MCQ-formatted,
and pulled from sources where the perceptual-similarity signal pmcache
exploits is plausible (Charades, Kinetics-style action clips, etc.).

## What we filter for

`pmcache` wins on **low cross-frame visual change** and benefits more
from many *cold-start* runs than from many warm-replay runs (see the
project README on the metric anatomy). MVBench gives us many short
cold-start clips, so by default we pick from tasks whose source
material trends to single-shot or static-camera footage:

  - `object_existence`, `object_interaction`, `object_shuffle`     (Charades / STAR)
  - `action_sequence`, `action_prediction`, `action_localization`  (Charades / STAR / Sta)
  - `scene_transition`, `episodic_reasoning`                       (TVQA / ScanNet)

Override via `--tasks` if you want different tasks.

## Output

For each downloaded video we append entries to `qa.jsonl` in the
pmcache schema, normalizing the MCQ structure to match
`prepare_videomme.py`:

    {
      "video": "<filename>.mp4",
      "question": "...",
      "gold_answer": "the cat",          # text of the correct candidate
      "mcq_answer": "B",                 # derived letter (A/B/C/...)
      "mcq_options": ["A. dog", "B. the cat", ...],
      "task_type": "object_existence",
      "video_meta": {"task": "...", "hf_repo": "OpenGVLab/MVBench"},
    }

## Two download modes

  1. **Pre-downloaded local videos** — pass `--video_src_dir <path>`
     and we search it recursively for each entry's video filename.
     Recommended for repeat runs (no re-download per task).

  2. **On-the-fly HF download** — without `--video_src_dir`, we pull
     each video via `huggingface_hub.hf_hub_download` from the
     `OpenGVLab/MVBench` dataset repo, using a known task→subfolder
     map (`TASK_VIDEO_DIRS`). Some tasks are gated/missing in that
     map; those skip with a warning. Use `--video_src_dir` for those.

## Usage

    pip install -e '.[mvbench]'

    # Default: 10 videos across the cache-friendly task set.
    python -m eval.datasets.prepare_mvbench --output_dir workspace/videos

    # One specific task, more videos.
    python -m eval.datasets.prepare_mvbench --output_dir workspace/videos \\
        --tasks object_existence --n 30

    # Use pre-downloaded videos (recommended for repeat runs).
    python -m eval.datasets.prepare_mvbench --output_dir workspace/videos \\
        --video_src_dir /content/mvbench_videos --tasks object_interaction
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from pmcache._lazy import require


# ---------------------------------------------------------------------------
# MVBench task → video subfolder map (within OpenGVLab/MVBench HF repo)
# ---------------------------------------------------------------------------
#
# Derived from the official MVBench README + dataset.json. NOT every
# task is present here — some live in externally-hosted source datasets
# that MVBench doesn't redistribute (Kinetics, NTU RGB+D). Pass
# `--video_src_dir` for those.

TASK_VIDEO_DIRS: dict[str, str] = {
    # STAR / Charades
    "action_sequence":     "video/star/Charades_v1_480/",
    "action_prediction":   "video/star/Charades_v1_480/",
    "action_localization": "video/sta/sta_video/",
    "object_existence":    "video/charades/Charades_v1_480/",
    "object_interaction":  "video/star/Charades_v1_480/",
    # Something-Something v2
    "action_antonym":      "video/ssv2_video/",
    # Perception Test
    "object_shuffle":      "video/perception/videos/",
    "action_count":        "video/perception/videos/",
    "state_change":        "video/perception/videos/",
    "character_order":     "video/perception/videos/",
    # CLEVRER
    "moving_count":        "video/clevrer/video_validation/",
    "moving_direction":    "video/clevrer/video_validation/",
    "moving_attribute":    "video/clevrer/video_validation/",
    "counterfactual_inference": "video/clevrer/video_validation/",
    # FunQA (unexpected actions)
    "unexpected_action":   "video/FunQA_test/test/",
    # ScanNet (3D scene)
    "scene_transition":    "video/scene_qa/video/",
    # TVQA (long-form TV)
    "episodic_reasoning":  "video/tvqa/frames_fps3_hq/",
    # Tasks NOT mapped (require --video_src_dir):
    #   fine_grained_action  — Moments_in_Time
    #   fine_grained_pose    — NTU RGB+D
    #   egocentric_navigation — VLN-QA
}

DEFAULT_TASKS_PREFERRED: tuple[str, ...] = (
    "object_existence",
    "object_interaction",
    "action_sequence",
    "action_prediction",
    "scene_transition",
)

DATASET_REPO = "OpenGVLab/MVBench"


# ---------------------------------------------------------------------------
# HF metadata loading
# ---------------------------------------------------------------------------


def _load_task(task: str, hf_token: str | None = None) -> list[dict[str, Any]]:
    """Return MVBench rows for one task as a list of dicts.

    Loads `datasets.load_dataset("OpenGVLab/MVBench", task, split="test")`
    (MVBench only has one split, named `test`). Materializes to a list of
    plain dicts so callers can iterate without holding the HF Dataset
    wrapper open.
    """
    datasets = require("datasets", "mvbench")
    ds = datasets.load_dataset(DATASET_REPO, task, split="test", token=hf_token)
    rows: list[dict[str, Any]] = []
    for r in ds:
        # Every config exposes `video`, `question`, `candidates`, `answer`.
        # Some configs add `start`/`end`/`bound`/`task_type` etc. We pass
        # through what's present and stamp `task_type` from the task name
        # if missing.
        item = dict(r)
        item.setdefault("task_type", task)
        rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# MCQ normalization (candidates list → letter-prefixed options + letter answer)
# ---------------------------------------------------------------------------


def _normalize_mcq(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an MVBench entry's {candidates, answer} into the pmcache shape.

    Returns a dict with `mcq_options`, `mcq_answer` (letter), `gold_answer`
    (text of correct candidate), or **None** when the entry is malformed
    (e.g., empty candidates, answer doesn't match any candidate).

    Strategy for finding the correct letter:
      1. If `answer` is already a letter A–Z, use it directly.
      2. Otherwise, find the candidate whose text matches `answer` (exact
         first, then case-insensitive strip-match, then substring).
      3. Bail (return None) if no candidate matches — we'd rather skip
         the row than grade against a non-existent letter.
    """
    candidates = entry.get("candidates") or []
    answer = entry.get("answer")
    if not candidates or answer is None:
        return None

    # MVBench answers are usually full-text. But occasionally a config
    # gives a letter or an int index — handle both defensively.
    correct_idx: int | None = None
    if isinstance(answer, int):
        if 0 <= answer < len(candidates):
            correct_idx = answer
    elif isinstance(answer, str):
        a = answer.strip()
        is_letter = len(a) == 1 and a.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if is_letter:
            letter_idx = ord(a.upper()) - ord("A")
            if letter_idx < len(candidates):
                correct_idx = letter_idx
            # Single-letter answer that doesn't index into `candidates`
            # is unrecoverable — do NOT fall through to substring match
            # (a bare "E" would substring-match "red" via the lone 'e').
        else:
            # Multi-char text — try exact, then case-insensitive, then
            # bidirectional substring (tolerates trailing punctuation).
            for i, c in enumerate(candidates):
                if str(c).strip() == a:
                    correct_idx = i
                    break
            if correct_idx is None:
                a_low = a.casefold()
                for i, c in enumerate(candidates):
                    if str(c).strip().casefold() == a_low:
                        correct_idx = i
                        break
            if correct_idx is None:
                a_low = a.casefold()
                for i, c in enumerate(candidates):
                    cl = str(c).strip().casefold()
                    if cl and (cl in a_low or a_low in cl):
                        correct_idx = i
                        break

    if correct_idx is None:
        return None

    letter = chr(ord("A") + correct_idx)
    mcq_options = [
        f"{chr(ord('A') + i)}. {str(c).strip()}" for i, c in enumerate(candidates)
    ]
    gold = str(candidates[correct_idx]).strip()
    return {
        "mcq_options": mcq_options,
        "mcq_answer": letter,
        "gold_answer": gold,
    }


# ---------------------------------------------------------------------------
# Selection (pure function — easy to test)
# ---------------------------------------------------------------------------


def select_entries(
    rows: Iterable[dict[str, Any]],
    n: int = 10,
    max_questions_per_video: int = 5,
) -> list[dict[str, Any]]:
    """Pick up to `n` distinct videos from `rows`, grouping questions by video.

    Returns a list of `{video_filename, task_type, questions: [...]}` dicts.
    Each `questions` entry already carries normalized MCQ fields. Entries
    that fail `_normalize_mcq` (malformed candidates / answer) are dropped.

    `max_questions_per_video` caps how many questions we keep per video —
    MVBench is mostly 1 Q/video but some sources have multiple questions
    sharing a clip, and unlimited would skew the resulting qa.jsonl.
    """
    per_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_field = row.get("video") or row.get("video_path")
        if not video_field:
            continue
        # Normalize: keep just the filename (some configs include subdirs).
        video_filename = Path(str(video_field)).name
        # Some MVBench configs reference video frame folders (e.g., TVQA
        # gives "house_md_season_05_episode_19_clip_03" which is a
        # frame-folder name, not a single .mp4). We skip those — the
        # eval harness expects single video files.
        if not Path(video_filename).suffix:
            continue

        norm = _normalize_mcq(row)
        if norm is None:
            continue

        bucket = per_video.setdefault(video_filename, {
            "video_filename": video_filename,
            "task_type": row.get("task_type"),
            "video_field": str(video_field),  # original (may include subdirs)
            "questions": [],
        })
        if len(bucket["questions"]) >= max_questions_per_video:
            continue
        bucket["questions"].append({
            "question": row.get("question") or row.get("query") or "",
            "task_type": row.get("task_type"),
            **norm,
        })

    # Keep stable ordering by video filename so reruns are deterministic.
    candidates = sorted(per_video.values(), key=lambda v: v["video_filename"])
    return [v for v in candidates if v["questions"]][:n]


# ---------------------------------------------------------------------------
# Video sourcing — local search OR HF hub download
# ---------------------------------------------------------------------------


def _find_local_video(video_filename: str, src_dir: Path) -> Path | None:
    """Recursively search `src_dir` for a file matching `video_filename`."""
    # Common cases first: exact-path match, then immediate-child match.
    direct = src_dir / video_filename
    if direct.is_file():
        return direct
    # Recursive search — bounded by what's in the dir, no fancy globbing.
    # MVBench videos are organized in shallow subfolders.
    matches = list(src_dir.rglob(video_filename))
    if matches:
        return matches[0]
    # Try matching stem (filename without extension), in case the source
    # has a different container — only when extension differs.
    stem = Path(video_filename).stem
    for ext in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
        alt = src_dir / f"{stem}{ext}"
        if alt.is_file():
            return alt
        alt_matches = list(src_dir.rglob(f"{stem}{ext}"))
        if alt_matches:
            return alt_matches[0]
    return None


def _hf_download_video(
    task: str,
    video_filename: str,
    dest: Path,
    hf_token: str | None = None,
) -> bool:
    """Download `video_filename` for `task` from the MVBench HF repo.

    Returns True if `dest` exists with non-zero size afterward. Skips
    silently (returns False) if the task isn't mapped — the caller is
    expected to fall back to `--video_src_dir`.
    """
    subdir = TASK_VIDEO_DIRS.get(task)
    if not subdir:
        return False
    hub = require("huggingface_hub", "mvbench")
    # Try the canonical task subfolder first, then a few common variants
    # (some MVBench configs put videos one level deeper).
    candidates = [
        f"{subdir}{video_filename}",
        f"{subdir.rstrip('/')}/videos/{video_filename}",
    ]
    for hub_path in candidates:
        try:
            local = hub.hf_hub_download(
                repo_id=DATASET_REPO,
                filename=hub_path,
                repo_type="dataset",
                token=hf_token,
            )
        except Exception:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # `hf_hub_download` returns a cached path; copy into our dest so
        # the eval harness sees a stable filename in `videos_dir`.
        src = Path(local)
        if src.resolve() == dest.resolve():
            return dest.is_file() and dest.stat().st_size > 0
        try:
            dest.write_bytes(src.read_bytes())
        except OSError as e:
            print(f"  ! could not copy {src} -> {dest}: {e}", file=sys.stderr)
            return False
        return dest.is_file() and dest.stat().st_size > 0
    return False


# ---------------------------------------------------------------------------
# qa.jsonl writer
# ---------------------------------------------------------------------------


def write_qa_jsonl(
    selected: list[dict[str, Any]],
    successfully_sourced: set[str],
    qa_path: Path,
) -> int:
    """Write one JSONL row per (video, question) for sourced videos."""
    written = 0
    with open(qa_path, "w", encoding="utf-8") as f:
        for video in selected:
            vf = video["video_filename"]
            if vf not in successfully_sourced:
                continue
            for q in video["questions"]:
                entry = {
                    "video": vf,
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "mcq_answer": q["mcq_answer"],
                    "mcq_options": q["mcq_options"],
                    "task_type": q.get("task_type") or video.get("task_type"),
                    "video_meta": {
                        "task": video.get("task_type"),
                        "hf_repo": DATASET_REPO,
                    },
                }
                f.write(json.dumps(entry) + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(
    output_dir: str | Path,
    n: int = 10,
    tasks: Iterable[str] | None = None,
    video_src_dir: str | Path | None = None,
    max_questions_per_video: int = 5,
    hf_token: str | None = None,
    skip_existing: bool = True,
) -> Path:
    """Download an MVBench subset + write qa.jsonl. Returns the qa.jsonl path.

    Args:
        output_dir: Where to place qa.jsonl + downloaded videos.
        n: Number of videos to keep TOTAL across all selected tasks.
        tasks: Iterable of MVBench task names. None → `DEFAULT_TASKS_PREFERRED`.
        video_src_dir: If set, search this directory recursively for each
            video instead of downloading. Skips HF download entirely.
        max_questions_per_video: Cap per-video question count.
        hf_token: HF token for gated metadata access (most configs are public).
        skip_existing: Don't re-download / re-copy videos already on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(video_src_dir) if video_src_dir else None
    chosen_tasks = list(tasks) if tasks else list(DEFAULT_TASKS_PREFERRED)

    # Pool metadata across all requested tasks, then pick `n` videos.
    pooled_rows: list[dict[str, Any]] = []
    for task in chosen_tasks:
        print(f"loading MVBench task '{task}' from {DATASET_REPO}...")
        try:
            task_rows = _load_task(task, hf_token=hf_token)
        except Exception as e:
            print(f"  ! failed to load task '{task}': {e}", file=sys.stderr)
            continue
        print(f"  got {len(task_rows)} rows")
        pooled_rows.extend(task_rows)

    selected = select_entries(
        pooled_rows, n=n, max_questions_per_video=max_questions_per_video
    )
    print(f"selected {len(selected)} videos across {len(chosen_tasks)} task(s)")

    sourced: set[str] = set()
    for i, video in enumerate(selected, start=1):
        vf = video["video_filename"]
        dest = output_dir / vf
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            print(f"  [{i}/{len(selected)}] skip (exists): {vf}")
            sourced.add(vf)
            continue
        if src_dir is not None:
            found = _find_local_video(vf, src_dir)
            if found is None:
                print(f"  [{i}/{len(selected)}] not found in {src_dir}: {vf}",
                      file=sys.stderr)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_bytes(found.read_bytes())
            except OSError as e:
                print(f"  ! copy failed for {vf}: {e}", file=sys.stderr)
                continue
            print(f"  [{i}/{len(selected)}] copied {found.name}")
            sourced.add(vf)
            continue
        # HF download fallback.
        task = video.get("task_type") or ""
        print(f"  [{i}/{len(selected)}] downloading {vf} (task={task})")
        if _hf_download_video(task, vf, dest, hf_token=hf_token):
            sourced.add(vf)
        else:
            print(
                f"  ! could not source {vf} (task '{task}' not mapped or HF "
                f"path missing). Re-run with --video_src_dir.",
                file=sys.stderr,
            )

    qa_path = output_dir / "qa.jsonl"
    written = write_qa_jsonl(selected, sourced, qa_path)
    print(f"wrote {written} QA entries from {len(sourced)} videos -> {qa_path}")
    return qa_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an MVBench subset for pmcache.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, default=10, help="Total videos to keep")
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="MVBench task names (default: cache-friendly preset)",
    )
    parser.add_argument(
        "--video_src_dir", default=None,
        help="Pre-downloaded MVBench videos root (skips HF download)",
    )
    parser.add_argument(
        "--max_questions_per_video", type=int, default=5,
        help="Cap per-video question count",
    )
    parser.add_argument("--hf_token", default=None, help="HF token (if gated)")
    args = parser.parse_args()
    main(
        args.output_dir,
        n=args.n,
        tasks=args.tasks,
        video_src_dir=args.video_src_dir,
        max_questions_per_video=args.max_questions_per_video,
        hf_token=args.hf_token,
    )
