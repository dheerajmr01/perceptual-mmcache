"""Prepare demo videos + qa.jsonl for the pmcache benchmark.

Two modes:

  --mode synth  (default)
      Generates 5 deterministic synthetic mp4s via cv2, designed to
      stress each region of the perceptual cache:

        static.mp4        constant frame              all hit anchor
        slideshow.mp4     5 stills × 2s each          5 buckets (scene cuts)
        slow_pan.mp4      1-pixel/frame gradient pan  continuous near-anchor
        random_noise.mp4  fresh noise every frame     worst case (no hits)
        talking_head.mp4  mostly static + small ROI   most frames hit

      No network access needed — the full eval pipeline runs offline.

  --mode manifest --manifest path/to/manifest.json
      Downloads videos from a user-supplied JSON manifest. Schema:
          {
            "videos": [
              {
                "filename": "intro.mp4",
                "url": "https://example.com/intro.mp4",
                "questions": [
                  {"question": "...", "gold_answer": "..."},
                  ...
                ]
              },
              ...
            ]
          }

In both modes the output dir gets the mp4s plus a qa.jsonl with
3-5 questions per video.

Runnable as a script or as a function:
    python -m eval.datasets.prepare_demo_videos --output_dir /tmp/vids
    from eval.datasets.prepare_demo_videos import main
    main("/tmp/vids", mode="synth")
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Synthetic-video generation
# -----------------------------------------------------------------------------


def _writer(path: Path, fps: int, size: int):
    """Open a cv2 VideoWriter with the mp4v fourcc."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (size, size))


def _gen_static(path: Path, fps: int = 30, seconds: int = 4, size: int = 128) -> None:
    """Constant gray frame. Every frame should alias to the same anchor."""
    w = _writer(path, fps, size)
    frame = np.full((size, size, 3), 128, dtype=np.uint8)
    for _ in range(fps * seconds):
        w.write(frame)
    w.release()


def _gen_slideshow(path: Path, fps: int = 30, seconds_per_slide: int = 2, size: int = 128) -> None:
    """5 distinct full-color slides, 2s each. Exercises scene cuts."""
    w = _writer(path, fps, size)
    palette = [
        (40, 40, 200),    # red-ish
        (40, 200, 40),    # green-ish
        (200, 200, 40),   # yellow-ish
        (40, 200, 200),   # cyan-ish
        (200, 40, 200),   # magenta-ish
    ]
    for r, g, b in palette:
        frame = np.full((size, size, 3), (b, g, r), dtype=np.uint8)  # cv2 = BGR
        for _ in range(fps * seconds_per_slide):
            w.write(frame)
    w.release()


def _gen_slow_pan(path: Path, fps: int = 30, seconds: int = 4, size: int = 128) -> None:
    """Gradient that pans 1px/frame. Adjacent frames near-anchor."""
    w = _writer(path, fps, size)
    # Make a 256-wide gradient and slide a `size`-wide window across it.
    total = fps * seconds
    src = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (size, 1))
    src = np.stack([src, src, src], axis=-1)
    src = np.concatenate([src, src], axis=1)  # tile so we don't run out
    for i in range(total):
        x = i % (src.shape[1] - size)
        frame = src[:, x : x + size, :]
        w.write(frame)
    w.release()


def _gen_random_noise(path: Path, fps: int = 30, seconds: int = 4, size: int = 128) -> None:
    """Fresh uniform noise per frame. Worst case for perceptual cache."""
    w = _writer(path, fps, size)
    rng = np.random.RandomState(0)
    for _ in range(fps * seconds):
        frame = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
        w.write(frame)
    w.release()


def _gen_talking_head(path: Path, fps: int = 30, seconds: int = 4, size: int = 128) -> None:
    """Mostly static frame with a small animated ROI ('mouth')."""
    import cv2

    w = _writer(path, fps, size)
    base = np.full((size, size, 3), 90, dtype=np.uint8)
    cv2.circle(base, (size // 2, size // 2 - 10), 30, (200, 180, 160), -1)  # face
    for i in range(fps * seconds):
        frame = base.copy()
        # Small mouth that opens/closes — affects ~1% of pixels
        opening = abs((i % fps) - fps // 2) * 0.6 / fps
        cv2.ellipse(
            frame,
            (size // 2, size // 2 + 15),
            (12, max(1, int(opening * 30))),
            0, 0, 360, (40, 40, 40), -1,
        )
        w.write(frame)
    w.release()


# -----------------------------------------------------------------------------
# QA for synthetic videos — paired with each generator above
# -----------------------------------------------------------------------------

_SYNTH_QA: dict[str, list[dict[str, str]]] = {
    "static.mp4": [
        {"question": "Is anything moving in this video?", "gold_answer": "no"},
        {"question": "What dominant color fills the screen?", "gold_answer": "gray"},
        {"question": "Is the camera static?", "gold_answer": "yes"},
    ],
    "slideshow.mp4": [
        {"question": "How many distinct slides are shown?", "gold_answer": "5"},
        {"question": "Are the colors of the slides identical?", "gold_answer": "no"},
        {"question": "Is this a continuous scene or multiple cuts?", "gold_answer": "cuts"},
    ],
    "slow_pan.mp4": [
        {"question": "Is the camera moving?", "gold_answer": "yes"},
        {"question": "Are the frames changing gradually?", "gold_answer": "yes"},
        {"question": "Is there a sudden scene cut?", "gold_answer": "no"},
    ],
    "random_noise.mp4": [
        {"question": "Is there a clear subject?", "gold_answer": "no"},
        {"question": "Is each frame visually distinct from the previous one?", "gold_answer": "yes"},
        {"question": "Could you describe a recognizable object?", "gold_answer": "no"},
    ],
    "talking_head.mp4": [
        {"question": "Is there a person-like figure visible?", "gold_answer": "yes"},
        {"question": "Is the camera moving?", "gold_answer": "no"},
        {"question": "Is most of the frame static across time?", "gold_answer": "yes"},
    ],
}


_SYNTH_GENERATORS = {
    "static.mp4": _gen_static,
    "slideshow.mp4": _gen_slideshow,
    "slow_pan.mp4": _gen_slow_pan,
    "random_noise.mp4": _gen_random_noise,
    "talking_head.mp4": _gen_talking_head,
}


# -----------------------------------------------------------------------------
# Manifest-mode downloader
# -----------------------------------------------------------------------------


def _download(url: str, dest: Path, timeout: float = 30.0) -> None:
    """Stream `url` to `dest` with a basic UA. Skips if dest exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "pmcache/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while chunk := r.read(8192):
            f.write(chunk)


def _from_manifest(manifest_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    """Download all videos in the manifest and return the qa entries."""
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    qa_entries: list[dict[str, Any]] = []
    for v in spec["videos"]:
        dest = output_dir / v["filename"]
        print(f"  downloading {v['filename']} <- {v['url']}")
        _download(v["url"], dest)
        for q in v.get("questions", []):
            qa_entries.append({"video": v["filename"], **q})
    return qa_entries


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def main(
    output_dir: str | Path,
    mode: str = "synth",
    manifest: str | Path | None = None,
) -> Path:
    """Populate `output_dir` with demo videos + qa.jsonl. Returns qa.jsonl path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_entries: list[dict[str, Any]] = []

    if mode == "synth":
        print(f"synth-mode: writing {len(_SYNTH_GENERATORS)} videos to {output_dir}")
        for filename, generator in _SYNTH_GENERATORS.items():
            dest = output_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  skip (exists): {filename}")
            else:
                print(f"  generating: {filename}")
                generator(dest)
            for q in _SYNTH_QA[filename]:
                qa_entries.append({"video": filename, **q})

    elif mode == "manifest":
        if manifest is None:
            raise ValueError("manifest path required when mode='manifest'")
        qa_entries = _from_manifest(Path(manifest), output_dir)

    else:
        raise ValueError(f"unknown mode: {mode!r} (use 'synth' or 'manifest')")

    qa_path = output_dir / "qa.jsonl"
    with open(qa_path, "w", encoding="utf-8") as f:
        for entry in qa_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"wrote {len(qa_entries)} QA entries -> {qa_path}")
    return qa_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/generate pmcache demo videos.")
    parser.add_argument("--output_dir", required=True, help="Where to write videos + qa.jsonl")
    parser.add_argument("--mode", default="synth", choices=("synth", "manifest"))
    parser.add_argument("--manifest", default=None, help="JSON manifest (manifest mode only)")
    args = parser.parse_args()
    main(args.output_dir, mode=args.mode, manifest=args.manifest)
