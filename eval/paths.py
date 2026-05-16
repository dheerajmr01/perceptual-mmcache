"""Central path constants for the pmcache benchmark harness.

Set the workspace once — either via the `PMCACHE_WORKSPACE` env var or
by calling `set_workspace(path)` from code — and every downstream
path (videos, results, sweep, report) resolves consistently.

Typical use in a Colab notebook or a script:

    from eval import paths
    paths.set_workspace("/content/drive/MyDrive/pmcache")

    from eval.run_baseline import run_baseline_benchmark
    run_baseline_benchmark(
        videos_dir=paths.VIDEOS_DIR,
        qa_file=paths.QA_FILE,
        output_path=paths.BASELINE_PATH,
        model="Qwen/Qwen2-VL-2B-Instruct",
    )

Defaults (when `PMCACHE_WORKSPACE` is unset):
    workspace      = ./workspace
    VIDEOS_DIR     = ./workspace/videos
    RESULTS_DIR    = ./workspace/results
    SWEEP_DIR      = ./workspace/results/sweep
    REPORT_DIR     = ./workspace/results/report
    QA_FILE        = ./workspace/videos/qa.jsonl
    BASELINE_PATH  = ./workspace/results/baseline.jsonl
    PERCEPTUAL_PATH = ./workspace/results/perceptual.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path

# Module-level mutables. Importers re-read these after set_workspace().
WORKSPACE: Path
VIDEOS_DIR: Path
RESULTS_DIR: Path
SWEEP_DIR: Path
REPORT_DIR: Path
QA_FILE: Path
BASELINE_PATH: Path
PERCEPTUAL_PATH: Path
SMOKE_TEST_OUTPUT: Path


def _resolve(base: Path) -> dict[str, Path]:
    return {
        "WORKSPACE":        base,
        "VIDEOS_DIR":       base / "videos",
        "RESULTS_DIR":      base / "results",
        "SWEEP_DIR":        base / "results" / "sweep",
        "REPORT_DIR":       base / "results" / "report",
        "QA_FILE":          base / "videos" / "qa.jsonl",
        "BASELINE_PATH":    base / "results" / "baseline.jsonl",
        "PERCEPTUAL_PATH":  base / "results" / "perceptual.jsonl",
        "SMOKE_TEST_OUTPUT": base / "smoke.json",
    }


def set_workspace(path: str | os.PathLike) -> None:
    """Reset all path constants to be rooted at `path`.

    Does NOT create the directories — call `ensure_dirs()` for that.
    Idempotent and safe to call multiple times.
    """
    resolved = _resolve(Path(path))
    globals().update(resolved)


def ensure_dirs() -> None:
    """Create WORKSPACE + every subdirectory if missing."""
    for key in ("WORKSPACE", "VIDEOS_DIR", "RESULTS_DIR", "SWEEP_DIR", "REPORT_DIR"):
        Path(globals()[key]).mkdir(parents=True, exist_ok=True)


def workspace() -> Path:
    """Current workspace root."""
    return WORKSPACE


# Initialize at import. Honor the env var if set, else use ./workspace
# relative to the current working directory.
set_workspace(os.environ.get("PMCACHE_WORKSPACE", "workspace"))
