"""eval.run_baseline — vLLM + LMCache with bytewise mm_hash only.

Drives the standard LMCache integration (no perceptual shim) across the
QA dataset and writes per-frame metrics to JSONL. Adjacent video frames
are byte-different even when ~99% similar, so this run should produce
near-0% cross-frame cache hit rate — the baseline pmcache aims to beat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval._runner import run_benchmark


def run_baseline_benchmark(
    videos_dir: str | Path,
    qa_file: str | Path,
    output_path: str | Path,
    model: str,
    fps: float = 1.0,
    mock_vlm: bool = False,
    max_model_len: int | None = None,
    vllm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the bytewise-mm_hash baseline. Returns a summary dict.

    See `eval._runner.run_benchmark` for the JSONL schema.
    """
    return run_benchmark(
        videos_dir=videos_dir,
        qa_file=qa_file,
        output_path=output_path,
        model=model,
        pmcache=None,
        fps=fps,
        mock_vlm=mock_vlm,
        variant_label="baseline",
        max_model_len=max_model_len,
        vllm_kwargs=vllm_kwargs,
    )
