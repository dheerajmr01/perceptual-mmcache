"""eval.run_perceptual — vLLM + LMCache with the perceptual shim enabled.

Same harness as run_baseline; additionally, every sampled frame is fed
through `PerceptualMMCache.prepare_image` which tags the PIL image with
an anchor UUID (EXIF `ImageID`). vLLM's `MultiModalHasher` reads that
UUID and uses it as the mm_hash, so LMCache sees an exact-match hit on
its existing path for adjacent / near-duplicate frames.

This is the variant we're benchmarking against the baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval._runner import run_benchmark
from pmcache.config import PMCacheConfig
from pmcache.lmcache_shim import PerceptualMMCache

if TYPE_CHECKING:
    from pmcache.verifier import DinoV2Verifier


def run_perceptual_benchmark(
    videos_dir: str | Path,
    qa_file: str | Path,
    output_path: str | Path,
    model: str,
    tau: float = 0.98,
    k: int = 5,
    fps: float = 1.0,
    mock_vlm: bool = False,
    verifier: "DinoV2Verifier | None" = None,
    reset_per_video: bool = False,
    max_model_len: int | None = None,
    vllm_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the perceptual variant. Returns a summary dict.

    Args:
        tau: DINOv2 cosine threshold (Tier 2).
        k: max Hamming distance for pHash match (Tier 1).
        verifier: optional pre-built verifier (lets the threshold sweep
            reuse one DinoV2 instance across multiple τ values; tests
            pass a FakeVerifier).
        reset_per_video: if True, buckets are cleared between videos so
            each video is benchmarked cold. Default False matches
            production behavior (cross-video bucket reuse can still
            help, e.g. on a slideshow used across many videos).
    """
    config = PMCacheConfig(enabled=True, tau=tau, k=k)
    pmcache = PerceptualMMCache(config, verifier=verifier)

    return run_benchmark(
        videos_dir=videos_dir,
        qa_file=qa_file,
        output_path=output_path,
        model=model,
        pmcache=pmcache,
        fps=fps,
        mock_vlm=mock_vlm,
        reset_pmcache_per_video=reset_per_video,
        variant_label="perceptual",
        max_model_len=max_model_len,
        vllm_kwargs=vllm_kwargs,
    )
