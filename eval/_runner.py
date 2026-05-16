"""Shared benchmark loop used by run_baseline + run_perceptual.

Public API:
    run_benchmark(videos_dir, qa_file, output_path, model,
                  pmcache=None, fps=1.0, mock_vlm=False) -> dict

Writes one JSONL line per (video, question) pair containing per-frame
arrays, then returns a summary dict. JSONL schema:

    {
      "video": "static.mp4",
      "question": "Is anything moving?",
      "gold_answer": "no",
      "answer": "yes",
      "num_frames": 30,
      "mm_hashes": ["hex0", "hex1", ...],   # bytewise (baseline) or aliased (perceptual)
      "cache_hits": int, "cache_misses": int,
      "ttft_s": float,
      "tier1_hits": int, "tier2_hits": int, "tier2_rejects": int,  # 0 for baseline
      "misses": int, "aliased_hashes": int,                        # pmcache counters
      "variant": "baseline" | "perceptual",
    }
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.utils import (
    iter_videos,
    kv_bytes_per_token,
    load_qa,
    load_vlm,
    sample_frames,
    vllm_generate_multimodal,
)

_MCQ_LETTER_RE = re.compile(r"\b([A-D])\b")

if TYPE_CHECKING:
    from PIL import Image

    from pmcache.lmcache_shim import PerceptualMMCache


def _bytewise_hash(image: "Image.Image") -> str:
    """Mimic vLLM's MultiModalHasher behavior on a PIL Image.

    vLLM serializes `{"mode", "data": raw pixel array}` and runs
    BLAKE-something. We approximate with BLAKE2b on the same bytes,
    which is enough to reproduce LMCache's "byte-different → cache
    miss" pattern in mock mode. In real mode (mock_vlm=False) we don't
    use this — vLLM does the real hashing internally.
    """
    raw = image.tobytes()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _format_mcq_prompt(question: str, options: list[str]) -> str:
    """Build a Video-MME-style MCQ prompt.

    Options come in as `["A. text", "B. text", ...]` — already letter-prefixed.
    The trailing instruction nudges the model to emit just the letter so
    `_parse_letter` can grade reliably.
    """
    opts = "\n".join(options)
    return (
        f"{question}\n{opts}\n"
        "Answer with only the letter (A, B, C, or D) of the correct option."
    )


def _parse_letter(text: str) -> str | None:
    """Extract the first standalone A/B/C/D letter from `text`.

    Returns the uppercase letter or None. Handles common formats:
    `"A"`, `"A."`, `"The answer is B"`, `"(C)"`. Falls back to None when
    the model rambled without picking an option.
    """
    if not text:
        return None
    m = _MCQ_LETTER_RE.search(text.upper())
    return m.group(1) if m else None


def _frame_hash(image: "Image.Image") -> str:
    """Hash an (already-tagged-or-not) frame the way LMCache will see it.

    If the image carries an EXIF ImageID UUID (the perceptual shim
    tagged it), use those bytes — that's what vLLM's MultiModalHasher
    will use too. Otherwise fall back to the bytewise pixel hash.
    """
    from pmcache.lmcache_shim import _existing_uuid

    uid = _existing_uuid(image)
    if uid is not None:
        return uid.hex
    return _bytewise_hash(image)


def run_benchmark(
    videos_dir: str | Path,
    qa_file: str | Path,
    output_path: str | Path,
    model: str,
    pmcache: "PerceptualMMCache | None" = None,
    fps: float = 1.0,
    mock_vlm: bool = False,
    reset_pmcache_per_video: bool = False,
    variant_label: str | None = None,
    max_model_len: int | None = None,
    vllm_kwargs: dict[str, Any] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Drive vLLM/MockVLM over the QA set. Returns a summary dict.

    Args:
        pmcache: If None, this is the baseline run. If provided, every
            sampled frame is fed through `pmcache.prepare_image` before
            the VLM call (EXIF-tagged with the appropriate anchor UUID).
        reset_pmcache_per_video: If True (and pmcache is not None),
            call `pmcache.reset()` between videos so each video is
            benchmarked cold. Default False matches production behavior.
        variant_label: Override the "variant" field in the JSONL.
            Defaults to "perceptual" when pmcache is set, else "baseline".
        max_model_len: Override vLLM's max sequence length. Needed when
            sampling many frames per video (fps × duration × ~1k image
            tokens/frame can exceed the model's default 32768).
        vllm_kwargs: Extra kwargs forwarded to `vllm.LLM(...)` (e.g.
            `gpu_memory_utilization`, `tensor_parallel_size`).
    """
    videos_dir = Path(videos_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    variant = variant_label or ("perceptual" if pmcache is not None else "baseline")
    vlm_extra: dict[str, Any] = dict(vllm_kwargs or {})
    if max_model_len is not None:
        vlm_extra.setdefault("max_model_len", max_model_len)
    llm = load_vlm(model, mock_vlm=mock_vlm, **vlm_extra)
    bytes_per_tok = 0 if mock_vlm else kv_bytes_per_token(llm)
    qa = load_qa(qa_file)

    # Cache frames per video so we don't re-decode for every question on
    # the same video.
    frame_cache: dict[str, list["Image.Image"]] = {}
    available_videos = {p.name for p in iter_videos(videos_dir)}

    n_rows = 0
    t_run_start = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as out:
        for entry in qa:
            video_name = entry["video"]
            if video_name not in available_videos:
                # Skip QA rows for videos we don't have on disk.
                continue

            if video_name not in frame_cache:
                if pmcache is not None and reset_pmcache_per_video:
                    pmcache.reset()
                raw_frames = sample_frames(
                    videos_dir / video_name, fps=fps, max_frames=max_frames
                )
                if pmcache is not None:
                    raw_frames = [
                        pmcache.prepare_image(f, video_id=video_name) for f in raw_frames
                    ]
                frame_cache[video_name] = raw_frames

            frames = frame_cache[video_name]
            hashes = [_frame_hash(f) for f in frames]

            mcq_options = entry.get("mcq_options")
            mcq_answer = entry.get("mcq_answer")
            if mcq_options:
                prompt = _format_mcq_prompt(entry["question"], mcq_options)
            else:
                prompt = entry["question"]

            # MockVLM uses hashes to emulate LMCache hit/miss; real vLLM
            # ignores the kwarg (we don't pass it through).
            gen_stats: dict[str, int] = {
                "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0
            }
            if mock_vlm:
                result = llm.generate(prompt, frames, image_hashes=hashes)
                answer = result["answer"]
                ttft_s = result["ttft_s"]
                cache_hits = result["cache_hits"]
                cache_misses = result["cache_misses"]
            else:
                answer, ttft_s, gen_stats = vllm_generate_multimodal(
                    llm, question=prompt, images=frames
                )
                # Without LMCache log-scraping (deferred), report hits/misses
                # via hash repetition within this single call.
                cache_hits = len(hashes) - len(set(hashes))
                cache_misses = len(set(hashes))

            predicted_letter = _parse_letter(answer) if mcq_options else None
            correct: bool | None = None
            if mcq_options and mcq_answer:
                correct = predicted_letter == mcq_answer.strip().upper()

            prompt_tokens = gen_stats.get("prompt_tokens", 0)
            cached_tokens = gen_stats.get("cached_tokens", 0)
            kv_bytes_total = prompt_tokens * bytes_per_tok
            kv_bytes_cached = cached_tokens * bytes_per_tok
            kv_bytes_recomputed = max(0, kv_bytes_total - kv_bytes_cached)

            row: dict[str, Any] = {
                "video": video_name,
                "question": entry["question"],
                "gold_answer": entry.get("gold_answer"),
                "answer": answer,
                "num_frames": len(frames),
                "mm_hashes": hashes,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "ttft_s": ttft_s,
                "variant": variant,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "output_tokens": gen_stats.get("output_tokens", 0),
                "kv_bytes_total": kv_bytes_total,
                "kv_bytes_cached": kv_bytes_cached,
                "kv_bytes_recomputed": kv_bytes_recomputed,
                "kv_bytes_per_token": bytes_per_tok,
            }
            if mcq_options:
                row["mcq_options"] = mcq_options
                row["mcq_answer"] = mcq_answer
                row["predicted_letter"] = predicted_letter
                row["correct"] = correct
            # Snapshot pmcache metrics if present.
            if pmcache is not None:
                m = pmcache.metrics
                row.update(
                    tier1_hits=m.tier1_hits,
                    tier2_hits=m.tier2_hits,
                    tier2_rejects=m.tier2_rejects,
                    misses=m.misses,
                    aliased_hashes=m.aliased_hashes,
                    pmcache_hit_rate=m.hit_rate(),
                )
            else:
                row.update(
                    tier1_hits=0,
                    tier2_hits=0,
                    tier2_rejects=0,
                    misses=0,
                    aliased_hashes=0,
                    pmcache_hit_rate=0.0,
                )

            out.write(json.dumps(row) + "\n")
            n_rows += 1

    elapsed = time.perf_counter() - t_run_start
    summary = {
        "variant": variant,
        "rows_written": n_rows,
        "videos_seen": len(frame_cache),
        "elapsed_s": elapsed,
        "output_path": str(output_path),
    }
    if pmcache is not None:
        m = pmcache.metrics
        summary.update(
            frames_seen=m.frames_seen,
            tier1_hits=m.tier1_hits,
            tier2_hits=m.tier2_hits,
            tier2_rejects=m.tier2_rejects,
            aliased_hashes=m.aliased_hashes,
            pmcache_hit_rate=m.hit_rate(),
        )
    return summary
