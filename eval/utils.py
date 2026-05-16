"""eval.utils — frame sampling, mock VLM, smoke test helper.

Public surface used by the Colab notebook + by every `run_*_benchmark`:
    - `sample_frames(video_path, fps)`        CPU-side, opencv-backed
    - `MockVLM`                                deterministic stand-in for vLLM
    - `load_vlm(model, mock_vlm)`              factory: real vLLM or MockVLM
    - `smoke_test_vlm(model, workspace)`       GPU pre-flight (Colab only)

VLM imports (torch, transformers, vllm) stay lazy — `import eval.utils`
must work on the CPU laptop without the [vlm] extra installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from pmcache._lazy import require

if TYPE_CHECKING:
    from PIL import Image


# ---------------------------------------------------------------------------
# Frame sampling — pure CPU, no VLM needed
# ---------------------------------------------------------------------------


def sample_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
) -> list["Image.Image"]:
    """Decode `video_path` and return PIL Images sampled at `fps`.

    Uses opencv-python-headless. Returns RGB PIL Images (cv2 reads BGR;
    we convert) so callers can feed them straight to vLLM / PIL EXIF
    tagging.

    fps=1.0 means one frame per video-second. If the source is 30 fps,
    we keep every 30th frame. Always emits at least 1 frame for any
    non-empty video.

    If `max_frames` is set and the fps-sampled list exceeds it, take a
    uniform-stride subsample so the kept frames span the full duration
    (better signal for video QA than just truncating to the first N).
    """
    import cv2  # imported lazily — opencv brings in a lot of native code
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / fps)))
        frames: list[Image.Image] = []
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb, mode="RGB"))
            idx += 1
        if max_frames is not None and len(frames) > max_frames > 0:
            n = len(frames)
            # Uniform-stride pick: indices spread across [0, n-1].
            picks = [round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)] \
                if max_frames > 1 else [n // 2]
            frames = [frames[i] for i in picks]
        return frames
    finally:
        cap.release()


def iter_videos(videos_dir: str | Path) -> Iterator[Path]:
    """Yield video files in `videos_dir`, sorted, with stable extensions."""
    exts = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
    root = Path(videos_dir)
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            yield p


# ---------------------------------------------------------------------------
# QA file loading
# ---------------------------------------------------------------------------


def load_qa(qa_file: str | Path) -> list[dict[str, str]]:
    """Read qa.jsonl into a list of {video, question, gold_answer} dicts."""
    out: list[dict[str, str]] = []
    with open(qa_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Mock VLM — used by every eval function when `mock_vlm=True`
# ---------------------------------------------------------------------------


class MockVLM:
    """Deterministic stand-in for vLLM, for CPU-only harness exercises.

    Behavior:
      - `generate(prompt, images)` returns a stable canned answer
        (defaults to "yes"); per-prompt overrides supported via
        `answers={prompt_substring: response, ...}`.
      - Records every call into `self.calls` for assertions.
      - Reports a synthetic TTFT proportional to the number of
        cache-miss images (so the eval harness's plotting code has
        non-trivial signal even on CPU).
    """

    def __init__(
        self,
        canned_answer: str = "yes",
        answers: dict[str, str] | None = None,
        per_image_ms: float = 5.0,
    ) -> None:
        self.canned_answer = canned_answer
        self.answers = answers or {}
        self.per_image_ms = per_image_ms
        self.calls: list[dict[str, Any]] = []
        # Tracks which mm_hashes we've "seen" — emulates LMCache hit
        # behavior at the harness level so mock runs produce sensible
        # hit/miss telemetry.
        self._seen_hashes: set[str] = set()

    def generate(
        self,
        prompt: str,
        images: list["Image.Image"],
        image_hashes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return {answer, ttft_s, cache_hits, cache_misses}.

        `image_hashes` lets the caller emulate the LMCache hit/miss
        decision: any hash we've previously seen counts as a hit (no
        synthetic latency); each new hash adds `per_image_ms` to TTFT.
        """
        start = time.perf_counter()
        hashes = image_hashes if image_hashes is not None else [
            str(id(img)) for img in images
        ]
        hits = sum(1 for h in hashes if h in self._seen_hashes)
        misses = len(hashes) - hits
        self._seen_hashes.update(hashes)

        # Synthetic latency only for misses (the hit path is "free").
        synthetic_delay = (misses * self.per_image_ms) / 1000.0
        time.sleep(min(synthetic_delay, 0.05))  # cap so tests stay fast
        ttft = time.perf_counter() - start

        answer = self.canned_answer
        for keyword, override in self.answers.items():
            if keyword.lower() in prompt.lower():
                answer = override
                break

        result = {
            "answer": answer,
            "ttft_s": ttft,
            "cache_hits": hits,
            "cache_misses": misses,
            "num_images": len(images),
        }
        self.calls.append({"prompt": prompt, **result})
        return result


# ---------------------------------------------------------------------------
# Real vLLM loader — used by run_baseline / run_perceptual on Colab
# ---------------------------------------------------------------------------


_SITECUSTOMIZE_CODE = '''"""pmcache: neutralize vllm's suppress_stdout in spawn'd subprocesses.

Auto-imported by Python's `site` module whenever the directory holding
this file is on PYTHONPATH. We can't reliably keep sys.stdout from
becoming ipykernel.iostream.OutStream in the subprocess (something
reattaches it after sitecustomize runs in Colab), but we CAN intercept
the moment `vllm.utils.system_utils` gets imported and replace its
`suppress_stdout` context manager with a no-op — which is what causes
the crash in the first place. Its only purpose is silencing native
C++ NCCL/cuda prints, which Colab can't capture anyway.
"""
import sys, os, io, contextlib, builtins

@contextlib.contextmanager
def _noop_suppress():
    yield

def _patch_vllm_system_utils():
    mod = sys.modules.get("vllm.utils.system_utils")
    if mod is None or getattr(mod, "_pmcache_patched", False):
        return
    mod.suppress_stdout = _noop_suppress
    if hasattr(mod, "suppress_stderr"):
        mod.suppress_stderr = _noop_suppress
    mod._pmcache_patched = True

_real_import = builtins.__import__

def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _real_import(name, globals, locals, fromlist, level)
    if "vllm.utils.system_utils" in sys.modules:
        _patch_vllm_system_utils()
    return mod

builtins.__import__ = _patched_import

def _ensure_fileno(stream, fd):
    try:
        stream.fileno()
        return stream
    except (io.UnsupportedOperation, OSError, AttributeError):
        try:
            return os.fdopen(os.dup(fd), "w", buffering=1, closefd=True)
        except OSError:
            return stream

sys.stdout = _ensure_fileno(sys.stdout, 1)
sys.stderr = _ensure_fileno(sys.stderr, 2)
'''


def _patch_ipykernel_stdout_for_vllm() -> None:
    """Stop vllm.utils.system_utils.suppress_stdout from crashing on
    ipykernel-wrapped sys.stdout, in this process AND in any spawn'd
    subprocess vllm starts after this call.

    Background: vllm's distributed init calls `suppress_stdout` to
    silence native C++ NCCL prints. The context manager dups
    `sys.stdout.fileno()`. In Colab, sys.stdout in both the parent
    kernel AND vllm's V1 EngineCore subprocess is
    `ipykernel.iostream.OutStream`, whose `fileno()` always raises
    `io.UnsupportedOperation`.

    Things that DON'T work (tried in prior turns):
      - Instance monkey-patch `sys.stdout.fileno = ...` → OutStream
        inherits from io.TextIOBase, which rejects new instance attrs.
      - Class monkey-patch `OutStream.fileno = ...` in parent → spawn'd
        subprocess has its own sys.modules and re-imports the
        unmodified class.
      - sitecustomize that only swaps `sys.stdout` to FD-backed → some
        code in the spawned subprocess reattaches OutStream AFTER
        sitecustomize runs, so the swap gets undone.

    What works:
      - Patch `vllm.utils.system_utils.suppress_stdout` to a no-op
        (it's only suppressing native prints that Colab can't capture
        anyway). Do it in the parent's already-loaded module, AND via
        a `builtins.__import__` wrapper installed by sitecustomize so
        the subprocess catches the patch at the moment vllm imports
        system_utils. The no-op makes the broken fileno() call moot.
      - As belt-and-suspenders, also try to keep
        VLLM_ENABLE_V1_MULTIPROCESSING=0 (in load_vlm before import)
        so vllm falls back to the in-process engine; if vllm honors
        it, the subprocess isn't spawned at all.

    Side effects:
      - sys.stdout/stderr in parent get swapped to FD-backed wrappers
        if their fileno was broken — Colab captures FD 1/2 separately
        so output still appears.
      - PYTHONPATH gains a tmp-dir entry that lives for the kernel
        session.
    """
    import contextlib
    import io
    import os
    import sys
    import tempfile

    def _ensure_fileno(stream, fd):  # type: ignore[no-untyped-def]
        try:
            stream.fileno()
            return stream
        except (io.UnsupportedOperation, OSError, AttributeError):
            try:
                return os.fdopen(os.dup(fd), "w", buffering=1, closefd=True)
            except OSError:
                return stream

    sys.stdout = _ensure_fileno(sys.stdout, 1)
    sys.stderr = _ensure_fileno(sys.stderr, 2)

    # If vllm.utils.system_utils is already loaded in this process,
    # neuter its suppress_stdout right now. (Idempotent via flag.)
    mod = sys.modules.get("vllm.utils.system_utils")
    if mod is not None and not getattr(mod, "_pmcache_patched", False):
        @contextlib.contextmanager
        def _noop_suppress():  # type: ignore[no-untyped-def]
            yield

        mod.suppress_stdout = _noop_suppress  # type: ignore[attr-defined]
        if hasattr(mod, "suppress_stderr"):
            mod.suppress_stderr = _noop_suppress  # type: ignore[attr-defined]
        mod._pmcache_patched = True  # type: ignore[attr-defined]

    sc_dir = os.environ.get("PMCACHE_SITECUSTOMIZE_DIR")
    if not sc_dir or not os.path.isfile(os.path.join(sc_dir, "sitecustomize.py")):
        sc_dir = tempfile.mkdtemp(prefix="pmcache_sc_")
        with open(os.path.join(sc_dir, "sitecustomize.py"), "w", encoding="utf-8") as f:
            f.write(_SITECUSTOMIZE_CODE)
        os.environ["PMCACHE_SITECUSTOMIZE_DIR"] = sc_dir

    existing = os.environ.get("PYTHONPATH", "")
    parts = existing.split(os.pathsep) if existing else []
    if sc_dir not in parts:
        os.environ["PYTHONPATH"] = sc_dir + (os.pathsep + existing if existing else "")


def load_vlm(model: str, mock_vlm: bool = False, **kwargs: Any) -> Any:
    """Return either a `MockVLM` or a real vLLM `LLM` instance.

    Real vLLM construction is deferred until `mock_vlm=False` AND
    `vllm` is importable, so the CPU laptop never accidentally tries
    to load the GPU stack.

    Extra `**kwargs` are forwarded to `vllm.LLM(...)` (e.g.
    `gpu_memory_utilization`, `max_model_len`, `tensor_parallel_size`).
    """
    if mock_vlm:
        return MockVLM()
    import os

    # Belt-and-suspenders: tell vllm to skip the EngineCore subprocess
    # if it honors this. Must be set BEFORE `import vllm` because vllm
    # reads it at engine-args resolution time.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Install subprocess-side fix BEFORE vllm import so the
    # sitecustomize/PYTHONPATH is in place if vllm decides to spawn
    # anyway, AND so any subprocess inherits PYTHONPATH from us.
    _patch_ipykernel_stdout_for_vllm()

    vllm = require("vllm", "vlm")

    # Re-run to patch the now-loaded vllm.utils.system_utils in the
    # parent (the function is idempotent — the flag short-circuits
    # double application).
    _patch_ipykernel_stdout_for_vllm()

    return vllm.LLM(model=model, **kwargs)


# ---------------------------------------------------------------------------
# Real vLLM multimodal generate — used by smoke test + benchmark runner
# ---------------------------------------------------------------------------


def kv_bytes_per_token(llm: Any) -> int:
    """Best-effort: bytes of KV cache used per prompt token for `llm`.

    KV cache per token = 2 (K + V) × num_hidden_layers ×
                         num_key_value_heads × head_dim × dtype_bytes

    Strategy:
      1. Prefer vLLM's own `ModelConfig` getters — they handle the
         nested-config quirks (Qwen3-VL puts LM dims under `text_config`,
         LLaVA under `llm_config`, etc.) for us.
      2. Fall back to probing `hf_config` directly, walking common
         nested children (`text_config`, `llm_config`, `thinker_config`)
         until a child with `num_hidden_layers` shows up.

    Returns 0 on any failure rather than raising.
    """
    model_config = None
    try:
        model_config = llm.llm_engine.model_config
    except Exception:
        return 0

    dtype = getattr(model_config, "dtype", None)
    dtype_bytes = getattr(dtype, "itemsize", None)
    if dtype_bytes is None:
        dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}.get(str(dtype), 2)

    # Strategy 1: vLLM's getters.
    try:
        parallel_config = llm.llm_engine.vllm_config.parallel_config
        num_layers = model_config.get_num_layers(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_dim = model_config.get_head_size()
        if num_layers and num_kv_heads and head_dim:
            return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    except Exception:
        pass

    # Strategy 2: walk hf_config + common nested LM sub-configs.
    try:
        hf = model_config.hf_config
        candidates = [
            hf,
            getattr(hf, "text_config", None),
            getattr(hf, "llm_config", None),
            getattr(hf, "thinker_config", None),
            getattr(hf, "language_config", None),
        ]
        for cfg in candidates:
            if cfg is None:
                continue
            num_layers = getattr(cfg, "num_hidden_layers", None)
            num_attn = getattr(cfg, "num_attention_heads", None)
            num_kv = getattr(cfg, "num_key_value_heads", num_attn)
            hidden = getattr(cfg, "hidden_size", None)
            if num_layers and num_kv and hidden and num_attn:
                head_dim = getattr(cfg, "head_dim", hidden // num_attn)
                return 2 * num_layers * num_kv * head_dim * dtype_bytes
    except Exception:
        pass

    return 0


def vllm_generate_multimodal(
    llm: Any,
    question: str,
    images: list["Image.Image"],
    max_tokens: int = 64,
) -> tuple[str, float, dict[str, int]]:
    """Run one inference through a real `vllm.LLM` with PIL images.

    Returns `(answer_text, elapsed_seconds, stats)` where stats has:
        prompt_tokens   — total prompt length (text + image placeholders)
        cached_tokens   — prompt tokens served from KV cache (prefix +
                          LMCache hits, when the engine reports them)
        output_tokens   — tokens generated by the model

    vllm's sync `generate()` returns after EOS, so elapsed is total
    generation time — used as a TTFT proxy. Consistent across baseline
    and perceptual runs, so relative comparisons hold.

    Builds the prompt via the tokenizer's chat template so the
    vision-token placeholders are inserted correctly for whichever VLM
    is loaded (Qwen2-VL, LLaVA, etc.).
    """
    vllm = require("vllm", "vlm")  # SamplingParams + LLM types

    tokenizer = llm.get_tokenizer()
    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampling_params = vllm.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    t0 = time.perf_counter()
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images if len(images) != 1 else images[0]},
        },
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - t0

    text = ""
    stats: dict[str, int] = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
    if outputs:
        ro = outputs[0]
        if ro.outputs:
            text = ro.outputs[0].text.strip()
            stats["output_tokens"] = len(ro.outputs[0].token_ids) if getattr(
                ro.outputs[0], "token_ids", None
            ) else 0
        if getattr(ro, "prompt_token_ids", None):
            stats["prompt_tokens"] = len(ro.prompt_token_ids)
        # vLLM v0.5+ exposes num_cached_tokens on RequestOutput; older
        # versions stash it on .metrics. Try both, default 0.
        cached = getattr(ro, "num_cached_tokens", None)
        if cached is None and getattr(ro, "metrics", None):
            cached = getattr(ro.metrics, "num_cached_tokens", None)
        if cached is not None:
            stats["cached_tokens"] = int(cached)
    return text, elapsed, stats


# ---------------------------------------------------------------------------
# GPU pre-flight smoke test (Colab only)
# ---------------------------------------------------------------------------


def smoke_test_vlm(model: str, workspace: str | Path) -> dict[str, Any]:
    """Load the model once + run a single inference. Logs to {workspace}/smoke.json.

    Intended for the Colab notebook: catches OOM / version-mismatch
    failures early, before committing to long benchmark runs. On the
    CPU laptop this will raise via `require("vllm", "vlm")` — that's
    intentional (the laptop has no GPU; the function exists so the
    Colab notebook can call it).
    """
    from PIL import Image  # safe — pillow is a core dep

    llm = load_vlm(model, mock_vlm=False)
    test_image = Image.new("RGB", (224, 224), color="white")
    text, elapsed, _stats = vllm_generate_multimodal(
        llm,
        question="Describe this image in one word.",
        images=[test_image],
        max_tokens=20,
    )

    out = {
        "model": model,
        "smoke_elapsed_s": elapsed,
        "result": text[:200] if text else None,
    }
    Path(workspace).mkdir(parents=True, exist_ok=True)
    (Path(workspace) / "smoke.json").write_text(json.dumps(out, indent=2))
    return out
