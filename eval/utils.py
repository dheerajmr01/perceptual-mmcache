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


def sample_frames(video_path: str | Path, fps: float = 1.0) -> list["Image.Image"]:
    """Decode `video_path` and return PIL Images sampled at `fps`.

    Uses opencv-python-headless. Returns RGB PIL Images (cv2 reads BGR;
    we convert) so callers can feed them straight to vLLM / PIL EXIF
    tagging.

    fps=1.0 means one frame per video-second. If the source is 30 fps,
    we keep every 30th frame. Always emits at least 1 frame for any
    non-empty video.
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


def _patch_ipykernel_stdout_for_vllm() -> None:
    """Give sys.stdout/sys.stderr a real fileno() in Jupyter/Colab.

    vllm's distributed init calls sys.stdout.fileno() (via
    vllm.utils.system_utils.suppress_stdout) to dup the FD. Inside an
    ipykernel kernel, sys.stdout is ipykernel.iostream.OutStream which
    raises io.UnsupportedOperation. The OS-level stdout FD really is
    1/2 (ipykernel wraps the Python-side write, not the FD), so
    reporting that is safe — and the forked EngineCore subprocess
    inherits the patched stream so it gets through init too.

    We patch the OutStream *class* method (not the instance) because
    ipykernel's OutStream inherits from io.TextIOBase, which doesn't
    accept arbitrary instance attribute assignment — so an instance
    monkey-patch silently no-ops. A class-level patch lands on every
    existing AND future instance (including ones the EngineCore fork
    sees).
    """
    import io
    import sys

    try:
        from ipykernel.iostream import OutStream  # type: ignore[import-not-found]
    except ImportError:
        OutStream = None  # not running inside ipykernel — nothing to patch

    if OutStream is not None and not getattr(OutStream, "_pmcache_fileno_patched", False):
        _orig_fileno = OutStream.fileno

        def _safe_fileno(self):  # type: ignore[no-untyped-def]
            try:
                return _orig_fileno(self)
            except (io.UnsupportedOperation, OSError, AttributeError):
                # stderr-ish name → FD 2; everything else → FD 1.
                name = getattr(self, "name", "") or ""
                return 2 if "stderr" in str(name).lower() else 1

        OutStream.fileno = _safe_fileno  # type: ignore[method-assign]
        OutStream._pmcache_fileno_patched = True  # type: ignore[attr-defined]

    # Instance fallback for non-OutStream wrappers (e.g. Colab's
    # additional capture layers). Wrapped in try/except because some
    # stream types reject new attributes.
    for stream, fd in ((sys.stdout, 1), (sys.stderr, 2)):
        try:
            stream.fileno()
            continue
        except (OSError, AttributeError, io.UnsupportedOperation):
            pass
        try:
            stream.fileno = lambda fd=fd: fd  # type: ignore[method-assign]
        except (AttributeError, TypeError):
            pass


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
    vllm = require("vllm", "vlm")
    _patch_ipykernel_stdout_for_vllm()
    return vllm.LLM(model=model, **kwargs)


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
    t0 = time.perf_counter()
    result = llm.generate(
        prompt="Describe this image in one word.",
        images=[test_image],
    ) if hasattr(llm, "generate") else None
    elapsed = time.perf_counter() - t0

    out = {
        "model": model,
        "smoke_elapsed_s": elapsed,
        "result": str(result)[:200] if result else None,
    }
    Path(workspace).mkdir(parents=True, exist_ok=True)
    (Path(workspace) / "smoke.json").write_text(json.dumps(out, indent=2))
    return out
