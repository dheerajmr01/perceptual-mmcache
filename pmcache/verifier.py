"""Tier 2: DINOv2-small cosine verifier.

Loads `facebook/dinov2-small` once on `device` (default "cpu") and
exposes:
  - `embed(image)` -> L2-normalized np.ndarray (384-d).
  - `cosine(a, b)` -> float dot product (since inputs are normalized).

torch / transformers are NEVER imported at module top level so
`import pmcache.verifier` works on a CPU laptop without the `[vlm]`
extra installed. The first call to `embed()` triggers the lazy load
via `pmcache._lazy.require`.

DINOv2 architecture note: we take the CLS token of the last hidden
state and L2-normalize. This is the standard image-similarity head for
DINOv2 (matches the recipe in the paper's benchmark code).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pmcache._lazy import require

if TYPE_CHECKING:
    from PIL import Image


class DinoV2Verifier:
    """Embed images with DINOv2-small and compare via cosine similarity.

    Thread-safety: not safe across threads (single-producer assumption,
    matching LMCache's per-worker connector lifecycle).
    """

    DEFAULT_MODEL = "facebook/dinov2-small"
    EMBED_DIM = 384  # DINOv2-small CLS-token dim

    def __init__(self, device: str = "cpu", model_name: str | None = None) -> None:
        self.device = device
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Any = None
        self._processor: Any = None
        # Heavy imports + model download are deferred to first use so the
        # constructor stays cheap (importable on CPU laptop, instantiable
        # in unit tests via mocks).

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch = require("torch", "vlm")
        transformers = require("transformers", "vlm")
        self._processor = transformers.AutoImageProcessor.from_pretrained(self.model_name)
        self._model = transformers.AutoModel.from_pretrained(self.model_name)
        self._model = self._model.to(self.device).eval()
        # Pin requires_grad off — pure inference path.
        for p in self._model.parameters():
            p.requires_grad_(False)
        # Stash for embed() — avoid repeated `require` lookups in the hot path.
        self._torch = torch

    def embed(self, image: "Image.Image") -> np.ndarray:
        """Return an L2-normalized embedding (np.ndarray, shape (384,))."""
        self._ensure_loaded()
        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # CLS token of the last hidden state — standard DINOv2 image head.
        cls = outputs.last_hidden_state[:, 0]  # shape (1, EMBED_DIM)
        norm = cls.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cls = cls / norm
        return cls.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        Pure numpy — no torch dependency. If inputs are already
        L2-normalized (as `embed()` returns) this is just the dot
        product; we re-normalize defensively so the function also works
        on non-normalized inputs (e.g. hand-built unit-test vectors).
        Result is clamped to [-1.0, 1.0] for downstream numerical safety
        (floating-point can produce 1.0 + epsilon).
        """
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        sim = float(np.dot(a, b) / (na * nb))
        return max(-1.0, min(1.0, sim))
