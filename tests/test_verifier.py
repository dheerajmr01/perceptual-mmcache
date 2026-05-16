"""Step 3 tests — DINOv2 verifier.

Two test classes:
  - `TestCosine` — pure numpy unit tests, always run on the CPU laptop.
  - `TestDinoV2Integration` — loads `facebook/dinov2-small` for real.
    Auto-skips when `torch` / `transformers` aren't installed (i.e. the
    laptop doesn't have the `[vlm]` extra). Runs on Colab during
    pre-flight `pytest tests/ -q`.

Spec acceptance (integration):
    - same image  -> cosine ≈ 1.0
    - similar     -> cosine > 0.98
    - different   -> cosine < 0.9
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from pmcache.verifier import DinoV2Verifier


# --- pure cosine math (always runs) -----------------------------------------


class TestCosine:
    def test_identical_unit_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert DinoV2Verifier.cosine(v, v) == pytest.approx(1.0)

    def test_opposite_unit_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert DinoV2Verifier.cosine(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_unit_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert DinoV2Verifier.cosine(a, b) == pytest.approx(0.0)

    def test_non_normalized_inputs_work(self):
        a = np.array([3.0, 4.0])           # length 5
        b = np.array([6.0, 8.0])           # length 10, same direction
        assert DinoV2Verifier.cosine(a, b) == pytest.approx(1.0)

    def test_zero_vector_returns_zero(self):
        zero = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 0.0, 0.0])
        assert DinoV2Verifier.cosine(zero, v) == 0.0
        assert DinoV2Verifier.cosine(v, zero) == 0.0

    def test_clamped_to_unit_interval(self):
        """Floating-point can push cosine to 1.0 + epsilon; we clamp."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = rng.standard_normal(384).astype(np.float32)
            sim = DinoV2Verifier.cosine(v, v)
            assert -1.0 <= sim <= 1.0
            assert sim == pytest.approx(1.0, abs=1e-6)

    def test_known_cosine_value(self):
        # 60° apart: cos = 0.5
        a = np.array([1.0, 0.0])
        b = np.array([0.5, np.sqrt(3) / 2])
        assert DinoV2Verifier.cosine(a, b) == pytest.approx(0.5)


# --- DinoV2Verifier import-time invariants (no model load) ------------------


class TestVerifierImport:
    def test_constructor_does_not_load_model(self):
        """Constructing should be cheap — no torch import, no HF download."""
        v = DinoV2Verifier(device="cpu")
        assert v._model is None
        assert v._processor is None
        assert v.device == "cpu"
        assert v.model_name == "facebook/dinov2-small"

    def test_constructor_does_not_pull_in_torch(self):
        import sys
        # If a previous test loaded torch, we can't unload it. Only assert
        # the constructor doesn't *newly* import it.
        torch_loaded_before = "torch" in sys.modules
        DinoV2Verifier(device="cpu")
        torch_loaded_after = "torch" in sys.modules
        if not torch_loaded_before:
            assert not torch_loaded_after, (
                "DinoV2Verifier.__init__ pulled in torch — "
                "model load must stay in _ensure_loaded"
            )

    def test_embed_dim_constant(self):
        assert DinoV2Verifier.EMBED_DIM == 384


# --- DinoV2 integration tests (require [vlm]) -------------------------------
#
# Skipped automatically when torch / transformers aren't installed (the
# default CPU-laptop dev loop). On Colab and on any environment with
# `pip install -e '.[vlm]'`, these run and validate the spec assertions.


def _photo_like(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (size, size), dtype=np.int32).astype(np.uint8)
    blurred = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=24))
    arr = np.stack([np.asarray(blurred)] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture(scope="module")
def verifier():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    v = DinoV2Verifier(device="cpu")
    v._ensure_loaded()
    return v


class TestDinoV2Integration:
    """Spec compliance — load real DINOv2 once, embed three frames."""

    def test_embed_shape_and_normalization(self, verifier):
        img = _photo_like(seed=0)
        emb = verifier.embed(img)
        assert emb.shape == (DinoV2Verifier.EMBED_DIM,)
        assert emb.dtype == np.float32
        # L2-normalized
        assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-5)

    def test_same_image_cosine_is_one(self, verifier):
        img = _photo_like(seed=1)
        a = verifier.embed(img)
        b = verifier.embed(img)
        assert verifier.cosine(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_similar_image_cosine_above_tau(self, verifier):
        """Spec: similar → cosine > 0.98."""
        anchor = _photo_like(seed=2)
        # Mild perturbation: blur + brightness shift, no content change.
        arr = np.asarray(anchor).astype(np.int16) + 8
        nearly = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        a = verifier.embed(anchor)
        b = verifier.embed(nearly)
        sim = verifier.cosine(a, b)
        assert sim > 0.98, f"similar-image cosine was {sim:.4f}, expected > 0.98"

    def test_different_image_cosine_below_threshold(self, verifier):
        """Spec: different → cosine < 0.9."""
        a = verifier.embed(_photo_like(seed=10))
        b = verifier.embed(_photo_like(seed=999))
        sim = verifier.cosine(a, b)
        assert sim < 0.9, f"different-image cosine was {sim:.4f}, expected < 0.9"
