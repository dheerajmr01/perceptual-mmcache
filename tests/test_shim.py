"""Step 5 tests — perceptual shim (PerceptualMMCache).

We do NOT need torch for these — a FakeVerifier with attach-by-image
returns canned embeddings, exercising every branch of `prepare_image`
without ever loading DINOv2.

Spec coverage:
    - disabled config → pass-through
    - first frame → new bucket, fresh UUID, miss
    - Tier-1 match + Tier-2 ≥ τ → alias to anchor UUID, hit
    - Tier-1 match + Tier-2 < τ → new bucket, false-positive metric
    - Tier-1 miss → new bucket
    - already-tagged image → idempotent fast-path
    - build_pmcache(enabled=False) → None
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from PIL import Image, ImageFilter

from pmcache.config import PMCacheConfig
from pmcache.lmcache_shim import (
    PerceptualMMCache,
    _existing_uuid,
    _tag_with_uuid,
    build_pmcache,
)
from pmcache.verifier import DinoV2Verifier  # for cosine math only


# --- helpers ---------------------------------------------------------------


def _photo_like(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (size, size), dtype=np.int32).astype(np.uint8)
    blurred = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=24))
    return Image.fromarray(np.stack([np.asarray(blurred)] * 3, axis=-1), mode="RGB")


def _near_duplicate(image: Image.Image, dx: int = 1) -> Image.Image:
    """1-pixel translation (no wrap), as in test_hashing fixtures."""
    arr = np.asarray(image)
    return Image.fromarray(
        np.pad(arr[:, :-dx], ((0, 0), (dx, 0), (0, 0)), mode="edge")
    )


class FakeVerifier:
    """Test stand-in for DinoV2Verifier — embeddings come from attach()."""

    EMBED_DIM = 384

    def __init__(self) -> None:
        self._embs: dict[int, np.ndarray] = {}
        self.embed_calls: int = 0

    def attach(self, image: Image.Image, emb) -> None:
        v = np.asarray(emb, dtype=np.float32)
        self._embs[id(image)] = v

    def embed(self, image: Image.Image) -> np.ndarray:
        self.embed_calls += 1
        if id(image) not in self._embs:
            raise AssertionError(
                f"FakeVerifier: no embedding attached for image id={id(image)} — "
                "call verifier.attach(img, emb) before prepare_image(img)."
            )
        return self._embs[id(image)]

    # Re-use the real cosine math so tests track production behavior.
    cosine = staticmethod(DinoV2Verifier.cosine)


def _unit_vec(idx: int, dim: int = 384) -> np.ndarray:
    """A canonical basis vector — cosine(unit[i], unit[j]) = 0 for i != j."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _slightly_rotated(v: np.ndarray, by: int = 1) -> np.ndarray:
    """Mix a tiny amount of orthogonal direction in to drop cosine just below 1."""
    out = v.copy()
    other = np.zeros_like(v)
    other[(np.argmax(v) + 1) % len(v)] = 1.0
    mixed = (1.0 - 0.001 * by) * out + 0.001 * by * other
    return mixed / np.linalg.norm(mixed)


# --- disabled path ---------------------------------------------------------


class TestDisabled:
    def test_disabled_config_is_passthrough(self):
        verifier = FakeVerifier()
        pm = PerceptualMMCache(
            PMCacheConfig(enabled=False), verifier=verifier
        )
        img = _photo_like(0)
        out = pm.prepare_image(img)
        # Pass-through means we return the input object itself, untagged.
        assert out is img
        assert _existing_uuid(out) is None
        # Verifier should never have been called.
        assert verifier.embed_calls == 0
        # Metrics should not have ticked.
        assert pm.metrics.frames_seen == 0

    def test_build_pmcache_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("PMCACHE_ENABLED", "false")
        assert build_pmcache() is None

    def test_build_pmcache_returns_instance_when_enabled(self, monkeypatch):
        monkeypatch.setenv("PMCACHE_ENABLED", "true")
        pm = build_pmcache(verifier=FakeVerifier())
        assert isinstance(pm, PerceptualMMCache)


# --- first frame (no candidates) -------------------------------------------


class TestNewBucket:
    def test_first_frame_gets_fresh_uuid(self):
        verifier = FakeVerifier()
        img = _photo_like(0)
        verifier.attach(img, _unit_vec(0))

        pm = PerceptualMMCache(PMCacheConfig(tau=0.98, k=5), verifier=verifier)
        out = pm.prepare_image(img)

        # New UUID assigned
        u = _existing_uuid(out)
        assert u is not None
        # Metrics
        assert pm.metrics.frames_seen == 1
        assert pm.metrics.misses == 1
        assert pm.metrics.aliased_hashes == 0
        # Store now has one bucket whose anchor_mm_hash == hex(uuid)
        assert len(pm.store) == 1
        anchor = pm.store.buckets[0]
        assert anchor.anchor_mm_hash == u.hex


# --- Tier 1 hit + Tier 2 hit → alias ---------------------------------------


class TestVerifiedHit:
    def test_near_duplicate_aliased_to_anchor_uuid(self):
        verifier = FakeVerifier()
        anchor = _photo_like(0)
        near = _near_duplicate(anchor)
        emb = _unit_vec(0)
        verifier.attach(anchor, emb)
        verifier.attach(near, _slightly_rotated(emb, by=1))  # cosine ≈ 0.999

        pm = PerceptualMMCache(PMCacheConfig(tau=0.98, k=5), verifier=verifier)
        out_anchor = pm.prepare_image(anchor)
        out_near = pm.prepare_image(near)

        anchor_uuid = _existing_uuid(out_anchor)
        near_uuid = _existing_uuid(out_near)
        assert anchor_uuid is not None
        assert near_uuid == anchor_uuid, (
            "near-duplicate should have been aliased to the anchor's UUID"
        )
        # Metrics: 1 miss (anchor), 1 hit (near)
        assert pm.metrics.misses == 1
        assert pm.metrics.tier2_hits == 1
        assert pm.metrics.aliased_hashes == 1
        # Store still has 1 bucket (no new bucket on hit)
        assert len(pm.store) == 1


# --- Tier 1 hit + Tier 2 reject → false positive ---------------------------


class TestFalsePositive:
    def test_low_cosine_creates_new_bucket_and_records_false_positive(self):
        verifier = FakeVerifier()
        anchor = _photo_like(0)
        # Pick a near-duplicate so Tier-1 *does* hit, then attach an
        # orthogonal embedding so Tier-2 rejects.
        near = _near_duplicate(anchor)
        verifier.attach(anchor, _unit_vec(0))
        verifier.attach(near, _unit_vec(1))  # cosine == 0.0 < τ

        pm = PerceptualMMCache(PMCacheConfig(tau=0.98, k=5), verifier=verifier)
        pm.prepare_image(anchor)
        out_near = pm.prepare_image(near)

        # Tier-2 rejected → fresh UUID, distinct from anchor.
        anchor_uuid = uuid.UUID(hex=pm.store.buckets[0].anchor_mm_hash)
        near_uuid = _existing_uuid(out_near)
        assert near_uuid != anchor_uuid

        # Metrics: false_positive bumps tier1_hits, tier2_rejects, misses.
        assert pm.metrics.tier1_hits == 1
        assert pm.metrics.tier2_rejects == 1
        # misses counts both the very first anchor frame AND the false-positive
        # outcome of the second frame (which also created a new bucket).
        assert pm.metrics.misses == 2
        # Two distinct buckets now.
        assert len(pm.store) == 2


# --- Tier 1 miss → new bucket ----------------------------------------------


class TestTier1Miss:
    def test_far_apart_phash_makes_new_bucket(self):
        verifier = FakeVerifier()
        a = _photo_like(0)
        b = _photo_like(999)  # different pHash, well outside K=5
        verifier.attach(a, _unit_vec(0))
        verifier.attach(b, _unit_vec(1))

        pm = PerceptualMMCache(PMCacheConfig(tau=0.98, k=5), verifier=verifier)
        out_a = pm.prepare_image(a)
        out_b = pm.prepare_image(b)

        assert _existing_uuid(out_a) != _existing_uuid(out_b)
        assert len(pm.store) == 2
        # Both counted as misses — no Tier-1 candidates were even
        # considered for the second frame.
        assert pm.metrics.misses == 2
        assert pm.metrics.tier1_hits == 0


# --- idempotent fast-path --------------------------------------------------


class TestIdempotent:
    def test_already_tagged_image_passes_through(self):
        verifier = FakeVerifier()
        pm = PerceptualMMCache(PMCacheConfig(tau=0.98), verifier=verifier)
        img = _photo_like(0)
        u = uuid.uuid4()
        pre_tagged = _tag_with_uuid(img, u)

        out = pm.prepare_image(pre_tagged)

        assert _existing_uuid(out) == u
        # Fast-path: no verifier call, no new bucket.
        assert verifier.embed_calls == 0
        assert len(pm.store) == 0
        # Recorded as a hit (the caller deliberately reused this UUID).
        assert pm.metrics.tier2_hits == 1


# --- end-to-end metrics dump ------------------------------------------------


class TestMetricsDump:
    def test_metrics_dump_writes_json(self, tmp_path):
        verifier = FakeVerifier()
        pm = PerceptualMMCache(PMCacheConfig(tau=0.98), verifier=verifier)
        img = _photo_like(0)
        verifier.attach(img, _unit_vec(0))
        pm.prepare_image(img, video_id="vid-1")

        out_path = tmp_path / "metrics.json"
        pm.metrics.dump(out_path)
        text = out_path.read_text()
        assert '"frames_seen": 1' in text
        assert '"vid-1"' in text
