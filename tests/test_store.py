"""Step 4 tests — BucketStore.

Acceptance from the spec: hit / miss / eviction.

These tests do *not* touch DINOv2 — we hand the store synthetic
embeddings (np.ones(384)). The store doesn't compute embeddings; it
just bookkeeps them. Tier-2 verification logic lives in the shim
(Step 5).
"""

from __future__ import annotations

import numpy as np
import pytest

from pmcache.hashing import phash
from pmcache.store import BucketEntry, BucketStore
from PIL import Image, ImageFilter


# --- helpers ---------------------------------------------------------------


def _photo_like(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (size, size), dtype=np.int32).astype(np.uint8)
    blurred = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=24))
    arr = np.stack([np.asarray(blurred)] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _fake_emb(seed: int = 0, dim: int = 384) -> np.ndarray:
    """A normalized fake DINOv2 embedding for store-level tests."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# --- spec: hit / miss ------------------------------------------------------


class TestHitMiss:
    def test_empty_store_returns_no_candidates(self):
        store = BucketStore(k=5)
        assert store.candidates(0xCAFE) == []

    def test_hit_after_insert(self):
        """Spec: insert a bucket, then a matching pHash hits it."""
        store = BucketStore(k=5)
        h = phash(_photo_like(seed=0))
        entry = store.insert_new_bucket(h, _fake_emb(0), mm_hash="aaa")

        cands = store.candidates(h)
        assert len(cands) == 1
        assert cands[0].bucket_id == entry.bucket_id
        assert cands[0].anchor_mm_hash == "aaa"

    def test_miss_for_unrelated_phash(self):
        """Spec: an unrelated pHash returns no candidates."""
        store = BucketStore(k=5)
        store.insert_new_bucket(phash(_photo_like(0)), _fake_emb(0), "aaa")

        # A pHash that's deliberately far in Hamming
        far = phash(_photo_like(seed=999))
        cands = store.candidates(far)
        assert cands == []


# --- aliasing & fast-path lookup -------------------------------------------


class TestAlias:
    def test_alias_returns_anchor_mm_hash(self):
        store = BucketStore(k=5)
        h = phash(_photo_like(seed=1))
        entry = store.insert_new_bucket(h, _fake_emb(1), mm_hash="anchor-1")

        anchor = store.alias(entry, "frame-2")
        assert anchor == "anchor-1"

    def test_lookup_alias_after_aliasing(self):
        """Spec: aliased mm_hash → anchor lookup fast-path hits."""
        store = BucketStore(k=5)
        h = phash(_photo_like(seed=2))
        entry = store.insert_new_bucket(h, _fake_emb(2), mm_hash="anchor-2")
        store.alias(entry, "frame-3")
        store.alias(entry, "frame-4")

        assert store.lookup_alias("anchor-2") == "anchor-2"  # anchor itself
        assert store.lookup_alias("frame-3") == "anchor-2"
        assert store.lookup_alias("frame-4") == "anchor-2"
        assert store.lookup_alias("never-seen") is None

    def test_alias_dedupes_repeated_mm_hash(self):
        """Re-aliasing the same mm_hash should not grow the member list."""
        store = BucketStore(k=5, bucket_capacity=3)
        entry = store.insert_new_bucket(
            phash(_photo_like(3)), _fake_emb(3), mm_hash="anchor"
        )
        store.alias(entry, "frame-a")
        store.alias(entry, "frame-a")
        store.alias(entry, "frame-a")
        # anchor + frame-a → 2 members
        assert entry.members == ["anchor", "frame-a"]


# --- eviction --------------------------------------------------------------


class TestEviction:
    def test_per_bucket_member_lru_evicts_oldest(self):
        """Spec: bucket-level member LRU evicts oldest when over capacity."""
        store = BucketStore(k=5, bucket_capacity=3)
        entry = store.insert_new_bucket(
            phash(_photo_like(0)), _fake_emb(0), mm_hash="anchor"
        )
        # anchor is already member[0]; add 3 more members → over capacity by 1
        store.alias(entry, "m1")
        store.alias(entry, "m2")
        store.alias(entry, "m3")

        # Oldest ("anchor") should have been evicted from members AND alias map
        assert "anchor" not in entry.members
        assert entry.members == ["m1", "m2", "m3"]
        assert store.lookup_alias("anchor") is None
        # But the anchor's mm_hash is still the bucket's anchor_mm_hash
        assert entry.anchor_mm_hash == "anchor"

    def test_bucket_level_lru_evicts_oldest_bucket(self):
        """Spec: bucket-level LRU drops least-recently-used bucket when full."""
        store = BucketStore(k=5, max_buckets=3)
        e0 = store.insert_new_bucket(phash(_photo_like(0)), _fake_emb(0), "h0")
        e1 = store.insert_new_bucket(phash(_photo_like(1)), _fake_emb(1), "h1")
        e2 = store.insert_new_bucket(phash(_photo_like(2)), _fake_emb(2), "h2")
        assert len(store) == 3

        # Insert a 4th — e0 (oldest) should be evicted.
        e3 = store.insert_new_bucket(phash(_photo_like(3)), _fake_emb(3), "h3")
        assert len(store) == 3
        ids = {b.bucket_id for b in store.buckets}
        assert e0.bucket_id not in ids
        assert {e1.bucket_id, e2.bucket_id, e3.bucket_id} == ids

    def test_eviction_rebuilds_index(self):
        """After bucket eviction, candidates() must not return the evicted bucket."""
        store = BucketStore(k=5, max_buckets=2)
        e_first = store.insert_new_bucket(
            phash(_photo_like(0)), _fake_emb(0), "h0"
        )
        first_phash = e_first.anchor_phash
        store.insert_new_bucket(phash(_photo_like(1)), _fake_emb(1), "h1")
        store.insert_new_bucket(phash(_photo_like(2)), _fake_emb(2), "h2")
        # e_first should have been evicted

        # Querying with its anchor pHash should NOT return e_first.
        cands = store.candidates(first_phash)
        assert e_first.bucket_id not in {c.bucket_id for c in cands}

    def test_bucket_eviction_clears_alias_entries(self):
        store = BucketStore(k=5, max_buckets=2)
        e0 = store.insert_new_bucket(
            phash(_photo_like(0)), _fake_emb(0), "h0"
        )
        store.alias(e0, "h0-frame-1")
        store.alias(e0, "h0-frame-2")
        assert store.lookup_alias("h0-frame-1") == "h0"

        # Fill the store so e0 evicts.
        store.insert_new_bucket(phash(_photo_like(1)), _fake_emb(1), "h1")
        store.insert_new_bucket(phash(_photo_like(2)), _fake_emb(2), "h2")

        assert store.lookup_alias("h0") is None
        assert store.lookup_alias("h0-frame-1") is None
        assert store.lookup_alias("h0-frame-2") is None

    def test_candidates_touches_lru_order(self):
        """Querying a bucket should refresh its LRU position."""
        store = BucketStore(k=5, max_buckets=2)
        e0 = store.insert_new_bucket(phash(_photo_like(0)), _fake_emb(0), "h0")
        store.insert_new_bucket(phash(_photo_like(1)), _fake_emb(1), "h1")

        # Touch e0 via candidates() — now e1 is the LRU.
        cands = store.candidates(e0.anchor_phash)
        assert e0.bucket_id in {c.bucket_id for c in cands}

        # Insert a new bucket → e1 should evict, not e0.
        store.insert_new_bucket(phash(_photo_like(2)), _fake_emb(2), "h2")
        surviving_ids = {b.bucket_id for b in store.buckets}
        assert e0.bucket_id in surviving_ids


# --- import hygiene --------------------------------------------------------


class TestImportHygiene:
    def test_store_module_does_not_import_torch(self, torch_not_imported):
        import pmcache.store  # noqa: F401
