"""Step 2 tests — pHash + BK-tree bucket index.

Acceptance from the spec:
    - same image → same bucket
    - 1-pixel shift → same bucket
    - different image → different bucket
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from pmcache.hashing import BucketIndex, _PHASH_MASK, hamming, phash


# --- fixtures ---------------------------------------------------------------
#
# pHash is engineered for *natural* images (photos, video frames). On pure
# synthetic patterns (high-frequency sinusoids, sharp checkerboards) the
# hash is brittle to small changes because DCT coefficients hover near the
# median threshold. We mimic real video by Gaussian-blurring random noise
# — this produces smooth, low-frequency, photo-like content where pHash
# behaves the way it does in production.


def _photo_like_frame(seed: int = 0, size: int = 256) -> Image.Image:
    """Heavily-blurred random noise — close enough to a real video frame
    that pHash is stable under small translations (as in production)."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (size, size), dtype=np.int32).astype(np.uint8)
    blurred = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=24))
    arr = np.stack([np.asarray(blurred)] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _shifted(image: Image.Image, dx: int = 1) -> Image.Image:
    """Translate `image` right by `dx` pixels with edge replication
    (no `np.roll` wrap-around, which would inject a sharp discontinuity
    and dominate the pHash)."""
    arr = np.asarray(image)
    if dx <= 0:
        return image
    return Image.fromarray(
        np.pad(arr[:, :-dx], ((0, 0), (dx, 0), (0, 0)), mode="edge")
    )


# --- phash ------------------------------------------------------------------


def test_phash_is_64bit_int():
    img = _photo_like_frame()
    h = phash(img)
    assert isinstance(h, int)
    assert 0 <= h <= _PHASH_MASK


def test_phash_deterministic():
    img = _photo_like_frame(seed=3)
    assert phash(img) == phash(img)


def test_hamming_zero_for_identical():
    h = phash(_photo_like_frame(seed=5))
    assert hamming(h, h) == 0


def test_hamming_symmetric():
    a = phash(_photo_like_frame(seed=1))
    b = phash(_photo_like_frame(seed=2))
    assert hamming(a, b) == hamming(b, a)


# --- bucket index — acceptance criteria from the spec -----------------------


def test_identical_image_same_bucket():
    """Spec: same image → same bucket."""
    idx = BucketIndex(k=5)
    img = _photo_like_frame()
    idx.add(phash(img), bucket_id=0)

    hits = idx.query(phash(img))
    assert hits == [0]


def test_one_pixel_shift_same_bucket():
    """Spec: 1-pixel shift → same bucket.

    pHash works on a 32x32 downsample so a 1-pixel shift on 128x128
    should be within a couple of bits at most — well under K=5.
    """
    idx = BucketIndex(k=5)
    anchor = _photo_like_frame(seed=4)
    idx.add(phash(anchor), bucket_id=42)

    shifted = _shifted(anchor, dx=1)
    hits = idx.query(phash(shifted))
    assert 42 in hits, (
        f"1-pixel shift produced Hamming distance "
        f"{hamming(phash(anchor), phash(shifted))} — should be ≤ 5"
    )


def test_different_image_different_bucket():
    """Spec: different image → different bucket.

    Two unrelated low-frequency patterns should produce pHashes that
    are clearly outside the K=5 Hamming ball.
    """
    idx = BucketIndex(k=5)
    img_a = _photo_like_frame(seed=0)
    img_b = _photo_like_frame(seed=137)
    idx.add(phash(img_a), bucket_id=1)

    hits = idx.query(phash(img_b))
    assert 1 not in hits, (
        f"unrelated images had Hamming distance "
        f"{hamming(phash(img_a), phash(img_b))} — expected > 5"
    )


# --- bucket index — finer-grained invariants --------------------------------


def test_empty_index_returns_empty():
    idx = BucketIndex(k=5)
    assert idx.query(0xDEADBEEF) == []


def test_query_returns_multiple_candidates_sorted_by_distance():
    """When several buckets are within range, closest comes first."""
    idx = BucketIndex(k=10)
    h = phash(_photo_like_frame(seed=0))
    # Insert anchors at controlled Hamming distances from `h`.
    idx.add(h, bucket_id=100)          # distance 0
    idx.add(h ^ 0b1, bucket_id=101)    # distance 1
    idx.add(h ^ 0b111, bucket_id=103)  # distance 3

    hits = idx.query(h, max_dist=10)
    assert hits == [100, 101, 103]


def test_query_respects_max_dist_override():
    idx = BucketIndex(k=10)
    h = phash(_photo_like_frame(seed=2))
    idx.add(h, bucket_id=1)
    idx.add(h ^ ((1 << 4) - 1), bucket_id=2)  # distance 4

    assert idx.query(h, max_dist=0) == [1]
    assert sorted(idx.query(h, max_dist=10)) == [1, 2]


def test_index_len_tracks_inserts():
    idx = BucketIndex(k=5)
    assert len(idx) == 0
    idx.add(phash(_photo_like_frame(seed=1)), bucket_id=1)
    idx.add(phash(_photo_like_frame(seed=2)), bucket_id=2)
    assert len(idx) == 2


def test_phash_value_masked_to_64_bits():
    """Inserting an over-sized int shouldn't break lookup."""
    idx = BucketIndex(k=0)
    big = (1 << 70) | 0xCAFE  # bits above 63 should be discarded
    idx.add(big, bucket_id=7)
    # Query with the 64-bit-truncated value still finds it.
    assert idx.query(big & _PHASH_MASK) == [7]
