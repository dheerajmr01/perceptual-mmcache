"""Tier 1: perceptual hashing + BK-tree bucket index.

`phash` wraps `imagehash.phash` and returns a 64-bit int (easier to hand
off to LMCache's hex-based mm_hash later, and trivially serializable).

`BucketIndex` wraps `pybktree.BKTree` so we can find all bucket anchors
whose pHash is within `k` Hamming of a query in O(log n) average. The
BK-tree stores `(phash, bucket_id)` tuples; the distance function
operates on the pHash component only.

Note on eviction: `pybktree.BKTree` exposes `add` / `find` only — no
`remove`. The BucketStore (Step 4) handles eviction by rebuilding the
tree, which is fine at the scales we target (`max_buckets ≤ 4096`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagehash
import pybktree

if TYPE_CHECKING:
    from PIL import Image


_PHASH_MASK = (1 << 64) - 1


def phash(image: "Image.Image") -> int:
    """Compute the 64-bit perceptual hash of a PIL image as an int.

    Uses `imagehash.phash` with the library's default 8x8 hash size
    (DCT of a 32x32 downsample, thresholded against the median of the
    top-left 8x8 DCT block). Robust to small translations, scaling, and
    brightness shifts — exactly the changes we expect between adjacent
    video frames.
    """
    return int(str(imagehash.phash(image)), 16) & _PHASH_MASK


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit pHash values."""
    return ((a ^ b) & _PHASH_MASK).bit_count()


def _tree_distance(item_a: tuple[int, int], item_b: tuple[int, int]) -> int:
    """BK-tree distance fn over (phash, bucket_id) tuples."""
    return hamming(item_a[0], item_b[0])


class BucketIndex:
    """BK-tree over bucket anchor pHashes, for Hamming-ball lookups.

    Stores `(phash, bucket_id)` tuples; queries return the list of
    bucket_ids whose anchor pHash is within `max_dist` Hamming.
    """

    def __init__(self, k: int = 5) -> None:
        """`k` is the default Hamming radius for `query`."""
        self.k = k
        self._tree: pybktree.BKTree = pybktree.BKTree(_tree_distance)
        self._size = 0

    def add(self, phash_value: int, bucket_id: int) -> None:
        """Insert a bucket anchor."""
        self._tree.add((phash_value & _PHASH_MASK, bucket_id))
        self._size += 1

    def query(self, phash_value: int, max_dist: int | None = None) -> list[int]:
        """Return bucket_ids whose anchor pHash is within `max_dist` Hamming.

        Results are sorted by ascending Hamming distance (closest first),
        which is the order the BucketStore wants when picking a Tier-2
        verification candidate.
        """
        radius = self.k if max_dist is None else max_dist
        if self._size == 0:
            return []
        query_item = (phash_value & _PHASH_MASK, -1)
        # pybktree returns [(dist, item), ...] sorted by dist asc
        return [item[1] for _, item in self._tree.find(query_item, radius)]

    def __len__(self) -> int:
        return self._size
