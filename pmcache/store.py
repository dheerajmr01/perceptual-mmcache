"""BucketStore — perceptual buckets with anchor + LRU members.

Combines:
  - `BucketIndex` (Tier 1 BK-tree) for pHash candidate lookup.
  - Per-bucket member LRU (which exact `mm_hash`es have been aliased
    into this bucket — bounded by `bucket_capacity`).
  - Bucket-level LRU (which buckets we keep at all — bounded by
    `max_buckets`).

Tier 2 (DINOv2 cosine verification) is orchestrated by the caller (the
shim), not here — that keeps the metrics seam clean (a Tier-1 candidate
that fails Tier 2 verification is a meaningful "false positive avoided"
signal we don't want to lose).

Thread-safety: not safe across threads. Matches LMCache's per-worker
connector lifecycle (single producer per BucketStore instance).

Eviction note: `pybktree.BKTree` doesn't support removal, so bucket
eviction triggers an index rebuild. At max_buckets ≤ 4096 this is
cheap (microseconds) and only happens once per `bucket_capacity`
frames on average.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from pmcache.hashing import BucketIndex

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BucketEntry:
    """One perceptual bucket.

    Anchor is the first frame admitted; Tier-2 verification compares
    candidate frames against `anchor_embedding`. `members` is an LRU
    list of bytewise mm_hashes that have been aliased into this bucket
    (oldest at index 0).
    """

    bucket_id: int
    anchor_phash: int
    anchor_embedding: "np.ndarray"
    anchor_mm_hash: str
    members: list[str] = field(default_factory=list)


class BucketStore:
    """Combined Tier-1 index + bucket bookkeeping + alias map."""

    def __init__(
        self,
        k: int = 5,
        bucket_capacity: int = 128,
        max_buckets: int = 4096,
    ) -> None:
        self.k = k
        self.bucket_capacity = bucket_capacity
        self.max_buckets = max_buckets
        self._index = BucketIndex(k=k)
        # OrderedDict gives us O(1) LRU: most-recent at the end.
        self._buckets: "OrderedDict[int, BucketEntry]" = OrderedDict()
        # Fast-path alias map for exact-mm_hash re-occurrence (avoids
        # re-running DINOv2 for a frame we've already seen).
        self._alias_to_anchor: dict[str, str] = {}
        self._next_bucket_id = 0

    # --- public API ---------------------------------------------------------

    def candidates(self, phash_value: int) -> list[BucketEntry]:
        """Tier-1 candidates whose anchor pHash is within `k` Hamming.

        Touches each candidate as recently-used (so a bucket that keeps
        getting verification-matched against doesn't get evicted while
        a stale anchor sits at the top).
        """
        bucket_ids = self._index.query(phash_value)
        out: list[BucketEntry] = []
        for bid in bucket_ids:
            entry = self._buckets.get(bid)
            if entry is None:
                # Stale index entry from a prior eviction; tolerate it.
                # (Shouldn't happen — we rebuild on eviction — but defensive.)
                continue
            self._buckets.move_to_end(bid)
            out.append(entry)
        return out

    def insert_new_bucket(
        self,
        phash_value: int,
        embedding: "np.ndarray",
        mm_hash: str,
    ) -> BucketEntry:
        """Create a new bucket anchored on this frame.

        Also seeds the alias map with `mm_hash -> mm_hash` so a repeated
        view of the anchor frame itself takes the fast path.
        Evicts the LRU bucket (and rebuilds the index) if at capacity.
        """
        bucket_id = self._next_bucket_id
        self._next_bucket_id += 1
        entry = BucketEntry(
            bucket_id=bucket_id,
            anchor_phash=phash_value,
            anchor_embedding=embedding,
            anchor_mm_hash=mm_hash,
            members=[mm_hash],
        )
        self._buckets[bucket_id] = entry  # appended → most-recent
        self._index.add(phash_value, bucket_id)
        self._alias_to_anchor[mm_hash] = mm_hash
        self._maybe_evict_lru_bucket()
        return entry

    def alias(self, bucket: BucketEntry, mm_hash: str) -> str:
        """Record `mm_hash` as a member of `bucket`. Returns anchor_mm_hash.

        If the member list is at `bucket_capacity`, evicts the oldest
        member (and removes its alias-map entry).
        """
        # If this mm_hash is already a member, just refresh its LRU position.
        if mm_hash in bucket.members:
            bucket.members.remove(mm_hash)
        bucket.members.append(mm_hash)
        self._alias_to_anchor[mm_hash] = bucket.anchor_mm_hash
        if len(bucket.members) > self.bucket_capacity:
            evicted = bucket.members.pop(0)
            # Only drop the alias entry if it still points to *this* bucket's
            # anchor (a later bucket could have claimed the same mm_hash).
            if self._alias_to_anchor.get(evicted) == bucket.anchor_mm_hash:
                self._alias_to_anchor.pop(evicted, None)
        self._buckets.move_to_end(bucket.bucket_id)
        return bucket.anchor_mm_hash

    def lookup_alias(self, mm_hash: str) -> Optional[str]:
        """Fast-path: if `mm_hash` has been aliased before, return the anchor.

        Used by the shim to skip pHash + DINOv2 work when the *exact*
        bytewise mm_hash has been processed earlier in this session.
        """
        return self._alias_to_anchor.get(mm_hash)

    # --- internals ----------------------------------------------------------

    def _maybe_evict_lru_bucket(self) -> None:
        """Evict the least-recently-used bucket, rebuilding the BK-tree.

        Triggered only when `len(self._buckets) > max_buckets`. Cost is
        O(n log n) on n = max_buckets but n is bounded (default 4096) so
        the rebuild is sub-millisecond.
        """
        if len(self._buckets) <= self.max_buckets:
            return
        evicted_id, evicted_entry = self._buckets.popitem(last=False)
        # Drop alias entries that pointed at this bucket's anchor.
        anchor_hash = evicted_entry.anchor_mm_hash
        # Iterate over a snapshot since we mutate during iteration.
        for member in list(evicted_entry.members):
            if self._alias_to_anchor.get(member) == anchor_hash:
                self._alias_to_anchor.pop(member, None)
        # Rebuild the BK-tree from surviving buckets (pybktree has no remove).
        self._index = BucketIndex(k=self.k)
        for bid, entry in self._buckets.items():
            self._index.add(entry.anchor_phash, bid)

    # --- introspection (for tests + metrics) --------------------------------

    def __len__(self) -> int:
        return len(self._buckets)

    @property
    def buckets(self) -> list[BucketEntry]:
        """LRU-ordered view (oldest first). Read-only — for tests/metrics."""
        return list(self._buckets.values())
