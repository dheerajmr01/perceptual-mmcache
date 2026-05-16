"""Metrics — hit / miss / false-positive counters with JSON dump.

Single-producer (the perceptual shim runs inside the LMCache worker,
one producer per BucketStore instance), so counters are plain ints —
no locking. `dump()` snapshots atomically via `dataclasses.asdict`.

Implemented in Step 5 (alongside the shim) since the shim is the only
caller — folding the two together avoided one round of "later step
references earlier step's stubs."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Metrics:
    """Running counters for the perceptual cache.

    Naming:
        frames_seen     total prepare_image() calls
        tier1_hits      pHash matched at least one candidate bucket
        tier2_hits      cosine ≥ τ on a Tier-1 candidate (== aliased_hashes)
        tier2_rejects   Tier-1 hit but cosine < τ (false positive avoided)
        misses          no Tier-1 candidate at all (and the Tier-1+Tier-2
                        rejected case also counts as a miss outcome)
        aliased_hashes  bytewise hashes rewritten to the anchor UUID
        per_video       optional bookkeeping (video_id -> sub-counters)
    """

    frames_seen: int = 0
    tier1_hits: int = 0
    tier2_hits: int = 0
    tier2_rejects: int = 0
    misses: int = 0
    aliased_hashes: int = 0
    per_video: dict[str, dict[str, int]] = field(default_factory=dict)

    # --- counter increments (called by the shim) ----------------------------

    def saw_frame(self, video_id: str | None = None) -> None:
        self.frames_seen += 1
        if video_id:
            self._video(video_id)["frames_seen"] += 1

    def hit(self, video_id: str | None = None) -> None:
        """Tier-2 verified hit: bytewise mm_hash rewritten to anchor UUID."""
        self.tier1_hits += 1
        self.tier2_hits += 1
        self.aliased_hashes += 1
        if video_id:
            v = self._video(video_id)
            v["tier1_hits"] += 1
            v["tier2_hits"] += 1
            v["aliased_hashes"] += 1

    def miss(self, video_id: str | None = None) -> None:
        """No Tier-1 candidate — new bucket inserted."""
        self.misses += 1
        if video_id:
            self._video(video_id)["misses"] += 1

    def false_positive(self, video_id: str | None = None) -> None:
        """Tier-1 candidate(s) existed but all failed Tier-2 verification."""
        self.tier1_hits += 1
        self.tier2_rejects += 1
        self.misses += 1
        if video_id:
            v = self._video(video_id)
            v["tier1_hits"] += 1
            v["tier2_rejects"] += 1
            v["misses"] += 1

    # --- helpers ------------------------------------------------------------

    def _video(self, video_id: str) -> dict[str, int]:
        return self.per_video.setdefault(
            video_id,
            {
                "frames_seen": 0,
                "tier1_hits": 0,
                "tier2_hits": 0,
                "tier2_rejects": 0,
                "misses": 0,
                "aliased_hashes": 0,
            },
        )

    def hit_rate(self) -> float:
        """Fraction of frames that resulted in an aliased mm_hash."""
        if self.frames_seen == 0:
            return 0.0
        return self.aliased_hashes / self.frames_seen

    def dump(self, path: str | Path) -> None:
        """Write a JSON snapshot of all counters."""
        Path(path).write_text(json.dumps(asdict(self), indent=2))
