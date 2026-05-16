"""PMCacheConfig — central config with env-var overrides.

Read once at shim install time. Env vars (`PMCACHE_ENABLED`,
`PMCACHE_TAU`, `PMCACHE_K`) let the Colab notebook flip behavior between
the baseline and perceptual runs without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_bool(s: str | None, default: bool) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PMCacheConfig:
    """Runtime config for the pmcache shim.

    Attributes:
        enabled: master switch. When False, the shim is a no-op pass-through.
        tau: DINOv2 cosine threshold for Tier 2 verification (higher = stricter).
        k: max Hamming distance for Tier 1 pHash bucket match.
        bucket_capacity: max members per perceptual bucket (LRU eviction within).
        max_buckets: max total buckets (LRU eviction across).
    """

    enabled: bool = True
    tau: float = 0.98
    k: int = 5
    bucket_capacity: int = 128
    max_buckets: int = 4096

    @classmethod
    def from_env(cls) -> "PMCacheConfig":
        """Build a config from PMCACHE_* environment variables."""
        return cls(
            enabled=_parse_bool(os.environ.get("PMCACHE_ENABLED"), default=True),
            tau=float(os.environ.get("PMCACHE_TAU", cls.tau)),
            k=int(os.environ.get("PMCACHE_K", cls.k)),
            bucket_capacity=int(
                os.environ.get("PMCACHE_BUCKET_CAPACITY", cls.bucket_capacity)
            ),
            max_buckets=int(os.environ.get("PMCACHE_MAX_BUCKETS", cls.max_buckets)),
        )
