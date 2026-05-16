"""Perceptual shim — alias bytewise mm_hashes to perceptual-bucket UUIDs.

## How the alias actually reaches LMCache

vLLM's `MultiModalHasher` (`vllm/multimodal/hasher.py`) checks for
`image.getexif()[Image.ExifTags.Base.ImageID]` first. If a `uuid.UUID`
is present, the hasher uses **only** those UUID bytes and skips pixel
serialization. This is a public extension point.

So instead of subclassing `LMCacheConnectorV1` or monkeypatching
`apply_mm_hashes_to_token_ids`, we tag the image with an anchor UUID
*before* it reaches vLLM. vLLM hashes the UUID → LMCache sees an
exact mm_hash match → its existing KV-reuse path serves the hit.

Net result: zero LMCache changes, zero vLLM internals touched, just
standard PIL EXIF metadata.

## Lifecycle per frame

    prepare_image(image, video_id=None)
        |
        ├─ fast path: same UUID already attached?       → return image as-is
        |
        ├─ pHash(image) → Tier-1 candidates from store
        |
        ├─ no candidates           → new bucket, fresh UUID,         miss
        |
        ├─ for each candidate (closest pHash first):
        |     embedding = verifier.embed(image)
        |     if cosine(emb, anchor_emb) >= τ:
        |         alias → tag with anchor UUID                       HIT
        |         return
        |
        └─ all candidates failed Tier-2                              false_positive
              (new bucket with fresh UUID)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from PIL import Image

from pmcache.config import PMCacheConfig
from pmcache.hashing import phash
from pmcache.metrics import Metrics
from pmcache.store import BucketEntry, BucketStore

if TYPE_CHECKING:
    from pmcache.verifier import DinoV2Verifier


# EXIF tag id for ImageID. We import the value (not the enum) at module load
# so we don't need a fresh PIL lookup per frame. The numeric value (0xA420)
# is stable in the EXIF spec.
_IMAGE_ID_TAG = Image.ExifTags.Base.ImageID.value


def _tag_with_uuid(image: Image.Image, anchor_uuid: uuid.UUID) -> Image.Image:
    """Return a copy of `image` with EXIF ImageID set to `anchor_uuid`.

    PIL's EXIF API mutates an existing exif object; we copy the image
    so callers can pass a shared frame through multiple times without
    aliasing surprises.
    """
    tagged = image.copy()
    exif = tagged.getexif()
    exif[_IMAGE_ID_TAG] = anchor_uuid
    # Note: we don't call image.save() — the EXIF tag lives on the in-memory
    # Image object and that's what vLLM's MultiModalHasher inspects.
    return tagged


def _existing_uuid(image: Image.Image) -> Optional[uuid.UUID]:
    """Read back the ImageID UUID if `image` was already tagged."""
    try:
        exif = image.getexif()
    except Exception:
        return None
    val = exif.get(_IMAGE_ID_TAG)
    if isinstance(val, uuid.UUID):
        return val
    return None


class PerceptualMMCache:
    """The perceptual shim. Construct once per inference worker / run.

    Verifier is constructor-injected (use `pmcache.verifier.DinoV2Verifier`
    in production, a `FakeVerifier` in tests). Lazy-instantiated to the
    real DinoV2Verifier on first use if None.
    """

    def __init__(
        self,
        config: PMCacheConfig,
        verifier: "DinoV2Verifier | None" = None,
        metrics: Metrics | None = None,
        store: BucketStore | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or Metrics()
        self.store = store or BucketStore(
            k=config.k,
            bucket_capacity=config.bucket_capacity,
            max_buckets=config.max_buckets,
        )
        self._verifier = verifier  # lazy if None

    def reset(self) -> None:
        """Clear all perceptual buckets and metrics — keep the verifier.

        Use this between videos when you want each video benchmarked
        cold (no cross-video bucket reuse). Cheap: just discards the
        BucketStore + Metrics; the loaded DinoV2 model is preserved.
        """
        self.store = BucketStore(
            k=self.config.k,
            bucket_capacity=self.config.bucket_capacity,
            max_buckets=self.config.max_buckets,
        )
        self.metrics = Metrics()

    def _get_verifier(self) -> "DinoV2Verifier":
        if self._verifier is None:
            # Lazy import so the shim module itself stays torch-free.
            from pmcache.verifier import DinoV2Verifier

            self._verifier = DinoV2Verifier(device="cpu")
        return self._verifier

    def prepare_image(
        self,
        image: Image.Image,
        video_id: str | None = None,
    ) -> Image.Image:
        """Return a copy of `image` tagged with the appropriate ImageID UUID.

        Pass-through (returns input untouched) if the shim is disabled.
        Idempotent: if `image` is already tagged with a UUID we recognize,
        we don't redo the work.
        """
        if not self.config.enabled:
            return image

        self.metrics.saw_frame(video_id)

        # Fast path: already-tagged image. We trust the upstream caller.
        if (existing := _existing_uuid(image)) is not None:
            # Re-record as a hit (the caller deliberately reused this UUID).
            self.metrics.hit(video_id)
            return image

        h = phash(image)
        candidates = self.store.candidates(h)

        if not candidates:
            new_uuid = uuid.uuid4()
            emb = self._get_verifier().embed(image)
            self.store.insert_new_bucket(h, emb, mm_hash=new_uuid.hex)
            self.metrics.miss(video_id)
            return _tag_with_uuid(image, new_uuid)

        # Tier 2: verify against candidates in pHash-closest-first order.
        emb = self._get_verifier().embed(image)
        for entry in candidates:
            sim = self._get_verifier().cosine(emb, entry.anchor_embedding)
            if sim >= self.config.tau:
                # Verified hit — alias this frame's mm_hash to the anchor.
                anchor_uuid = uuid.UUID(hex=entry.anchor_mm_hash)
                # Record an alias keyed by something unique per frame — use
                # the pHash hex; collisions only occur for identical frames,
                # which is exactly when re-aliasing is correct anyway.
                self.store.alias(entry, mm_hash=f"{h:016x}")
                self.metrics.hit(video_id)
                return _tag_with_uuid(image, anchor_uuid)

        # All Tier-1 candidates rejected at Tier 2.
        new_uuid = uuid.uuid4()
        self.store.insert_new_bucket(h, emb, mm_hash=new_uuid.hex)
        self.metrics.false_positive(video_id)
        return _tag_with_uuid(image, new_uuid)


def build_pmcache(
    config: PMCacheConfig | None = None,
    verifier: "DinoV2Verifier | None" = None,
) -> PerceptualMMCache | None:
    """Factory: build the shim from env (or explicit config).

    Returns `None` if `config.enabled is False` so the eval runners can
    pattern-match on it (`if pmcache is None: ...`) instead of carrying
    a no-op object through the hot path.
    """
    cfg = config or PMCacheConfig.from_env()
    if not cfg.enabled:
        return None
    return PerceptualMMCache(cfg, verifier=verifier)
