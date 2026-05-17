import { Code2, Terminal, Rocket, FileCode } from "lucide-react";

const TAG_CODE = `# pmcache/lmcache_shim.py — the entire EXIF-tagging fast path
def _tag_with_uuid(image: Image.Image, anchor_uuid: uuid.UUID) -> Image.Image:
    """Return a copy of \`image\` with EXIF ImageID set to \`anchor_uuid\`.

    PIL's \`image.copy()\` does NOT preserve EXIF — copy first, then write.
    """
    tagged = image.copy()
    exif = tagged.getexif()
    exif[_IMAGE_ID_TAG] = anchor_uuid
    return tagged`;

const PREPARE_CODE = `def prepare_image(self, image, video_id=None):
    if not self.config.enabled:
        return image

    self.metrics.saw_frame(video_id)

    # Fast path: already tagged (e.g. cached PIL handle).
    if (existing := _existing_uuid(image)) is not None:
        self.metrics.hit(video_id)
        return image

    # Tier 1 — pHash bucket candidates.
    h = phash(image)
    candidates = self.store.candidates(h)

    if not candidates:
        new_uuid = uuid.uuid4()
        emb = self._get_verifier().embed(image)
        self.store.insert_new_bucket(h, emb, mm_hash=new_uuid.hex)
        self.metrics.miss(video_id)
        return _tag_with_uuid(image, new_uuid)

    # Tier 2 — DINOv2 cosine verification.
    emb = self._get_verifier().embed(image)
    for entry in candidates:
        sim = self._get_verifier().cosine(emb, entry.anchor_embedding)
        if sim >= self.config.tau:
            anchor_uuid = uuid.UUID(hex=entry.anchor_mm_hash)
            self.store.alias(entry, mm_hash=f"{h:016x}")
            self.metrics.hit(video_id)
            return _tag_with_uuid(image, anchor_uuid)

    # All candidates failed Tier 2 — fresh bucket.
    new_uuid = uuid.uuid4()
    self.store.insert_new_bucket(h, emb, mm_hash=new_uuid.hex)
    self.metrics.false_positive(video_id)
    return _tag_with_uuid(image, new_uuid)`;

const VLLM_CODE = `# vllm/multimodal/hasher.py (verbatim, vLLM source)
# This is the public extension point we hook into:
def _serialize_item(obj):
    if isinstance(obj, Image.Image):
        exif = obj.getexif()
        if (Image.ExifTags.Base.ImageID in exif and
                isinstance(exif[Image.ExifTags.Base.ImageID], uuid.UUID)):
            # mm_hash uses ONLY the UUID bytes — pixel serialization is
            # skipped entirely.
            return (exif[Image.ExifTags.Base.ImageID].bytes,)
        # ... fallback: serialize pixel array
`;

const RUN_CODE = `# Colab Pro · A100-40GB
pip install -e '.[mvbench,eval]'
python -m eval.datasets.prepare_videomme --output_dir workspace/videos --n 10
python -c "
from eval.run_baseline import run_baseline_benchmark
from eval.run_perceptual import run_perceptual_benchmark
from eval import paths

run_baseline_benchmark(paths.VIDEOS_DIR, paths.QA_FILE, paths.BASELINE_PATH,
                      model='Qwen/Qwen3-VL-2B-Instruct', fps=1.0)
run_perceptual_benchmark(paths.VIDEOS_DIR, paths.QA_FILE, paths.PERCEPTUAL_PATH,
                        model='Qwen/Qwen3-VL-2B-Instruct',
                        tau=0.98, k=5, fps=1.0)
"`;

export default function Implementation() {
  return (
    <section id="implementation" className="section">
      <div className="eyebrow">Implementation</div>
      <h2 className="section-title mt-2">The whole shim, in 30 lines</h2>
      <p className="section-subtitle">
        No subclasses. No monkeypatches. Just PIL EXIF and a two-tier
        candidate-then-verify pipeline. The full module is{" "}
        <code className="font-mono text-perceptual">pmcache/lmcache_shim.py</code>{" "}
        in the repo.
      </p>

      <div className="mt-10 space-y-6">
        <CodeBlock
          icon={<FileCode className="h-4 w-4 text-perceptual" />}
          title="The EXIF tagger (3 lines that matter)"
          subtitle="PIL EXIF is the public vLLM extension point we hook into"
          code={TAG_CODE}
          tone="perceptual"
        />

        <CodeBlock
          icon={<Code2 className="h-4 w-4 text-perceptual" />}
          title="prepare_image — the full two-tier flow"
          subtitle="Tier 1 (CPU pHash + BK-tree) → Tier 2 (GPU DINOv2 cosine) → tag with anchor UUID"
          code={PREPARE_CODE}
          tone="perceptual"
        />

        <CodeBlock
          icon={<Rocket className="h-4 w-4 text-highlight" />}
          title="Why this works — vLLM's MultiModalHasher"
          subtitle="Public hashing logic. If ImageID is a UUID, vLLM skips pixel hashing entirely."
          code={VLLM_CODE}
          tone="highlight"
        />

        <CodeBlock
          icon={<Terminal className="h-4 w-4 text-ocean" />}
          title="Run it yourself"
          subtitle="Three commands. The Colab notebook bundles GPU setup + plots."
          code={RUN_CODE}
          tone="ocean"
        />
      </div>
    </section>
  );
}

function CodeBlock({
  icon,
  title,
  subtitle,
  code,
  tone = "perceptual",
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  code: string;
  tone?: "perceptual" | "highlight" | "ocean";
}) {
  const accent = {
    perceptual: "border-l-perceptual",
    highlight: "border-l-highlight",
    ocean: "border-l-ocean",
  }[tone];
  return (
    <div className={`card border-l-4 ${accent} overflow-hidden`}>
      <div className="flex items-center gap-2 px-5 py-3 border-b border-cream-200">
        {icon}
        <div>
          <div className="text-sm font-medium text-ink-900">{title}</div>
          <div className="text-xs text-ink-400">{subtitle}</div>
        </div>
      </div>
      <pre className="px-5 py-4 overflow-x-auto text-xs leading-relaxed font-mono text-ink-700 bg-cream-50">
        <code>{code}</code>
      </pre>
    </div>
  );
}
