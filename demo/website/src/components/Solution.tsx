import { Layers, ShieldCheck, Tag, CheckCircle2 } from "lucide-react";

export default function Solution() {
  return (
    <section id="solution" className="section">
      <div className="eyebrow">The solution</div>
      <h2 className="section-title mt-2">
        Two-tier perceptual matching → one anchor UUID → vLLM&apos;s public extension point.
      </h2>
      <p className="section-subtitle">
        pmcache tags each PIL image with an EXIF{" "}
        <code className="font-mono text-perceptual">ImageID</code> UUID{" "}
        <em>before</em> vLLM sees it. vLLM&apos;s{" "}
        <code className="font-mono">MultiModalHasher</code> already prioritizes
        EXIF UUID over pixel bytes (a public extension point), so similar
        frames sharing a UUID hit LMCache&apos;s existing exact-match path. No
        subclasses, no monkeypatches, no internal API risk.
      </p>

      <div className="grid md:grid-cols-3 gap-4 mt-10">
        <SolutionStep
          step="1"
          icon={<Layers className="h-5 w-5 text-perceptual" />}
          title="Tier 1 — pHash candidate retrieval"
          body="64-bit perceptual hash (DCT-based) of each frame, indexed in a BK-tree. Find all anchors within Hamming distance K=5 in O(log n)."
          tech="imagehash + pybktree · microseconds CPU"
        />
        <SolutionStep
          step="2"
          icon={<ShieldCheck className="h-5 w-5 text-perceptual" />}
          title="Tier 2 — DINOv2 cosine verification"
          body="Confirm visual identity with DINOv2-small embedding cosine ≥ τ=0.98. Rejects perceptual-hash false positives (different content that happens to hash close)."
          tech="DINOv2-small · ~30 ms GPU per frame"
        />
        <SolutionStep
          step="3"
          icon={<Tag className="h-5 w-5 text-perceptual" />}
          title="Tag with anchor UUID, hand to vLLM"
          body="Set EXIF ImageID = anchor's UUID on a copy of the PIL image. vLLM's MultiModalHasher reads the UUID and uses it as the mm_hash — LMCache treats the frame as a cache hit."
          tech="PIL EXIF · zero LMCache changes"
        />
      </div>

      <div className="card p-6 mt-10">
        <div className="flex items-center gap-2 mb-3">
          <CheckCircle2 className="h-5 w-5 text-perceptual" />
          <h3 className="text-lg font-semibold text-ink-900">
            Why this design wins
          </h3>
        </div>
        <ul className="grid md:grid-cols-2 gap-x-8 gap-y-3 text-sm text-ink-700">
          <li className="flex gap-3">
            <span className="text-perceptual mt-1">▸</span>
            <span>
              <strong className="text-ink-900">Public extension point.</strong>{" "}
              We use what vLLM already supports — no subclass to maintain
              across releases.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-perceptual mt-1">▸</span>
            <span>
              <strong className="text-ink-900">Two-tier safety net.</strong>{" "}
              pHash alone has false positives; DINOv2 alone is too expensive
              per frame. Combining them gets recall + precision + speed.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-perceptual mt-1">▸</span>
            <span>
              <strong className="text-ink-900">Decoupled from vLLM internals.</strong>{" "}
              The shim sits in the model-input pipeline. No LMCache changes,
              no engine code, no risk of break-on-upgrade.
            </span>
          </li>
          <li className="flex gap-3">
            <span className="text-perceptual mt-1">▸</span>
            <span>
              <strong className="text-ink-900">Tunable per workload.</strong> τ
              controls precision (higher = safer); K controls recall (wider =
              more candidates). Default settings hold accuracy with measurable
              speed-up.
            </span>
          </li>
        </ul>
      </div>
    </section>
  );
}

function SolutionStep({
  step,
  icon,
  title,
  body,
  tech,
}: {
  step: string;
  icon: React.ReactNode;
  title: string;
  body: string;
  tech: string;
}) {
  return (
    <div className="card p-6 card-hover relative">
      <div className="absolute -top-3 -left-3 grid h-8 w-8 place-items-center rounded-full bg-perceptual text-cream-50 text-sm font-bold shadow-matte">
        {step}
      </div>
      <div className="mb-3 flex items-center gap-2">{icon}</div>
      <h3 className="text-base font-semibold text-ink-900">{title}</h3>
      <p className="mt-2 text-sm text-ink-500 leading-relaxed">{body}</p>
      <div className="mt-3 text-xs font-mono text-ink-400 border-t border-cream-200 pt-3">
        {tech}
      </div>
    </div>
  );
}
