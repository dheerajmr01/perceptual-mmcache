import { ArrowDown, ArrowRight } from "lucide-react";

export default function Architecture() {
  return (
    <section id="architecture" className="section">
      <div className="eyebrow">Architecture</div>
      <h2 className="section-title mt-2">From PIL frame to LMCache hit</h2>
      <p className="section-subtitle">
        The full data flow. Anything in the upper boxes runs on CPU; the
        Tier-2 path uses the GPU (DINOv2-small) only when a Tier-1 candidate
        exists.
      </p>

      <div className="mt-10 card p-8 overflow-x-auto">
        <FlowDiagram />
      </div>
    </section>
  );
}

function FlowDiagram() {
  return (
    <div className="min-w-[700px] mx-auto">
      {/* Row 1: Input */}
      <div className="flex justify-center">
        <Node label="sample_frames(video, fps=1)" sub="opencv → PIL Image[]" tone="input" />
      </div>
      <Arrow />

      {/* Row 2: Shim entry */}
      <div className="flex justify-center">
        <Node
          label="PerceptualMMCache.prepare_image()"
          sub="EXIF ImageID UUID approach"
          tone="shim"
          wide
        />
      </div>
      <Arrow />

      {/* Row 3: Two-tier matching */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <TierBox
          step="Tier 1 — pHash"
          tech="imagehash + pybktree · CPU"
          desc="phash(image) → BK-tree.find(Hamming ≤ K). Returns candidate anchors."
          tone="tier1"
        />
        <TierBox
          step="Tier 2 — DINOv2 cosine"
          tech="DINOv2-small · GPU"
          desc="For each candidate: cosine(verifier.embed(image), anchor_embedding) ≥ τ?"
          tone="tier2"
        />
      </div>

      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
        <Outcome
          tag="hit"
          color="perceptual"
          label="Aliased"
          detail="EXIF ImageID := anchor UUID"
        />
        <Outcome
          tag="reject"
          color="neutral"
          label="False positive avoided"
          detail="new bucket with fresh UUID"
        />
        <Outcome
          tag="miss"
          color="highlight"
          label="No candidates"
          detail="new bucket with fresh UUID"
        />
      </div>

      <Arrow />

      {/* Row 4: vLLM hashing */}
      <div className="flex justify-center">
        <Node
          label="vLLM MultiModalHasher reads EXIF"
          sub="if ImageID is uuid.UUID → mm_hash = uuid.bytes (skips pixel hash)"
          tone="vllm"
          wide
        />
      </div>
      <Arrow />

      {/* Row 5: LMCache */}
      <div className="flex justify-center">
        <Node
          label="LMCache exact-match path"
          sub="same mm_hash for all aliased frames → KV blocks reused"
          tone="lmcache"
          wide
        />
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex justify-center my-3">
      <ArrowDown className="h-5 w-5 text-ink-300" />
    </div>
  );
}

function Node({
  label,
  sub,
  tone,
  wide,
}: {
  label: string;
  sub?: string;
  tone: "input" | "shim" | "vllm" | "lmcache";
  wide?: boolean;
}) {
  const palette = {
    input: "border-ocean/30 bg-ocean/5 text-ink-800",
    shim: "border-perceptual/30 bg-perceptual/5 text-ink-800",
    vllm: "border-highlight/30 bg-highlight/5 text-ink-800",
    lmcache: "border-ink-300 bg-cream-100 text-ink-800",
  }[tone];
  return (
    <div
      className={`rounded-xl border ${palette} px-5 py-3 text-center shadow-matte ${
        wide ? "min-w-[420px]" : "min-w-[280px]"
      }`}
    >
      <div className="font-mono font-semibold text-sm">{label}</div>
      {sub && <div className="text-xs text-ink-500 mt-1">{sub}</div>}
    </div>
  );
}

function TierBox({
  step,
  tech,
  desc,
  tone,
}: {
  step: string;
  tech: string;
  desc: string;
  tone: "tier1" | "tier2";
}) {
  const palette = {
    tier1: "border-perceptual/30 bg-perceptual/[0.04]",
    tier2: "border-ocean/30 bg-ocean/[0.04]",
  }[tone];
  return (
    <div className={`rounded-xl border ${palette} p-4`}>
      <div className="flex items-center justify-between">
        <div className="font-semibold text-ink-900 text-sm">{step}</div>
        <ArrowRight className="h-3.5 w-3.5 text-ink-300" />
      </div>
      <div className="mt-1 text-[11px] uppercase tracking-wider text-ink-400 font-mono">
        {tech}
      </div>
      <p className="mt-2 text-xs text-ink-700 leading-relaxed">{desc}</p>
    </div>
  );
}

function Outcome({
  tag,
  color,
  label,
  detail,
}: {
  tag: string;
  color: "perceptual" | "highlight" | "neutral";
  label: string;
  detail: string;
}) {
  const palette = {
    perceptual: "border-perceptual/40 bg-perceptual/10 text-ink-900",
    highlight: "border-highlight/40 bg-highlight/10 text-ink-900",
    neutral: "border-ink-200 bg-cream-100 text-ink-700",
  }[color];
  return (
    <div className={`rounded-lg border ${palette} px-3 py-2`}>
      <div className="text-[10px] uppercase tracking-widest opacity-70">{tag}</div>
      <div className="font-medium text-sm mt-0.5">{label}</div>
      <div className="text-xs opacity-80 mt-0.5">{detail}</div>
    </div>
  );
}
