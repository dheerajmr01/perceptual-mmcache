import { ArrowDown, Zap, Database, Target } from "lucide-react";
import { DELTAS, formatBytes, formatPercent, formatSeconds } from "../data/metrics";

export default function Hero() {
  return (
    <section id="hero" className="section pt-12 pb-24">
      <div className="flex flex-col items-start gap-6">
        <span className="badge-highlight">
          <Zap className="h-3.5 w-3.5" />
          Hackathon track · Acceleration
        </span>

        <h1 className="text-5xl md:text-7xl font-bold tracking-tight leading-[1.05]">
          <span className="text-ink-900">Accelerate video-LLMs by</span>
          <br />
          <span className="gradient-text">ignoring frames you've already seen.</span>
        </h1>

        <p className="max-w-3xl text-lg md:text-xl text-ink-500 leading-relaxed">
          <strong className="text-ink-800">pmcache</strong> (perceptual mm-cache)
          aliases visually-similar video frames to a single anchor{" "}
          <code className="text-perceptual font-mono">mm_hash</code>, so vLLM's
          vision encoder runs <em>once per anchor</em> instead of once per frame.
          Zero changes to vLLM internals, zero accuracy loss, measurable
          speed-up and KV-memory savings.
        </p>

        <div className="flex flex-wrap gap-3 mt-2">
          <a
            href="#results"
            className="inline-flex items-center gap-2 rounded-lg bg-perceptual px-5 py-3 text-sm font-semibold text-cream-50 hover:bg-perceptual/90 transition-colors shadow-matte"
          >
            See the benchmark results
            <ArrowDown className="h-4 w-4" />
          </a>
          <a
            href="#architecture"
            className="inline-flex items-center gap-2 rounded-lg border border-cream-300 bg-white px-5 py-3 text-sm font-semibold text-ink-800 hover:bg-cream-100 transition-colors"
          >
            How it works
          </a>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full mt-10">
          <StatCard
            icon={<Zap className="h-5 w-5 text-perceptual" />}
            label="TTFT acceleration"
            value={formatPercent(-DELTAS.ttftTotalDeltaPct, 1, true)}
            sub={`${formatSeconds(DELTAS.ttftTotalSavedSec, 2)} saved over 27 calls`}
            accent="perceptual"
          />
          <StatCard
            icon={<Database className="h-5 w-5 text-highlight" />}
            label="Vision KV pressure reduced"
            value={formatPercent(-DELTAS.visionKvDeltaPct, 1, true)}
            sub={`${formatBytes(DELTAS.visionKvSavedBytes, 1)} freed`}
            accent="highlight"
          />
          <StatCard
            icon={<Target className="h-5 w-5 text-ocean" />}
            label="Accuracy delta"
            value={`${formatPercent(DELTAS.accuracyDeltaPp, 1, true)}pp`}
            sub="parity preserved · 55.6% on both"
            accent="ocean"
          />
        </div>
      </div>
    </section>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
  accent: "perceptual" | "highlight" | "ocean";
}) {
  const left = {
    perceptual: "border-l-perceptual",
    highlight: "border-l-highlight",
    ocean: "border-l-ocean",
  }[accent];
  return (
    <div className={`card p-6 border-l-4 ${left}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-wider text-ink-400 font-medium">
          {label}
        </div>
        {icon}
      </div>
      <div className="stat-num text-4xl md:text-5xl text-ink-900">{value}</div>
      <div className="mt-2 text-sm text-ink-500">{sub}</div>
    </div>
  );
}
