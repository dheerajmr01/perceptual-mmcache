import { TrendingDown, Database, Target, Zap, Layers } from "lucide-react";
import {
  BASELINE_SUMMARY,
  DELTAS,
  PERCEPTUAL_SUMMARY,
  formatBytes,
  formatPercent,
  formatSeconds,
} from "../data/metrics";

export default function HeadlineMetrics() {
  return (
    <section id="results" className="section">
      <div className="eyebrow">Results</div>
      <h2 className="section-title mt-2">The acceleration story, in three numbers</h2>
      <p className="section-subtitle">
        Aggregated across 27 (video × question) pairs, 9 unique videos, 2,643
        sampled frames. Same model, same prompts — only the EXIF UUID tagging
        differs.
      </p>

      <div className="grid md:grid-cols-3 gap-4 mt-10">
        <BigStat
          icon={<Zap className="h-5 w-5 text-perceptual" />}
          title="TTFT acceleration"
          delta={formatPercent(-DELTAS.ttftTotalDeltaPct, 1, true)}
          left={{
            label: "baseline TTFT total",
            value: formatSeconds(BASELINE_SUMMARY.ttftTotal, 2),
          }}
          right={{
            label: "perceptual TTFT total",
            value: formatSeconds(PERCEPTUAL_SUMMARY.ttftTotal, 2),
          }}
          footer={`${formatSeconds(DELTAS.ttftTotalSavedSec, 2)} saved`}
          accent="perceptual"
        />

        <BigStat
          icon={<Database className="h-5 w-5 text-highlight" />}
          title="Vision KV pressure"
          delta={formatPercent(-DELTAS.visionKvDeltaPct, 1, true)}
          left={{
            label: "baseline pressure",
            value: formatBytes(BASELINE_SUMMARY.visionKvPressure, 1),
          }}
          right={{
            label: "perceptual pressure",
            value: formatBytes(PERCEPTUAL_SUMMARY.visionKvPressure, 1),
          }}
          footer={`${formatBytes(DELTAS.visionKvSavedBytes, 1)} freed`}
          accent="highlight"
        />

        <BigStat
          icon={<Target className="h-5 w-5 text-ocean" />}
          title="Accuracy parity"
          delta={`${formatPercent(DELTAS.accuracyDeltaPp, 1, true)}pp`}
          left={{
            label: "baseline accuracy",
            value: formatPercent(BASELINE_SUMMARY.accuracy * 100, 1),
          }}
          right={{
            label: "perceptual accuracy",
            value: formatPercent(PERCEPTUAL_SUMMARY.accuracy * 100, 1),
          }}
          footer="MCQ A/B/C/D · letter match"
          accent="ocean"
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
        <Small
          icon={<TrendingDown className="h-4 w-4 text-perceptual" />}
          label="Unique frames saved"
          value={`${DELTAS.uniqueFramesSaved.toLocaleString()}`}
          sub={`${BASELINE_SUMMARY.uniqueFrames} → ${PERCEPTUAL_SUMMARY.uniqueFrames}`}
        />
        <Small
          icon={<Layers className="h-4 w-4 text-perceptual" />}
          label="Dedup rate (intra-prompt)"
          value={formatPercent(PERCEPTUAL_SUMMARY.dedupRate * 100, 1)}
          sub={`${formatPercent(BASELINE_SUMMARY.dedupRate * 100, 1)} → ${formatPercent(PERCEPTUAL_SUMMARY.dedupRate * 100, 1)} (${formatPercent(DELTAS.dedupRateDeltaPp, 1, true)}pp)`}
        />
        <Small
          icon={<Zap className="h-4 w-4 text-perceptual" />}
          label="TTFT p50"
          value={formatSeconds(PERCEPTUAL_SUMMARY.ttftP50, 2)}
          sub={`baseline ${formatSeconds(BASELINE_SUMMARY.ttftP50, 2)}`}
        />
        <Small
          icon={<Zap className="h-4 w-4 text-perceptual" />}
          label="TTFT p99"
          value={formatSeconds(PERCEPTUAL_SUMMARY.ttftP99, 2)}
          sub={`baseline ${formatSeconds(BASELINE_SUMMARY.ttftP99, 2)}`}
        />
      </div>
    </section>
  );
}

function BigStat({
  icon,
  title,
  delta,
  left,
  right,
  footer,
  accent,
}: {
  icon: React.ReactNode;
  title: string;
  delta: string;
  left: { label: string; value: string };
  right: { label: string; value: string };
  footer: string;
  accent: "perceptual" | "highlight" | "ocean";
}) {
  const border = {
    perceptual: "border-l-perceptual",
    highlight: "border-l-highlight",
    ocean: "border-l-ocean",
  }[accent];
  const text = {
    perceptual: "text-perceptual",
    highlight: "text-highlight",
    ocean: "text-ocean",
  }[accent];
  return (
    <div className={`card p-6 border-l-4 ${border}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {icon}
          <div className="text-sm font-medium text-ink-800">{title}</div>
        </div>
      </div>
      <div className={`text-5xl font-mono font-semibold ${text}`}>{delta}</div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-md border border-cream-200 bg-cream-50 p-3">
          <div className="text-ink-400 uppercase tracking-wider">{left.label}</div>
          <div className="font-mono text-ink-900 mt-1">{left.value}</div>
        </div>
        <div className="rounded-md border border-cream-200 bg-cream-50 p-3">
          <div className="text-ink-400 uppercase tracking-wider">{right.label}</div>
          <div className="font-mono text-ink-900 mt-1">{right.value}</div>
        </div>
      </div>
      <div className="mt-3 text-xs text-ink-400">{footer}</div>
    </div>
  );
}

function Small({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 text-xs text-ink-500">
        {icon}
        <span>{label}</span>
      </div>
      <div className="mt-1 text-2xl font-mono text-ink-900">{value}</div>
      <div className="text-[11px] text-ink-400 mt-0.5">{sub}</div>
    </div>
  );
}
