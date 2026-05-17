import {
  BASELINE_SUMMARY,
  DELTAS,
  PERCEPTUAL_SUMMARY,
  formatBytes,
  formatPercent,
  formatSeconds,
} from "../data/metrics";

interface Row {
  metric: string;
  baseline: string;
  perceptual: string;
  delta: string;
  highlight?: "good" | "neutral";
  hint?: string;
}

export default function MetricsTable() {
  const rows: Row[] = [
    {
      metric: "rows",
      baseline: `${BASELINE_SUMMARY.rows}`,
      perceptual: `${PERCEPTUAL_SUMMARY.rows}`,
      delta: "—",
      highlight: "neutral",
    },
    {
      metric: "accuracy",
      baseline: formatPercent(BASELINE_SUMMARY.accuracy * 100, 1),
      perceptual: formatPercent(PERCEPTUAL_SUMMARY.accuracy * 100, 1),
      delta: `${formatPercent(DELTAS.accuracyDeltaPp, 1, true)}pp`,
      highlight: "neutral",
      hint: "MCQ letter match · parity preserved",
    },
    {
      metric: "mm_hash dedup rate (intra-prompt)",
      baseline: formatPercent(BASELINE_SUMMARY.dedupRate * 100, 1),
      perceptual: formatPercent(PERCEPTUAL_SUMMARY.dedupRate * 100, 1),
      delta: `${formatPercent(DELTAS.dedupRateDeltaPp, 1, true)}pp`,
      highlight: "good",
      hint: "pmcache's direct win — hash repetition within a single prompt",
    },
    {
      metric: "TTFT p50",
      baseline: formatSeconds(BASELINE_SUMMARY.ttftP50, 4),
      perceptual: formatSeconds(PERCEPTUAL_SUMMARY.ttftP50, 4),
      delta: `${((PERCEPTUAL_SUMMARY.ttftP50 - BASELINE_SUMMARY.ttftP50) * 1000).toFixed(1)} ms`,
      highlight: "good",
    },
    {
      metric: "TTFT p99",
      baseline: formatSeconds(BASELINE_SUMMARY.ttftP99, 4),
      perceptual: formatSeconds(PERCEPTUAL_SUMMARY.ttftP99, 4),
      delta: `${((PERCEPTUAL_SUMMARY.ttftP99 - BASELINE_SUMMARY.ttftP99) * 1000).toFixed(1)} ms`,
      highlight: "good",
    },
    {
      metric: "TTFT total (sum)",
      baseline: formatSeconds(BASELINE_SUMMARY.ttftTotal, 2),
      perceptual: formatSeconds(PERCEPTUAL_SUMMARY.ttftTotal, 2),
      delta: `${formatSeconds(DELTAS.ttftTotalSavedSec * -1, 2)} (${formatPercent(DELTAS.ttftTotalDeltaPct, 1, true)})`,
      highlight: "good",
      hint: "wall-clock acceleration",
    },
    {
      metric: "total frames sampled",
      baseline: BASELINE_SUMMARY.totalFrames.toLocaleString(),
      perceptual: PERCEPTUAL_SUMMARY.totalFrames.toLocaleString(),
      delta: "—",
    },
    {
      metric: "unique frames (post-alias)",
      baseline: BASELINE_SUMMARY.uniqueFrames.toLocaleString(),
      perceptual: PERCEPTUAL_SUMMARY.uniqueFrames.toLocaleString(),
      delta: `${DELTAS.uniqueFramesSaved} fewer`,
      highlight: "good",
    },
    {
      metric: "vision KV pressure (unique-frame est.)",
      baseline: formatBytes(BASELINE_SUMMARY.visionKvPressure, 2),
      perceptual: formatBytes(PERCEPTUAL_SUMMARY.visionKvPressure, 2),
      delta: `${formatBytes(DELTAS.visionKvSavedBytes, 2)} saved (${formatPercent(DELTAS.visionKvDeltaPct, 1, true)})`,
      highlight: "good",
      hint: "Σ(unique_frames × prompt_tok/frame × KV_bytes/tok) — the metric pmcache reduces",
    },
    {
      metric: "KV bytes recomputed (vLLM prefix-cache)",
      baseline: formatBytes(BASELINE_SUMMARY.kvBytesRecomputed, 2),
      perceptual: formatBytes(PERCEPTUAL_SUMMARY.kvBytesRecomputed, 2),
      delta: "0.0 B saved (+0.0%)",
      hint: "vLLM only credits cross-call prefix hits · symmetric in both variants",
    },
    {
      metric: "KV bytes total (prompt)",
      baseline: formatBytes(BASELINE_SUMMARY.kvBytesTotal, 2),
      perceptual: formatBytes(PERCEPTUAL_SUMMARY.kvBytesTotal, 2),
      delta: "—",
    },
  ];

  return (
    <section className="section">
      <div className="eyebrow">Full metric table</div>
      <h2 className="section-title mt-2">Every number from the run</h2>
      <p className="section-subtitle">
        The same table the analyzer writes to{" "}
        <code className="font-mono">REPORT.md</code>. Highlighted rows are the
        ones that move with pmcache; non-highlighted rows are invariants or
        cross-call metrics that are blind to within-prompt aliasing.
      </p>

      <div className="mt-8 card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-cream-100">
            <tr>
              <th className="table-th">Metric</th>
              <th className="table-th text-right">Baseline</th>
              <th className="table-th text-right">Perceptual</th>
              <th className="table-th text-right">Δ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.metric}
                className={
                  r.highlight === "good"
                    ? "bg-perceptual/[0.04]"
                    : "hover:bg-cream-50"
                }
              >
                <td className="table-td">
                  <div className="text-ink-900">{r.metric}</div>
                  {r.hint && (
                    <div className="text-xs text-ink-400 mt-0.5">{r.hint}</div>
                  )}
                </td>
                <td className="table-td text-right font-mono text-ink-700">
                  {r.baseline}
                </td>
                <td className="table-td text-right font-mono text-ink-900">
                  {r.perceptual}
                </td>
                <td
                  className={`table-td text-right font-mono ${
                    r.highlight === "good" ? "text-perceptual" : "text-ink-400"
                  }`}
                >
                  {r.delta}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
