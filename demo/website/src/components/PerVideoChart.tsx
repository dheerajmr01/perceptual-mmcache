import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import { perVideoStats } from "../data/metrics";

const COLOR_BASELINE = "#78736b";
const COLOR_PERCEPTUAL = "#3f7d57";
const COLOR_NEGATIVE = "#a8444d";
const COLOR_GRID = "#ebe1cb";
const COLOR_AXIS = "#78736b";

const tooltipStyle = {
  background: "#ffffff",
  border: "1px solid #ebe1cb",
  borderRadius: 8,
  fontSize: 12,
  color: "#2d2925",
  boxShadow:
    "0 1px 0 0 rgba(31, 29, 25, 0.05), 0 4px 12px -2px rgba(31, 29, 25, 0.08)",
};
const tooltipLabelStyle = { color: "#5a554d" };
const legendStyle = { fontSize: 12, color: "#5a554d" };

export default function PerVideoChart() {
  const stats = perVideoStats();

  const ttftData = stats.map((s) => ({
    video: s.video.replace(".mp4", ""),
    baseline: +s.baseTtft.toFixed(3),
    perceptual: +s.percTtft.toFixed(3),
    speedupPct: s.baseTtft > 0 ? ((s.baseTtft - s.percTtft) / s.baseTtft) * 100 : 0,
  }));

  const uniqueData = stats.map((s) => ({
    video: s.video.replace(".mp4", ""),
    baseline: +s.baseUnique.toFixed(1),
    perceptual: +s.percUnique.toFixed(1),
    totalFrames: +s.baseTotalFrames.toFixed(0),
    reduction: s.baseUnique - s.percUnique,
  }));

  return (
    <section className="section">
      <div className="eyebrow">Per-video breakdown</div>
      <h2 className="section-title mt-2">Where the wins land</h2>
      <p className="section-subtitle">
        Some videos have more cross-frame redundancy than others (talking-head
        lectures vs. dynamic action). The aliasing payoff scales with that
        redundancy, which you can read straight off the right chart.
      </p>

      <div className="grid lg:grid-cols-2 gap-4 mt-10">
        <div className="card p-4 md:p-6">
          <div className="text-sm font-medium text-ink-900 mb-1">
            Mean TTFT per video
          </div>
          <div className="text-xs text-ink-400 mb-3">
            Lower is faster · averaged across questions on the same video
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={ttftData}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} />
              <XAxis
                dataKey="video"
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                angle={-30}
                textAnchor="end"
                height={60}
              />
              <YAxis
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                label={{
                  value: "TTFT (s)",
                  angle: -90,
                  position: "insideLeft",
                  fill: COLOR_AXIS,
                  fontSize: 12,
                }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipLabelStyle}
                formatter={(v: number, name: string) => {
                  if (name === "speedupPct") return [`${v.toFixed(1)}%`, "speedup"];
                  return [`${v.toFixed(3)} s`, name];
                }}
              />
              <Legend wrapperStyle={legendStyle} />
              <Bar dataKey="baseline" fill={COLOR_BASELINE} radius={[2, 2, 0, 0]} />
              <Bar dataKey="perceptual" fill={COLOR_PERCEPTUAL} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card p-4 md:p-6">
          <div className="text-sm font-medium text-ink-900 mb-1">
            Mean unique mm_hashes per video
          </div>
          <div className="text-xs text-ink-400 mb-3">
            Lower = more frames collapsed to anchors
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={uniqueData}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} />
              <XAxis
                dataKey="video"
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                angle={-30}
                textAnchor="end"
                height={60}
              />
              <YAxis
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                label={{
                  value: "unique mm_hashes / call",
                  angle: -90,
                  position: "insideLeft",
                  fill: COLOR_AXIS,
                  fontSize: 12,
                }}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipLabelStyle}
              />
              <Legend wrapperStyle={legendStyle} />
              <Bar dataKey="baseline" fill={COLOR_BASELINE} radius={[2, 2, 0, 0]} />
              <Bar dataKey="perceptual" fill={COLOR_PERCEPTUAL} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="mt-8 card p-4 md:p-6">
        <div className="text-sm font-medium text-ink-900 mb-1">
          Per-video TTFT speedup
        </div>
        <div className="text-xs text-ink-400 mb-3">
          (baseline − perceptual) / baseline, expressed as %
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={ttftData}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} />
            <XAxis
              dataKey="video"
              stroke={COLOR_AXIS}
              tick={{ fontSize: 11, fill: COLOR_AXIS }}
              angle={-30}
              textAnchor="end"
              height={60}
            />
            <YAxis
              stroke={COLOR_AXIS}
              tick={{ fontSize: 11, fill: COLOR_AXIS }}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
              formatter={(v: number) => `${v.toFixed(1)}%`}
            />
            <Bar dataKey="speedupPct" radius={[2, 2, 0, 0]}>
              {ttftData.map((entry, idx) => (
                <Cell
                  key={`c-${idx}`}
                  fill={entry.speedupPct >= 0 ? COLOR_PERCEPTUAL : COLOR_NEGATIVE}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
