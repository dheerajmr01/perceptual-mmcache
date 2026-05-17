import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BASELINE, PERCEPTUAL } from "../data/results";

// Chart palette tokens — matte cream theme
const COLOR_BASELINE = "#78736b";
const COLOR_PERCEPTUAL = "#3f7d57";
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

function histogram(values: number[], bins: number): { x: number; count: number }[] {
  if (!values.length) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const buckets = Array.from({ length: bins }, (_, i) => ({
    x: min + step * i,
    count: 0,
  }));
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor((v - min) / step));
    buckets[idx].count += 1;
  }
  return buckets;
}

function cdf(values: number[]): { x: number; y: number }[] {
  if (!values.length) return [];
  const s = [...values].sort((a, b) => a - b);
  return s.map((x, i) => ({ x, y: (i + 1) / s.length }));
}

export default function TTFTChart() {
  const baseTtfts = BASELINE.map((r) => r.ttft_s);
  const percTtfts = PERCEPTUAL.map((r) => r.ttft_s);

  const bins = 18;
  const baseHist = histogram(baseTtfts, bins);
  const percHist = histogram(percTtfts, bins);
  const histData = baseHist.map((b, i) => ({
    x: b.x.toFixed(2),
    baseline: b.count,
    perceptual: percHist[i]?.count ?? 0,
  }));

  const baseCdf = cdf(baseTtfts);
  const percCdf = cdf(percTtfts);
  const combined: { x: number; baseline?: number; perceptual?: number }[] = [];
  const xs = [
    ...new Set([...baseCdf.map((p) => p.x), ...percCdf.map((p) => p.x)]),
  ].sort((a, b) => a - b);
  for (const x of xs) {
    const b = baseCdf.filter((p) => p.x <= x);
    const p = percCdf.filter((p) => p.x <= x);
    combined.push({
      x: +x.toFixed(3),
      baseline: b.length ? b[b.length - 1].y : 0,
      perceptual: p.length ? p[p.length - 1].y : 0,
    });
  }

  return (
    <section className="section">
      <div className="eyebrow">TTFT distribution</div>
      <h2 className="section-title mt-2">The shift, not just the average</h2>
      <p className="section-subtitle">
        Histogram: where the per-call latencies cluster. CDF: at any TTFT
        budget, the perceptual curve sits above baseline — i.e., more calls
        finish under that budget. The CDF curve to the left = faster.
      </p>

      <div className="grid lg:grid-cols-2 gap-4 mt-10">
        <div className="card p-4 md:p-6">
          <div className="text-sm font-medium text-ink-900 mb-3">Histogram</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={histData}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} />
              <XAxis
                dataKey="x"
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                label={{
                  value: "TTFT (s)",
                  position: "insideBottom",
                  offset: -5,
                  fill: COLOR_AXIS,
                  fontSize: 12,
                }}
              />
              <YAxis
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
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

        <div className="card p-4 md:p-6">
          <div className="text-sm font-medium text-ink-900 mb-3">
            Cumulative distribution (CDF)
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={combined}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLOR_GRID} />
              <XAxis
                dataKey="x"
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                label={{
                  value: "TTFT (s)",
                  position: "insideBottom",
                  offset: -5,
                  fill: COLOR_AXIS,
                  fontSize: 12,
                }}
              />
              <YAxis
                stroke={COLOR_AXIS}
                tick={{ fontSize: 11, fill: COLOR_AXIS }}
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipLabelStyle}
                formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
              <Legend wrapperStyle={legendStyle} />
              <Line
                type="stepAfter"
                dataKey="baseline"
                stroke={COLOR_BASELINE}
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="stepAfter"
                dataKey="perceptual"
                stroke={COLOR_PERCEPTUAL}
                strokeWidth={2.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
