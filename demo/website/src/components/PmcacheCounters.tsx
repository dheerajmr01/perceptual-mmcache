import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { Activity, Layers, Filter, AlertOctagon } from "lucide-react";
import { pmcacheTotals } from "../data/metrics";

const COLOR_PERCEPTUAL = "#3f7d57";
const COLOR_HIGHLIGHT = "#a06425";
const COLOR_OCEAN = "#385c7e";

const tooltipStyle = {
  background: "#ffffff",
  border: "1px solid #ebe1cb",
  borderRadius: 8,
  fontSize: 12,
  color: "#2d2925",
  boxShadow:
    "0 1px 0 0 rgba(31, 29, 25, 0.05), 0 4px 12px -2px rgba(31, 29, 25, 0.08)",
};
const legendStyle = { fontSize: 12, color: "#5a554d" };

export default function PmcacheCounters() {
  const t = pmcacheTotals();

  const outcomes = [
    { name: "Aliased (Tier-2 hit)", value: t.tier2Hits, color: COLOR_PERCEPTUAL },
    {
      name: "False positive caught (Tier-2 reject)",
      value: t.tier2Rejects,
      color: COLOR_HIGHLIGHT,
    },
    { name: "New anchor (miss)", value: t.misses, color: COLOR_OCEAN },
  ].filter((o) => o.value > 0);

  return (
    <section className="section">
      <div className="eyebrow">Perceptual shim · cumulative counters</div>
      <h2 className="section-title mt-2">What the shim actually did</h2>
      <p className="section-subtitle">
        Running totals from{" "}
        <code className="font-mono">PerceptualMMCache.metrics</code> — every
        frame the shim saw, classified by which tier resolved it. Aliased
        frames are the win; rejected candidates are the safety net.
      </p>

      <div className="grid md:grid-cols-4 gap-3 mt-10">
        <Counter
          icon={<Activity className="h-4 w-4 text-perceptual" />}
          label="Frames seen"
          value={t.framesSeen.toLocaleString()}
          sub="prepare_image() calls (deduped per video)"
        />
        <Counter
          icon={<Layers className="h-4 w-4 text-perceptual" />}
          label="Aliased (Tier-2 hits)"
          value={t.tier2Hits.toLocaleString()}
          sub="cosine ≥ 0.98 → mm_hash := anchor UUID"
        />
        <Counter
          icon={<Filter className="h-4 w-4 text-highlight" />}
          label="Tier-2 rejects"
          value={t.tier2Rejects.toLocaleString()}
          sub="pHash candidate, cosine < τ"
        />
        <Counter
          icon={<AlertOctagon className="h-4 w-4 text-ocean" />}
          label="pmcache hit rate"
          value={`${(t.hitRate * 100).toFixed(1)}%`}
          sub="aliased / frames seen"
        />
      </div>

      <div className="card p-6 mt-6">
        <div className="text-sm font-medium text-ink-900 mb-1">
          Outcome breakdown
        </div>
        <div className="text-xs text-ink-400 mb-3">
          Frame-by-frame classification by the shim
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={outcomes}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={110}
              paddingAngle={3}
              label={(e) => `${e.name}: ${e.value}`}
              labelLine={{ stroke: "#a8a39a" }}
            >
              {outcomes.map((o, i) => (
                <Cell key={i} fill={o.color} stroke="#ffffff" strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={legendStyle} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Counter({
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
