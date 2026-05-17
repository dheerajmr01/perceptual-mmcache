import { BASELINE, PERCEPTUAL, RunRow } from "./results";

export interface Summary {
  rows: number;
  totalFrames: number;
  uniqueFrames: number;
  ttftTotal: number;
  ttftMean: number;
  ttftP50: number;
  ttftP99: number;
  kvBytesTotal: number;
  kvBytesRecomputed: number;
  visionKvPressure: number;
  accuracy: number;
  dedupRate: number;
}

function percentile(values: number[], p: number): number {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const k = (s.length - 1) * p;
  const f = Math.floor(k);
  const c = Math.min(f + 1, s.length - 1);
  if (f === c) return s[f];
  return s[f] + (s[c] - s[f]) * (k - f);
}

function visionKvPressureBytes(rows: RunRow[]): number {
  // Σ (unique_frames × prompt_tokens_per_frame × kv_bytes_per_token)
  let total = 0;
  for (const r of rows) {
    if (!r.kv_bytes_per_token || !r.prompt_tokens || !r.num_frames) continue;
    const perFrameTok = r.prompt_tokens / r.num_frames;
    total += Math.round(perFrameTok * r.unique_mm_hashes * r.kv_bytes_per_token);
  }
  return total;
}

export function summarize(rows: RunRow[]): Summary {
  if (!rows.length) {
    return {
      rows: 0,
      totalFrames: 0,
      uniqueFrames: 0,
      ttftTotal: 0,
      ttftMean: 0,
      ttftP50: 0,
      ttftP99: 0,
      kvBytesTotal: 0,
      kvBytesRecomputed: 0,
      visionKvPressure: 0,
      accuracy: 0,
      dedupRate: 0,
    };
  }
  const ttfts = rows.map((r) => r.ttft_s);
  const totalFrames = rows.reduce((s, r) => s + r.num_frames, 0);
  const uniqueFrames = rows.reduce((s, r) => s + r.unique_mm_hashes, 0);
  const correct = rows.reduce((s, r) => s + (r.correct ? 1 : 0), 0);
  return {
    rows: rows.length,
    totalFrames,
    uniqueFrames,
    ttftTotal: ttfts.reduce((a, b) => a + b, 0),
    ttftMean: ttfts.reduce((a, b) => a + b, 0) / ttfts.length,
    ttftP50: percentile(ttfts, 0.5),
    ttftP99: percentile(ttfts, 0.99),
    kvBytesTotal: rows.reduce((s, r) => s + r.kv_bytes_total, 0),
    kvBytesRecomputed: rows.reduce((s, r) => s + r.kv_bytes_recomputed, 0),
    visionKvPressure: visionKvPressureBytes(rows),
    accuracy: correct / rows.length,
    dedupRate: (totalFrames - uniqueFrames) / Math.max(1, totalFrames),
  };
}

export const BASELINE_SUMMARY = summarize(BASELINE);
export const PERCEPTUAL_SUMMARY = summarize(PERCEPTUAL);

export const DELTAS = {
  ttftTotalDeltaPct:
    BASELINE_SUMMARY.ttftTotal > 0
      ? ((PERCEPTUAL_SUMMARY.ttftTotal - BASELINE_SUMMARY.ttftTotal) /
          BASELINE_SUMMARY.ttftTotal) *
        100
      : 0,
  ttftTotalSavedSec: BASELINE_SUMMARY.ttftTotal - PERCEPTUAL_SUMMARY.ttftTotal,
  visionKvDeltaPct:
    BASELINE_SUMMARY.visionKvPressure > 0
      ? ((PERCEPTUAL_SUMMARY.visionKvPressure - BASELINE_SUMMARY.visionKvPressure) /
          BASELINE_SUMMARY.visionKvPressure) *
        100
      : 0,
  visionKvSavedBytes:
    BASELINE_SUMMARY.visionKvPressure - PERCEPTUAL_SUMMARY.visionKvPressure,
  uniqueFramesSaved:
    BASELINE_SUMMARY.uniqueFrames - PERCEPTUAL_SUMMARY.uniqueFrames,
  accuracyDeltaPp:
    (PERCEPTUAL_SUMMARY.accuracy - BASELINE_SUMMARY.accuracy) * 100,
  dedupRateDeltaPp:
    (PERCEPTUAL_SUMMARY.dedupRate - BASELINE_SUMMARY.dedupRate) * 100,
};

export interface PerVideoStat {
  video: string;
  baseTtft: number;
  percTtft: number;
  baseUnique: number;
  percUnique: number;
  baseTotalFrames: number;
  questions: number;
}

function average(xs: number[]): number {
  if (!xs.length) return 0;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

export function perVideoStats(): PerVideoStat[] {
  const videos = Array.from(
    new Set([...BASELINE, ...PERCEPTUAL].map((r) => r.video)),
  ).sort();

  return videos.map((video) => {
    const b = BASELINE.filter((r) => r.video === video);
    const p = PERCEPTUAL.filter((r) => r.video === video);
    return {
      video,
      baseTtft: average(b.map((r) => r.ttft_s)),
      percTtft: average(p.map((r) => r.ttft_s)),
      baseUnique: average(b.map((r) => r.unique_mm_hashes)),
      percUnique: average(p.map((r) => r.unique_mm_hashes)),
      baseTotalFrames: average(b.map((r) => r.num_frames)),
      questions: b.length,
    };
  });
}

// Cumulative perceptual-shim counters (from the last perceptual row in
// the run — pmcache metrics are running totals).
export function pmcacheTotals() {
  const last = PERCEPTUAL[PERCEPTUAL.length - 1];
  if (!last) {
    return {
      tier1Hits: 0,
      tier2Hits: 0,
      tier2Rejects: 0,
      aliasedHashes: 0,
      hitRate: 0,
      framesSeen: 0,
      misses: 0,
    };
  }
  // Estimate framesSeen by summing num_frames across questions where the
  // same video appeared (frames are decoded once per video, not per Q).
  // The shim sees each frame exactly once per video.
  const videoFrames: Record<string, number> = {};
  for (const r of PERCEPTUAL) {
    if (!(r.video in videoFrames)) videoFrames[r.video] = r.num_frames;
  }
  const framesSeen = Object.values(videoFrames).reduce((a, b) => a + b, 0);
  const aliased = last.tier2_hits; // tier2_hits == aliased_hashes by definition
  const rejects = last.tier2_rejects;
  const misses = Math.max(0, framesSeen - aliased - rejects);
  return {
    tier1Hits: last.tier1_hits,
    tier2Hits: last.tier2_hits,
    tier2Rejects: last.tier2_rejects,
    aliasedHashes: last.aliased_hashes,
    hitRate: aliased / Math.max(1, framesSeen),
    framesSeen,
    misses,
  };
}

export function formatBytes(n: number, digits = 2): string {
  if (n === 0) return "0 B";
  const sign = n < 0 ? "-" : "";
  let x = Math.abs(n);
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let u = 0;
  while (x >= 1024 && u < units.length - 1) {
    x /= 1024;
    u++;
  }
  return `${sign}${x.toFixed(digits)} ${units[u]}`;
}

export function formatPercent(n: number, digits = 1, withSign = false): string {
  const sign = withSign ? (n >= 0 ? "+" : "") : "";
  return `${sign}${n.toFixed(digits)}%`;
}

export function formatSeconds(n: number, digits = 2): string {
  if (Math.abs(n) < 1) return `${(n * 1000).toFixed(0)} ms`;
  return `${n.toFixed(digits)} s`;
}
