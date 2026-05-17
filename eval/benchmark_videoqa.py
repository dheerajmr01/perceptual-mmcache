"""eval.benchmark_videoqa — aggregate baseline + perceptual + sweep into a report.

Loads JSONLs, computes headline metrics, renders matplotlib plots, writes REPORT.md.

Public API (called by pmcache_colab.ipynb):
    analyze_results(baseline_path, perceptual_path, sweep_dir, output_dir)

Heavy deps (matplotlib, pandas) are lazy-imported via `pmcache._lazy.require`
so `import eval.benchmark_videoqa` works on the CPU laptop without `[eval]`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pmcache._lazy import require


# ---------------------------------------------------------------------------
# Loading + per-row metrics
# ---------------------------------------------------------------------------


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit_rate(row: dict[str, Any]) -> float:
    n = row["num_frames"] or 1
    return row["cache_hits"] / n


def _accuracy(rows: list[dict[str, Any]]) -> float:
    """Grade each row. Prefers the MCQ letter match (`correct` field set
    by the runner when `mcq_options` is present); falls back to the old
    lowercased substring match for non-MCQ datasets (synthetic test fixtures).
    """
    if not rows:
        return 0.0
    n_correct = 0
    for r in rows:
        if "correct" in r:
            if r["correct"]:
                n_correct += 1
            continue
        gold = (r.get("gold_answer") or "").strip().lower()
        ans = (r.get("answer") or "").strip().lower()
        if gold and gold in ans:
            n_correct += 1
    return n_correct / len(rows)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0, "accuracy": 0.0, "dedup_rate": 0.0, "hit_rate": 0.0,
            "ttft_p50": 0.0, "ttft_p99": 0.0, "ttft_total": 0.0, "total_frames": 0,
            "kv_bytes_total": 0, "kv_bytes_cached": 0, "kv_bytes_recomputed": 0,
            "kv_token_cache_rate": 0.0,
            "unique_frames_total": 0, "vision_kv_pressure_bytes": 0,
        }
    ttfts = [r["ttft_s"] for r in rows]
    hits = sum(r["cache_hits"] for r in rows)
    misses = sum(r["cache_misses"] for r in rows)
    total = hits + misses
    kv_total = sum(r.get("kv_bytes_total", 0) for r in rows)
    kv_cached = sum(r.get("kv_bytes_cached", 0) for r in rows)
    kv_recomp = sum(r.get("kv_bytes_recomputed", 0) for r in rows)
    prompt_toks = sum(r.get("prompt_tokens", 0) for r in rows)
    cached_toks = sum(r.get("cached_tokens", 0) for r in rows)

    # Vision KV pressure: bytes vLLM would have to (re)allocate for the
    # *unique* image-token chunks in each row. Approximates the real
    # pmcache benefit: aliased frames collapse to one anchor mm_hash, so
    # each row only contributes len(set(mm_hashes)) chunks worth of
    # vision KV instead of num_frames. The vLLM prefix-cache stat
    # (cached_tokens) doesn't capture this — it's a cross-call metric.
    unique_frames_total = 0
    vision_kv_pressure_bytes = 0
    for r in rows:
        mm = r.get("mm_hashes") or []
        n_frames = r.get("num_frames", len(mm)) or 1
        unique = len(set(mm)) if mm else n_frames
        unique_frames_total += unique
        bpt = r.get("kv_bytes_per_token", 0)
        if bpt and r.get("prompt_tokens"):
            # Per-frame token chunk × bytes-per-token × unique frames
            per_frame_tok = r["prompt_tokens"] / n_frames
            vision_kv_pressure_bytes += int(per_frame_tok * unique * bpt)

    dedup_rate = hits / total if total else 0.0
    return {
        "n_rows": len(rows),
        "accuracy": _accuracy(rows),
        "dedup_rate": dedup_rate,
        # Back-compat alias: threshold_sweep + tests still read "hit_rate".
        "hit_rate": dedup_rate,
        "ttft_p50": _percentile(ttfts, 0.50),
        "ttft_p99": _percentile(ttfts, 0.99),
        "ttft_total": sum(ttfts),
        "total_frames": sum(r["num_frames"] for r in rows),
        "total_cache_hits": hits,
        "total_cache_misses": misses,
        "kv_bytes_total": kv_total,
        "kv_bytes_cached": kv_cached,
        "kv_bytes_recomputed": kv_recomp,
        "kv_token_cache_rate": (cached_toks / prompt_toks) if prompt_toks else 0.0,
        "unique_frames_total": unique_frames_total,
        "vision_kv_pressure_bytes": vision_kv_pressure_bytes,
    }


# ---------------------------------------------------------------------------
# Plotting (matplotlib — lazy-imported)
# ---------------------------------------------------------------------------


def _plot_ttft_distribution(
    baseline_rows: list[dict[str, Any]],
    perceptual_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    plt = require("matplotlib.pyplot", "eval")

    b = [r["ttft_s"] for r in baseline_rows]
    p = [r["ttft_s"] for r in perceptual_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 20
    if b:
        ax.hist(b, bins=bins, alpha=0.5, label=f"baseline (n={len(b)})")
    if p:
        ax.hist(p, bins=bins, alpha=0.5, label=f"perceptual (n={len(p)})")
    ax.set_xlabel("TTFT (s)")
    ax.set_ylabel("count")
    ax.set_title("TTFT distribution — baseline vs perceptual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_hit_rate_by_video(
    baseline_rows: list[dict[str, Any]],
    perceptual_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    plt = require("matplotlib.pyplot", "eval")
    np_ = require("numpy", "eval")

    videos = sorted({r["video"] for r in baseline_rows} | {r["video"] for r in perceptual_rows})
    if not videos:
        return

    def avg_hit(rows: list[dict[str, Any]], video: str) -> float:
        rs = [r for r in rows if r["video"] == video]
        if not rs:
            return 0.0
        return sum(_hit_rate(r) for r in rs) / len(rs)

    b_vals = [avg_hit(baseline_rows, v) for v in videos]
    p_vals = [avg_hit(perceptual_rows, v) for v in videos]

    x = np_.arange(len(videos))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, len(videos) * 1.2), 4))
    ax.bar(x - width / 2, b_vals, width, label="baseline", color="#888")
    ax.bar(x + width / 2, p_vals, width, label="perceptual", color="#4a90e2")
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=20, ha="right")
    ax.set_ylabel("cache hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-frame cache hit rate per video")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    baseline_agg: dict[str, Any],
    perceptual_agg: dict[str, Any],
    sweep_summary_path: Path | None,
    plots_written: list[str],
    output_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# pmcache benchmark report\n")
    lines.append("Generated by `eval.benchmark_videoqa.analyze_results`.\n")

    lines.append("## Headline metrics\n")
    lines.append("| metric | baseline | perceptual | Δ |")
    lines.append("|---|---:|---:|---:|")

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    def fmt_s(x: float) -> str:
        return f"{x:.4f}s"

    def fmt_bytes(n: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PiB"

    lines.append(
        f"| rows | {baseline_agg['n_rows']} | {perceptual_agg['n_rows']} | — |"
    )
    lines.append(
        f"| accuracy | {fmt_pct(baseline_agg['accuracy'])} | "
        f"{fmt_pct(perceptual_agg['accuracy'])} | "
        f"{(perceptual_agg['accuracy'] - baseline_agg['accuracy']) * 100:+.1f}pp |"
    )
    lines.append(
        f"| mm_hash dedup rate (intra-prompt) | {fmt_pct(baseline_agg['dedup_rate'])} | "
        f"{fmt_pct(perceptual_agg['dedup_rate'])} | "
        f"{(perceptual_agg['dedup_rate'] - baseline_agg['dedup_rate']) * 100:+.1f}pp |"
    )
    lines.append(
        f"| TTFT p50 | {fmt_s(baseline_agg['ttft_p50'])} | "
        f"{fmt_s(perceptual_agg['ttft_p50'])} | "
        f"{(perceptual_agg['ttft_p50'] - baseline_agg['ttft_p50']) * 1000:+.1f}ms |"
    )
    lines.append(
        f"| TTFT p99 | {fmt_s(baseline_agg['ttft_p99'])} | "
        f"{fmt_s(perceptual_agg['ttft_p99'])} | "
        f"{(perceptual_agg['ttft_p99'] - baseline_agg['ttft_p99']) * 1000:+.1f}ms |"
    )
    b_ttft_sum = baseline_agg.get("ttft_total", 0.0)
    p_ttft_sum = perceptual_agg.get("ttft_total", 0.0)
    ttft_pct = ((b_ttft_sum - p_ttft_sum) / b_ttft_sum * 100) if b_ttft_sum else 0.0
    lines.append(
        f"| TTFT total (sum) | {fmt_s(b_ttft_sum)} | {fmt_s(p_ttft_sum)} | "
        f"{(p_ttft_sum - b_ttft_sum) * 1000:+.1f}ms ({-ttft_pct:+.1f}%) |"
    )
    lines.append(
        f"| total frames | {baseline_agg['total_frames']} | "
        f"{perceptual_agg['total_frames']} | — |"
    )
    b_uniq = baseline_agg.get("unique_frames_total", 0)
    p_uniq = perceptual_agg.get("unique_frames_total", 0)
    uniq_saved = b_uniq - p_uniq
    lines.append(
        f"| unique frames (post-alias) | {b_uniq} | {p_uniq} | "
        f"{uniq_saved} fewer |"
    )
    b_press = baseline_agg.get("vision_kv_pressure_bytes", 0)
    p_press = perceptual_agg.get("vision_kv_pressure_bytes", 0)
    press_saved = b_press - p_press
    press_pct = (press_saved / b_press * 100) if b_press else 0.0
    lines.append(
        f"| vision KV pressure (unique-frame est.) | {fmt_bytes(b_press)} | "
        f"{fmt_bytes(p_press)} | "
        f"{fmt_bytes(abs(press_saved))} saved ({press_pct:+.1f}%) |"
    )
    b_recomp = baseline_agg.get("kv_bytes_recomputed", 0)
    p_recomp = perceptual_agg.get("kv_bytes_recomputed", 0)
    b_total = baseline_agg.get("kv_bytes_total", 0)
    saved_bytes = b_recomp - p_recomp
    saved_pct = (saved_bytes / b_recomp * 100) if b_recomp else 0.0
    lines.append(
        f"| KV bytes recomputed (vLLM prefix-cache) | {fmt_bytes(b_recomp)} | "
        f"{fmt_bytes(p_recomp)} | "
        f"{fmt_bytes(abs(saved_bytes))} saved ({saved_pct:+.1f}%) |"
    )
    lines.append(
        f"| KV bytes total (prompt) | {fmt_bytes(b_total)} | "
        f"{fmt_bytes(perceptual_agg.get('kv_bytes_total', 0))} | — |"
    )
    lines.append(
        f"| LMCache prefix-cache hit rate | {fmt_pct(baseline_agg.get('kv_token_cache_rate', 0.0))} | "
        f"{fmt_pct(perceptual_agg.get('kv_token_cache_rate', 0.0))} | "
        f"{(perceptual_agg.get('kv_token_cache_rate', 0.0) - baseline_agg.get('kv_token_cache_rate', 0.0)) * 100:+.1f}pp |"
    )
    lines.append("")
    lines.append(
        "> **Reading the table.** *mm_hash dedup rate* counts intra-prompt "
        "hash repetition — pmcache's direct win. *Vision KV pressure* is the "
        "estimated KV bytes for the unique image-token chunks per prompt "
        "(unique frames × prompt-tok-per-frame × KV-bytes-per-token); this "
        "is the metric pmcache reduces. *KV bytes recomputed* and *LMCache "
        "prefix-cache hit rate* come from vLLM's `num_cached_tokens` and "
        "only credit cross-call prefix hits — both variants benefit "
        "symmetrically when subsequent questions on the same video replay "
        "the cached prefix, so they show ~0 delta even when pmcache is "
        "aliasing aggressively.\n"
    )

    if plots_written:
        lines.append("## Plots\n")
        for png in plots_written:
            lines.append(f"![{png}]({png})\n")

    if sweep_summary_path and sweep_summary_path.exists():
        sweep = json.loads(sweep_summary_path.read_text())
        lines.append("## Threshold sweep\n")
        lines.append("| τ | accuracy | hit rate | aliased | misses |")
        lines.append("|---:|---:|---:|---:|---:|")
        for entry in sweep.get("results", []):
            lines.append(
                f"| {entry['tau']:.2f} | {fmt_pct(entry['accuracy'])} | "
                f"{fmt_pct(entry['hit_rate'])} | {entry['aliased_hashes']} | "
                f"{entry['misses']} |"
            )
        if "recommended_tau" in sweep:
            lines.append(
                f"\n**Recommended τ:** `{sweep['recommended_tau']}` — "
                f"max hit rate where accuracy ≥ baseline − 1pp."
            )
        lines.append("")

    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_results(
    baseline_path: str | Path,
    perceptual_path: str | Path,
    sweep_dir: str | Path | None,
    output_dir: str | Path,
    mock_vlm: bool = False,
) -> dict[str, Any]:
    """Load all JSONLs, render plots, write REPORT.md. Returns the aggregates."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_jsonl(baseline_path) if Path(baseline_path).exists() else []
    perceptual_rows = _load_jsonl(perceptual_path) if Path(perceptual_path).exists() else []

    baseline_agg = _aggregate(baseline_rows)
    perceptual_agg = _aggregate(perceptual_rows)

    plots_written: list[str] = []
    try:
        ttft_png = output_dir / "ttft_distribution.png"
        _plot_ttft_distribution(baseline_rows, perceptual_rows, ttft_png)
        plots_written.append(ttft_png.name)

        hit_png = output_dir / "hit_rate_by_video.png"
        _plot_hit_rate_by_video(baseline_rows, perceptual_rows, hit_png)
        plots_written.append(hit_png.name)
    except ModuleNotFoundError:
        # `[eval]` extra not installed — emit a text-only report.
        pass

    sweep_summary_path = None
    if sweep_dir is not None:
        sweep_summary_path = Path(sweep_dir) / "sweep_summary.json"
        sweep_plot = Path(sweep_dir) / "sweep_plot.png"
        if sweep_plot.exists():
            # Copy into output_dir so the report references a stable local file.
            (output_dir / sweep_plot.name).write_bytes(sweep_plot.read_bytes())
            plots_written.append(sweep_plot.name)

    _write_report(
        baseline_agg, perceptual_agg, sweep_summary_path, plots_written, output_dir
    )

    return {
        "baseline": baseline_agg,
        "perceptual": perceptual_agg,
        "plots_written": plots_written,
        "report_path": str(output_dir / "REPORT.md"),
    }
