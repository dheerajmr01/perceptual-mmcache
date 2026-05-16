"""eval.threshold_sweep — sweep DINOv2 cosine threshold τ.

For each τ in `taus`, run the perceptual benchmark, collect headline
metrics, and write per-τ JSONLs plus a summary JSON and a matplotlib
plot of accuracy + hit rate vs τ.

The "recommended τ" is the largest hit rate where accuracy is within
1 percentage point of the baseline. If no baseline is supplied, we
recommend the τ with the highest hit rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from eval.benchmark_videoqa import _aggregate, _load_jsonl
from eval.run_perceptual import run_perceptual_benchmark
from pmcache._lazy import require

if TYPE_CHECKING:
    from pmcache.verifier import DinoV2Verifier


def _plot_sweep(results: list[dict[str, Any]], out_path: Path) -> None:
    plt = require("matplotlib.pyplot", "eval")

    taus = [r["tau"] for r in results]
    acc = [r["accuracy"] for r in results]
    hit = [r["hit_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(taus, acc, "o-", color="#4a90e2", label="accuracy")
    ax1.set_xlabel("τ (cosine threshold)")
    ax1.set_ylabel("accuracy", color="#4a90e2")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#4a90e2")
    ax2 = ax1.twinx()
    ax2.plot(taus, hit, "s-", color="#e25c4a", label="hit rate")
    ax2.set_ylabel("cache hit rate", color="#e25c4a")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#e25c4a")
    fig.suptitle("pmcache τ sweep — accuracy vs cache hit rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _recommend_tau(
    results: list[dict[str, Any]],
    baseline_accuracy: float | None,
) -> float | None:
    """Pick the largest hit-rate τ whose accuracy is within 1pp of baseline.

    Tie-break by smallest τ (looser threshold = more frames pass Tier 2).
    """
    if not results:
        return None
    if baseline_accuracy is None:
        # No baseline → pick max hit rate.
        best = max(results, key=lambda r: (r["hit_rate"], -r["tau"]))
        return best["tau"]
    floor = baseline_accuracy - 0.01  # 1pp tolerance
    candidates = [r for r in results if r["accuracy"] >= floor]
    if not candidates:
        # Fallback: pick the highest-accuracy τ.
        best = max(results, key=lambda r: r["accuracy"])
        return best["tau"]
    best = max(candidates, key=lambda r: (r["hit_rate"], -r["tau"]))
    return best["tau"]


def sweep_thresholds(
    videos_dir: str | Path,
    qa_file: str | Path,
    taus: Sequence[float],
    output_dir: str | Path,
    model: str,
    mock_vlm: bool = False,
    fps: float = 1.0,
    k: int = 5,
    verifier: "DinoV2Verifier | None" = None,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Sweep τ over `taus`. Writes per-τ JSONLs + summary + plot.

    Output layout:
        {output_dir}/sweep_tau_{tau:.2f}.jsonl     one per τ
        {output_dir}/sweep_summary.json            aggregate + recommended τ
        {output_dir}/sweep_plot.png                accuracy + hit rate vs τ

    Args:
        baseline_path: if provided, use the baseline JSONL's accuracy
            as the floor for recommending τ. Without it we recommend
            max hit-rate τ.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for tau in taus:
        out_path = output_dir / f"sweep_tau_{tau:.2f}.jsonl"
        run_perceptual_benchmark(
            videos_dir=videos_dir,
            qa_file=qa_file,
            output_path=out_path,
            model=model,
            tau=tau,
            k=k,
            fps=fps,
            mock_vlm=mock_vlm,
            verifier=verifier,
        )
        rows = _load_jsonl(out_path)
        agg = _aggregate(rows)
        results.append({
            "tau": float(tau),
            "accuracy": agg["accuracy"],
            "hit_rate": agg["hit_rate"],
            "ttft_p50": agg["ttft_p50"],
            "ttft_p99": agg["ttft_p99"],
            "aliased_hashes": sum(r.get("aliased_hashes", 0) for r in rows),
            "misses": sum(r.get("misses", 0) for r in rows),
            "jsonl_path": str(out_path),
        })

    baseline_accuracy = None
    if baseline_path and Path(baseline_path).exists():
        baseline_accuracy = _aggregate(_load_jsonl(baseline_path))["accuracy"]
    recommended = _recommend_tau(results, baseline_accuracy)

    summary = {
        "taus": list(map(float, taus)),
        "baseline_accuracy": baseline_accuracy,
        "recommended_tau": recommended,
        "results": results,
    }
    (output_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        _plot_sweep(results, output_dir / "sweep_plot.png")
    except ModuleNotFoundError:
        pass

    return summary
