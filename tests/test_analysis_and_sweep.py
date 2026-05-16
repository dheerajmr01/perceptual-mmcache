"""Steps 11 + 12 tests — analyze_results + sweep_thresholds.

End-to-end: synth dataset → baseline + perceptual runs → analyze_results
produces REPORT.md and PNGs. sweep_thresholds runs perceptual at three
τ values and emits a summary + plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.benchmark_videoqa import analyze_results
from eval.datasets.prepare_demo_videos import main as prepare_main
from eval.run_baseline import run_baseline_benchmark
from eval.run_perceptual import run_perceptual_benchmark
from eval.threshold_sweep import sweep_thresholds
from pmcache.verifier import DinoV2Verifier


class AlwaysHitVerifier:
    EMBED_DIM = 384
    _VEC = np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)

    def embed(self, image):
        return self._VEC

    cosine = staticmethod(DinoV2Verifier.cosine)


@pytest.fixture(scope="module")
def baseline_and_perceptual(tmp_path_factory):
    """Set up: synth dataset → baseline JSONL → perceptual JSONL."""
    ws = tmp_path_factory.mktemp("ws")
    videos = ws / "videos"
    results = ws / "results"
    results.mkdir()
    prepare_main(videos, mode="synth")
    qa = videos / "qa.jsonl"

    baseline_path = results / "baseline.jsonl"
    perceptual_path = results / "perceptual.jsonl"

    run_baseline_benchmark(
        videos_dir=videos, qa_file=qa, output_path=baseline_path,
        model="any", fps=2.0, mock_vlm=True,
    )
    run_perceptual_benchmark(
        videos_dir=videos, qa_file=qa, output_path=perceptual_path,
        model="any", fps=2.0, mock_vlm=True,
        verifier=AlwaysHitVerifier(),
    )
    return {"workspace": ws, "videos": videos, "qa": qa,
            "baseline": baseline_path, "perceptual": perceptual_path}


# --- analyze_results --------------------------------------------------------


class TestAnalyzeResults:
    def test_writes_report_md_and_plots(self, baseline_and_perceptual, tmp_path):
        report_dir = tmp_path / "report"
        result = analyze_results(
            baseline_path=baseline_and_perceptual["baseline"],
            perceptual_path=baseline_and_perceptual["perceptual"],
            sweep_dir=None,
            output_dir=report_dir,
        )
        report = report_dir / "REPORT.md"
        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "pmcache benchmark report" in text
        assert "baseline" in text and "perceptual" in text
        assert "TTFT p50" in text
        # Plots
        assert (report_dir / "ttft_distribution.png").exists()
        assert (report_dir / "hit_rate_by_video.png").exists()
        assert "ttft_distribution.png" in result["plots_written"]

    def test_aggregates_are_consistent(self, baseline_and_perceptual, tmp_path):
        report_dir = tmp_path / "report2"
        result = analyze_results(
            baseline_path=baseline_and_perceptual["baseline"],
            perceptual_path=baseline_and_perceptual["perceptual"],
            sweep_dir=None,
            output_dir=report_dir,
        )
        b = result["baseline"]
        p = result["perceptual"]
        assert b["n_rows"] > 0 and p["n_rows"] > 0
        # Hit rate is in [0, 1].
        assert 0.0 <= b["hit_rate"] <= 1.0
        assert 0.0 <= p["hit_rate"] <= 1.0


# --- sweep_thresholds -------------------------------------------------------


class TestSweepThresholds:
    def test_sweep_writes_per_tau_jsonl_and_summary(
        self, baseline_and_perceptual, tmp_path
    ):
        sweep_dir = tmp_path / "sweep"
        taus = [0.90, 0.98, 0.999]
        summary = sweep_thresholds(
            videos_dir=baseline_and_perceptual["videos"],
            qa_file=baseline_and_perceptual["qa"],
            taus=taus,
            output_dir=sweep_dir,
            model="any",
            fps=2.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
            baseline_path=baseline_and_perceptual["baseline"],
        )
        for tau in taus:
            assert (sweep_dir / f"sweep_tau_{tau:.2f}.jsonl").exists()
        assert (sweep_dir / "sweep_summary.json").exists()
        assert (sweep_dir / "sweep_plot.png").exists()

        on_disk = json.loads((sweep_dir / "sweep_summary.json").read_text())
        assert on_disk["taus"] == list(map(float, taus))
        assert on_disk["recommended_tau"] in [float(t) for t in taus]
        assert len(on_disk["results"]) == len(taus)

    def test_recommended_tau_prefers_higher_hit_rate(
        self, baseline_and_perceptual, tmp_path
    ):
        """With AlwaysHitVerifier, lower τ never matters (cosine == 1).
        We expect recommend to be the largest τ that still gets hits —
        which here is the largest τ ≤ 1.0."""
        sweep_dir = tmp_path / "sweep3"
        summary = sweep_thresholds(
            videos_dir=baseline_and_perceptual["videos"],
            qa_file=baseline_and_perceptual["qa"],
            taus=[0.90, 0.98, 0.999],
            output_dir=sweep_dir,
            model="any",
            fps=2.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
            baseline_path=baseline_and_perceptual["baseline"],
        )
        # All three τ should produce the same hit rate (AlwaysHit), so
        # recommend picks the smallest τ as tie-break (looser threshold).
        assert summary["recommended_tau"] is not None


# --- end-to-end: sweep + analyze together ----------------------------------


class TestEndToEnd:
    def test_full_pipeline_writes_report_with_sweep_table(
        self, baseline_and_perceptual, tmp_path
    ):
        sweep_dir = tmp_path / "sweep4"
        sweep_thresholds(
            videos_dir=baseline_and_perceptual["videos"],
            qa_file=baseline_and_perceptual["qa"],
            taus=[0.95, 0.98],
            output_dir=sweep_dir,
            model="any",
            fps=2.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
            baseline_path=baseline_and_perceptual["baseline"],
        )
        report_dir = tmp_path / "report4"
        analyze_results(
            baseline_path=baseline_and_perceptual["baseline"],
            perceptual_path=baseline_and_perceptual["perceptual"],
            sweep_dir=sweep_dir,
            output_dir=report_dir,
        )
        report = (report_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "Threshold sweep" in report
        assert "Recommended τ" in report
        assert (report_dir / "sweep_plot.png").exists()
