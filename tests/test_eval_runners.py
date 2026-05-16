"""Steps 9 + 10 tests — eval/run_baseline.py + eval/run_perceptual.py.

End-to-end runs on the synthetic videos with mock_vlm=True. No torch,
no vllm — a FakeVerifier provides DINOv2 stand-in embeddings.

Spec acceptance:
    - Both runners produce well-formed JSONL with the documented schema.
    - Both runners write a summary dict that the report step can consume.
    - On static.mp4 (all frames identical pixel-wise) the perceptual run
      produces aliased mm_hashes.
    - On random_noise.mp4 the perceptual run gets 0 hits (each frame is
      a fresh bucket — worst case for perceptual cache).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.datasets.prepare_demo_videos import main as prepare_main
from eval.run_baseline import run_baseline_benchmark
from eval.run_perceptual import run_perceptual_benchmark
from pmcache.verifier import DinoV2Verifier


class AlwaysHitVerifier:
    """FakeVerifier that returns a constant embedding → cosine == 1.0.

    With this verifier, every Tier-1 candidate also passes Tier-2 — so
    pHash behavior alone determines whether a frame is aliased. Useful
    for testing the runner's plumbing without depending on real DINOv2.
    """

    EMBED_DIM = 384
    _VEC = np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)

    def embed(self, image):
        return self._VEC

    cosine = staticmethod(DinoV2Verifier.cosine)


@pytest.fixture(scope="module")
def synth_dataset(tmp_path_factory):
    """Generate the 5 synth videos + qa.jsonl once for all runner tests."""
    workspace = tmp_path_factory.mktemp("dataset")
    prepare_main(workspace, mode="synth")
    return workspace


# --- run_baseline ----------------------------------------------------------


class TestBaseline:
    def test_baseline_writes_one_row_per_question(self, synth_dataset, tmp_path):
        out = tmp_path / "baseline.jsonl"
        summary = run_baseline_benchmark(
            videos_dir=synth_dataset,
            qa_file=synth_dataset / "qa.jsonl",
            output_path=out,
            model="any-model",
            fps=2.0,
            mock_vlm=True,
        )
        assert out.exists()
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == summary["rows_written"] > 0

        # Each row carries the documented schema
        required = {
            "video", "question", "gold_answer", "answer", "num_frames",
            "mm_hashes", "cache_hits", "cache_misses", "ttft_s",
            "tier1_hits", "tier2_hits", "tier2_rejects", "misses",
            "aliased_hashes", "pmcache_hit_rate", "variant",
        }
        assert set(rows[0]).issuperset(required)
        assert rows[0]["variant"] == "baseline"

    def test_baseline_does_not_alias_any_hashes(self, synth_dataset, tmp_path):
        """No pmcache → tier1_hits and aliased_hashes always 0."""
        out = tmp_path / "baseline.jsonl"
        run_baseline_benchmark(
            videos_dir=synth_dataset,
            qa_file=synth_dataset / "qa.jsonl",
            output_path=out,
            model="any-model",
            fps=2.0,
            mock_vlm=True,
        )
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        for r in rows:
            assert r["tier1_hits"] == 0
            assert r["aliased_hashes"] == 0


# --- run_perceptual --------------------------------------------------------


class TestPerceptual:
    def test_perceptual_aliases_on_static_video(self, synth_dataset, tmp_path):
        """static.mp4 has identical frames → all but the first should alias."""
        # Build a qa file containing only the static video.
        static_qa = tmp_path / "qa_static.jsonl"
        static_qa.write_text(
            json.dumps({
                "video": "static.mp4",
                "question": "Anything moving?",
                "gold_answer": "no",
            }) + "\n"
        )

        out = tmp_path / "perceptual_static.jsonl"
        summary = run_perceptual_benchmark(
            videos_dir=synth_dataset,
            qa_file=static_qa,
            output_path=out,
            model="any-model",
            tau=0.98,
            k=5,
            fps=4.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
        )

        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        # First frame → miss; the rest should all alias.
        assert row["aliased_hashes"] >= row["num_frames"] - 1
        assert row["tier1_hits"] >= row["num_frames"] - 1
        # Summary echoes the counters.
        assert summary["aliased_hashes"] >= row["num_frames"] - 1
        assert summary["variant"] == "perceptual"

    def test_perceptual_does_not_alias_on_random_noise(self, synth_dataset, tmp_path):
        """random_noise.mp4 → no two frames share a pHash bucket."""
        noise_qa = tmp_path / "qa_noise.jsonl"
        noise_qa.write_text(
            json.dumps({
                "video": "random_noise.mp4",
                "question": "Clear subject?",
                "gold_answer": "no",
            }) + "\n"
        )

        out = tmp_path / "perceptual_noise.jsonl"
        run_perceptual_benchmark(
            videos_dir=synth_dataset,
            qa_file=noise_qa,
            output_path=out,
            model="any-model",
            fps=4.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
        )

        row = json.loads(out.read_text().splitlines()[0])
        # AlwaysHitVerifier means *if* Tier-1 matches we'd alias — but
        # random noise frames have pHashes spread > K=5 apart, so no
        # Tier-1 candidates exist → no aliases.
        assert row["aliased_hashes"] == 0
        # Every frame is a fresh bucket.
        assert row["misses"] >= row["num_frames"]

    def test_perceptual_uses_supplied_verifier_not_dinov2(self, synth_dataset, tmp_path):
        """Sanity check: passing a FakeVerifier means no real DINOv2 load."""
        import sys

        out = tmp_path / "perceptual.jsonl"
        run_perceptual_benchmark(
            videos_dir=synth_dataset,
            qa_file=synth_dataset / "qa.jsonl",
            output_path=out,
            model="any-model",
            fps=2.0,
            mock_vlm=True,
            verifier=AlwaysHitVerifier(),
        )
        # We never imported torch / transformers / vllm during the run.
        assert "torch" not in sys.modules
        assert "transformers" not in sys.modules
        assert "vllm" not in sys.modules


# --- baseline vs perceptual differential -----------------------------------


class TestBaselineVsPerceptual:
    def test_perceptual_beats_baseline_on_static(self, synth_dataset, tmp_path):
        """On static.mp4, perceptual hit rate > baseline hit rate."""
        static_qa = tmp_path / "qa.jsonl"
        static_qa.write_text(
            json.dumps({
                "video": "static.mp4",
                "question": "Anything moving?",
                "gold_answer": "no",
            }) + "\n"
        )

        baseline_out = tmp_path / "baseline.jsonl"
        run_baseline_benchmark(
            videos_dir=synth_dataset, qa_file=static_qa,
            output_path=baseline_out, model="any-model",
            fps=4.0, mock_vlm=True,
        )
        perceptual_out = tmp_path / "perceptual.jsonl"
        run_perceptual_benchmark(
            videos_dir=synth_dataset, qa_file=static_qa,
            output_path=perceptual_out, model="any-model",
            fps=4.0, mock_vlm=True, verifier=AlwaysHitVerifier(),
        )

        baseline_row = json.loads(baseline_out.read_text().splitlines()[0])
        perceptual_row = json.loads(perceptual_out.read_text().splitlines()[0])

        # Both should have the same number of frames.
        assert baseline_row["num_frames"] == perceptual_row["num_frames"]
        # Static video is byte-identical → baseline also gets hits.
        # We just assert perceptual got AT LEAST as many.
        assert perceptual_row["cache_hits"] >= baseline_row["cache_hits"]
        # But aliased_hashes is the unique pmcache signal.
        assert perceptual_row["aliased_hashes"] > 0
