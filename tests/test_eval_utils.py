"""Step 7 tests — eval/utils.py.

We verify on CPU:
    - `sample_frames` decodes a synthetic mp4 written via cv2 at the
      requested fps.
    - `iter_videos` enumerates known video extensions, sorted.
    - `load_qa` reads qa.jsonl.
    - `MockVLM` returns deterministic answers, tracks calls, and
      emulates LMCache hit/miss behavior so synthetic TTFTs differ
      between baseline (all misses) and perceptual (mostly hits).
    - `load_vlm(mock_vlm=True)` returns a `MockVLM` without trying to
      import `vllm`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.utils import (
    MockVLM,
    iter_videos,
    load_qa,
    load_vlm,
    sample_frames,
)


# --- synthetic video helper -------------------------------------------------


def _write_video(path: Path, fps: int = 30, seconds: int = 2, size: int = 64) -> None:
    """Write a small mp4 at known fps using cv2 (mp4v fourcc).

    Frames cycle through gray levels so a sampler that picks every Nth
    frame gets a visually distinct frame each time.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
    try:
        total = fps * seconds
        for i in range(total):
            shade = int(255 * (i / max(total - 1, 1)))
            frame = np.full((size, size, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


# --- sample_frames ----------------------------------------------------------


class TestSampleFrames:
    def test_sample_frames_at_1_fps(self, tmp_path):
        v = tmp_path / "test.mp4"
        _write_video(v, fps=30, seconds=2)
        frames = sample_frames(v, fps=1.0)
        # 2-second video at 1 fps → expect ~2 frames.
        assert 1 <= len(frames) <= 3
        # Each frame is an RGB PIL Image.
        from PIL import Image
        for f in frames:
            assert isinstance(f, Image.Image)
            assert f.mode == "RGB"

    def test_sample_frames_at_higher_fps(self, tmp_path):
        """fps=5.0 on a 30-fps 2-second clip should yield ~10 frames."""
        v = tmp_path / "test.mp4"
        _write_video(v, fps=30, seconds=2)
        frames_1fps = sample_frames(v, fps=1.0)
        frames_5fps = sample_frames(v, fps=5.0)
        assert len(frames_5fps) > len(frames_1fps)

    def test_sample_frames_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sample_frames(tmp_path / "does-not-exist.mp4")


# --- iter_videos -----------------------------------------------------------


class TestIterVideos:
    def test_iter_videos_finds_known_extensions(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"")
        (tmp_path / "b.mov").write_bytes(b"")
        (tmp_path / "c.webm").write_bytes(b"")
        (tmp_path / "ignore.txt").write_bytes(b"")

        got = [p.name for p in iter_videos(tmp_path)]
        assert got == ["a.mp4", "b.mov", "c.webm"]


# --- load_qa --------------------------------------------------------------


class TestLoadQA:
    def test_load_qa_parses_jsonl(self, tmp_path):
        f = tmp_path / "qa.jsonl"
        f.write_text(
            json.dumps({"video": "a.mp4", "question": "Q1?", "gold_answer": "yes"}) + "\n"
            + json.dumps({"video": "b.mp4", "question": "Q2?", "gold_answer": "no"}) + "\n"
        )
        qa = load_qa(f)
        assert len(qa) == 2
        assert qa[0]["question"] == "Q1?"
        assert qa[1]["gold_answer"] == "no"

    def test_load_qa_skips_blank_lines(self, tmp_path):
        f = tmp_path / "qa.jsonl"
        f.write_text(
            "\n"
            + json.dumps({"video": "a.mp4", "question": "Q1?", "gold_answer": "yes"})
            + "\n\n"
        )
        assert len(load_qa(f)) == 1


# --- MockVLM ---------------------------------------------------------------


class TestMockVLM:
    def test_canned_answer_default(self):
        m = MockVLM()
        from PIL import Image
        out = m.generate("anything?", [Image.new("RGB", (8, 8))])
        assert out["answer"] == "yes"
        assert out["ttft_s"] >= 0
        assert out["num_images"] == 1

    def test_keyword_overrides(self):
        m = MockVLM(answers={"color": "blue", "count": "3"})
        from PIL import Image
        img = Image.new("RGB", (8, 8))
        assert m.generate("What is the dominant color?", [img])["answer"] == "blue"
        assert m.generate("How many?", [img])["answer"] == "yes"  # no match -> default
        assert m.generate("count please", [img])["answer"] == "3"

    def test_records_calls(self):
        m = MockVLM()
        from PIL import Image
        img = Image.new("RGB", (8, 8))
        m.generate("Q1", [img])
        m.generate("Q2", [img])
        assert len(m.calls) == 2
        assert m.calls[0]["prompt"] == "Q1"

    def test_image_hashes_drive_hit_miss(self):
        """Repeat hashes count as hits (no synthetic latency); new hashes are misses."""
        m = MockVLM(per_image_ms=1.0)
        from PIL import Image
        img = Image.new("RGB", (8, 8))
        first = m.generate("Q?", [img], image_hashes=["aaa"])
        assert first["cache_hits"] == 0
        assert first["cache_misses"] == 1
        second = m.generate("Q?", [img], image_hashes=["aaa"])
        assert second["cache_hits"] == 1
        assert second["cache_misses"] == 0


# --- load_vlm --------------------------------------------------------------


class TestLoadVLM:
    def test_load_vlm_with_mock_returns_mockvlm(self):
        llm = load_vlm("any-model", mock_vlm=True)
        assert isinstance(llm, MockVLM)
        # Did not pull in vllm
        assert "vllm" not in sys.modules

    def test_load_vlm_without_mock_raises_friendly_error_when_vllm_missing(self):
        """On the CPU laptop vllm isn't installed — expect the lazy
        require's friendly error, not a bare ImportError."""
        try:
            load_vlm("any-model", mock_vlm=False)
        except ModuleNotFoundError as e:
            assert "[vlm]" in str(e)
        else:
            # On Colab vllm IS installed — accept either outcome.
            pass


# --- import hygiene --------------------------------------------------------


class TestImportHygiene:
    def test_eval_utils_does_not_import_torch_or_vllm(self, torch_not_imported):
        # `torch_not_imported` fixture already asserts torch absent.
        # Confirm vllm too.
        assert "vllm" not in sys.modules
