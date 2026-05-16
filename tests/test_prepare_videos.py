"""Step 8 tests — prepare_demo_videos.py.

Synth mode end-to-end: tmp dir → 5 videos + qa.jsonl, then verify each
video is decodable + the QA file has 3 entries per video.

Manifest mode is exercised via a fake HTTP fetch (monkeypatched
`urllib.request.urlopen`) so we don't hit the network in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets.prepare_demo_videos import (
    _SYNTH_GENERATORS,
    _SYNTH_QA,
    main,
)
from eval.utils import sample_frames


class TestSynthMode:
    def test_synth_mode_writes_all_videos(self, tmp_path):
        qa_path = main(tmp_path, mode="synth")
        for filename in _SYNTH_GENERATORS:
            assert (tmp_path / filename).exists()
            assert (tmp_path / filename).stat().st_size > 0

        assert qa_path == tmp_path / "qa.jsonl"
        assert qa_path.exists()

    def test_qa_jsonl_has_3_entries_per_video(self, tmp_path):
        main(tmp_path, mode="synth")
        entries = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text().splitlines()]
        expected = sum(len(qs) for qs in _SYNTH_QA.values())
        assert len(entries) == expected
        for e in entries:
            assert set(e) >= {"video", "question", "gold_answer"}

    def test_synth_videos_are_decodable(self, tmp_path):
        """Every generated video can be sampled by eval.utils.sample_frames."""
        main(tmp_path, mode="synth")
        for filename in _SYNTH_GENERATORS:
            frames = sample_frames(tmp_path / filename, fps=2.0)
            assert len(frames) >= 1, f"{filename} produced 0 sampled frames"

    def test_synth_mode_idempotent(self, tmp_path):
        """Re-running should not regenerate existing videos."""
        main(tmp_path, mode="synth")
        first_sizes = {f: (tmp_path / f).stat().st_size for f in _SYNTH_GENERATORS}
        main(tmp_path, mode="synth")  # again
        second_sizes = {f: (tmp_path / f).stat().st_size for f in _SYNTH_GENERATORS}
        assert first_sizes == second_sizes


class TestManifestMode:
    def test_manifest_mode_downloads_and_writes_qa(self, tmp_path, monkeypatch):
        # Fake the network: any urlopen returns a 100-byte blob.
        from eval.datasets import prepare_demo_videos as mod

        captured_urls = []

        class FakeResp:
            def __init__(self, payload):
                self._buf = payload
                self._pos = 0

            def read(self, n=-1):
                if self._pos >= len(self._buf):
                    return b""
                end = len(self._buf) if n < 0 else min(self._pos + n, len(self._buf))
                chunk = self._buf[self._pos:end]
                self._pos = end
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url if hasattr(req, "full_url") else str(req))
            return FakeResp(b"x" * 100)

        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({
            "videos": [
                {
                    "filename": "a.mp4",
                    "url": "https://example.com/a.mp4",
                    "questions": [
                        {"question": "Q1?", "gold_answer": "A1"},
                        {"question": "Q2?", "gold_answer": "A2"},
                    ],
                },
            ]
        }))

        main(tmp_path, mode="manifest", manifest=manifest)
        assert (tmp_path / "a.mp4").exists()
        assert captured_urls == ["https://example.com/a.mp4"]

        qa = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text().splitlines()]
        assert len(qa) == 2
        assert qa[0]["video"] == "a.mp4"

    def test_manifest_mode_requires_manifest_path(self, tmp_path):
        with pytest.raises(ValueError, match="manifest path required"):
            main(tmp_path, mode="manifest")


class TestUnknownMode:
    def test_unknown_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown mode"):
            main(tmp_path, mode="bogus")
