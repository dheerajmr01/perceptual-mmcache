"""Tests for prepare_videomme — selection logic + qa.jsonl construction.

No HuggingFace download, no yt-dlp call. We hand `select_videos` a
synthetic list of Video-MME-shaped rows, and we monkeypatch
`_download_video` to fake successful downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets import prepare_videomme as vm
from eval.datasets.prepare_videomme import (
    DEFAULT_DOMAINS_PREFERRED,
    DEFAULT_TASK_TYPES_PREFERRED,
    _mcq_to_gold_answer,
    select_videos,
    write_qa_jsonl,
)


# --- synthetic metadata helper ---------------------------------------------


def _row(
    video_id: str,
    question_id: str,
    *,
    duration: str = "short",
    domain: str = "Knowledge",
    task_type: str = "Information Synopsis",
    question: str = "Q?",
    options=("A. red", "B. blue", "C. green", "D. yellow"),
    answer: str = "B",
):
    return {
        "video_id": video_id,
        "videoID": "yt" + video_id,
        "url": f"https://youtu.be/yt{video_id}",
        "duration": duration,
        "domain": domain,
        "sub_category": "subcat",
        "task_type": task_type,
        "question_id": question_id,
        "question": question,
        "options": list(options),
        "answer": answer,
    }


# --- _mcq_to_gold_answer ----------------------------------------------------


class TestGoldAnswer:
    def test_strips_prefix(self):
        q = {"options": ["A. red", "B. blue", "C. green", "D. yellow"], "answer": "B"}
        assert _mcq_to_gold_answer(q) == "blue"

    def test_handles_no_prefix(self):
        q = {"options": ["red", "blue", "green", "yellow"], "answer": "C"}
        assert _mcq_to_gold_answer(q) == "green"

    def test_falls_back_to_letter_when_options_missing(self):
        assert _mcq_to_gold_answer({"options": [], "answer": "A"}) == "A"

    def test_lowercases_letter_ignored(self):
        q = {"options": ["A) one", "B) two", "C) three", "D) four"], "answer": "d"}
        assert _mcq_to_gold_answer(q) == "four"


# --- select_videos ----------------------------------------------------------


class TestSelectVideos:
    def test_filters_to_short_duration(self):
        rows = [
            _row("aaa", "q1", duration="short"),
            _row("bbb", "q2", duration="medium"),
            _row("ccc", "q3", duration="long"),
        ]
        out = select_videos(rows, n=10, duration="short")
        ids = [v["video_id"] for v in out]
        assert ids == ["aaa"]

    def test_groups_questions_per_video(self):
        rows = [
            _row("aaa", "q1"),
            _row("aaa", "q2"),
            _row("aaa", "q3"),
            _row("bbb", "q4"),
        ]
        out = select_videos(rows, n=10)
        per_video = {v["video_id"]: v for v in out}
        assert len(per_video["aaa"]["questions"]) == 3
        assert len(per_video["bbb"]["questions"]) == 1

    def test_caps_questions_per_video(self):
        rows = [_row("aaa", f"q{i}") for i in range(10)]
        out = select_videos(rows, n=1, max_questions_per_video=3)
        assert len(out[0]["questions"]) == 3

    def test_returns_at_most_n_videos(self):
        rows = [_row(f"vid{i:03}", f"q{i}") for i in range(20)]
        out = select_videos(rows, n=5)
        assert len(out) == 5

    def test_ranks_cache_friendly_first(self):
        """A row in a preferred domain+task should outrank one in neither."""
        rows = [
            _row("plain", "q1", domain="Off-list", task_type="Off-list"),
            _row("ideal", "q2",
                 domain=next(iter(DEFAULT_DOMAINS_PREFERRED)),
                 task_type=next(iter(DEFAULT_TASK_TYPES_PREFERRED))),
        ]
        out = select_videos(rows, n=2)
        assert out[0]["video_id"] == "ideal"
        assert out[1]["video_id"] == "plain"

    def test_domain_filter_excludes_questions(self):
        rows = [
            _row("aaa", "q1", domain="X"),
            _row("aaa", "q2", domain="Y"),
        ]
        out = select_videos(rows, n=10, domains={"X"})
        assert len(out) == 1
        assert len(out[0]["questions"]) == 1
        assert out[0]["questions"][0]["question_id"] == "q1"

    def test_videos_with_no_surviving_questions_dropped(self):
        rows = [
            _row("aaa", "q1", domain="X"),
            _row("bbb", "q2", domain="Y"),
        ]
        out = select_videos(rows, n=10, domains={"X"})
        assert [v["video_id"] for v in out] == ["aaa"]


# --- write_qa_jsonl ---------------------------------------------------------


class TestWriteQAJsonl:
    def test_only_writes_downloaded_videos(self, tmp_path):
        selected = [{
            "video_id": "aaa", "videoID": "ytaaa",
            "url": "https://youtu.be/ytaaa",
            "domain": "Knowledge", "sub_category": "x",
            "questions": [
                {"question_id": "q1", "question": "Q1?",
                 "options": ["A. one", "B. two", "C. three", "D. four"],
                 "answer": "B", "task_type": "Info"},
            ],
        }, {
            "video_id": "bbb", "videoID": "ytbbb",
            "url": "https://youtu.be/ytbbb",
            "domain": "Knowledge", "sub_category": "x",
            "questions": [
                {"question_id": "q2", "question": "Q2?",
                 "options": ["A. cat", "B. dog", "C. fish", "D. bird"],
                 "answer": "A", "task_type": "Info"},
            ],
        }]
        qa = tmp_path / "qa.jsonl"
        n = write_qa_jsonl(selected, {"aaa.mp4"}, qa)
        assert n == 1
        rows = [json.loads(l) for l in qa.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["video"] == "aaa.mp4"
        assert rows[0]["gold_answer"] == "two"
        assert rows[0]["mcq_answer"] == "B"
        assert rows[0]["mcq_options"] == ["A. one", "B. two", "C. three", "D. four"]
        assert rows[0]["video_meta"]["url"] == "https://youtu.be/ytaaa"

    def test_writes_one_jsonl_row_per_question(self, tmp_path):
        selected = [{
            "video_id": "aaa", "videoID": "ytaaa",
            "url": "https://youtu.be/ytaaa",
            "domain": "K", "sub_category": "x",
            "questions": [
                {"question_id": f"q{i}", "question": f"Q{i}?",
                 "options": ["A. x", "B. y", "C. z", "D. w"],
                 "answer": "A", "task_type": "Info"}
                for i in range(3)
            ],
        }]
        qa = tmp_path / "qa.jsonl"
        n = write_qa_jsonl(selected, {"aaa.mp4"}, qa)
        assert n == 3


# --- main (mocked HF + yt-dlp) ---------------------------------------------


class TestMainEndToEnd:
    def test_main_with_mocked_hf_and_ytdlp(self, tmp_path, monkeypatch):
        """Mock both _load_metadata and _download_video so we test the
        end-to-end glue without hitting the network."""

        fake_rows = [
            _row("aaa", "q1"),
            _row("aaa", "q2"),
            _row("bbb", "q3"),
            _row("ccc", "q4", duration="medium"),  # filtered out
        ]

        monkeypatch.setattr(vm, "_load_metadata", lambda hf_token=None: fake_rows)

        def fake_download(url, dest, max_height=360):
            # Simulate a successful download by writing a 4-byte file.
            dest.write_bytes(b"fake")
            return True

        monkeypatch.setattr(vm, "_download_video", fake_download)

        qa_path = vm.main(tmp_path, n=10, duration="short")
        assert qa_path == tmp_path / "qa.jsonl"
        # Both aaa.mp4 and bbb.mp4 should be downloaded.
        assert (tmp_path / "aaa.mp4").exists()
        assert (tmp_path / "bbb.mp4").exists()
        assert not (tmp_path / "ccc.mp4").exists()

        rows = [json.loads(l) for l in qa_path.read_text().splitlines()]
        # 2 questions for aaa + 1 for bbb = 3
        assert len(rows) == 3
        videos_seen = {r["video"] for r in rows}
        assert videos_seen == {"aaa.mp4", "bbb.mp4"}

    def test_main_skips_existing_files(self, tmp_path, monkeypatch):
        (tmp_path / "aaa.mp4").write_bytes(b"already-here")

        monkeypatch.setattr(vm, "_load_metadata", lambda hf_token=None: [
            _row("aaa", "q1"),
        ])
        downloaded_called = []
        monkeypatch.setattr(vm, "_download_video",
                            lambda *a, **kw: downloaded_called.append(a) or True)

        vm.main(tmp_path, n=10, duration="short")
        # download was NOT called because the file already exists
        assert downloaded_called == []
        # And the qa.jsonl still references the existing file
        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text().splitlines()]
        assert rows[0]["video"] == "aaa.mp4"

    def test_main_skips_failed_downloads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vm, "_load_metadata", lambda hf_token=None: [
            _row("aaa", "q1"),
            _row("bbb", "q2"),
        ])

        def flaky(url, dest, max_height=360):
            # Only aaa downloads successfully.
            if "aaa" in url:
                dest.write_bytes(b"fake")
                return True
            return False

        monkeypatch.setattr(vm, "_download_video", flaky)
        vm.main(tmp_path, n=10, duration="short")

        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text().splitlines()]
        videos = {r["video"] for r in rows}
        assert videos == {"aaa.mp4"}


# --- friendly error when extras aren't installed ---------------------------


class TestExtrasGuard:
    def test_load_metadata_requires_videomme_extra(self, monkeypatch):
        """If `datasets` isn't importable, raise a friendly error."""
        import sys
        monkeypatch.setitem(sys.modules, "datasets", None)
        # Force the lazy require to fail by stubbing it.
        import pmcache._lazy as lazy
        orig = lazy.require

        def fake_require(module, extra):
            if module == "datasets":
                raise ModuleNotFoundError(
                    f"pmcache requires `{module}` for this code path. "
                    f"Install the `[{extra}]` extra:\n"
                    f"    pip install -e '.[{extra}]'"
                )
            return orig(module, extra)

        monkeypatch.setattr(lazy, "require", fake_require)
        monkeypatch.setattr(vm, "require", fake_require)

        with pytest.raises(ModuleNotFoundError, match=r"\[videomme\]"):
            vm._load_metadata()
