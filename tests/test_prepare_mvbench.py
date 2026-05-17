"""Tests for prepare_mvbench — MCQ normalization, selection, video sourcing.

No HuggingFace download, no hub fetch. We hand `select_entries` synthetic
MVBench-shaped rows, and we monkeypatch the HF loader + video-download
helpers to fake their results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.datasets import prepare_mvbench as mv
from eval.datasets.prepare_mvbench import (
    DEFAULT_TASKS_PREFERRED,
    TASK_VIDEO_DIRS,
    _find_local_video,
    _normalize_mcq,
    select_entries,
    write_qa_jsonl,
)


# --- synthetic MVBench row -------------------------------------------------


def _row(
    *,
    video: str = "clip1.mp4",
    question: str = "What is happening?",
    candidates=("a cat sits", "a dog runs", "a bird flies", "a fish swims"),
    answer="a dog runs",
    task_type: str = "object_interaction",
):
    return {
        "video": video,
        "question": question,
        "candidates": list(candidates),
        "answer": answer,
        "task_type": task_type,
    }


# --- _normalize_mcq --------------------------------------------------------


class TestNormalizeMCQ:
    def test_text_answer_matches_candidate(self):
        out = _normalize_mcq(_row(candidates=["red", "blue", "green"], answer="blue"))
        assert out["mcq_answer"] == "B"
        assert out["mcq_options"] == ["A. red", "B. blue", "C. green"]
        assert out["gold_answer"] == "blue"

    def test_text_answer_case_insensitive(self):
        out = _normalize_mcq(_row(candidates=["Red", "Blue", "Green"], answer="BLUE"))
        assert out["mcq_answer"] == "B"

    def test_text_answer_substring_fallback(self):
        # MVBench sometimes has trailing punctuation; substring match
        # rescues it.
        out = _normalize_mcq(_row(
            candidates=["a person walks.", "a person runs.", "nothing"],
            answer="a person walks",
        ))
        assert out["mcq_answer"] == "A"

    def test_integer_answer(self):
        out = _normalize_mcq(_row(candidates=["red", "blue", "green"], answer=2))
        assert out["mcq_answer"] == "C"
        assert out["gold_answer"] == "green"

    def test_letter_answer(self):
        out = _normalize_mcq(_row(candidates=["red", "blue", "green"], answer="C"))
        assert out["mcq_answer"] == "C"

    def test_lowercase_letter_answer(self):
        out = _normalize_mcq(_row(candidates=["red", "blue", "green"], answer="c"))
        assert out["mcq_answer"] == "C"

    def test_five_options_produces_e(self):
        out = _normalize_mcq(_row(
            candidates=["a", "b", "c", "d", "e"], answer="e",
        ))
        assert out["mcq_answer"] == "E"
        assert len(out["mcq_options"]) == 5

    def test_empty_candidates_returns_none(self):
        assert _normalize_mcq(_row(candidates=[], answer="anything")) is None

    def test_unmatched_answer_returns_none(self):
        out = _normalize_mcq(_row(
            candidates=["red", "blue"], answer="purple-not-in-list",
        ))
        assert out is None

    def test_letter_beyond_candidates_returns_none(self):
        # Answer "E" with only 3 candidates — should not happen, but
        # defensively returns None rather than crashing.
        assert _normalize_mcq(_row(
            candidates=["red", "blue", "green"], answer="E",
        )) is None

    def test_int_out_of_range_returns_none(self):
        assert _normalize_mcq(_row(
            candidates=["red", "blue"], answer=99,
        )) is None


# --- select_entries --------------------------------------------------------


class TestSelectEntries:
    def test_groups_by_video_filename(self):
        rows = [
            _row(video="clip1.mp4", question="Q1"),
            _row(video="clip1.mp4", question="Q2", answer="a cat sits"),
            _row(video="clip2.mp4", question="Q3"),
        ]
        out = select_entries(rows, n=10)
        per = {v["video_filename"]: v for v in out}
        assert len(per) == 2
        assert len(per["clip1.mp4"]["questions"]) == 2
        assert len(per["clip2.mp4"]["questions"]) == 1

    def test_strips_path_to_filename(self):
        rows = [_row(video="some/sub/dir/clip.mp4")]
        out = select_entries(rows, n=10)
        assert out[0]["video_filename"] == "clip.mp4"

    def test_skips_frame_folder_entries(self):
        # MVBench's TVQA/episodic_reasoning sometimes lists frame folders
        # without extensions — we can't feed those to the eval harness,
        # so we skip them.
        rows = [
            _row(video="house_md_ep01_clip03"),  # no extension → skip
            _row(video="clip.mp4"),               # ok
        ]
        out = select_entries(rows, n=10)
        assert [v["video_filename"] for v in out] == ["clip.mp4"]

    def test_drops_malformed_mcq(self):
        rows = [
            _row(video="ok.mp4"),
            _row(video="bad.mp4", candidates=[], answer="x"),
        ]
        out = select_entries(rows, n=10)
        assert [v["video_filename"] for v in out] == ["ok.mp4"]

    def test_caps_max_questions_per_video(self):
        rows = [_row(video="clip.mp4", question=f"Q{i}") for i in range(10)]
        out = select_entries(rows, n=10, max_questions_per_video=3)
        assert len(out) == 1
        assert len(out[0]["questions"]) == 3

    def test_returns_at_most_n_videos(self):
        rows = [_row(video=f"clip{i:03d}.mp4") for i in range(20)]
        out = select_entries(rows, n=5)
        assert len(out) == 5

    def test_skips_rows_missing_video_field(self):
        rows = [
            {"question": "no video here", "candidates": ["a", "b"], "answer": "a"},
            _row(video="clip.mp4"),
        ]
        out = select_entries(rows, n=10)
        assert [v["video_filename"] for v in out] == ["clip.mp4"]

    def test_stable_ordering_across_runs(self):
        rows = [_row(video=v) for v in ("z.mp4", "a.mp4", "m.mp4")]
        out1 = [v["video_filename"] for v in select_entries(rows, n=10)]
        out2 = [v["video_filename"] for v in select_entries(rows, n=10)]
        assert out1 == out2 == ["a.mp4", "m.mp4", "z.mp4"]


# --- _find_local_video -----------------------------------------------------


class TestFindLocalVideo:
    def test_direct_child(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"x")
        assert _find_local_video("clip.mp4", tmp_path) == f

    def test_nested(self, tmp_path):
        sub = tmp_path / "task" / "sub"
        sub.mkdir(parents=True)
        f = sub / "clip.mp4"
        f.write_bytes(b"x")
        assert _find_local_video("clip.mp4", tmp_path) == f

    def test_returns_none_when_missing(self, tmp_path):
        assert _find_local_video("nope.mp4", tmp_path) is None

    def test_alternative_extension(self, tmp_path):
        # User has clip.webm but metadata says clip.mp4 — we recover.
        f = tmp_path / "clip.webm"
        f.write_bytes(b"x")
        assert _find_local_video("clip.mp4", tmp_path) == f


# --- write_qa_jsonl --------------------------------------------------------


class TestWriteQAJsonl:
    def test_only_writes_sourced_videos(self, tmp_path):
        selected = [
            {
                "video_filename": "got.mp4",
                "task_type": "object_existence",
                "questions": [{
                    "question": "Q1?",
                    "task_type": "object_existence",
                    "mcq_options": ["A. yes", "B. no"],
                    "mcq_answer": "A",
                    "gold_answer": "yes",
                }],
            },
            {
                "video_filename": "missing.mp4",
                "task_type": "object_existence",
                "questions": [{
                    "question": "Q2?",
                    "task_type": "object_existence",
                    "mcq_options": ["A. yes", "B. no"],
                    "mcq_answer": "B",
                    "gold_answer": "no",
                }],
            },
        ]
        qa = tmp_path / "qa.jsonl"
        n = write_qa_jsonl(selected, {"got.mp4"}, qa)
        assert n == 1
        rows = [json.loads(l) for l in qa.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["video"] == "got.mp4"
        assert rows[0]["mcq_answer"] == "A"
        assert rows[0]["video_meta"]["hf_repo"] == "OpenGVLab/MVBench"

    def test_one_row_per_question(self, tmp_path):
        selected = [{
            "video_filename": "clip.mp4",
            "task_type": "object_interaction",
            "questions": [
                {
                    "question": f"Q{i}?",
                    "task_type": "object_interaction",
                    "mcq_options": ["A. x", "B. y"],
                    "mcq_answer": "A",
                    "gold_answer": "x",
                } for i in range(4)
            ],
        }]
        qa = tmp_path / "qa.jsonl"
        n = write_qa_jsonl(selected, {"clip.mp4"}, qa)
        assert n == 4


# --- main (mocked HF + sourcing) -------------------------------------------


class TestMainEndToEnd:
    def test_local_src_dir_copies_videos(self, tmp_path, monkeypatch):
        """Pre-downloaded videos: we copy from src_dir into output_dir."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.mp4").write_bytes(b"video-a")
        (src / "b.mp4").write_bytes(b"video-b")

        fake_rows = {
            "object_existence": [
                _row(video="a.mp4", task_type="object_existence"),
                _row(video="b.mp4", task_type="object_existence"),
                _row(video="missing.mp4", task_type="object_existence"),
            ],
        }
        monkeypatch.setattr(
            mv, "_load_task",
            lambda task, hf_token=None: fake_rows.get(task, []),
        )

        out = tmp_path / "out"
        qa_path = mv.main(
            out, n=10, tasks=["object_existence"], video_src_dir=src,
        )
        assert qa_path == out / "qa.jsonl"
        assert (out / "a.mp4").exists()
        assert (out / "b.mp4").exists()
        assert not (out / "missing.mp4").exists()
        rows = [json.loads(l) for l in qa_path.read_text(encoding="utf-8").splitlines()]
        assert {r["video"] for r in rows} == {"a.mp4", "b.mp4"}

    def test_hf_download_path(self, tmp_path, monkeypatch):
        """No video_src_dir → calls _hf_download_video."""
        monkeypatch.setattr(
            mv, "_load_task",
            lambda task, hf_token=None: [_row(video="x.mp4", task_type=task)],
        )

        downloads: list[tuple[str, str]] = []
        def fake_dl(task, vf, dest, hf_token=None):
            downloads.append((task, vf))
            dest.write_bytes(b"fake-video")
            return True
        monkeypatch.setattr(mv, "_hf_download_video", fake_dl)

        out = tmp_path / "out"
        mv.main(out, n=10, tasks=["object_existence"])
        assert downloads == [("object_existence", "x.mp4")]
        assert (out / "x.mp4").exists()

    def test_skips_when_hf_download_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            mv, "_load_task",
            lambda task, hf_token=None: [
                _row(video="ok.mp4"), _row(video="fail.mp4"),
            ],
        )
        def flaky_dl(task, vf, dest, hf_token=None):
            if "ok" in vf:
                dest.write_bytes(b"x")
                return True
            return False
        monkeypatch.setattr(mv, "_hf_download_video", flaky_dl)
        mv.main(tmp_path, n=10, tasks=["object_existence"])
        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {r["video"] for r in rows} == {"ok.mp4"}

    def test_skips_existing_videos(self, tmp_path, monkeypatch):
        (tmp_path / "a.mp4").write_bytes(b"already-here")

        monkeypatch.setattr(
            mv, "_load_task",
            lambda task, hf_token=None: [_row(video="a.mp4")],
        )
        called: list[str] = []
        monkeypatch.setattr(
            mv, "_hf_download_video",
            lambda *a, **kw: called.append("dl") or True,
        )
        mv.main(tmp_path, n=10, tasks=["object_existence"])
        assert called == []
        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[0]["video"] == "a.mp4"

    def test_handles_task_load_failure_gracefully(self, tmp_path, monkeypatch):
        """If one task config fails to load (e.g., gated), keep going."""
        def loader(task, hf_token=None):
            if task == "broken":
                raise RuntimeError("gated config")
            return [_row(video=f"{task}.mp4", task_type=task)]
        monkeypatch.setattr(mv, "_load_task", loader)
        monkeypatch.setattr(
            mv, "_hf_download_video",
            lambda task, vf, dest, hf_token=None: (dest.write_bytes(b"x") or True),
        )
        mv.main(tmp_path, n=10, tasks=["broken", "object_existence"])
        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
        # Only object_existence survived.
        assert {r["video"] for r in rows} == {"object_existence.mp4"}

    def test_pools_across_tasks_and_caps_n(self, tmp_path, monkeypatch):
        def loader(task, hf_token=None):
            return [_row(video=f"{task}_{i}.mp4", task_type=task) for i in range(3)]
        monkeypatch.setattr(mv, "_load_task", loader)
        monkeypatch.setattr(
            mv, "_hf_download_video",
            lambda task, vf, dest, hf_token=None: (dest.write_bytes(b"x") or True),
        )
        mv.main(tmp_path, n=4, tasks=["object_existence", "action_sequence"])
        # 3 + 3 candidates, capped to n=4.
        rows = [json.loads(l) for l in (tmp_path / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len({r["video"] for r in rows}) == 4


# --- task→subfolder map invariants -----------------------------------------


class TestTaskMap:
    def test_preferred_tasks_are_all_mapped(self):
        """The defaults must have a downloadable subfolder, or HF mode breaks."""
        for task in DEFAULT_TASKS_PREFERRED:
            assert task in TASK_VIDEO_DIRS, (
                f"preferred task '{task}' missing from TASK_VIDEO_DIRS — "
                f"HF fallback would fail without --video_src_dir"
            )

    def test_all_dirs_have_trailing_slash(self):
        """We concatenate `{subdir}{filename}` — a missing slash silently
        produces a wrong path."""
        for task, subdir in TASK_VIDEO_DIRS.items():
            assert subdir.endswith("/"), (
                f"TASK_VIDEO_DIRS[{task!r}] should end with '/'; got {subdir!r}"
            )


# --- friendly error when extras aren't installed --------------------------


class TestExtrasGuard:
    def test_load_task_requires_mvbench_extra(self, monkeypatch):
        """If `datasets` isn't importable, raise a friendly error pointing
        at the [mvbench] extra."""
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
        monkeypatch.setattr(mv, "require", fake_require)

        with pytest.raises(ModuleNotFoundError, match=r"\[mvbench\]"):
            mv._load_task("object_existence")
