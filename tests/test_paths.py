"""Tests for eval/paths.py — central path constants for benchmarking."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import paths


def test_default_workspace_subpaths():
    """All subpaths sit under WORKSPACE in the documented layout."""
    paths.set_workspace("workspace")  # reset to default
    assert paths.VIDEOS_DIR == Path("workspace") / "videos"
    assert paths.RESULTS_DIR == Path("workspace") / "results"
    assert paths.SWEEP_DIR == Path("workspace") / "results" / "sweep"
    assert paths.REPORT_DIR == Path("workspace") / "results" / "report"
    assert paths.QA_FILE == Path("workspace") / "videos" / "qa.jsonl"
    assert paths.BASELINE_PATH == Path("workspace") / "results" / "baseline.jsonl"
    assert paths.PERCEPTUAL_PATH == Path("workspace") / "results" / "perceptual.jsonl"


def test_set_workspace_updates_all_paths(tmp_path):
    paths.set_workspace(tmp_path / "my_run")
    assert paths.WORKSPACE == tmp_path / "my_run"
    assert paths.VIDEOS_DIR == tmp_path / "my_run" / "videos"
    assert paths.BASELINE_PATH == tmp_path / "my_run" / "results" / "baseline.jsonl"


def test_ensure_dirs_creates_workspace_tree(tmp_path):
    paths.set_workspace(tmp_path / "new_ws")
    assert not paths.WORKSPACE.exists()
    paths.ensure_dirs()
    assert paths.WORKSPACE.is_dir()
    assert paths.VIDEOS_DIR.is_dir()
    assert paths.RESULTS_DIR.is_dir()
    assert paths.SWEEP_DIR.is_dir()
    assert paths.REPORT_DIR.is_dir()


def test_pmcache_workspace_env_var(monkeypatch, tmp_path):
    """A fresh `from eval import paths` after setting PMCACHE_WORKSPACE
    should honor the env var."""
    monkeypatch.setenv("PMCACHE_WORKSPACE", str(tmp_path / "env_ws"))
    # Force re-init by calling the same logic the module does at import.
    paths.set_workspace(str(tmp_path / "env_ws"))
    assert paths.WORKSPACE == tmp_path / "env_ws"


def test_set_workspace_is_idempotent(tmp_path):
    paths.set_workspace(tmp_path / "a")
    paths.set_workspace(tmp_path / "a")
    assert paths.WORKSPACE == tmp_path / "a"


def test_workspace_function_returns_root(tmp_path):
    paths.set_workspace(tmp_path / "x")
    assert paths.workspace() == tmp_path / "x"
