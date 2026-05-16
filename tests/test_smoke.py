"""Step 1 smoke tests — scaffolding is healthy.

These tests do NOT exercise pmcache behavior (that's Steps 2-13). They
only verify that `pip install -e '.[dev]'` produced an importable
package, public eval API signatures match the Colab notebook, and no
heavy deps leaked into the import-cheap path.
"""

from __future__ import annotations

import inspect
import sys


def test_pmcache_imports_without_torch(torch_not_imported):
    import pmcache  # noqa: F401

    assert pmcache.__version__ == "0.1.0"


def test_pmcache_submodules_import_without_torch(torch_not_imported):
    import pmcache.config  # noqa: F401
    import pmcache.hashing  # noqa: F401
    import pmcache.lmcache_shim  # noqa: F401
    import pmcache.metrics  # noqa: F401
    import pmcache.store  # noqa: F401
    import pmcache.verifier  # noqa: F401

    assert "torch" not in sys.modules, (
        f"Importing pmcache submodules pulled in torch: "
        f"check which module has a top-level torch import."
    )


def test_eval_modules_import_without_torch(torch_not_imported):
    import eval.benchmark_videoqa  # noqa: F401
    import eval.run_baseline  # noqa: F401
    import eval.run_perceptual  # noqa: F401
    import eval.threshold_sweep  # noqa: F401
    import eval.utils  # noqa: F401

    assert "torch" not in sys.modules
    assert "vllm" not in sys.modules
    assert "transformers" not in sys.modules


def test_eval_public_api_signatures_match_notebook():
    """Every eval entry point must accept mock_vlm and the params the
    Colab notebook passes by keyword."""
    from eval.benchmark_videoqa import analyze_results
    from eval.run_baseline import run_baseline_benchmark
    from eval.run_perceptual import run_perceptual_benchmark
    from eval.threshold_sweep import sweep_thresholds
    from eval.utils import smoke_test_vlm

    for fn in (
        run_baseline_benchmark,
        run_perceptual_benchmark,
        sweep_thresholds,
        analyze_results,
    ):
        assert "mock_vlm" in inspect.signature(fn).parameters, (
            f"{fn.__name__} must accept mock_vlm for CPU-only tests"
        )

    # smoke_test_vlm has fixed (model, workspace) — no mock_vlm by design
    sig = inspect.signature(smoke_test_vlm)
    assert "model" in sig.parameters
    assert "workspace" in sig.parameters

    # Notebook calls baseline with fps=, model=
    sig = inspect.signature(run_baseline_benchmark)
    assert "fps" in sig.parameters
    assert "model" in sig.parameters

    # Notebook calls perceptual with tau=, k=, fps=, model=
    sig = inspect.signature(run_perceptual_benchmark)
    for p in ("tau", "k", "fps", "model"):
        assert p in sig.parameters, f"run_perceptual_benchmark missing {p}"

    # Sweep takes a list of taus + output_dir
    sig = inspect.signature(sweep_thresholds)
    for p in ("taus", "output_dir", "model"):
        assert p in sig.parameters, f"sweep_thresholds missing {p}"


def test_config_from_env_respects_overrides(monkeypatch):
    """PMCacheConfig.from_env() must pick up PMCACHE_* overrides."""
    from pmcache.config import PMCacheConfig

    monkeypatch.setenv("PMCACHE_TAU", "0.95")
    monkeypatch.setenv("PMCACHE_K", "7")
    monkeypatch.setenv("PMCACHE_ENABLED", "false")

    cfg = PMCacheConfig.from_env()
    assert cfg.tau == 0.95
    assert cfg.k == 7
    assert cfg.enabled is False


def test_config_defaults_when_env_unset(monkeypatch):
    from pmcache.config import PMCacheConfig

    for var in ("PMCACHE_TAU", "PMCACHE_K", "PMCACHE_ENABLED",
                "PMCACHE_BUCKET_CAPACITY", "PMCACHE_MAX_BUCKETS"):
        monkeypatch.delenv(var, raising=False)

    cfg = PMCacheConfig.from_env()
    assert cfg.tau == 0.98
    assert cfg.k == 5
    assert cfg.enabled is True
    assert cfg.bucket_capacity == 128
    assert cfg.max_buckets == 4096


def test_lazy_require_raises_friendly_error_for_missing_module():
    """_lazy.require should raise a clear pip-install hint, not bare ImportError."""
    from pmcache._lazy import require

    try:
        require("definitely_not_a_real_module_xyz", extra="vlm")
    except ModuleNotFoundError as e:
        msg = str(e)
        assert "[vlm]" in msg, f"error should mention the extra: {msg}"
        assert "pip install" in msg, f"error should give an install hint: {msg}"
    else:
        raise AssertionError("require() should have raised for a missing module")
