"""Shared pytest fixtures + invariants for pmcache tests."""

from __future__ import annotations

import os
import sys

# Force matplotlib's non-interactive Agg backend before any pyplot import.
# On Windows the default TkAgg backend can fail when tests recreate
# figures across module boundaries ("Tcl wasn't installed properly").
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


@pytest.fixture
def torch_not_imported() -> None:
    """Assert torch is NOT in sys.modules.

    The pmcache CPU-laptop development loop depends on torch never being
    pulled in eagerly. If a test that uses this fixture finds torch
    already loaded, someone added a top-level `import torch` to a module
    that's part of the import-cheap path — fix that, don't loosen this.
    """
    if "torch" in sys.modules:
        pytest.fail(
            "torch was imported eagerly. Find the `import torch` at "
            "module top level and move it inside the function that "
            "needs it (use pmcache._lazy.require)."
        )
