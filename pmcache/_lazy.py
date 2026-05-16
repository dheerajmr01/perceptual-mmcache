"""Lazy import helpers with friendly missing-dep errors.

Use `require("torch", extra="vlm")` instead of `import torch` at module
top level, so that `import pmcache.verifier` works on a CPU-only laptop
without the `[vlm]` extra installed and only fails the moment the user
actually calls into the VLM-dependent code path.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """Import `module`, raising a friendly error if missing.

    Args:
        module: dotted module name, e.g. "torch" or "transformers".
        extra: pyproject optional-dependency group that ships it,
            e.g. "vlm" or "eval".

    Returns:
        The imported module.

    Raises:
        ModuleNotFoundError: with a message pointing to the install command.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"pmcache requires `{module}` for this code path. "
            f"Install the `[{extra}]` extra:\n"
            f"    pip install -e '.[{extra}]'"
        ) from e
