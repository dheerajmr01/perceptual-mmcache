"""Throwaway: AST-parse every code cell in the notebook to catch syntax errors.

Ignored by tests/CI — lives here just so the build_colab_notebook.py
workflow has a verifiable smoke test. Safe to delete.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "pmcache_colab.ipynb"


def _strip_magics(src: str) -> str:
    """Remove IPython %magic and !shell lines (incl. \\-continuations)."""
    out = []
    in_cont = False
    for line in src.split("\n"):
        if in_cont:
            in_cont = line.rstrip().endswith("\\")
            continue
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            in_cont = line.rstrip().endswith("\\")
            continue
        out.append(line)
    return "\n".join(out)


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    errs = []
    n_code = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        src = c["source"]
        if isinstance(src, list):
            src = "".join(src)
        cleaned = _strip_magics(src)
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            errs.append((i, str(e), cleaned))
    if errs:
        for i, msg, snippet in errs:
            print(f"CELL {i}: {msg}")
            print(snippet)
            print("---")
        return 1
    print(f"All {n_code} code cells parse OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
