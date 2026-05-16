"""eval — benchmark and analysis entry points.

Kept import-light: each module's heavy deps (torch, vllm, transformers,
matplotlib, etc.) are pulled in lazily inside functions, so the Colab
notebook can do `from eval.run_baseline import run_baseline_benchmark`
without paying for unrelated stacks at import time.

Note: this package is named `eval` to match the Colab notebook's import
sites. Inside this package, never write `eval = ...` as a local variable
or you will shadow Python's builtin.
"""
