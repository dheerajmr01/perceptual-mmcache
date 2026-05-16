"""pmcache — perceptual-similarity extension to LMCache for video LLMs.

Top-level package. Deliberately keeps imports minimal so that
`import pmcache` on a CPU-only laptop does not trigger any heavy deps
(torch / transformers / vllm / lmcache). Submodules are imported lazily
by callers.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
