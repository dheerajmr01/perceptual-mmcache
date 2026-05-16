"""Gradio demo — live UI for the hackathon presentation.

Step 13 will fill in. UI surface:
    - upload video
    - ask question
    - live hit-rate bar (Tier 1 / Tier 2)
    - per-frame status (hit / miss / aliased)
    - final answer + TTFT

Run on Colab with share=True to get a public URL.
"""

from __future__ import annotations


def launch_demo(model: str, workspace: str, share: bool = True) -> None:
    """Launch the Gradio app. Blocks until the server is killed.

    Args:
        model: HF model id (Qwen2-VL-2B-Instruct in the Colab default).
        workspace: writable dir for upload cache, frame thumbs, metrics.
        share: True to enable Gradio's public tunnel.
    """
    raise NotImplementedError(
        "pmcache step 13: build Gradio Blocks UI, wire to "
        "run_perceptual_benchmark via PMCACHE_* env vars, "
        "stream per-frame status to the UI"
    )
