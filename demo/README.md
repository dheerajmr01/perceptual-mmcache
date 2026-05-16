# pmcache Gradio demo

Live UI for the hackathon presentation. Upload a video, ask a question,
watch per-frame hit-rate light up as the perceptual cache kicks in.

Run on Colab (needs GPU for Qwen2-VL) after the main notebook finishes:

```python
from demo.gradio_app import launch_demo
launch_demo(model="Qwen/Qwen2-VL-2B-Instruct", workspace="/content/drive/MyDrive/pmcache", share=True)
```

`share=True` exposes a public `gradio.live` tunnel — perfect for sending
judges a live link during the demo.
