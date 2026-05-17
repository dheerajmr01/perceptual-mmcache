import { Github, Zap, BookOpen } from "lucide-react";

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-cream-200 bg-cream-100/60">
      <div className="mx-auto max-w-6xl px-6 py-10">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-lg bg-perceptual/10 ring-1 ring-perceptual/25">
              <Zap className="h-5 w-5 text-perceptual" />
            </div>
            <div>
              <div className="text-sm font-semibold text-ink-900">pmcache</div>
              <div className="text-xs text-ink-500">
                perceptual-mmcache · acceleration-track demo
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 text-sm">
            <a
              href="https://github.com/dheerajmr01/perceptual-mmcache"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-cream-200 bg-white px-3 py-1.5 text-ink-700 hover:text-ink-900 hover:bg-cream-50 transition-colors"
            >
              <Github className="h-4 w-4" />
              Source code
            </a>
            <a
              href="https://github.com/dheerajmr01/perceptual-mmcache/blob/main/pmcache_colab.ipynb"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-md border border-perceptual/30 bg-perceptual/10 px-3 py-1.5 text-perceptual hover:bg-perceptual/15 transition-colors"
            >
              <BookOpen className="h-4 w-4" />
              Colab notebook
            </a>
          </div>
        </div>

        <div className="mt-8 grid md:grid-cols-3 gap-4 text-xs text-ink-500">
          <div>
            <div className="text-ink-700 font-medium mb-2">Built with</div>
            <ul className="space-y-1 font-mono">
              <li>vllm + lmcache</li>
              <li>imagehash · pybktree</li>
              <li>DINOv2-small (facebook/dinov2)</li>
              <li>PIL EXIF · Qwen3-VL-2B-Instruct</li>
            </ul>
          </div>
          <div>
            <div className="text-ink-700 font-medium mb-2">Benchmarked on</div>
            <ul className="space-y-1 font-mono">
              <li>Colab Pro · NVIDIA A100-40GB</li>
              <li>Video-MME (test split, short-duration filter)</li>
              <li>9 videos × 3 questions = 27 calls</li>
              <li>2,643 frames sampled total</li>
            </ul>
          </div>
          <div>
            <div className="text-ink-700 font-medium mb-2">Acceleration track</div>
            <p className="leading-relaxed">
              This project addresses video-LLM inference acceleration through
              perceptual frame deduplication. The win is measurable in TTFT
              and KV memory pressure, with zero accuracy regression and zero
              vLLM internals touched.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-cream-200 text-xs text-ink-400 font-mono">
          © 2026 · pmcache demo site · auto-generated from JSONL benchmark
          results via{" "}
          <code className="text-ink-500">
            demo/website/scripts/build_data.py
          </code>
        </div>
      </div>
    </footer>
  );
}
