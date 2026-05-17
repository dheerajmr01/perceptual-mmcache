import { Cpu, Database, Settings, FileVideo } from "lucide-react";
import { BASELINE_SUMMARY, PERCEPTUAL_SUMMARY } from "../data/metrics";
import { SMOKE, VIDEO_META } from "../data/results";

export default function BenchmarkSetup() {
  const videos = Object.keys(VIDEO_META);
  return (
    <section id="setup" className="section">
      <div className="eyebrow">Benchmark setup</div>
      <h2 className="section-title mt-2">How the numbers were measured</h2>
      <p className="section-subtitle">
        Same model, same questions, same frame sampling — only one variable
        flipped between the two runs: whether each frame goes through{" "}
        <code className="font-mono">PerceptualMMCache.prepare_image()</code> first.
      </p>

      <div className="grid md:grid-cols-2 gap-4 mt-10">
        <Card icon={<Cpu className="h-5 w-5 text-perceptual" />} title="Model">
          <KV k="VLM" v={SMOKE.model ?? "Qwen/Qwen3-VL-2B-Instruct"} />
          <KV k="vLLM backend" v="vllm + LMCache (Colab Pro A100-40G)" />
          <KV k="DINOv2 verifier" v="DINOv2-small (lazy-loaded)" />
          <KV k="Smoke test" v={`${SMOKE.smoke_elapsed_s?.toFixed(2) ?? "?"} s end-to-end`} />
        </Card>

        <Card icon={<Settings className="h-5 w-5 text-perceptual" />} title="Hyperparameters">
          <KV k="τ (cosine threshold)" v="0.98" />
          <KV k="K (pHash Hamming cap)" v="5" />
          <KV k="FPS" v="1.0 frames / video-second" />
          <KV k="Frame cap" v="up to ~120 per video" />
        </Card>

        <Card icon={<FileVideo className="h-5 w-5 text-perceptual" />} title="Dataset">
          <KV k="Source" v="Video-MME (lmms-lab/Video-MME, test split)" />
          <KV k="Filter" v="duration=short · Knowledge / Humanity & History / Lit & Art" />
          <KV k="Videos" v={`${videos.length}`} />
          <KV k="Questions" v={`${BASELINE_SUMMARY.rows} (≈3 per video, MCQ A–D)`} />
        </Card>

        <Card icon={<Database className="h-5 w-5 text-perceptual" />} title="What we measured">
          <KV k="Total frames sampled" v={`${BASELINE_SUMMARY.totalFrames.toLocaleString()}`} />
          <KV
            k="Unique mm_hashes — baseline"
            v={`${BASELINE_SUMMARY.uniqueFrames.toLocaleString()} (${((BASELINE_SUMMARY.uniqueFrames / BASELINE_SUMMARY.totalFrames) * 100).toFixed(1)}%)`}
          />
          <KV
            k="Unique mm_hashes — perceptual"
            v={`${PERCEPTUAL_SUMMARY.uniqueFrames.toLocaleString()} (${((PERCEPTUAL_SUMMARY.uniqueFrames / PERCEPTUAL_SUMMARY.totalFrames) * 100).toFixed(1)}%)`}
          />
          <KV
            k="KV bytes / token (Qwen3-VL-2B)"
            v="112 KiB"
          />
        </Card>
      </div>
    </section>
  );
}

function Card({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card p-6 card-hover">
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <h3 className="text-base font-semibold text-ink-900">{title}</h3>
      </div>
      <dl className="space-y-2">{children}</dl>
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between items-baseline gap-4">
      <dt className="text-sm text-ink-500">{k}</dt>
      <dd className="text-sm font-mono text-ink-900 text-right">{v}</dd>
    </div>
  );
}
