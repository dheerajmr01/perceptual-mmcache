import { Film, Cpu, HardDrive } from "lucide-react";

export default function Problem() {
  return (
    <section id="problem" className="section">
      <div className="eyebrow">The problem</div>
      <h2 className="section-title mt-2">
        Video LLMs re-encode every frame, even when the frames are nearly identical.
      </h2>
      <p className="section-subtitle">
        vLLM&apos;s <code className="font-mono text-perceptual">MultiModalHasher</code>{" "}
        hashes raw pixel bytes. Two adjacent frames of a talking-head video are
        99% identical to the human eye, but byte-different — so they produce
        two different <code className="font-mono">mm_hash</code> values. The
        vision encoder runs once per frame. KV cache stores one block per
        frame. TTFT scales linearly with frame count.
      </p>

      <div className="grid md:grid-cols-3 gap-4 mt-10">
        <ProblemCard
          icon={<Film className="h-5 w-5 text-highlight" />}
          title="Frame-level redundancy is enormous"
          body="At 1 fps, an 87-frame video typically has only 20–25 visually-distinct keyframes. The rest are near-duplicates within Hamming distance 5 of a neighbor."
        />
        <ProblemCard
          icon={<Cpu className="h-5 w-5 text-highlight" />}
          title="Vision encoder pays the full bill"
          body="ViT-style encoders compute ~221 vision tokens per frame regardless of similarity. Running 87 times means 87 × ViT forward passes, even when only 7 of them produce different embeddings."
        />
        <ProblemCard
          icon={<HardDrive className="h-5 w-5 text-highlight" />}
          title="KV cache pressure compounds"
          body="LMCache keys each frame's KV by its mm_hash. Byte-different frames → distinct cache entries → each frame's KV stored, transferred, and retrieved separately."
        />
      </div>

    </section>
  );
}

function ProblemCard({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="card p-6 card-hover">
      <div className="mb-3">{icon}</div>
      <h3 className="text-base font-semibold text-ink-900">{title}</h3>
      <p className="mt-2 text-sm text-ink-500 leading-relaxed">{body}</p>
    </div>
  );
}
