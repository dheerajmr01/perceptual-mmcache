import { useMemo, useState } from "react";
import { Check, X, ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { BASELINE, PERCEPTUAL, VIDEO_META, RunRow } from "../data/results";
import { formatSeconds } from "../data/metrics";

interface Pair {
  video: string;
  question: string;
  baseline: RunRow;
  perceptual: RunRow;
}

export default function DetailedTable() {
  const pairs = useMemo<Pair[]>(() => {
    const out: Pair[] = [];
    for (let i = 0; i < BASELINE.length && i < PERCEPTUAL.length; i++) {
      const b = BASELINE[i];
      const p = PERCEPTUAL[i];
      if (b.video !== p.video || b.question !== p.question) continue;
      out.push({ video: b.video, question: b.question, baseline: b, perceptual: p });
    }
    return out;
  }, []);

  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  return (
    <section id="detailed" className="section">
      <div className="eyebrow">Detailed results</div>
      <h2 className="section-title mt-2">All 27 (video × question) rows</h2>
      <p className="section-subtitle">
        Each row pairs a baseline call with its perceptual counterpart on the
        same video + question. Click a row to expand its full prediction +
        gold answer + MCQ options.
      </p>

      <div className="mt-8 card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-cream-100">
              <tr>
                <th className="table-th w-6"></th>
                <th className="table-th">Video</th>
                <th className="table-th">Question</th>
                <th className="table-th text-right">Frames</th>
                <th className="table-th text-right">Unique (B → P)</th>
                <th className="table-th text-right">TTFT (B → P)</th>
                <th className="table-th text-right">Speedup</th>
                <th className="table-th text-center">B✓</th>
                <th className="table-th text-center">P✓</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((row, i) => {
                const speedup =
                  row.baseline.ttft_s > 0
                    ? ((row.baseline.ttft_s - row.perceptual.ttft_s) /
                        row.baseline.ttft_s) *
                      100
                    : 0;
                const isOpen = expanded.has(i);
                const url = VIDEO_META[row.video]?.url;
                return (
                  <>
                    <tr
                      key={`r-${i}`}
                      className={`cursor-pointer ${
                        isOpen ? "bg-cream-100" : "hover:bg-cream-50"
                      }`}
                      onClick={() => {
                        const next = new Set(expanded);
                        if (next.has(i)) next.delete(i);
                        else next.add(i);
                        setExpanded(next);
                      }}
                    >
                      <td className="table-td">
                        {isOpen ? (
                          <ChevronDown className="h-3.5 w-3.5 text-ink-400" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5 text-ink-400" />
                        )}
                      </td>
                      <td className="table-td">
                        <div className="flex items-center gap-1.5">
                          <span className="font-mono text-ink-900">
                            {row.video.replace(".mp4", "")}
                          </span>
                          {url && (
                            <a
                              href={url}
                              target="_blank"
                              rel="noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="text-ink-400 hover:text-ink-700"
                              title="open source video"
                            >
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          )}
                        </div>
                      </td>
                      <td className="table-td max-w-md">
                        <div className="text-ink-700 line-clamp-1">
                          {row.question}
                        </div>
                      </td>
                      <td className="table-td text-right font-mono text-ink-700">
                        {row.baseline.num_frames}
                      </td>
                      <td className="table-td text-right font-mono">
                        <span className="text-ink-400">
                          {row.baseline.unique_mm_hashes}
                        </span>
                        <span className="text-ink-300 mx-1">→</span>
                        <span className="text-perceptual">
                          {row.perceptual.unique_mm_hashes}
                        </span>
                      </td>
                      <td className="table-td text-right font-mono">
                        <span className="text-ink-400">
                          {formatSeconds(row.baseline.ttft_s, 2)}
                        </span>
                        <span className="text-ink-300 mx-1">→</span>
                        <span className="text-perceptual">
                          {formatSeconds(row.perceptual.ttft_s, 2)}
                        </span>
                      </td>
                      <td
                        className={`table-td text-right font-mono ${
                          speedup >= 0 ? "text-perceptual" : "text-[#a8444d]"
                        }`}
                      >
                        {speedup >= 0 ? "+" : ""}
                        {speedup.toFixed(1)}%
                      </td>
                      <td className="table-td text-center">
                        {row.baseline.correct ? (
                          <Check className="h-4 w-4 text-perceptual inline" />
                        ) : (
                          <X className="h-4 w-4 text-[#a8444d] inline" />
                        )}
                      </td>
                      <td className="table-td text-center">
                        {row.perceptual.correct ? (
                          <Check className="h-4 w-4 text-perceptual inline" />
                        ) : (
                          <X className="h-4 w-4 text-[#a8444d] inline" />
                        )}
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={`d-${i}`} className="bg-cream-50">
                        <td className="table-td" colSpan={9}>
                          <ExpandedDetail row={row} />
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function ExpandedDetail({ row }: { row: Pair }) {
  const meta = VIDEO_META[row.video];
  return (
    <div className="p-4 space-y-3">
      <div>
        <div className="text-xs uppercase tracking-wider text-ink-400 mb-1">
          Question
        </div>
        <div className="text-sm text-ink-800">{row.question}</div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-400 mb-1">
            MCQ options
          </div>
          <ul className="text-sm font-mono text-ink-700 space-y-1">
            {row.baseline.mcq_options.map((opt, i) => {
              const letter = opt.match(/^([A-E])\./)?.[1];
              const isGold = letter === row.baseline.mcq_answer;
              return (
                <li
                  key={i}
                  className={isGold ? "text-perceptual font-semibold" : ""}
                >
                  {opt} {isGold && " ← gold"}
                </li>
              );
            })}
          </ul>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-400 mb-1">
            Predictions
          </div>
          <dl className="text-sm space-y-1.5">
            <div className="flex gap-3">
              <dt className="text-ink-500 w-24">baseline</dt>
              <dd className="font-mono text-ink-700">
                {row.baseline.predicted_letter ?? "?"}{" "}
                <span className="text-ink-400">
                  ({row.baseline.answer.slice(0, 40)})
                </span>
              </dd>
            </div>
            <div className="flex gap-3">
              <dt className="text-ink-500 w-24">perceptual</dt>
              <dd className="font-mono text-ink-700">
                {row.perceptual.predicted_letter ?? "?"}{" "}
                <span className="text-ink-400">
                  ({row.perceptual.answer.slice(0, 40)})
                </span>
              </dd>
            </div>
            <div className="flex gap-3">
              <dt className="text-ink-500 w-24">gold</dt>
              <dd className="font-mono text-perceptual">
                {row.baseline.mcq_answer}{" "}
                <span className="text-ink-400">
                  ({row.baseline.gold_answer.slice(0, 40)})
                </span>
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {meta && (
        <div className="pt-2 border-t border-cream-200 text-xs text-ink-400 flex flex-wrap gap-x-4 gap-y-1">
          {meta.domain && (
            <span>
              domain: <span className="text-ink-600">{meta.domain}</span>
            </span>
          )}
          {meta.sub_category && (
            <span>
              sub-category:{" "}
              <span className="text-ink-600">{meta.sub_category}</span>
            </span>
          )}
          <span>
            prompt_tokens:{" "}
            <span className="text-ink-600 font-mono">{row.baseline.prompt_tokens}</span>
          </span>
          <span>
            cached_tokens (B → P):{" "}
            <span className="text-ink-600 font-mono">
              {row.baseline.cached_tokens} → {row.perceptual.cached_tokens}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
