# pmcache demo website

Static React site for the hackathon **acceleration**-track demo. Renders the
benchmark numbers from the JSONL results in a single-page, scroll-through
narrative: problem → solution → architecture → measured results → code.

## Quick start

```bash
cd demo/website
npm install
npm run dev      # http://localhost:5173
```

```bash
npm run build    # production build to dist/
npm run preview  # preview the production build
```

## Regenerating the data

The site reads from `src/data/results.ts`, which is auto-generated from the
JSONL files in
`pmcache-20260517T031128Z-3-001/pmcache/results/`. Re-run after a fresh
benchmark:

```bash
# from repo root
python demo/website/scripts/build_data.py

# or, with a custom results dir:
python demo/website/scripts/build_data.py \
    --results-dir path/to/your/results/folder
```

The script reads `baseline.jsonl`, `perceptual.jsonl`, `videos/qa.jsonl`, and
`smoke.json`, then writes a typed TS module the site consumes at build time.
No backend, no live API — everything is baked in at build.

## Structure

```
demo/website/
├── package.json
├── vite.config.ts
├── tsconfig.json / tsconfig.node.json
├── tailwind.config.js / postcss.config.js
├── index.html
├── public/
│   └── favicon.svg
├── scripts/
│   └── build_data.py           # JSONL → src/data/results.ts
└── src/
    ├── main.tsx
    ├── index.css                # Tailwind + a few card/badge utilities
    ├── App.tsx
    ├── data/
    │   ├── results.ts           # AUTO-GENERATED
    │   └── metrics.ts           # summarize() + delta helpers
    └── components/
        ├── NavBar.tsx
        ├── Hero.tsx
        ├── Problem.tsx
        ├── Solution.tsx
        ├── Architecture.tsx     # flow-diagram in CSS boxes (no SVG library)
        ├── BenchmarkSetup.tsx
        ├── HeadlineMetrics.tsx  # the 3 big numbers
        ├── MetricsTable.tsx     # mirror of REPORT.md table
        ├── TTFTChart.tsx        # histogram + CDF
        ├── PerVideoChart.tsx    # per-video TTFT + unique-frames + speedup
        ├── PmcacheCounters.tsx  # shim counters + pie chart
        ├── DetailedTable.tsx    # expandable per-(video, question) table
        ├── Implementation.tsx   # code snippets
        └── Footer.tsx
```

## Stack

- **React 18 + TypeScript** — strict mode on
- **Vite 5** — instant HMR, single-file prod build
- **Tailwind 3** — dark-theme tokens in `index.css`
- **Recharts** — bar / line / pie charts
- **lucide-react** — icons

## Color tokens (Tailwind)

- `baseline` = `slate-500` (#94a3b8) — gray, neutral
- `perceptual` = `emerald-400` (#34d399) — green, the win
- `highlight` = `amber-400` (#fbbf24) — amber, the headline pop
