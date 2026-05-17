import { Github, Zap } from "lucide-react";

const LINKS = [
  { href: "#problem", label: "Problem" },
  { href: "#solution", label: "Solution" },
  { href: "#architecture", label: "Architecture" },
  { href: "#results", label: "Results" },
  { href: "#implementation", label: "Implementation" },
];

export default function NavBar() {
  return (
    <header className="fixed inset-x-0 top-0 z-40 backdrop-blur-md bg-cream-50/85 border-b border-cream-200">
      <div className="mx-auto max-w-6xl px-6 h-16 flex items-center justify-between">
        <a href="#hero" className="flex items-center gap-2 group">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-perceptual/10 ring-1 ring-perceptual/25">
            <Zap className="h-4 w-4 text-perceptual" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-ink-900 tracking-tight">
              pmcache
            </div>
            <div className="text-[10px] text-ink-400 uppercase tracking-widest">
              acceleration track
            </div>
          </div>
        </a>

        <nav className="hidden md:flex items-center gap-1">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="px-3 py-1.5 text-sm text-ink-500 hover:text-ink-900 transition-colors rounded-md hover:bg-cream-100"
            >
              {l.label}
            </a>
          ))}
        </nav>

        <a
          href="https://github.com/dheerajmr01/perceptual-mmcache"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-cream-200 bg-white px-3 py-1.5 text-sm text-ink-700 hover:bg-cream-100 hover:text-ink-900 transition-colors"
        >
          <Github className="h-4 w-4" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
      </div>
    </header>
  );
}
