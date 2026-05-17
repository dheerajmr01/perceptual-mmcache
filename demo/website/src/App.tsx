import Hero from "./components/Hero";
import Problem from "./components/Problem";
import Solution from "./components/Solution";
import Architecture from "./components/Architecture";
import BenchmarkSetup from "./components/BenchmarkSetup";
import HeadlineMetrics from "./components/HeadlineMetrics";
import TTFTChart from "./components/TTFTChart";
import PerVideoChart from "./components/PerVideoChart";
import MetricsTable from "./components/MetricsTable";
import PmcacheCounters from "./components/PmcacheCounters";
import DetailedTable from "./components/DetailedTable";
import Implementation from "./components/Implementation";
import Footer from "./components/Footer";
import NavBar from "./components/NavBar";

export default function App() {
  return (
    <div className="relative min-h-screen">
      {/* Soft warm wash behind the hero (matte, no glow). */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[600px] bg-gradient-to-b from-cream-100 via-cream-50 to-cream-50"
      />
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 grid-bg opacity-40" />

      <NavBar />

      <main className="pt-16">
        <Hero />
        <Problem />
        <Solution />
        <Architecture />
        <BenchmarkSetup />
        <HeadlineMetrics />
        <MetricsTable />
        <TTFTChart />
        <PerVideoChart />
        <PmcacheCounters />
        <DetailedTable />
        <Implementation />
      </main>

      <Footer />
    </div>
  );
}
