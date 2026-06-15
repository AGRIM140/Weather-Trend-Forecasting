"use client";
// frontend/app/page.tsx
//
// Root dashboard page. Manages the tab state, location selection, and
// fetches KPI data for the Overview tab. ForecastChart and
// ExplainabilityChart own their own data fetching so they remain
// independently cacheable and cancellable.

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Database,
  Globe2,
  MapPin,
  Thermometer,
  TrendingUp,
  Wind,
} from "lucide-react";

import Sidebar from "@/components/Sidebar";
import KpiCard from "@/components/KpiCard";
import ForecastChart from "@/components/ForecastChart";
import ExplainabilityChart from "@/components/ExplainabilityChart";
import { fetchKpis, fetchLocations } from "@/lib/api";
import type { KpiResponse, NavTab } from "@/types";

// ── Overview Tab ────────────────────────────────────────────────────────────
function OverviewTab({
  kpis,
  loading,
}: {
  kpis: KpiResponse | null;
  loading: boolean;
}) {
  return (
    <div className="space-y-6 animate-fade-up">
      {/* ── Hero row: five primary KPIs ── */}
      <section>
        <p className="section-label mb-3">Dataset at a Glance</p>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
          <KpiCard
            label="Total Observations"
            value={kpis?.total_observations ?? 0}
            sub="After preprocessing"
            accent
            loading={loading}
          />
          <KpiCard
            label="Locations"
            value={kpis?.location_count ?? 0}
            sub={`${kpis?.country_count ?? 0} countries`}
            loading={loading}
          />
          <KpiCard
            label="Engineered Features"
            value={kpis?.feature_count ?? 0}
            sub="Incl. lag · rolling · cyclical"
            loading={loading}
          />
          <KpiCard
            label="Date Range"
            value={
              kpis
                ? `${kpis.date_range_start.slice(0, 7)} → ${kpis.date_range_end.slice(0, 7)}`
                : "—"
            }
            sub="Hourly readings"
            loading={loading}
          />
          <KpiCard
            label="Most Observed"
            value={kpis?.most_observed_location ?? "—"}
            sub="Highest observation count"
            loading={loading}
          />
        </div>
      </section>

      {/* ── Weather statistics ── */}
      <section>
        <p className="section-label mb-3">Global Weather Statistics</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <KpiCard
            label="Avg Temperature"
            value={kpis ? `${kpis.avg_temperature_celsius.toFixed(1)} °C` : "—"}
            sub={
              kpis
                ? `${kpis.min_temperature_celsius}° – ${kpis.max_temperature_celsius}°`
                : ""
            }
            accent
            loading={loading}
          />
          <KpiCard
            label="Avg Humidity"
            value={kpis ? `${kpis.avg_humidity.toFixed(0)}%` : "—"}
            sub="Global mean"
            loading={loading}
          />
          <KpiCard
            label="Avg Pressure"
            value={kpis ? `${kpis.avg_pressure_mb.toFixed(0)} mb` : "—"}
            sub="Mean sea-level"
            loading={loading}
          />
          <KpiCard
            label="Avg Wind Speed"
            value={kpis ? `${kpis.avg_wind_kph.toFixed(1)} kph` : "—"}
            sub="Surface wind"
            loading={loading}
          />
        </div>
      </section>

      {/* ── Outlier stats ── */}
      <section>
        <p className="section-label mb-3">Data Quality</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <KpiCard
            label="IQR Outlier Rows"
            value={kpis ? `${kpis.outlier_iqr_pct.toFixed(2)}%` : "—"}
            sub="Tukey fences k = 1.5 · Retained in training"
            trend="neutral"
            loading={loading}
          />
          <KpiCard
            label="Isolation Forest Anomalies"
            value={kpis ? `${kpis.outlier_isoforest_pct.toFixed(2)}%` : "—"}
            sub="contamination = 0.01 · Retained in training"
            trend="neutral"
            loading={loading}
          />
        </div>
      </section>

      {/* ── Architecture summary ── */}
      <section>
        <p className="section-label mb-3">Pipeline Architecture</p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            {
              icon: Database,
              title: "ETL & Preprocessing",
              body: "Panel-aware groupby imputation, dual outlier detection (IQR + Isolation Forest), label encoding, StandardScaler.",
            },
            {
              icon: Activity,
              title: "Feature Engineering",
              body: "Lag-1 · Lag-7, rolling mean/std 3d & 7d, cyclical sin/cos encoding — applied to 5 exogenous variables.",
            },
            {
              icon: BarChart3,
              title: "Ensemble Forecasting",
              body: "ARIMA · Prophet · Random Forest · Extra Trees · XGBoost · LightGBM — inverse-RMSE weighted ensemble.",
            },
            {
              icon: TrendingUp,
              title: "Explainability",
              body: "SHAP TreeExplainer (exact Shapley values) + permutation importance — unified combined ranking.",
            },
          ].map(({ icon: Icon, title, body }) => (
            <div key={title} className="card p-4">
              <div className="flex items-center gap-2 mb-2.5">
                <div className="h-7 w-7 rounded-lg bg-signal-muted flex items-center justify-center">
                  <Icon className="h-3.5 w-3.5 text-signal" />
                </div>
                <p className="text-xs font-semibold text-ink-primary">{title}</p>
              </div>
              <p className="text-2xs text-ink-secondary leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

// ── Tab content shells ───────────────────────────────────────────────────────
function ForecastTab({ location }: { location: string }) {
  return (
    <div className="animate-fade-up">
      <div className="mb-4 flex items-center gap-2">
        <MapPin className="h-4 w-4 text-signal" />
        <h2 className="text-sm font-semibold text-ink-primary">
          Forecasting — <span className="text-signal">{location}</span>
        </h2>
      </div>
      <ForecastChart location={location} />
    </div>
  );
}

function ExplainabilityTab({ location }: { location: string }) {
  return (
    <div className="animate-fade-up">
      <div className="mb-4 flex items-center gap-2">
        <MapPin className="h-4 w-4 text-signal" />
        <h2 className="text-sm font-semibold text-ink-primary">
          Explainability — <span className="text-signal">{location}</span>
        </h2>
      </div>
      <ExplainabilityChart location={location} />
    </div>
  );
}

// ── Page header ─────────────────────────────────────────────────────────────
function Header({ activeTab, location }: { activeTab: NavTab; location: string }) {
  const titles: Record<NavTab, { label: string; icon: React.ElementType }> = {
    overview: { label: "Dataset Overview", icon: Globe2 },
    forecast: { label: "30-Day Forecast", icon: Thermometer },
    explainability: { label: "Model Explainability", icon: Wind },
  };
  const { label, icon: Icon } = titles[activeTab];

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-surface-hover bg-surface-muted/60 backdrop-blur-sm sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <Icon className="h-4 w-4 text-signal" />
        <h1 className="text-sm font-semibold text-ink-primary">{label}</h1>
        <span className="hidden md:inline text-2xs text-ink-tertiary">
          ·{" "}
          <span className="text-data">{location}</span>
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="h-1.5 w-1.5 rounded-full bg-positive animate-pulse" />
        <span className="text-2xs text-ink-tertiary">API connected</span>
      </div>
    </header>
  );
}

// ── Root page ────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<NavTab>("overview");
  const [location, setLocation] = useState<string>("");
  const [locations, setLocations] = useState<string[]>([]);
  const [kpis, setKpis] = useState<KpiResponse | null>(null);
  const [kpiLoading, setKpiLoading] = useState(true);
  const [locLoading, setLocLoading] = useState(true);

  // Load locations once on mount
  useEffect(() => {
    fetchLocations()
      .then((res) => {
        setLocations(res.locations);
        // Default to "London" if available, otherwise first location
        const def = res.locations.includes("London")
          ? "London"
          : res.locations[0] ?? "";
        setLocation(def);
      })
      .catch(console.error)
      .finally(() => setLocLoading(false));
  }, []);

  // Load KPIs once on mount
  useEffect(() => {
    setKpiLoading(true);
    fetchKpis()
      .then(setKpis)
      .catch(console.error)
      .finally(() => setKpiLoading(false));
  }, []);

  const handleLocationChange = useCallback((loc: string) => {
    setLocation(loc);
  }, []);

  return (
    <div className="flex min-h-dvh">
      {/* ── Fixed Sidebar ── */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        location={location}
        locations={locations}
        onLocationChange={handleLocationChange}
        isLoading={locLoading}
      />

      {/* ── Main content area ── */}
      <div className="flex flex-1 flex-col ml-64">
        <Header activeTab={activeTab} location={location} />

        <main className="flex-1 overflow-auto p-6">
          {activeTab === "overview" && (
            <OverviewTab kpis={kpis} loading={kpiLoading} />
          )}
          {activeTab === "forecast" && location && (
            <ForecastTab location={location} />
          )}
          {activeTab === "explainability" && location && (
            <ExplainabilityTab location={location} />
          )}
        </main>
      </div>
    </div>
  );
}
