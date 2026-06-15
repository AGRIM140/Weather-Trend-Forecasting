"use client";
// frontend/components/ExplainabilityChart.tsx

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertCircle, BrainCircuit, Loader2, RefreshCw } from "lucide-react";
import { fetchExplainability } from "@/lib/api";
import type { ExplainabilityResponse, ShapFeature } from "@/types";

const TREE_MODELS = ["XGBoost", "LightGBM", "RandomForest", "ExtraTrees"] as const;
type TreeModel = (typeof TREE_MODELS)[number];

// ── Feature label helper: shorten long column names ────────────────────────
function shortLabel(feature: string): string {
  return feature
    .replace("temperature_celsius", "temp")
    .replace("_roll_mean_", "→mean")
    .replace("_roll_std_", "→std")
    .replace("_lag_", "→lag")
    .replace("air_quality_", "aq:")
    .replace("day_of_year", "doy")
    .replace("month_sin", "month·sin")
    .replace("month_cos", "month·cos");
}

// ── Colour ramp: dim slate → electric cyan based on normalized rank ─────────
function barColor(rank: number, total: number): string {
  const t = 1 - (rank - 1) / Math.max(total - 1, 1); // 1 = top, 0 = bottom
  // Interpolate #334155 → #22D3EE
  const r = Math.round(51  + t * (34  - 51));
  const g = Math.round(65  + t * (211 - 65));
  const b = Math.round(85  + t * (238 - 85));
  return `rgb(${r},${g},${b})`;
}

// ── Custom tooltip ──────────────────────────────────────────────────────────
function ShapTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ShapFeature }>;
}) {
  if (!active || !payload?.length) return null;
  const f = payload[0].payload;
  return (
    <div className="card border-surface-line px-3 py-2.5 text-xs shadow-xl max-w-xs">
      <p className="text-ink-primary font-semibold mb-2 break-words">{f.feature}</p>
      <div className="space-y-1 text-data">
        <div className="flex justify-between gap-4">
          <span className="text-ink-tertiary">Mean |SHAP|</span>
          <span className="text-signal">{f.mean_abs_shap.toFixed(5)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-ink-tertiary">Perm. importance</span>
          <span className="text-amber">{f.permutation_importance.toFixed(5)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-ink-tertiary">Combined rank</span>
          <span className="text-ink-primary">#{f.combined_rank}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-ink-tertiary">Combined score</span>
          <span className="text-ink-primary">{f.combined_score.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
interface ExplainabilityChartProps {
  location: string;
}

export default function ExplainabilityChart({ location }: ExplainabilityChartProps) {
  const [data, setData] = useState<ExplainabilityResponse | null>(null);
  const [model, setModel] = useState<TreeModel>("XGBoost");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    if (!location) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setLoading(true);
    setError(null);
    try {
      const result = await fetchExplainability(location, model);
      setData(result);
    } catch (err: unknown) {
      if ((err as Error)?.name !== "AbortError") {
        const msg = (err as Error)?.message ?? "Failed to load explainability data.";
        setError(
          msg.includes("409")
            ? "Run the Forecast tab first — the trained model is required."
            : msg,
        );
      }
    } finally {
      setLoading(false);
    }
  }, [location, model]);

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  // Prepare chart data (top 15, sorted ascending so highest bar is at top)
  const chartFeatures =
    data?.top_features
      .slice(0, 15)
      .sort((a, b) => b.combined_rank - a.combined_rank) ?? [];

  // ── Render states ──────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="card p-8 flex flex-col items-center justify-center gap-3 min-h-[420px]">
        <Loader2 className="h-8 w-8 text-signal animate-spin" />
        <p className="text-sm text-ink-secondary">Computing SHAP values…</p>
        <p className="text-2xs text-ink-tertiary">
          TreeExplainer on {model} — typically 10–30 seconds.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 flex flex-col items-center justify-center gap-3 min-h-[420px]">
        <AlertCircle className="h-8 w-8 text-negative" />
        <p className="text-sm font-medium text-ink-primary">Explainability failed</p>
        <p className="text-xs text-ink-secondary text-center max-w-sm">{error}</p>
        <button
          onClick={load}
          className="mt-2 flex items-center gap-2 rounded-lg bg-surface-hover px-4 py-2 text-xs font-medium text-ink-primary hover:bg-surface-line transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" /> Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const chartHeight = Math.max(320, chartFeatures.length * 28);

  return (
    <div className="space-y-4 animate-fade-up">
      {/* ── Header row ── */}
      <div className="card p-5">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-sm font-semibold text-ink-primary">
              Feature Importance — SHAP + Permutation
            </h2>
            <p className="text-2xs text-ink-tertiary mt-0.5">
              {location} · {data.model} · Top 15 features by combined rank
            </p>
          </div>
          {/* Model selector */}
          <div className="flex gap-1.5">
            {TREE_MODELS.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`
                  rounded-md px-2.5 py-1 text-2xs font-medium transition-colors
                  ${
                    model === m
                      ? "bg-signal/20 text-signal border border-signal/40"
                      : "text-ink-tertiary border border-surface-hover hover:text-ink-secondary"
                  }
                `}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* Exogenous impact callout */}
        <div className="flex items-center gap-3 rounded-lg border border-surface-hover bg-surface-base/60 px-3 py-2.5">
          <BrainCircuit className="h-4 w-4 text-signal flex-shrink-0" />
          <p className="text-xs text-ink-secondary">
            Exogenous variables (humidity, pressure, wind, precipitation) contribute{" "}
            <span className="text-data font-semibold text-signal">
              {(data.exogenous_shap_fraction * 100).toFixed(1)}%
            </span>{" "}
            of total SHAP impact — validating the multivariate design.
          </p>
        </div>
      </div>

      {/* ── SHAP bar chart ── */}
      <div className="card p-5">
        <p className="section-label mb-4">Mean |SHAP Value| per Feature (°C)</p>
        <ResponsiveContainer width="100%" height={chartHeight}>
          <BarChart
            data={chartFeatures}
            layout="vertical"
            margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1E293B"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fill: "#64748B", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={{ stroke: "#334155" }}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(3)}
            />
            <YAxis
              type="category"
              dataKey="feature"
              width={160}
              tick={{ fill: "#94A3B8", fontSize: 10, fontFamily: "var(--font-mono)" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={shortLabel}
            />
            <Tooltip content={<ShapTooltip />} cursor={{ fill: "#1E293B" }} />
            <Bar dataKey="mean_abs_shap" name="Mean |SHAP|" radius={[0, 3, 3, 0]}>
              {chartFeatures.map((entry, index) => (
                <Cell
                  key={entry.feature}
                  fill={barColor(index + 1, chartFeatures.length)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Unified ranking table ── */}
      <div className="card p-5">
        <p className="section-label mb-3">Unified Ranking — SHAP vs Permutation Agreement</p>
        <div className="overflow-x-auto rounded-card border border-surface-hover">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-surface-hover">
                {["Rank", "Feature", "SHAP Rank", "Perm. Rank", "Combined Score"].map((h) => (
                  <th
                    key={h}
                    className="px-3 py-2.5 text-left text-ink-tertiary font-medium whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.top_features.slice(0, 15).map((f, i) => (
                <tr
                  key={f.feature}
                  className={`border-b border-surface-hover/50 last:border-0 ${
                    i % 2 === 0 ? "" : "bg-surface-card/30"
                  }`}
                >
                  <td className="px-3 py-2 text-data font-semibold text-signal">
                    #{f.combined_rank}
                  </td>
                  <td className="px-3 py-2 font-mono text-2xs text-ink-secondary max-w-[220px] truncate">
                    {f.feature}
                  </td>
                  <td className="px-3 py-2 text-data text-ink-secondary">#{f.shap_rank}</td>
                  <td className="px-3 py-2 text-data text-ink-secondary">
                    #{f.permutation_rank}
                  </td>
                  <td className="px-3 py-2 text-data">
                    <div className="flex items-center gap-2">
                      <div
                        className="h-1.5 rounded-full bg-signal"
                        style={{ width: `${f.combined_score * 80}px` }}
                      />
                      <span className="text-ink-secondary">
                        {f.combined_score.toFixed(4)}
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
