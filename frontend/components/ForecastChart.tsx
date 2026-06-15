"use client";
// frontend/components/ForecastChart.tsx
//
// Signature visual element of the dashboard:
//   - Historical temperature rendered as a solid cyan area
//   - Forecast rendered as a dashed amber line with a semi-transparent fill
//   - Vertical "Forecast begins" reference line
//   - Per-model metrics comparison table beneath the chart
//   - CSV export button

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AlertCircle, Download, Loader2, RefreshCw } from "lucide-react";
import { fetchForecast } from "@/lib/api";
import type { ForecastResponse, ModelMetrics, TimeSeriesPoint } from "@/types";

// ── Chart data shape ────────────────────────────────────────────────────────
interface ChartPoint {
  date: string;
  historical: number | null;
  forecast: number | null;
  forecastFill: number | null;
}

function buildChartData(data: ForecastResponse): ChartPoint[] {
  const histPoints = data.historical.map((p) => ({
    date: p.date,
    historical: p.value,
    forecast: null,
    forecastFill: null,
  }));

  const forecastPoints = data.forecast.map((p) => ({
    date: p.date,
    historical: null,
    forecast: p.value,
    forecastFill: p.value,
  }));

  // Stitch the last historical point as the first forecast point for a
  // seamless line connection at the boundary.
  if (histPoints.length > 0 && forecastPoints.length > 0) {
    const last = histPoints[histPoints.length - 1];
    forecastPoints[0] = {
      ...forecastPoints[0],
      historical: last.historical,
    };
  }

  return [...histPoints, ...forecastPoints];
}

// ── Custom tooltip ──────────────────────────────────────────────────────────
function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const item = payload.find((p) => p.value !== null);
  if (!item) return null;

  return (
    <div className="card border-surface-line px-3 py-2 text-xs shadow-xl">
      <p className="text-ink-tertiary mb-1 font-medium">{label}</p>
      <p
        className="text-data font-semibold"
        style={{ color: item.color }}
      >
        {item.value.toFixed(2)} °C
      </p>
      <p className="text-2xs text-ink-tertiary mt-0.5 capitalize">
        {item.name === "historical" ? "Observed" : "Ensemble Forecast"}
      </p>
    </div>
  );
}

// ── Model metrics table ─────────────────────────────────────────────────────
const METRIC_KEYS: (keyof ModelMetrics)[] = ["MAE", "RMSE", "MAPE", "R2"];

function MetricsTable({ metrics }: { metrics: Record<string, ModelMetrics> }) {
  const models = Object.keys(metrics);
  if (models.length === 0) return null;

  // Find best (lowest MAE) model
  const bestModel = models.reduce((best, m) =>
    metrics[m].MAE < metrics[best].MAE ? m : best
  );

  return (
    <div className="mt-6">
      <p className="section-label mb-3">Model Comparison — Test Set</p>
      <div className="overflow-x-auto rounded-card border border-surface-hover">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-surface-hover">
              <th className="px-3 py-2.5 text-left text-ink-tertiary font-medium">Model</th>
              {METRIC_KEYS.map((k) => (
                <th key={k} className="px-3 py-2.5 text-right text-ink-tertiary font-medium">
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model, i) => {
              const m = metrics[model];
              const isBest = model === bestModel;
              return (
                <tr
                  key={model}
                  className={`
                    border-b border-surface-hover/50 last:border-0 transition-colors
                    ${isBest ? "bg-signal-muted/30" : i % 2 === 0 ? "bg-transparent" : "bg-surface-card/30"}
                  `}
                >
                  <td className="px-3 py-2.5 font-medium">
                    <span className={isBest ? "text-signal" : "text-ink-secondary"}>
                      {model}
                    </span>
                    {isBest && (
                      <span className="ml-2 rounded-full bg-signal/20 px-1.5 py-0.5 text-2xs text-signal font-semibold">
                        Best
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-right text-data">
                    <span className={isBest ? "text-signal font-semibold" : "text-ink-secondary"}>
                      {m.MAE.toFixed(3)}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 text-right text-data text-ink-secondary">
                    {m.RMSE.toFixed(3)}
                  </td>
                  <td className="px-3 py-2.5 text-right text-data text-ink-secondary">
                    {m.MAPE.toFixed(2)}%
                  </td>
                  <td className="px-3 py-2.5 text-right text-data">
                    <span className={m.R2 >= 0.9 ? "text-positive" : "text-amber"}>
                      {m.R2.toFixed(4)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
interface ForecastChartProps {
  location: string;
}

export default function ForecastChart({ location }: ForecastChartProps) {
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    if (!location) return;
    // Cancel any in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    try {
      const result = await fetchForecast(location);
      setData(result);
    } catch (err: unknown) {
      if ((err as Error)?.name !== "AbortError") {
        setError((err as Error)?.message ?? "Failed to load forecast.");
      }
    } finally {
      setLoading(false);
    }
  }, [location]);

  useEffect(() => {
    load();
    return () => abortRef.current?.abort();
  }, [load]);

  // CSV export
  const handleExport = () => {
    if (!data) return;
    const rows = [
      ["date", "type", "temperature_celsius"],
      ...data.historical.map((p) => [p.date, "historical", p.value]),
      ...data.forecast.map((p) => [p.date, "forecast", p.value]),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `forecast_${location.replace(/\s+/g, "_")}_30day.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ── Render states ────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="card p-8 flex flex-col items-center justify-center gap-3 min-h-[420px]">
        <Loader2 className="h-8 w-8 text-signal animate-spin" />
        <p className="text-sm text-ink-secondary">Training ensemble models…</p>
        <p className="text-2xs text-ink-tertiary">
          First run for a new location takes 15–60 seconds. Results are cached.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 flex flex-col items-center justify-center gap-3 min-h-[420px]">
        <AlertCircle className="h-8 w-8 text-negative" />
        <p className="text-sm font-medium text-ink-primary">Forecast failed</p>
        <p className="text-xs text-ink-secondary text-center max-w-xs">{error}</p>
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

  const chartData = buildChartData(data);
  const splitDate = data.historical[data.historical.length - 1]?.date;

  // Tick formatter: show every 7th point's date label
  const formatTick = (tick: string) => {
    try {
      return new Date(tick).toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
      });
    } catch {
      return tick;
    }
  };

  return (
    <div className="card p-5 animate-fade-up">
      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h2 className="text-sm font-semibold text-ink-primary">
            30-Day Temperature Forecast
          </h2>
          <p className="text-2xs text-ink-tertiary mt-0.5">
            {location} · Inverse-RMSE weighted ensemble ·{" "}
            <span className="text-data">
              Computed in {data.computation_time_seconds}s
            </span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={load}
            className="rounded-lg border border-surface-hover p-1.5 text-ink-tertiary hover:text-ink-primary hover:border-signal/40 transition-colors"
            title="Refresh forecast"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 rounded-lg border border-surface-hover px-3 py-1.5 text-2xs font-medium text-ink-secondary hover:text-ink-primary hover:border-signal/40 transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </button>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart
          data={chartData}
          margin={{ top: 4, right: 8, left: -8, bottom: 0 }}
        >
          <defs>
            <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#22D3EE" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#22D3EE" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="fcGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#FBBF24" stopOpacity={0.18} />
              <stop offset="95%" stopColor="#FBBF24" stopOpacity={0.01} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />

          <XAxis
            dataKey="date"
            tickFormatter={formatTick}
            interval={Math.floor(chartData.length / 8)}
            tick={{ fill: "#64748B", fontSize: 10, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#64748B", fontSize: 10, fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => `${v.toFixed(0)}°`}
            width={36}
          />

          <Tooltip content={<ChartTooltip />} />

          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            iconSize={7}
            wrapperStyle={{ fontSize: "11px", color: "#94A3B8", paddingBottom: "8px" }}
          />

          {/* Vertical marker at forecast start */}
          {splitDate && (
            <ReferenceLine
              x={splitDate}
              stroke="#334155"
              strokeDasharray="4 4"
              label={{
                value: "Forecast ›",
                position: "insideTopRight",
                fill: "#FBBF24",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
              }}
            />
          )}

          {/* Historical area */}
          <Area
            type="monotone"
            dataKey="historical"
            name="Historical"
            stroke="#22D3EE"
            strokeWidth={2}
            fill="url(#histGrad)"
            dot={false}
            activeDot={{ r: 4, fill: "#22D3EE", stroke: "#020617", strokeWidth: 2 }}
            connectNulls={false}
          />

          {/* Forecast fill */}
          <Area
            type="monotone"
            dataKey="forecastFill"
            name="forecast-fill"
            stroke="none"
            fill="url(#fcGrad)"
            dot={false}
            legendType="none"
            connectNulls={false}
          />

          {/* Forecast line */}
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke="#FBBF24"
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            activeDot={{ r: 4, fill: "#FBBF24", stroke: "#020617", strokeWidth: 2 }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Model metrics */}
      <MetricsTable metrics={data.metrics} />
    </div>
  );
}
