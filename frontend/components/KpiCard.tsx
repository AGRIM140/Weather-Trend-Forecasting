"use client";
// frontend/components/KpiCard.tsx

import { TrendingDown, TrendingUp } from "lucide-react";

interface KpiCardProps {
  label: string;
  value: string | number;
  sub?: string;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  accent?: boolean;    // electric cyan highlight border
  loading?: boolean;
}

export default function KpiCard({
  label,
  value,
  sub,
  trend,
  trendValue,
  accent = false,
  loading = false,
}: KpiCardProps) {
  if (loading) {
    return (
      <div className="card p-4 space-y-2 animate-pulse">
        <div className="h-3 w-24 rounded bg-surface-hover" />
        <div className="h-7 w-32 rounded bg-surface-hover" />
        <div className="h-3 w-20 rounded bg-surface-hover" />
      </div>
    );
  }

  return (
    <div
      className={`
        relative overflow-hidden p-4 rounded-card border transition-all duration-200
        hover:border-signal/40 group
        ${accent
          ? "bg-surface-card border-signal/30 shadow-signal"
          : "bg-surface-card border-surface-hover"
        }
      `}
    >
      {/* Subtle top-edge glow for accent cards */}
      {accent && (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-signal to-transparent opacity-60"
        />
      )}

      {/* Label */}
      <p className="section-label mb-2">{label}</p>

      {/* Value */}
      <p
        className={`text-data text-2xl font-semibold leading-none tracking-tight ${
          accent ? "text-signal" : "text-ink-primary"
        }`}
      >
        {typeof value === "number" ? value.toLocaleString() : value}
      </p>

      {/* Sub-label and optional trend */}
      <div className="mt-2 flex items-center gap-2">
        {sub && <p className="text-2xs text-ink-tertiary">{sub}</p>}
        {trend && trendValue && (
          <span
            className={`
              flex items-center gap-0.5 text-2xs font-medium text-data
              ${trend === "up" ? "text-positive" : trend === "down" ? "text-negative" : "text-neutral"}
            `}
          >
            {trend === "up" ? (
              <TrendingUp className="h-3 w-3" />
            ) : trend === "down" ? (
              <TrendingDown className="h-3 w-3" />
            ) : null}
            {trendValue}
          </span>
        )}
      </div>

      {/* Hover shimmer */}
      <div
        aria-hidden
        className="
          pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100
          bg-gradient-to-br from-signal/5 via-transparent to-transparent
          transition-opacity duration-300
        "
      />
    </div>
  );
}
