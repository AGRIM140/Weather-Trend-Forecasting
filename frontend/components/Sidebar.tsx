"use client";
// frontend/components/Sidebar.tsx

import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronDown,
  Globe,
  Wind,
} from "lucide-react";
import type { NavTab } from "@/types";

const PM_ACCELERATOR_MISSION =
  "[Note to Agrim: Paste the exact PM Accelerator mission wording here before publishing]";

const NAV_ITEMS: {
  id: NavTab;
  label: string;
  icon: React.ElementType;
  description: string;
}[] = [
  {
    id: "overview",
    label: "Overview",
    icon: Activity,
    description: "Dataset KPIs & summary statistics",
  },
  {
    id: "forecast",
    label: "Forecast",
    icon: BarChart3,
    description: "30-day ensemble temperature projection",
  },
  {
    id: "explainability",
    label: "Explainability",
    icon: BrainCircuit,
    description: "SHAP feature importance analysis",
  },
];

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  location: string;
  locations: string[];
  onLocationChange: (loc: string) => void;
  isLoading: boolean;
}

export default function Sidebar({
  activeTab,
  onTabChange,
  location,
  locations,
  onLocationChange,
  isLoading,
}: SidebarProps) {
  return (
    <aside className="fixed inset-y-0 left-0 z-40 flex w-64 flex-col bg-surface-muted border-r border-surface-hover">
      {/* ── Logo / Wordmark ── */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-surface-hover">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-signal-muted border border-signal/30">
          <Wind className="h-4 w-4 text-signal" />
        </div>
        <div>
          <p className="text-sm font-semibold text-ink-primary leading-tight">
            Weather Engine
          </p>
          <p className="text-2xs text-ink-tertiary">Global Forecasting</p>
        </div>
      </div>

      {/* ── PM Accelerator Mission ── */}
      <div className="mx-4 mt-4 rounded-card border border-signal/20 bg-signal-muted/40 p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <div className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse-ring" />
          <span className="section-label text-signal">PM Accelerator</span>
        </div>
        <p className="text-2xs text-ink-secondary leading-relaxed">
          {PM_ACCELERATOR_MISSION}
        </p>
      </div>

      {/* ── Location Selector ── */}
      <div className="px-4 mt-5">
        <label className="section-label block mb-1.5">Location</label>
        <div className="relative">
          <Globe className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-ink-tertiary" />
          <select
            value={location}
            onChange={(e) => onLocationChange(e.target.value)}
            disabled={isLoading || locations.length === 0}
            className="
              w-full appearance-none rounded-lg
              border border-surface-hover bg-surface-card
              py-2 pl-8 pr-8
              text-xs font-medium text-ink-primary
              focus:outline-none focus:ring-1 focus:ring-signal/50
              disabled:opacity-40 disabled:cursor-not-allowed
              transition-colors
            "
          >
            {locations.length === 0 ? (
              <option>Loading locations…</option>
            ) : (
              locations.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))
            )}
          </select>
          <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-ink-tertiary" />
        </div>
      </div>

      {/* ── Navigation ── */}
      <nav className="flex-1 px-3 mt-6 space-y-0.5">
        <p className="section-label px-2 mb-2">Navigation</p>
        {NAV_ITEMS.map(({ id, label, icon: Icon, description }) => {
          const active = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              className={`
                group w-full flex items-center gap-3 rounded-lg px-3 py-2.5
                text-left transition-all duration-150
                ${
                  active
                    ? "bg-signal-muted border border-signal/30 text-signal"
                    : "text-ink-secondary hover:bg-surface-hover hover:text-ink-primary border border-transparent"
                }
              `}
            >
              <Icon
                className={`h-4 w-4 flex-shrink-0 transition-colors ${
                  active ? "text-signal" : "text-ink-tertiary group-hover:text-ink-secondary"
                }`}
              />
              <div className="min-w-0">
                <p className="text-xs font-semibold leading-tight">{label}</p>
                <p
                  className={`text-2xs leading-tight mt-0.5 truncate ${
                    active ? "text-signal/70" : "text-ink-tertiary"
                  }`}
                >
                  {description}
                </p>
              </div>
              {active && (
                <div className="ml-auto h-1.5 w-1.5 rounded-full bg-signal flex-shrink-0" />
              )}
            </button>
          );
        })}
      </nav>

      {/* ── Footer ── */}
      <div className="px-5 py-4 border-t border-surface-hover">
        <p className="text-2xs text-ink-tertiary">
          FastAPI · Next.js · XGBoost · SHAP
        </p>
        <p className="text-2xs text-ink-tertiary mt-0.5">
          Global Weather Repository
        </p>
      </div>
    </aside>
  );
}
