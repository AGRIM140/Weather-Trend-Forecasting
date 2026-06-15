// frontend/tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      // ── Design token palette ─────────────────────────────────────────
      // Atmospheric / radar-room aesthetic:
      //   Slate depths for structure, electric cyan for signal,
      //   amber for secondary data, JetBrains Mono for all numbers.
      colors: {
        // Surface hierarchy
        surface: {
          base:  "#020617", // slate-950 — page bg
          muted: "#0F172A", // slate-900 — sidebar bg
          card:  "#1E293B", // slate-800 — card bg
          hover: "#334155", // slate-700 — hover / border
          line:  "#475569", // slate-600 — dividers
        },
        // Text hierarchy
        ink: {
          primary: "#F1F5F9", // slate-100
          secondary: "#94A3B8", // slate-400
          tertiary:  "#64748B", // slate-500
        },
        // Accent: electric cyan — the signal color
        signal: {
          DEFAULT: "#22D3EE", // cyan-400
          dim:     "#0891B2", // cyan-600
          glow:    "#67E8F9", // cyan-300 (for glows)
          muted:   "#164E63", // cyan-900 (for tinted backgrounds)
        },
        // Secondary accent: amber — weather warnings, secondary data
        amber: {
          DEFAULT: "#FBBF24",
          dim:     "#B45309",
          muted:   "#451A03",
        },
        // Semantic
        positive: "#34D399", // emerald-400
        negative: "#FB7185", // rose-400
        neutral:  "#94A3B8", // slate-400
      },
      fontFamily: {
        // Display + body: Inter (clean, legible at small sizes)
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        // Data values, numbers, axis labels: JetBrains Mono
        mono: ["var(--font-mono)", "Menlo", "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      borderRadius: {
        card: "0.625rem",
      },
      boxShadow: {
        signal: "0 0 0 1px #22D3EE33, 0 0 12px #22D3EE22",
        card:   "0 1px 3px 0 rgba(0,0,0,0.4), 0 1px 2px -1px rgba(0,0,0,0.4)",
      },
      keyframes: {
        "fade-up": {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-ring": {
          "0%":   { boxShadow: "0 0 0 0 rgba(34,211,238,0.35)" },
          "70%":  { boxShadow: "0 0 0 8px rgba(34,211,238,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(34,211,238,0)" },
        },
      },
      animation: {
        "fade-up":    "fade-up 0.4s ease-out both",
        "pulse-ring": "pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};

export default config;
