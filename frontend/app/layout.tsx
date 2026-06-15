// frontend/app/layout.tsx
import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Weather Forecast Engine | PM Accelerator",
  description:
    "Production-grade multivariate weather forecasting dashboard — XGBoost, LightGBM, SHAP explainability across 200+ global locations.",
  keywords: ["weather", "forecasting", "machine learning", "XGBoost", "SHAP", "PM Accelerator"],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#020617",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        {/* Preconnect to Google Fonts for Inter + JetBrains Mono */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="bg-surface-base antialiased">{children}</body>
    </html>
  );
}
