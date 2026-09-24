# Frontend Service — Adaptive Threat Intelligence

## Overview
This directory contains the Next.js (App Router) user interface for the Adaptive Threat Intelligence system. The interface is designed for security analysts and ML engineers to monitor network intrusion classifications, inspect SHAP feature attribution explanations, track concept and data drift metrics, and audit model performance transparently.

## Tech Stack
- **Framework**: Next.js 15+ (App Router)
- **Language**: TypeScript (Strict Mode)
- **Styling**: Tailwind CSS
- **Visualization**: Recharts
- **Iconography**: Lucide React

## Directory Structure
- `src/app/` — Next.js App Router pages and route handlers
- `src/components/` — Shared UI design components (tables, badges, alerts, layouts)
- `src/features/` — Domain-specific modules:
  - `threat-monitoring/` — Real-time telemetry feed and event logs
  - `explainability/` — SHAP summary plots, waterfall visualizations, and top-k contributing features
  - `drift-analysis/` — Kolmogorov-Smirnov / PSI drift tracking and threshold alarms
  - `model-registry/` — Active model versions, metrics, and rollback history
- `src/types/` — TypeScript data contract interfaces matching backend Pydantic schemas
- `src/lib/` — API client wrappers and utility functions

## Planned Setup (When entering Phase 5)
```bash
npm install
npm run dev
```

> **Note**: In accordance with Phase 0 constraints, frontend packages have not been installed and application code has not been built.
