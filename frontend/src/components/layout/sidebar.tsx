"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Compass,
  FlaskConical,
  Network,
  Shield,
  Sparkles,
} from "lucide-react";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    name: "Overview",
    href: "/",
    icon: Compass,
    description: "System health, readiness & telemetry snapshot",
  },
  {
    name: "Traffic Analysis",
    href: "/traffic",
    icon: Network,
    description: "Single-flow and batch CSV intrusion classification",
  },
  {
    name: "Explainability Lab",
    href: "/explainability",
    icon: Sparkles,
    description: "SHAP feature attributions and margin diagnostics",
  },
  {
    name: "Drift Monitor",
    href: "/drift",
    icon: Activity,
    description: "PSI/KS distribution divergence & adaptive alerts",
  },
  {
    name: "Model Laboratory",
    href: "/models",
    icon: FlaskConical,
    description: "Candidate 8 vs. Candidate 10 side-by-side audit",
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-slate-900 text-slate-200 flex flex-col shrink-0 border-r border-slate-800">
      {/* Brand Header */}
      <div className="p-5 border-b border-slate-800 flex items-center space-x-3">
        <div className="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-md">
          <Shield className="h-5 w-5" />
        </div>
        <div>
          <h1 className="font-semibold text-sm tracking-tight text-white">DataNova NIDS</h1>
          <p className="text-[11px] text-slate-400">Adaptive Threat Intelligence</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
        <div className="px-3 py-2 text-[10px] font-semibold tracking-wider text-slate-400 uppercase">
          Analyst Workspace
        </div>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-start space-x-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors ${
                isActive
                  ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 font-semibold"
                  : "text-slate-300 hover:bg-slate-800/60 hover:text-white"
              }`}
            >
              <Icon
                className={`h-4 w-4 mt-0.5 shrink-0 ${
                  isActive ? "text-indigo-400" : "text-slate-400"
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="truncate">{item.name}</div>
                <div className="text-[10px] text-slate-400 font-normal truncate">
                  {item.description}
                </div>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-slate-800 bg-slate-950/40 text-[11px] text-slate-400">
        <div className="flex items-center justify-between mb-1">
          <span className="font-medium text-slate-300">Phase 8 Prototype</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-300">
            v1.0.0
          </span>
        </div>
        <p className="text-[10px] text-slate-400 leading-tight">
          Tabular flow telemetry analysis. Sealed research models frozen.
        </p>
      </div>
    </aside>
  );
}
