"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BarChart3, ExternalLink, Loader2 } from "lucide-react";
import type { ReportListItem } from "@/lib/types";
import { formatDate, formatNum, formatPct, verdictStyle } from "@/lib/report-utils";
import { VerdictBadge } from "@/components/VerdictBadge";
import { ExternalDataBadge } from "@/components/ExternalDataBadge";
import { ReportNav } from "@/components/ReportNav";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// ── Recharts compare bar chart ────────────────────────────────────────────────
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

function CompareBarChart({
  data,
  dataKey,
  label,
  refLine,
  color,
  unit = "",
}: {
  data: { id: string; value: number }[];
  dataKey: string;
  label: string;
  refLine?: number;
  color: string;
  unit?: string;
}) {
  const chartData = data.map((d) => ({
    name: d.id.replace("exp_", "").slice(0, 13),
    [dataKey]: isFinite(d.value) ? parseFloat(d.value.toFixed(4)) : null,
    fullId: d.id,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-slate-600">{label}</CardTitle>
      </CardHeader>
      <CardContent className="pb-4">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 9, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) =>
                isFinite(v) ? `${v.toFixed(2)}${unit}` : "—"
              }
            />
            <Tooltip
              contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
              formatter={(v: number) => [
                isFinite(v) ? `${v.toFixed(4)}${unit}` : "—",
                label,
              ]}
              labelFormatter={(l) => `${l}`}
            />
            {refLine !== undefined && (
              <ReferenceLine y={refLine} stroke="#cbd5e1" strokeDasharray="4 3" />
            )}
            <Bar dataKey={dataKey} fill={color} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

// ── Main compare page ─────────────────────────────────────────────────────────

export default function ComparePage() {
  const [all, setAll] = useState<ReportListItem[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/reports")
      .then((r) => {
        if (!r.ok) throw new Error("Failed to load reports");
        return r.json();
      })
      .then((data: ReportListItem[]) => {
        setAll(data);
        // Pre-select all by default (up to 6)
        setSelected(new Set(data.slice(0, 6).map((r) => r.id)));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelected(new Set(all.map((r) => r.id)));
  }

  function clearAll() {
    setSelected(new Set());
  }

  const shown = all.filter((r) => selected.has(r.id));

  const daData = shown.map((r) => ({ id: r.id, value: r.directional_accuracy * 100 }));
  const sharpeData = shown.map((r) => ({ id: r.id, value: r.sharpe }));
  const icData = shown.map((r) => ({ id: r.id, value: r.ic }));
  const maeData = shown.map((r) => ({ id: r.id, value: r.mae * 100 }));

  return (
    <div className="min-h-screen bg-slate-50">
      <ReportNav activePage="compare" />

      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-slate-900">Compare Experiments</h1>
          <p className="mt-1 text-sm text-slate-500">
            Select reports below to compare metrics side-by-side.
          </p>
        </div>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading reports…
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {!loading && !error && all.length === 0 && (
          <div className="rounded-xl border-2 border-dashed border-slate-200 bg-white py-16 text-center">
            <BarChart3 className="mx-auto mb-4 h-10 w-10 text-slate-300" />
            <p className="text-sm font-medium text-slate-600">No reports to compare</p>
            <p className="mt-1 text-xs text-slate-400">
              Run <code className="rounded bg-slate-100 px-1">python run_pipeline.py</code> first.
            </p>
          </div>
        )}

        {!loading && all.length > 0 && (
          <>
            {/* ── Selection table ──────────────────────────────────── */}
            <Card className="mb-6">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">
                    Select experiments
                    <span className="ml-2 text-sm font-normal text-slate-400">
                      ({selected.size} selected)
                    </span>
                  </CardTitle>
                  <div className="flex gap-2 text-xs">
                    <button
                      onClick={selectAll}
                      className="text-slate-500 hover:text-slate-900 underline underline-offset-2"
                    >
                      Select all
                    </button>
                    <span className="text-slate-300">|</span>
                    <button
                      onClick={clearAll}
                      className="text-slate-500 hover:text-slate-900 underline underline-offset-2"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-slate-50 text-left">
                        <th className="px-4 py-2 w-10" />
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                          Experiment ID
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                          Dir. Acc.
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                          Sharpe
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                          IC
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                          Features
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                          Verdict
                        </th>
                        <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                          External
                        </th>
                        <th className="px-4 py-2 w-8" />
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {all.map((r) => (
                        <tr
                          key={r.id}
                          className={`transition-colors ${
                            selected.has(r.id) ? "bg-blue-50/40" : "hover:bg-slate-50"
                          }`}
                        >
                          <td className="px-4 py-2">
                            <input
                              type="checkbox"
                              checked={selected.has(r.id)}
                              onChange={() => toggleSelect(r.id)}
                              className="h-4 w-4 rounded border-slate-300 text-blue-600 accent-blue-600"
                            />
                          </td>
                          <td className="px-4 py-2 font-mono text-xs text-slate-700">
                            {r.id}
                          </td>
                          <td className="px-4 py-2 text-right font-mono font-semibold">
                            <span
                              className={
                                r.directional_accuracy >= 0.55
                                  ? "text-emerald-600"
                                  : r.directional_accuracy >= 0.5
                                  ? "text-yellow-600"
                                  : "text-red-600"
                              }
                            >
                              {formatPct(r.directional_accuracy)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right font-mono font-semibold">
                            <span
                              className={
                                r.sharpe >= 1
                                  ? "text-emerald-600"
                                  : r.sharpe >= 0
                                  ? "text-yellow-600"
                                  : "text-red-600"
                              }
                            >
                              {formatNum(r.sharpe, 3)}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-xs text-slate-600">
                            {formatNum(r.ic, 4)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-xs text-slate-600">
                            {r.n_features}
                          </td>
                          <td className="px-4 py-2">
                            <VerdictBadge verdict={r.verdict} size="sm" />
                          </td>
                          <td className="px-4 py-2">
                            <ExternalDataBadge enabled={r.external_data_enabled} size="sm" />
                          </td>
                          <td className="px-4 py-2">
                            <Link
                              href={`/reports/${r.id}`}
                              className="text-slate-300 hover:text-slate-600"
                              title="Open report"
                            >
                              <ExternalLink className="h-3.5 w-3.5" />
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* ── Charts ───────────────────────────────────────────── */}
            {shown.length > 0 ? (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <CompareBarChart
                  data={daData}
                  dataKey="da"
                  label="Directional Accuracy (%)"
                  refLine={50}
                  color="#3b82f6"
                  unit="%"
                />
                <CompareBarChart
                  data={sharpeData}
                  dataKey="sharpe"
                  label="Sharpe Ratio"
                  refLine={0}
                  color="#8b5cf6"
                />
                <CompareBarChart
                  data={icData}
                  dataKey="ic"
                  label="Information Coefficient"
                  refLine={0}
                  color="#10b981"
                />
                <CompareBarChart
                  data={maeData}
                  dataKey="mae"
                  label="MAE (× 100)"
                  color="#f59e0b"
                  unit=""
                />
              </div>
            ) : (
              <div className="rounded-xl border-2 border-dashed border-slate-200 bg-white py-12 text-center text-sm text-slate-400">
                Select at least one experiment to see charts.
              </div>
            )}

            {/* ── Summary table ─────────────────────────────────────── */}
            {shown.length > 0 && (
              <Card className="mt-6">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Metric Summary</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b bg-slate-50 text-left">
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Experiment
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            Dir. Acc.
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            Sharpe
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            RMSE
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            IC
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            Folds
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500 text-right">
                            OOS obs.
                          </th>
                          <th className="px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            Verdict
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {shown.map((r) => (
                          <tr key={r.id} className="hover:bg-slate-50">
                            <td className="px-4 py-2 font-mono text-xs">
                              <Link
                                href={`/reports/${r.id}`}
                                className="text-slate-700 hover:underline"
                              >
                                {r.id}
                              </Link>
                            </td>
                            <td className="px-4 py-2 text-right font-mono">
                              {formatPct(r.directional_accuracy)}
                            </td>
                            <td className="px-4 py-2 text-right font-mono">
                              {formatNum(r.sharpe, 3)}
                            </td>
                            <td className="px-4 py-2 text-right font-mono text-xs">
                              {/* mae used as proxy for now, rmse not in list item */}
                              —
                            </td>
                            <td className="px-4 py-2 text-right font-mono text-xs">
                              {formatNum(r.ic, 4)}
                            </td>
                            <td className="px-4 py-2 text-right text-xs text-slate-600">
                              {r.n_folds}
                            </td>
                            <td className="px-4 py-2 text-right text-xs text-slate-600">
                              {r.n_oos_observations.toLocaleString()}
                            </td>
                            <td className="px-4 py-2">
                              <VerdictBadge verdict={r.verdict} size="sm" />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </main>
    </div>
  );
}
