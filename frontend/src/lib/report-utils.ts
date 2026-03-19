import type {
  ExperimentReport,
  FoldSummary,
  ReportListItem,
  Verdict,
} from "./types";

export function toListItem(report: ExperimentReport): ReportListItem {
  const m = report.evaluation.aggregate_metrics;
  return {
    id: report.experiment_id,
    generated_at: report.generated_at,
    ticker: report.config.ticker,
    verdict: report.evaluation.verdict,
    directional_accuracy: m.directional_accuracy,
    sharpe: m.sharpe,
    mae: m.mae,
    ic: m.ic,
    n_features: report.modeling_result.n_features,
    n_folds: report.modeling_result.n_folds,
    n_oos_observations: report.modeling_result.n_oos_observations,
    external_data_enabled: report.config.external_data_enabled,
    data_start: report.data_summary.date_start,
    data_end: report.data_summary.date_end,
  };
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}

export function formatPct(v: number, decimals = 2): string {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function formatNum(v: number, decimals = 4): string {
  if (!isFinite(v)) return "—";
  return v.toFixed(decimals);
}

export type VerdictStyle = {
  label: string;
  bg: string;
  text: string;
  ring: string;
};

const VERDICT_STYLES: Record<Verdict, VerdictStyle> = {
  poor: {
    label: "POOR",
    bg: "bg-red-100",
    text: "text-red-700",
    ring: "ring-red-200",
  },
  marginal: {
    label: "MARGINAL",
    bg: "bg-yellow-100",
    text: "text-yellow-700",
    ring: "ring-yellow-200",
  },
  acceptable: {
    label: "ACCEPTABLE",
    bg: "bg-green-100",
    text: "text-green-700",
    ring: "ring-green-200",
  },
  strong: {
    label: "STRONG",
    bg: "bg-emerald-100",
    text: "text-emerald-700",
    ring: "ring-emerald-200",
  },
};

export function verdictStyle(verdict: string): VerdictStyle {
  return (
    VERDICT_STYLES[verdict as Verdict] ?? {
      label: verdict.toUpperCase(),
      bg: "bg-slate-100",
      text: "text-slate-700",
      ring: "ring-slate-200",
    }
  );
}

// Returns colour class for directional accuracy value
export function daColor(da: number): string {
  if (da >= 0.55) return "text-emerald-600";
  if (da >= 0.50) return "text-yellow-600";
  return "text-red-600";
}

// Returns colour class for Sharpe ratio value
export function sharpeColor(sharpe: number): string {
  if (!isFinite(sharpe)) return "text-slate-400";
  if (sharpe >= 1.0) return "text-emerald-600";
  if (sharpe >= 0.0) return "text-yellow-600";
  return "text-red-600";
}

// Returns colour class for IC value
export function icColor(ic: number): string {
  if (!isFinite(ic)) return "text-slate-400";
  if (ic >= 0.1) return "text-emerald-600";
  if (ic >= 0.0) return "text-yellow-600";
  return "text-red-600";
}

export interface ChartPoint {
  fold: number;
  label: string;  // "F0", "F1", ...
  da: number;     // directional accuracy as %
  sharpe: number;
  rmse: number;   // RMSE × 100 for readability
  ic: number;
}

export function toChartData(folds: FoldSummary[]): ChartPoint[] {
  return folds.map((f) => ({
    fold: f.fold,
    label: `F${f.fold}`,
    da: parseFloat((f.directional_accuracy * 100).toFixed(2)),
    sharpe: parseFloat(f.sharpe.toFixed(4)),
    rmse: parseFloat((f.rmse * 100).toFixed(4)),
    ic: parseFloat(f.ic.toFixed(4)),
  }));
}

export function targetKindLabel(kind: string): string {
  const MAP: Record<string, string> = {
    next_day_log_return: "Next-day log return",
    next_day_direction: "Next-day direction",
    next_5d_log_return: "5-day log return",
  };
  return MAP[kind] ?? kind;
}
