// TypeScript types mirroring the Python Pydantic schemas in src/agents/schemas.py

export interface ExternalSeriesConfig {
  name: string;
  source: string;
  symbol: string;
  lag_days: number;
  feature_type: string;
  frequency: string;
  api_key_env?: string;
  extra?: Record<string, unknown>;
}

export interface ExperimentConfig {
  ticker: string;
  start_date: string;
  end_date: string | null;
  interval: string;
  cache_dir: string;
  target_kind: string;
  horizon: number;
  return_windows: number[];
  ma_windows: number[];
  rsi_window: number;
  atr_window: number;
  volume_ma_window: number;
  initial_train_days: number;
  step_days: number;
  min_train_days: number;
  lgbm_params: Record<string, unknown>;
  external_data_enabled: boolean;
  external_series: ExternalSeriesConfig[];
  external_cache_dir: string;
  external_cache_ttl_hours: number;
  experiment_id: string;
}

export interface ExecutionPlan {
  experiment_id: string;
  ticker: string;
  target_kind: string;
  horizon: number;
  steps: string[];
  feature_config: Record<string, unknown>;
  backtest_config: Record<string, unknown>;
  lgbm_config: Record<string, unknown>;
  warnings: string[];
  created_at: string;
  llm_reasoning?: string | null;
}

export interface DataSummary {
  ticker: string;
  n_rows: number;
  date_start: string;
  date_end: string;
  trading_days_per_year: number;
  close_min: number;
  close_max: number;
  close_mean: number;
  volume_mean: number;
  missing_close_pct: number;
  sufficient_for_backtest: boolean;
  source: string;
}

export interface FoldSummary {
  fold: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  n_train: number;
  n_test: number;
  mae: number;
  rmse: number;
  directional_accuracy: number;
  sharpe: number;
  ic: number;
}

export interface ModelingResult {
  experiment_id: string;
  model_name: string;
  target_kind: string;
  n_features: number;
  n_folds: number;
  n_oos_observations: number;
  folds: FoldSummary[];
  feature_names: string[];
  top_features: string[];
  // External data visibility (empty arrays/object when external_data_enabled=false)
  external_columns: string[];
  external_missing_ratios: Record<string, number>;
}

export interface AggregateMetrics {
  mae: number;
  rmse: number;
  directional_accuracy: number;
  sharpe: number;
  ic: number;
}

export type Verdict = "poor" | "marginal" | "acceptable" | "strong";

export interface EvaluationReport {
  experiment_id: string;
  aggregate_metrics: AggregateMetrics;
  directional_accuracy_mean: number;
  sharpe_mean: number;
  verdict: Verdict;
  flags: string[];
  recommendations: string[];
  llm_interpretation?: string | null;
  llm_recommendations: string[];
}

export interface ExperimentReport {
  experiment_id: string;
  generated_at: string;
  config: ExperimentConfig;
  plan: ExecutionPlan;
  data_summary: DataSummary;
  modeling_result: ModelingResult;
  evaluation: EvaluationReport;
}

// Lightweight summary used in the reports list and compare page
export interface ReportListItem {
  id: string;
  generated_at: string;
  ticker: string;
  verdict: Verdict;
  directional_accuracy: number;
  sharpe: number;
  mae: number;
  ic: number;
  n_features: number;
  n_folds: number;
  n_oos_observations: number;
  external_data_enabled: boolean;
  n_ext_features: number;         // actual merged ext_* column count
  data_start: string;
  data_end: string;
}
