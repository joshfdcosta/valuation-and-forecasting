/**
 * Types for the static backtest snapshot produced by scripts/build_snapshot.py.
 *
 * The site reads this file and nothing else — no API, no server. Every number
 * in it came from a walk-forward backtest where the model only ever saw data
 * from strictly before the anchor it was predicting.
 */

export interface Candle {
  time: string
  open: number
  high: number
  low: number
  close: number
}

export interface PredictionRow {
  anchor: string
  time: string
  step: number
  model_version: string
  anchor_close: number
  pred_close: number
  pred_std: number
  baseline_close: number
  actual_close: number
  abs_error: number
  baseline_abs_error: number
  direction_correct: number
}

export interface Summary {
  n: number
  mae: number
  rmse: number
  mape_pct: number
  baseline_mae: number
  skill_vs_baseline_pct: number
  direction_accuracy_pct: number
}

export interface TrainingRun {
  version: string
  trained_at: string
  trigger: 'initial' | 'drift'
  n_samples: number
  val_loss: number
  reference_mae: number
  baseline_mae: number
  retired_at?: string
  retired_drift_pct?: number
}

export interface Snapshot {
  generated_at: string
  ticker: string
  config: {
    lookback: number
    horizon: number
    indicators: string[]
    train_span: number
    drift_threshold_pct: number
    interval: string
  }
  company: { name: string | null; currency: string; current_price: number | null }
  fundamentals: {
    revenue: number | null
    ebit_margin: number | null
    revenue_growth: number | null
    shares_outstanding: number | null
    net_debt: number | null
    capex_pct: number
    depreciation_pct: number
    nwc_change_pct: number
  }
  defaults: {
    tax_rate: number
    wacc: number
    terminal_growth: number
    forecast_years: number
  }
  candles: Candle[]
  backtest: {
    metrics: Summary
    by_horizon: (Summary & { step: number })[]
    error_trend: { period: string; mae: number; baseline_mae: number; n: number }[]
    training_runs: TrainingRun[]
    predictions: PredictionRow[]
    span: { from: string; to: string }
  }
}

export async function loadSnapshot(): Promise<Snapshot> {
  const res = await fetch(`${import.meta.env.BASE_URL}snapshot.json`)
  if (!res.ok) throw new Error(`snapshot.json failed to load (${res.status})`)
  return res.json()
}
