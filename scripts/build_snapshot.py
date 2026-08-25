"""Walk-forward backtest exported as a static JSON snapshot.

The site is a public portfolio piece, so it cannot depend on a running Python
server. This script does all the expensive, honest work once — offline — and
writes a snapshot the frontend reads as a static asset.

Crucially this is a real walk-forward backtest, not a replay of the training
fit. The model is trained on data strictly before each prediction anchor, and
refit whenever measured error drifts past the configured threshold. Every
number in the snapshot is an out-of-sample result.

Run: python -m scripts.build_snapshot AAPL
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import features as F
from src.data.fetch import get_candles, get_fundamentals
from src.models import lstm
from src.models.baseline import PersistenceBaseline
from src.pipeline.train import load_config

OUT = Path("web/public/snapshot.json")

# Walk-forward geometry. TRAIN_SPAN candles of history feed each fit; the model
# then forecasts forward until drift triggers a refit or the data runs out.
TRAIN_SPAN = 500
MIN_TRAIN_WINDOWS = 120
DRIFT_CHECK_EVERY = 10


def _fit(x: np.ndarray, y: np.ndarray, mcfg: dict, horizon: int):
    """Fit one model on a slice, holding out the tail for validation."""
    split = int(len(x) * 0.85)
    model = lstm.CandleLSTM(
        n_features=x.shape[-1],
        horizon=horizon,
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
    )
    history = lstm.train(
        model,
        x[:split],
        y[:split],
        x[split:],
        y[split:],
        epochs=mcfg["epochs"],
        batch_size=mcfg["batch_size"],
        lr=mcfg["lr"],
    )
    return model, history


def build(ticker: str) -> dict:
    cfg = load_config()
    fcfg, mcfg, rcfg = cfg["features"], cfg["model"], cfg["retrain"]
    lookback, horizon = fcfg["lookback"], fcfg["horizon"]
    indicators = fcfg["indicators"]

    print(f"[1/5] fetching {ticker}")
    candles = get_candles(
        ticker,
        interval=cfg["data"]["interval"],
        period=cfg["data"]["history_period"],
        cache_dir=cfg["data"]["cache_dir"],
    )
    fundamentals = get_fundamentals(ticker)

    print("[2/5] engineering features")
    feat = F.build_features(candles)
    x_all, y_abs_all, anchors = F.make_windows(feat, indicators, lookback, horizon)
    anchor_close_all = F.anchor_closes(feat, anchors)
    y_rel_all = F.to_relative_targets(y_abs_all, anchor_close_all)

    n = len(x_all)
    if n < TRAIN_SPAN + MIN_TRAIN_WINDOWS:
        raise SystemExit(f"only {n} windows available; need {TRAIN_SPAN + MIN_TRAIN_WINDOWS}")

    print(f"[3/5] walk-forward over {n - TRAIN_SPAN} anchors")
    rows: list[dict] = []
    runs: list[dict] = []
    model = None
    scaler = None
    version = 0
    reference_mae = None
    since_check = 0

    cursor = TRAIN_SPAN
    while cursor < n:
        # (Re)fit on the TRAIN_SPAN windows immediately before the cursor.
        # Nothing at or after the cursor has been seen by the model.
        if model is None:
            train_slice = slice(cursor - TRAIN_SPAN, cursor)
            xt, yt = x_all[train_slice], y_rel_all[train_slice]
            scaler = F.fit_scaler(xt)
            model, history = _fit(F.apply_scaler(scaler, xt), yt, mcfg, horizon)
            version += 1

            # Score the fresh fit on the last chunk of its own training span so
            # the drift detector has a reference that is comparable in units.
            ref_slice = slice(cursor - 60, cursor)
            ref_pred, _ = lstm.predict(
                model, F.apply_scaler(scaler, x_all[ref_slice]), mc_samples=10
            )
            ref_price = F.from_relative_targets(ref_pred, anchor_close_all[ref_slice])[:, :, 3]
            ref_actual = y_abs_all[ref_slice][:, :, 3]
            ref_base = anchor_close_all[ref_slice][:, None].repeat(horizon, axis=1)
            reference_mae = float(np.mean(np.abs(ref_price - ref_actual)))

            runs.append(
                {
                    "version": f"v{version}",
                    "trained_at": str(pd.Timestamp(anchors[cursor - 1]).date()),
                    "trigger": "initial" if version == 1 else "drift",
                    "n_samples": TRAIN_SPAN,
                    "val_loss": round(history["best_val_loss"], 6),
                    "reference_mae": round(reference_mae, 4),
                    "baseline_mae": round(float(np.mean(np.abs(ref_base - ref_actual))), 4),
                }
            )
            print(
                f"  fit v{version} @ {runs[-1]['trained_at']} "
                f"ref_mae={reference_mae:.3f} base={runs[-1]['baseline_mae']:.3f}"
            )
            since_check = 0

        # Predict this single anchor, out of sample.
        xs = F.apply_scaler(scaler, x_all[cursor : cursor + 1])
        pred_rel, std_rel = lstm.predict(model, xs, mc_samples=mcfg["mc_dropout_samples"])
        anchor_c = anchor_close_all[cursor : cursor + 1]
        pred_px = F.from_relative_targets(pred_rel, anchor_c)[0]
        std_px = (std_rel[:, :, 3] * anchor_c[:, None])[0]
        base_px = F.from_relative_targets(
            PersistenceBaseline().predict(xs, horizon), anchor_c
        )[0]
        actual_px = y_abs_all[cursor]

        anchor_ts = pd.Timestamp(anchors[cursor])
        for step in range(horizon):
            pred_close = float(pred_px[step][3])
            actual_close = float(actual_px[step][3])
            base_close = float(base_px[step][3])
            target_idx = feat.index.get_loc(anchor_ts) + step + 1
            rows.append(
                {
                    "anchor": str(anchor_ts.date()),
                    "time": str(feat.index[target_idx].date()),
                    "step": step + 1,
                    "model_version": f"v{version}",
                    "anchor_close": round(float(anchor_c[0]), 2),
                    "pred_close": round(pred_close, 2),
                    "pred_std": round(float(std_px[step]), 2),
                    "baseline_close": round(base_close, 2),
                    "actual_close": round(actual_close, 2),
                    "abs_error": round(abs(pred_close - actual_close), 4),
                    "baseline_abs_error": round(abs(base_close - actual_close), 4),
                    "direction_correct": int(
                        np.sign(pred_close - anchor_c[0]) == np.sign(actual_close - anchor_c[0])
                    ),
                }
            )

        cursor += 1
        since_check += 1

        # Drift check on a cadence, using only predictions already resolved.
        if since_check >= DRIFT_CHECK_EVERY and reference_mae:
            recent = rows[-rcfg["drift_lookback_candles"] * horizon :]
            recent_mae = float(np.mean([r["abs_error"] for r in recent]))
            drift_pct = (recent_mae - reference_mae) / reference_mae * 100
            if drift_pct > rcfg["drift_threshold_pct"]:
                print(f"  drift {drift_pct:+.1f}% @ {anchor_ts.date()} -> refit")
                runs[-1]["retired_at"] = str(anchor_ts.date())
                runs[-1]["retired_drift_pct"] = round(drift_pct, 2)
                model = None  # forces a refit on the next loop
            since_check = 0

    print(f"[4/5] scoring {len(rows)} predictions")
    df = pd.DataFrame(rows)

    def summarise(frame: pd.DataFrame) -> dict:
        mae = float(frame["abs_error"].mean())
        base = float(frame["baseline_abs_error"].mean())
        return {
            "n": int(len(frame)),
            "mae": round(mae, 3),
            "rmse": round(float(np.sqrt((frame["abs_error"] ** 2).mean())), 3),
            "mape_pct": round(float((frame["abs_error"] / frame["actual_close"]).mean() * 100), 3),
            "baseline_mae": round(base, 3),
            "skill_vs_baseline_pct": round((base - mae) / base * 100, 2) if base else 0.0,
            "direction_accuracy_pct": round(float(frame["direction_correct"].mean() * 100), 2),
        }

    overall = summarise(df)
    by_horizon = [
        {"step": int(step), **summarise(group)} for step, group in df.groupby("step")
    ]

    trend_src = df.copy()
    trend_src["time"] = pd.to_datetime(trend_src["time"])
    trend = (
        trend_src.set_index("time")
        .resample("ME")
        .agg(mae=("abs_error", "mean"), baseline_mae=("baseline_abs_error", "mean"), n=("abs_error", "size"))
        .dropna()
        .reset_index()
    )
    error_trend = [
        {
            "period": str(r["time"].date()),
            "mae": round(float(r["mae"]), 3),
            "baseline_mae": round(float(r["baseline_mae"]), 3),
            "n": int(r["n"]),
        }
        for _, r in trend.iterrows()
    ]

    print("[5/5] writing snapshot")
    chart = candles.tail(260)
    snapshot = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "ticker": ticker.upper(),
        "config": {
            "lookback": lookback,
            "horizon": horizon,
            "indicators": indicators,
            "train_span": TRAIN_SPAN,
            "drift_threshold_pct": rcfg["drift_threshold_pct"],
            "interval": cfg["data"]["interval"],
        },
        "company": {
            "name": fundamentals.get("name"),
            "currency": fundamentals.get("currency") or "USD",
            "current_price": fundamentals.get("current_price"),
        },
        "fundamentals": {
            k: fundamentals.get(k)
            for k in (
                "revenue",
                "ebit_margin",
                "revenue_growth",
                "shares_outstanding",
                "net_debt",
                "capex_pct",
                "depreciation_pct",
                "nwc_change_pct",
            )
        },
        "defaults": {
            "tax_rate": cfg["valuation"]["default_tax_rate"],
            "wacc": cfg["valuation"]["default_wacc"],
            "terminal_growth": cfg["valuation"]["default_terminal_growth"],
            "forecast_years": cfg["valuation"]["forecast_years"],
        },
        "candles": [
            {
                "time": str(ts.date()),
                "open": round(float(r["open"]), 2),
                "high": round(float(r["high"]), 2),
                "low": round(float(r["low"]), 2),
                "close": round(float(r["close"]), 2),
            }
            for ts, r in chart.iterrows()
        ],
        "backtest": {
            "metrics": overall,
            "by_horizon": by_horizon,
            "error_trend": error_trend,
            "training_runs": runs,
            # The full row set is large; ship the most recent slice for the
            # prediction-log table and the overlay chart.
            "predictions": rows[-400:],
            "span": {"from": rows[0]["time"], "to": rows[-1]["time"]},
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")

    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  span         {snapshot['backtest']['span']['from']} -> {snapshot['backtest']['span']['to']}")
    print(f"  predictions  {overall['n']}")
    print(f"  model MAE    {overall['mae']}")
    print(f"  baseline MAE {overall['baseline_mae']}")
    print(f"  skill        {overall['skill_vs_baseline_pct']:+}%")
    print(f"  direction    {overall['direction_accuracy_pct']}%")
    print(f"  refits       {len(runs)}")
    return snapshot


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
