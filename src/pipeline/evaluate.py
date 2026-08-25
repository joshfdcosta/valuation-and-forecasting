"""Reconcile logged predictions against the candles that actually printed.

Run this after each candle close. It fills in actuals, computes per-row error
against both the model and the persistence baseline, and rolls the results up
into the metrics the dashboard shows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.fetch import get_candles
from src.pipeline.train import load_config
from src.storage.db import Prediction, make_engine, open_predictions, utcnow


def reconcile(ticker: str, cfg: dict | None = None) -> dict:
    """Match unresolved predictions to real candles and score them."""
    cfg = cfg or load_config()
    engine = make_engine(cfg["storage"]["url"])

    candles = get_candles(
        ticker,
        interval=cfg["data"]["interval"],
        period="1y",
        cache_dir=cfg["data"]["cache_dir"],
        refresh=True,
    )

    resolved = 0
    with Session(engine) as session:
        for row in open_predictions(session, ticker):
            target = pd.Timestamp(row.target_ts)
            if target not in candles.index:
                # Holiday, half day, or the candle simply has not closed yet.
                continue
            actual = candles.loc[target]

            row.actual_open = float(actual["open"])
            row.actual_high = float(actual["high"])
            row.actual_low = float(actual["low"])
            row.actual_close = float(actual["close"])
            row.abs_error_close = abs(row.pred_close - row.actual_close)
            row.baseline_abs_error_close = abs(row.baseline_close - row.actual_close)

            predicted_move = row.pred_close - row.anchor_close
            actual_move = row.actual_close - row.anchor_close
            row.direction_correct = int(np.sign(predicted_move) == np.sign(actual_move))
            row.resolved_at = utcnow()
            resolved += 1

        session.commit()

    return {"ticker": ticker.upper(), "resolved": resolved, **metrics(ticker, cfg)}


def _resolved_frame(ticker: str, cfg: dict) -> pd.DataFrame:
    engine = make_engine(cfg["storage"]["url"])
    stmt = (
        select(Prediction)
        .where(Prediction.ticker == ticker.upper())
        .where(Prediction.actual_close.is_not(None))
        .order_by(Prediction.target_ts)
    )
    with Session(engine) as session:
        rows = list(session.scalars(stmt))

    return pd.DataFrame(
        [
            {
                "target_ts": r.target_ts,
                "model_version": r.model_version,
                "step": r.step,
                "abs_error_close": r.abs_error_close,
                "baseline_abs_error_close": r.baseline_abs_error_close,
                "direction_correct": r.direction_correct,
                "actual_close": r.actual_close,
            }
            for r in rows
        ]
    )


def metrics(ticker: str, cfg: dict | None = None, last_n: int | None = None) -> dict:
    """Headline numbers for the dashboard.

    `skill_vs_baseline` is the one to read first: positive means the model beat
    persistence, negative means it did not, and near-zero means the model is
    reproducing the last close and calling it a forecast.
    """
    cfg = cfg or load_config()
    df = _resolved_frame(ticker, cfg)
    if df.empty:
        return {"n_resolved": 0}

    if last_n:
        df = df.tail(last_n)

    mae = float(df["abs_error_close"].mean())
    base_mae = float(df["baseline_abs_error_close"].mean())
    rmse = float(np.sqrt((df["abs_error_close"] ** 2).mean()))
    mape = float((df["abs_error_close"] / df["actual_close"]).mean() * 100)

    return {
        "n_resolved": int(len(df)),
        "mae_close": round(mae, 4),
        "rmse_close": round(rmse, 4),
        "mape_close_pct": round(mape, 3),
        "baseline_mae_close": round(base_mae, 4),
        "skill_vs_baseline_pct": round((base_mae - mae) / base_mae * 100, 2)
        if base_mae
        else 0.0,
        "direction_accuracy_pct": round(df["direction_correct"].mean() * 100, 2),
    }


def error_trend(ticker: str, cfg: dict | None = None, freq: str = "W") -> pd.DataFrame:
    """MAE over time — the series the drift detector and the chart both read."""
    cfg = cfg or load_config()
    df = _resolved_frame(ticker, cfg)
    if df.empty:
        return pd.DataFrame(columns=["period", "mae_close", "baseline_mae_close", "n"])

    df["target_ts"] = pd.to_datetime(df["target_ts"])
    grouped = (
        df.set_index("target_ts")
        .resample(freq)
        .agg(
            mae_close=("abs_error_close", "mean"),
            baseline_mae_close=("baseline_abs_error_close", "mean"),
            n=("abs_error_close", "size"),
        )
        .dropna()
        .reset_index(names="period")
    )
    return grouped


if __name__ == "__main__":
    import sys

    print(json.dumps(reconcile(sys.argv[1] if len(sys.argv) > 1 else "AAPL"), indent=2))
