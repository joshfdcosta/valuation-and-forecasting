"""Drift detection and the retrain trigger.

This is the loop the whole project turns on:

    predict -> wait for the candle -> reconcile -> measure drift -> maybe retrain

Drift is measured as recent MAE against the MAE the model posted on its own
held-out test set at training time. If recent error has degraded past the
threshold, the regime has moved and the model gets refit on a fresh rolling
window.

Deliberately NOT online/incremental learning. Per-sample weight updates on
noisy financial data drift into catastrophic forgetting fast, and there is no
clean way to roll back a bad update. A periodic refit on a rolling window is
reproducible, auditable, and can be reverted to any prior artifact.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src.pipeline import evaluate
from src.pipeline.train import load_config, train_ticker
from src.storage.db import TrainingRun, make_engine


def check_drift(ticker: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    rcfg = cfg["retrain"]
    engine = make_engine(cfg["storage"]["url"])

    with Session(engine) as session:
        run = (
            session.query(TrainingRun)
            .filter(TrainingRun.ticker == ticker.upper())
            .order_by(TrainingRun.trained_at.desc())
            .first()
        )

    if run is None:
        return {"status": "no_model", "should_retrain": True, "reason": "never trained"}

    recent = evaluate.metrics(ticker, cfg, last_n=rcfg["drift_lookback_candles"])
    if recent.get("n_resolved", 0) < rcfg["drift_lookback_candles"]:
        return {
            "status": "insufficient_data",
            "should_retrain": False,
            "n_resolved": recent.get("n_resolved", 0),
            "needed": rcfg["drift_lookback_candles"],
        }

    reference = run.test_mae_close
    current = recent["mae_close"]
    drift_pct = (current - reference) / reference * 100 if reference else 0.0
    should = drift_pct > rcfg["drift_threshold_pct"]

    return {
        "status": "drifted" if should else "stable",
        "should_retrain": should,
        "model_version": run.model_version,
        "reference_mae": round(reference, 4),
        "recent_mae": round(current, 4),
        "drift_pct": round(drift_pct, 2),
        "threshold_pct": rcfg["drift_threshold_pct"],
        "skill_vs_baseline_pct": recent["skill_vs_baseline_pct"],
        "direction_accuracy_pct": recent["direction_accuracy_pct"],
    }


def run_cycle(ticker: str, cfg: dict | None = None, force: bool = False) -> dict:
    """One full feedback pass. Schedule this after each candle close."""
    cfg = cfg or load_config()

    reconciliation = evaluate.reconcile(ticker, cfg)
    drift = check_drift(ticker, cfg)

    retrained = None
    if force or drift["should_retrain"]:
        trigger = "drift" if drift["should_retrain"] else "manual"
        retrained = train_ticker(ticker, cfg, trigger=trigger)

    return {"reconciliation": reconciliation, "drift": drift, "retrained": retrained}


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    symbol = args[0] if args else "AAPL"
    print(json.dumps(run_cycle(symbol, force="--force" in args), indent=2, default=str))
