"""Fit a model for one ticker and record the run.

Called for the initial fit, on a schedule, and by the drift detector. The
`trigger` argument is what distinguishes them in the audit trail.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy.orm import Session

from src.data import features as F
from src.data.fetch import get_candles
from src.models import lstm
from src.models.baseline import PersistenceBaseline
from src.storage.db import TrainingRun, make_engine, utcnow

ARTIFACT_DIR = Path("data/models")


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _version(ticker: str) -> str:
    return f"{ticker.upper()}-{utcnow().strftime('%Y%m%dT%H%M%SZ')}"


def train_ticker(
    ticker: str,
    cfg: dict | None = None,
    trigger: str = "scheduled",
) -> dict:
    cfg = cfg or load_config()
    fcfg, mcfg, rcfg = cfg["features"], cfg["model"], cfg["retrain"]
    lookback, horizon = fcfg["lookback"], fcfg["horizon"]

    candles = get_candles(
        ticker,
        interval=cfg["data"]["interval"],
        period=cfg["data"]["history_period"],
        cache_dir=cfg["data"]["cache_dir"],
        refresh=True,
    )

    # Rolling window: recent regimes carry more signal than 2015 does. Keep
    # enough history either side of the window for indicators and targets.
    cutoff = candles.index.max() - timedelta(days=rcfg["rolling_window_days"])
    warmup = lookback + 40
    windowed = candles[candles.index >= cutoff]
    if len(windowed) < warmup + horizon + 50:
        windowed = candles.tail(warmup + horizon + 400)

    feat = F.build_features(windowed)
    x, y_abs, anchors = F.make_windows(feat, fcfg["indicators"], lookback, horizon)
    anchor_close = F.anchor_closes(feat, anchors)
    y = F.to_relative_targets(y_abs, anchor_close)

    tr, va, te = F.chronological_split(len(x))
    scaler = F.fit_scaler(x[tr])
    xs = F.apply_scaler(scaler, x)

    model = lstm.CandleLSTM(
        n_features=xs.shape[-1],
        horizon=horizon,
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
    )
    history = lstm.train(
        model,
        xs[tr],
        y[tr],
        xs[va],
        y[va],
        epochs=mcfg["epochs"],
        batch_size=mcfg["batch_size"],
        lr=mcfg["lr"],
    )

    # Held-out comparison against persistence, in price space.
    pred_rel, _ = lstm.predict(model, xs[te], mc_samples=mcfg["mc_dropout_samples"])
    base_rel = PersistenceBaseline().predict(xs[te], horizon)
    close_anchor = anchor_close[te]
    pred_price = F.from_relative_targets(pred_rel, close_anchor)[:, :, 3]
    base_price = F.from_relative_targets(base_rel, close_anchor)[:, :, 3]
    actual_price = y_abs[te][:, :, 3]

    test_mae = float(np.mean(np.abs(pred_price - actual_price)))
    base_mae = float(np.mean(np.abs(base_price - actual_price)))

    version = _version(ticker)
    artifact = ARTIFACT_DIR / f"{version}.pt"
    meta = {
        "ticker": ticker.upper(),
        "version": version,
        "n_features": xs.shape[-1],
        "horizon": horizon,
        "lookback": lookback,
        "hidden_size": mcfg["hidden_size"],
        "num_layers": mcfg["num_layers"],
        "dropout": mcfg["dropout"],
        "indicators": fcfg["indicators"],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "interval": cfg["data"]["interval"],
    }
    lstm.save(model, artifact, meta)

    engine = make_engine(cfg["storage"]["url"])
    with Session(engine) as session:
        session.add(
            TrainingRun(
                ticker=ticker.upper(),
                model_version=version,
                trigger=trigger,
                window_days=rcfg["rolling_window_days"],
                n_samples=len(x),
                val_loss=history["best_val_loss"],
                test_mae_close=test_mae,
                baseline_mae_close=base_mae,
                artifact_path=str(artifact),
            )
        )
        session.commit()

    return {
        "version": version,
        "artifact": str(artifact),
        "n_samples": len(x),
        "val_loss": history["best_val_loss"],
        "test_mae_close": test_mae,
        "baseline_mae_close": base_mae,
        "beats_baseline": test_mae < base_mae,
    }


if __name__ == "__main__":
    import sys

    result = train_ticker(sys.argv[1] if len(sys.argv) > 1 else "AAPL", trigger="initial")
    print(json.dumps(result, indent=2))
