"""Generate a forward forecast and log it before the outcome is known."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from src.data import features as F
from src.data.fetch import get_candles
from src.models import lstm
from src.models.baseline import PersistenceBaseline
from src.pipeline.train import load_config
from src.storage.db import Prediction, TrainingRun, make_engine


def _rebuild_scaler(meta: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(meta["scaler_mean"])
    scaler.scale_ = np.array(meta["scaler_scale"])
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler


def future_timestamps(last: pd.Timestamp, horizon: int, interval: str) -> list[pd.Timestamp]:
    """Approximate target timestamps for the predicted candles.

    Business days are a stand-in for a real exchange calendar — they overcount
    by roughly nine holidays a year. Swap in `pandas_market_calendars` before
    trusting the target timestamps for anything precise.
    """
    if interval.endswith("d"):
        return list(pd.bdate_range(last, periods=horizon + 1)[1:])
    freq = {"1h": "h", "1m": "min", "5m": "5min", "15m": "15min"}.get(interval, "h")
    return list(pd.date_range(last, periods=horizon + 1, freq=freq)[1:])


def enforce_ohlc_coherence(candles: np.ndarray) -> np.ndarray:
    """Clamp predicted candles so high is the max and low is the min.

    The model predicts the four legs independently, so nothing stops it
    returning a low above the open. That is not a candle. Clamping after the
    fact is the cheap fix; a constrained head (predict high/low as non-negative
    offsets from the body) is the better one.
    """
    out = candles.copy()
    body = np.stack([out[:, 0], out[:, 3]], axis=1)
    out[:, 1] = np.maximum(out[:, 1], body.max(axis=1))
    out[:, 2] = np.minimum(out[:, 2], body.min(axis=1))
    return out


def predict_ticker(ticker: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    engine = make_engine(cfg["storage"]["url"])

    with Session(engine) as session:
        run = (
            session.query(TrainingRun)
            .filter(TrainingRun.ticker == ticker.upper())
            .order_by(TrainingRun.trained_at.desc())
            .first()
        )
        if run is None:
            raise RuntimeError(f"no trained model for {ticker!r} — run train first")
        artifact, version = run.artifact_path, run.model_version

    model, meta = lstm.load(artifact)
    scaler = _rebuild_scaler(meta)
    lookback, horizon = meta["lookback"], meta["horizon"]

    candles = get_candles(
        ticker,
        interval=meta["interval"],
        period=cfg["data"]["history_period"],
        cache_dir=cfg["data"]["cache_dir"],
        refresh=True,
    )
    feat = F.build_features(candles)
    window = feat[meta["indicators"]].to_numpy(dtype=np.float32)[-lookback:]
    if len(window) < lookback:
        raise ValueError(f"only {len(window)} candles available, need {lookback}")

    x = F.apply_scaler(scaler, window[None, ...])
    anchor_ts = feat.index[-1]
    anchor_close = np.array([feat["close"].iloc[-1]], dtype=np.float32)

    mean_rel, std_rel = lstm.predict(
        model, x, mc_samples=cfg["model"]["mc_dropout_samples"]
    )
    prices = enforce_ohlc_coherence(F.from_relative_targets(mean_rel, anchor_close)[0])
    close_std = (std_rel[:, :, 3] * anchor_close[:, None])[0]
    baseline = F.from_relative_targets(
        PersistenceBaseline().predict(x, horizon), anchor_close
    )[0]

    targets = future_timestamps(anchor_ts, horizon, meta["interval"])
    rows = []
    with Session(engine) as session:
        for step, (ts, candle) in enumerate(zip(targets, prices), start=1):
            o, h, l, c = (float(v) for v in candle)
            session.merge(
                Prediction(
                    ticker=ticker.upper(),
                    model_version=version,
                    anchor_ts=anchor_ts.to_pydatetime(),
                    target_ts=ts.to_pydatetime(),
                    step=step,
                    anchor_close=float(anchor_close[0]),
                    pred_open=o,
                    pred_high=h,
                    pred_low=l,
                    pred_close=c,
                    pred_close_std=float(close_std[step - 1]),
                    baseline_close=float(baseline[step - 1][3]),
                )
            )
            rows.append(
                {
                    "step": step,
                    "target_ts": str(ts),
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "close_std": round(float(close_std[step - 1]), 2),
                    "baseline_close": round(float(baseline[step - 1][3]), 2),
                }
            )
        session.commit()

    return {
        "ticker": ticker.upper(),
        "model_version": version,
        "anchor_ts": str(anchor_ts),
        "anchor_close": round(float(anchor_close[0]), 2),
        "candles": rows,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(predict_ticker(sys.argv[1] if len(sys.argv) > 1 else "AAPL"), indent=2))
