"""Feature engineering and windowing for candle sequence models.

Two rules govern everything here:

1. No lookahead. Every feature at row t uses only data available at or before
   t. Indicators are computed on trailing windows and the target windows are
   sliced strictly after the input window.
2. Scaling is fit on train only. `fit_scaler` is called on the training split
   and the fitted scaler is reused for validation, test, and live inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMNS = ["open", "high", "low", "close"]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach trailing-window indicators to an OHLCV frame."""
    out = df.copy()
    out["returns"] = out["close"].pct_change()
    out["log_returns"] = np.log(out["close"]).diff()
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)
    out["volume_z"] = (
        (out["volume"] - out["volume"].rolling(20).mean())
        / out["volume"].rolling(20).std()
    )
    out["sma_ratio_20"] = out["close"] / out["close"].rolling(20).mean() - 1
    return out.dropna()


def make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Slice a feature frame into (X, y, anchor_timestamps).

    X: (n_samples, lookback, n_features)
    y: (n_samples, horizon, 4)  -- next `horizon` OHLC candles
    anchors: timestamp of the last candle in each input window, i.e. the moment
             the prediction is made. Predictions cover the candles after it.
    """
    features = df[feature_columns].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    xs, ys, anchors = [], [], []
    last_start = len(df) - lookback - horizon + 1
    for start in range(last_start):
        end = start + lookback
        xs.append(features[start:end])
        ys.append(targets[end : end + horizon])
        anchors.append(df.index[end - 1])

    if not xs:
        raise ValueError(
            f"not enough rows ({len(df)}) for lookback={lookback} horizon={horizon}"
        )
    return np.stack(xs), np.stack(ys), anchors


def to_relative_targets(y: np.ndarray, anchor_close: np.ndarray) -> np.ndarray:
    """Convert absolute OHLC targets to fractional offsets from the anchor close.

    Raw prices are non-stationary — a model trained on 2019 dollar levels does
    not transfer to 2026 ones. Training on offsets from the last known close
    keeps the target distribution stable and makes the naive baseline (all
    zeros) the explicit thing to beat.
    """
    return (y / anchor_close[:, None, None] - 1).astype(np.float32)


def from_relative_targets(y_rel: np.ndarray, anchor_close: np.ndarray) -> np.ndarray:
    """Inverse of `to_relative_targets` — back to price space."""
    return (y_rel + 1) * anchor_close[:, None, None]


def anchor_closes(df: pd.DataFrame, anchors: list[pd.Timestamp]) -> np.ndarray:
    return df.loc[anchors, "close"].to_numpy(dtype=np.float32)


def chronological_split(
    n: int, train: float = 0.7, val: float = 0.15
) -> tuple[slice, slice, slice]:
    """Time-ordered split. Never shuffle time series — it leaks the future."""
    i, j = int(n * train), int(n * (train + val))
    return slice(0, i), slice(i, j), slice(j, n)


def fit_scaler(x_train: np.ndarray) -> StandardScaler:
    n_features = x_train.shape[-1]
    return StandardScaler().fit(x_train.reshape(-1, n_features))


def apply_scaler(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    shape = x.shape
    flat = scaler.transform(x.reshape(-1, shape[-1]))
    return flat.reshape(shape).astype(np.float32)
