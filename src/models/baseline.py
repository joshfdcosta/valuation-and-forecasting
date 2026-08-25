"""Naive baselines.

The persistence baseline is the number that matters. On daily equity candles
it is very hard to beat, and any learned model that does not beat it is not
adding information. Report both, always.
"""

from __future__ import annotations

import numpy as np


class PersistenceBaseline:
    """Predict every future candle as flat at the anchor close.

    In relative-target space that is simply zeros.
    """

    name = "persistence"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        return self

    def predict(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return np.zeros((len(x), horizon, 4), dtype=np.float32)


class DriftBaseline:
    """Persistence plus the average per-candle drift seen in training."""

    name = "drift"

    def __init__(self) -> None:
        self.per_step_drift = 0.0

    def fit(self, x: np.ndarray, y_rel: np.ndarray) -> "DriftBaseline":
        # y_rel[:, h, 3] is the close offset h+1 candles ahead.
        steps = np.arange(1, y_rel.shape[1] + 1)
        self.per_step_drift = float(np.mean(y_rel[:, :, 3] / steps))
        return self

    def predict(self, x: np.ndarray, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=np.float32)
        path = (self.per_step_drift * steps)[None, :, None]
        return np.repeat(np.repeat(path, len(x), axis=0), 4, axis=2).astype(np.float32)
