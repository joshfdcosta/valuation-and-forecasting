import numpy as np

from src.pipeline.predict import enforce_ohlc_coherence, future_timestamps
import pandas as pd


def test_coherence_lifts_high_and_drops_low_to_bound_the_body():
    # low sits above the open, high sits below the close — both invalid.
    raw = np.array([[307.07, 310.0, 308.15, 311.73]], dtype=np.float32)
    fixed = enforce_ohlc_coherence(raw)
    o, h, l, c = fixed[0]
    assert h == max(o, h, l, c)
    assert l == min(o, h, l, c)
    assert h >= max(o, c) and l <= min(o, c)


def test_coherence_leaves_valid_candles_untouched():
    valid = np.array([[100.0, 105.0, 99.0, 103.0]], dtype=np.float32)
    assert np.array_equal(enforce_ohlc_coherence(valid), valid)


def test_daily_targets_skip_weekends():
    friday = pd.Timestamp("2026-08-21")
    targets = future_timestamps(friday, 3, "1d")
    assert targets[0] == pd.Timestamp("2026-08-24")
    assert all(t.weekday() < 5 for t in targets)
    assert all(t > friday for t in targets)
