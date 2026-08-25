import numpy as np
import pandas as pd
import pytest

from src.patterns.detect import PATTERN_META, detect_all
from src.patterns.study import _holm, forward_returns, run_study


def frame(rows):
    """rows: list of (open, high, low, close)."""
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def flat(n, price=100.0):
    return [(price, price + 0.5, price - 0.5, price)] * n


def test_every_detected_column_has_metadata():
    df = frame(flat(30))
    for col in detect_all(df).columns:
        assert col in PATTERN_META


def test_doji_fires_when_open_and_close_coincide():
    rows = flat(10) + [(100.0, 103.0, 97.0, 100.05)]
    out = detect_all(frame(rows))
    assert out["doji"].iloc[-1]


def test_doji_does_not_fire_on_a_full_bodied_candle():
    rows = flat(10) + [(100.0, 106.0, 99.0, 105.5)]
    assert not detect_all(frame(rows))["doji"].iloc[-1]


def test_hammer_requires_a_prior_downtrend():
    # Long lower wick, small body at the top of the range.
    hammer = (100.0, 100.4, 94.0, 100.2)

    falling = [(110.0 - i, 110.5 - i, 109.5 - i, 109.8 - i) for i in range(8)]
    assert detect_all(frame(falling + [hammer]))["hammer"].iloc[-1]

    rising = [(100.0 + i, 100.5 + i, 99.5 + i, 100.4 + i) for i in range(8)]
    assert not detect_all(frame(rising + [hammer]))["hammer"].iloc[-1]


def test_shooting_star_requires_a_prior_uptrend():
    star = (100.0, 106.0, 99.7, 100.2)
    rising = [(90.0 + i, 90.5 + i, 89.5 + i, 90.4 + i) for i in range(8)]
    assert detect_all(frame(rising + [star]))["shooting_star"].iloc[-1]


def test_bullish_engulfing_body_must_swallow_the_prior_body():
    rows = flat(6) + [(100.0, 100.2, 97.0, 97.5), (97.0, 101.5, 96.8, 101.0)]
    out = detect_all(frame(rows))
    assert out["bullish_engulfing"].iloc[-1]
    assert not out["bearish_engulfing"].iloc[-1]


def test_bearish_engulfing_is_the_mirror():
    rows = flat(6) + [(97.5, 100.2, 97.0, 100.0), (101.0, 101.5, 96.5, 96.8)]
    assert detect_all(frame(rows))["bearish_engulfing"].iloc[-1]


def test_morning_star_needs_fall_pause_then_recovery():
    rows = flat(6) + [
        (100.0, 100.5, 94.0, 94.5),   # decisive fall
        (94.0, 94.6, 93.4, 93.9),     # small-bodied pause, closes below
        (94.5, 99.0, 94.2, 98.5),     # rise back above the first body midpoint
    ]
    assert detect_all(frame(rows))["morning_star"].iloc[-1]


def test_evening_star_is_the_mirror():
    rows = flat(6) + [
        (94.5, 100.5, 94.2, 100.0),
        (100.4, 101.0, 100.1, 100.6),
        (100.0, 100.2, 95.0, 95.5),
    ]
    assert detect_all(frame(rows))["evening_star"].iloc[-1]


def test_three_white_soldiers_needs_three_rising_closes():
    rows = flat(6) + [
        (100.0, 102.1, 99.9, 102.0),
        (102.2, 104.1, 102.1, 104.0),
        (104.2, 106.1, 104.1, 106.0),
    ]
    out = detect_all(frame(rows))
    assert out["three_white_soldiers"].iloc[-1]
    assert not out["three_black_crows"].iloc[-1]


def test_detection_never_looks_ahead():
    """Truncating the future must not change a past detection."""
    rng = np.random.default_rng(0)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    df = frame(
        [
            (float(c * 0.995), float(c * 1.01), float(c * 0.99), float(c))
            for c in close
        ]
    )
    full = detect_all(df)
    truncated = detect_all(df.iloc[:150])
    pd.testing.assert_frame_equal(full.iloc[:150], truncated)


def test_forward_returns_align_to_the_future_not_the_past():
    close = pd.Series([100.0, 110.0, 121.0], index=pd.bdate_range("2024-01-01", periods=3))
    fwd = forward_returns(close, 1)
    assert fwd.iloc[0] == pytest.approx(10.0)
    assert fwd.iloc[1] == pytest.approx(10.0)
    assert pd.isna(fwd.iloc[2])  # no future left to measure


def test_holm_is_stricter_than_uncorrected_alpha():
    # 0.04 would pass a naive 0.05 test; against 5 hypotheses it must not.
    assert _holm([0.04, 0.5, 0.6, 0.7, 0.8]) == [False] * 5
    # A genuinely tiny p-value still survives.
    assert _holm([0.0001, 0.5, 0.6, 0.7, 0.8])[0]


def test_study_reports_no_edge_on_random_walk_data():
    """The critical guard: on data with no signal, almost nothing should survive."""
    rng = np.random.default_rng(3)
    n = 1200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    rows = []
    for c in close:
        drift = rng.normal(0, 0.004)
        o = c * (1 + drift)
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
        rows.append((float(o), float(hi), float(lo), float(c)))
    df = frame(rows)

    study = run_study(df, detect_all(df), seed=11)
    assert study["n_tests"] > 0
    # On a pure random walk a correct procedure finds essentially nothing.
    assert study["n_significant"] == 0
