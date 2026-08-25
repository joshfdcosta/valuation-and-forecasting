import numpy as np
import pandas as pd
import pytest

from src.data import features as F
from src.models.baseline import PersistenceBaseline
from src.valuation import dcf


def base_assumptions(**overrides):
    defaults = dict(
        revenue_base=1000.0,
        revenue_growth=0.08,
        ebit_margin=0.22,
        tax_rate=0.21,
        wacc=0.09,
        terminal_growth=0.025,
        forecast_years=5,
    )
    return dcf.Assumptions(**{**defaults, **overrides})


def test_wacc_must_exceed_terminal_growth():
    with pytest.raises(ValueError, match="must exceed terminal_growth"):
        base_assumptions(wacc=0.02, terminal_growth=0.03)


def test_forecast_has_one_row_per_year_and_decaying_discount_factors():
    forecast = dcf.project_free_cash_flow(base_assumptions())
    assert len(forecast) == 5
    assert forecast["discount_factor"].is_monotonic_decreasing
    assert forecast["revenue"].is_monotonic_increasing


def test_terminal_value_matches_gordon_growth_by_hand():
    # 100 * 1.025 / (0.09 - 0.025) = 1576.923...
    assert dcf.terminal_value(100.0, 0.09, 0.025) == pytest.approx(1576.9231, rel=1e-4)


def test_higher_wacc_lowers_enterprise_value():
    cheap = dcf.value(base_assumptions(wacc=0.08)).enterprise_value
    dear = dcf.value(base_assumptions(wacc=0.12)).enterprise_value
    assert dear < cheap


def test_per_share_value_nets_out_debt():
    with_debt = dcf.value(base_assumptions(), net_debt=500.0, shares_outstanding=100.0)
    without = dcf.value(base_assumptions(), net_debt=0.0, shares_outstanding=100.0)
    assert with_debt.fair_value_per_share == pytest.approx(
        without.fair_value_per_share - 5.0
    )


def test_sensitivity_grid_is_square_and_monotonic():
    grid = dcf.sensitivity(
        base_assumptions(),
        wacc_range=[0.08, 0.09, 0.10],
        growth_range=[0.02, 0.025, 0.03],
        net_debt=0.0,
        shares_outstanding=100.0,
    )
    assert grid.shape == (3, 3)
    # Down a column (rising WACC) value falls; across a row (rising g) it rises.
    assert grid[0.025].is_monotonic_decreasing
    assert grid.loc[0.09].is_monotonic_increasing


def _synthetic_candles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        },
        index=index,
    )


def test_windows_do_not_leak_the_future():
    feat = F.build_features(_synthetic_candles())
    cols = ["returns", "rsi_14", "atr_14"]
    x, y, anchors = F.make_windows(feat, cols, lookback=30, horizon=5)

    assert x.shape[1:] == (30, 3)
    assert y.shape[1:] == (5, 4)
    # The first predicted candle must sit strictly after the anchor timestamp.
    first_target = feat.index[feat.index.get_loc(anchors[0]) + 1]
    assert first_target > anchors[0]
    assert y[0][0][3] == pytest.approx(feat["close"].loc[first_target], rel=1e-5)


def test_relative_target_roundtrip():
    feat = F.build_features(_synthetic_candles())
    _, y_abs, anchors = F.make_windows(feat, ["returns", "rsi_14"], 30, 5)
    anchor_close = F.anchor_closes(feat, anchors)
    rel = F.to_relative_targets(y_abs, anchor_close)
    assert np.allclose(F.from_relative_targets(rel, anchor_close), y_abs, rtol=1e-4)


def test_persistence_baseline_is_flat_in_relative_space():
    pred = PersistenceBaseline().predict(np.zeros((7, 30, 2), dtype=np.float32), 5)
    assert pred.shape == (7, 5, 4)
    assert np.all(pred == 0)


def test_chronological_split_is_ordered_and_covers_everything():
    tr, va, te = F.chronological_split(100)
    assert (tr.start, tr.stop) == (0, 70)
    assert (va.start, va.stop) == (70, 85)
    assert (te.start, te.stop) == (85, 100)
