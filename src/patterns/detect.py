"""Candlestick pattern detection.

Vectorised pandas rules, not a model — every pattern is a deterministic
description of candle shape, so there is no lookahead risk: whether a hammer
occurred on day t depends only on days up to and including t.

These are approximations of the textbook definitions, not exact
reproductions. Real technical-analysis literature disagrees with itself on
thresholds (how small is a "small" upper shadow?), so the thresholds below are
one reasonable reading, documented inline. The point of this module is not to
nail the canonical definition — it is to test, honestly, whether shapes that
traders believe predict a move actually do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternMeta:
    label: str
    bias: str  # 'bullish' | 'bearish'
    description: str


PATTERN_META: dict[str, PatternMeta] = {
    "doji": PatternMeta(
        "Doji",
        "neutral",
        "Open and close land almost on top of each other — the session ended where it began, a tug of war with no winner.",
    ),
    "hammer": PatternMeta(
        "Hammer",
        "bullish",
        "A small body near the top of the range with a long lower wick, after a decline — sellers pushed price down and buyers pushed it back.",
    ),
    "shooting_star": PatternMeta(
        "Shooting star",
        "bearish",
        "A small body near the bottom of the range with a long upper wick, after a rise — buyers pushed price up and sellers pushed it back.",
    ),
    "bullish_engulfing": PatternMeta(
        "Bullish engulfing",
        "bullish",
        "A rising candle whose body fully swallows the prior falling candle's body — often read as buyers overwhelming the prior day's sellers.",
    ),
    "bearish_engulfing": PatternMeta(
        "Bearish engulfing",
        "bearish",
        "A falling candle whose body fully swallows the prior rising candle's body.",
    ),
    "piercing_line": PatternMeta(
        "Piercing line",
        "bullish",
        "A falling candle followed by a rising candle that opens below the prior low and closes above its midpoint.",
    ),
    "dark_cloud_cover": PatternMeta(
        "Dark cloud cover",
        "bearish",
        "A rising candle followed by a falling candle that opens above the prior high and closes below its midpoint.",
    ),
    "morning_star": PatternMeta(
        "Morning star",
        "bullish",
        "Three candles: a fall, a small-bodied pause, then a rise closing back into the first candle's body — a textbook reversal shape.",
    ),
    "evening_star": PatternMeta(
        "Evening star",
        "bearish",
        "The mirror of the morning star: a rise, a pause, then a fall.",
    ),
    "three_white_soldiers": PatternMeta(
        "Three white soldiers",
        "bullish",
        "Three consecutive rising candles, each closing higher than the last with small upper wicks — read as sustained buying pressure.",
    ),
    "three_black_crows": PatternMeta(
        "Three black crows",
        "bearish",
        "The mirror of three white soldiers: three consecutive falling candles.",
    ),
}


def detect_all(df: pd.DataFrame, trend_lookback: int = 5) -> pd.DataFrame:
    """Boolean occurrence matrix, one column per pattern, aligned to df.index.

    A True at row t means the pattern completed on day t — the last candle
    involved (one, two, or three candles back) is t itself.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bullish = c > o
    bearish = c < o

    # "Downtrend into the pattern" / "uptrend into the pattern": is the close
    # immediately before the pattern below/above where it was `trend_lookback`
    # sessions earlier. A blunt proxy for trend, not a technical-analysis
    # trend filter — documented as a simplification.
    prior_down = c.shift(1) < c.shift(1 + trend_lookback)
    prior_up = c.shift(1) > c.shift(1 + trend_lookback)

    out = pd.DataFrame(index=df.index)

    out["doji"] = body <= 0.1 * rng

    out["hammer"] = (
        (lower >= 0.6 * rng) & (body <= 0.3 * rng) & (upper <= 0.1 * rng) & prior_down
    )
    out["shooting_star"] = (
        (upper >= 0.6 * rng) & (body <= 0.3 * rng) & (lower <= 0.1 * rng) & prior_up
    )

    prev_bear, prev_bull = bearish.shift(1), bullish.shift(1)
    prev_o, prev_c = o.shift(1), c.shift(1)

    out["bullish_engulfing"] = prev_bear & bullish & (o <= prev_c) & (c >= prev_o)
    out["bearish_engulfing"] = prev_bull & bearish & (o >= prev_c) & (c <= prev_o)

    prev_mid = (prev_o + prev_c) / 2
    out["piercing_line"] = prev_bear & bullish & (o < l.shift(1)) & (c > prev_mid) & (c < prev_o)
    out["dark_cloud_cover"] = (
        prev_bull & bearish & (o > h.shift(1)) & (c < prev_mid) & (c > prev_o)
    )

    # Three-candle stars: candle 1 (two back) sets direction, candle 2 (one
    # back) is a small-bodied pause, candle 3 (today) reverses back past the
    # midpoint of candle 1.
    c2_bear, c2_bull = bearish.shift(2), bullish.shift(2)
    c2_o, c2_c = o.shift(2), c.shift(2)
    c2_mid = (c2_o + c2_c) / 2
    pause_small = body.shift(1) <= 0.3 * rng.shift(1)

    out["morning_star"] = c2_bear & pause_small & (c.shift(1) < c2_c) & bullish & (c > c2_mid)
    out["evening_star"] = c2_bull & pause_small & (c.shift(1) > c2_c) & bearish & (c < c2_mid)

    small_upper = upper <= 0.2 * rng
    small_lower = lower <= 0.2 * rng
    out["three_white_soldiers"] = (
        bullish
        & bullish.shift(1)
        & bullish.shift(2)
        & (c > c.shift(1))
        & (c.shift(1) > c.shift(2))
        & (o > o.shift(1))
        & (o.shift(1) > o.shift(2))
        & small_upper
    )
    out["three_black_crows"] = (
        bearish
        & bearish.shift(1)
        & bearish.shift(2)
        & (c < c.shift(1))
        & (c.shift(1) < c.shift(2))
        & (o < o.shift(1))
        & (o.shift(1) < o.shift(2))
        & small_lower
    )

    return out.fillna(False).astype(bool)
