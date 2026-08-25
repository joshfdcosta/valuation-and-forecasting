"""Does a candlestick pattern predict the next few days?

The test is deliberately conservative, because the naive version of this study
is one of the easiest ways to fool yourself in finance:

1. Forward returns from overlapping windows are NOT independent. A 5-day
   forward return starting today shares four days with the one starting
   tomorrow. A textbook t-test assumes independence and will happily report
   significance that is not there. We use a stationary bootstrap instead, which
   resamples blocks of contiguous days and so preserves the autocorrelation.

2. Testing eleven patterns at four horizons is forty-four chances to get a
   p < 0.05 by luck alone. At that rate you expect roughly two false positives
   from pure noise. Reporting the winner without correcting for that is the
   single most common way technical-analysis "edges" get manufactured, so we
   apply a Holm-Bonferroni correction across every test in the family.

The output is deliberately blunt about which effects survive both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)
N_BOOTSTRAP = 2000
BLOCK_MEAN_LEN = 10
ALPHA = 0.05


def forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Percentage return from close_t to close_{t+horizon}."""
    return (close.shift(-horizon) / close - 1) * 100


def _stationary_bootstrap_means(
    values: np.ndarray, sample_size: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Distribution of the mean of `sample_size` draws under the null.

    Blocks of geometric length are stitched together from the full return
    series, so each resample keeps the serial correlation of real returns. The
    null being tested: "a window of this many days, drawn without regard to
    whether the pattern occurred, has this mean."
    """
    n = len(values)
    p = 1.0 / BLOCK_MEAN_LEN
    out = np.empty(n_boot)

    for b in range(n_boot):
        idx = np.empty(sample_size, dtype=np.int64)
        i = rng.integers(0, n)
        for k in range(sample_size):
            idx[k] = i
            if rng.random() < p:
                i = rng.integers(0, n)  # start a new block
            else:
                i = (i + 1) % n  # continue this one
        out[b] = values[idx].mean()

    return out


def _holm(pvalues: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm-Bonferroni step-down. Returns which hypotheses survive."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    survives = [False] * m
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[i] <= threshold:
            survives[i] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return survives


def run_study(
    candles: pd.DataFrame,
    occurrences: pd.DataFrame,
    min_occurrences: int = 20,
    seed: int = 7,
) -> dict:
    """Test every pattern at every horizon, then correct for multiple testing."""
    rng = np.random.default_rng(seed)
    close = candles["close"]

    results: list[dict] = []

    for horizon in HORIZONS:
        fwd = forward_returns(close, horizon)
        valid = fwd.notna()
        baseline_pool = fwd[valid].to_numpy()
        baseline_mean = float(baseline_pool.mean())

        for pattern in occurrences.columns:
            hits = occurrences[pattern] & valid
            n = int(hits.sum())
            if n < min_occurrences:
                results.append(
                    {
                        "pattern": pattern,
                        "horizon": horizon,
                        "n": n,
                        "skipped": True,
                    }
                )
                continue

            observed = float(fwd[hits].mean())
            null_means = _stationary_bootstrap_means(baseline_pool, n, N_BOOTSTRAP, rng)

            # Two-sided: how often does chance produce a deviation from the
            # baseline at least as large as the one we observed?
            deviation = abs(observed - baseline_mean)
            null_deviation = np.abs(null_means - baseline_mean)
            p = float((null_deviation >= deviation).mean())

            results.append(
                {
                    "pattern": pattern,
                    "horizon": horizon,
                    "n": n,
                    "skipped": False,
                    "mean_return_pct": round(observed, 4),
                    "baseline_return_pct": round(baseline_mean, 4),
                    "edge_pct": round(observed - baseline_mean, 4),
                    "p_value": round(p, 4),
                    "hit_rate_pct": round(float((fwd[hits] > 0).mean() * 100), 2),
                    "baseline_hit_rate_pct": round(float((baseline_pool > 0).mean() * 100), 2),
                }
            )

    tested = [r for r in results if not r["skipped"]]
    survives = _holm([r["p_value"] for r in tested])
    for r, ok in zip(tested, survives):
        r["significant"] = bool(ok)
    for r in results:
        r.setdefault("significant", False)

    return {
        "results": results,
        "n_tests": len(tested),
        "n_significant": sum(1 for r in tested if r["significant"]),
        "alpha": ALPHA,
        "correction": "holm-bonferroni",
        "bootstrap": {
            "method": "stationary block bootstrap",
            "n_resamples": N_BOOTSTRAP,
            "mean_block_length": BLOCK_MEAN_LEN,
        },
        "horizons": list(HORIZONS),
        "min_occurrences": min_occurrences,
    }
