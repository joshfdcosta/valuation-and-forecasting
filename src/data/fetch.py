"""Market and fundamentals retrieval via yfinance, with a local parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(cache_dir: str, ticker: str, interval: str, period: str) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{ticker.upper()}_{interval}_{period}.parquet"


def get_candles(
    ticker: str,
    interval: str = "1d",
    period: str = "5y",
    cache_dir: str = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a tz-naive OHLCV frame indexed by timestamp, oldest first."""
    cache = _cache_path(cache_dir, ticker, interval, period)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"no candles returned for {ticker!r} ({interval}, {period})")

    df = raw.rename(columns=str.lower)[OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "timestamp"
    df = df.sort_index()
    df.to_parquet(cache)
    return df


def _cashflow_ratios(tkr: yf.Ticker, revenue: float | None) -> dict:
    """Capex, D&A, and net-working-capital-change as a share of revenue.

    Pulled from the most recent annual cash flow statement rather than
    hardcoded — an AAPL DCF and a capital-intensive industrial should not
    share the same 5%/4%/1% guess. Falls back to those defaults (matching
    `Assumptions`) only when the statement or a row is missing, since a
    single-column DCF should never hard-fail on a data gap.
    """
    defaults = {"capex_pct": 0.05, "depreciation_pct": 0.04, "nwc_change_pct": 0.01}
    if not revenue:
        return defaults

    try:
        cf = tkr.cashflow
    except Exception:
        return defaults
    if cf is None or cf.empty:
        return defaults

    latest = cf.iloc[:, 0]

    def row(*labels) -> float | None:
        for label in labels:
            if label in latest.index and pd.notna(latest[label]):
                return float(latest[label])
        return None

    # yfinance reports capex and NWC change as negative (cash outflow).
    capex = row("Capital Expenditure", "Purchase Of PPE")
    depreciation = row("Depreciation And Amortization", "Depreciation Amortization Depletion")
    nwc_change = row("Change In Working Capital")

    return {
        "capex_pct": abs(capex) / revenue if capex is not None else defaults["capex_pct"],
        "depreciation_pct": depreciation / revenue
        if depreciation is not None
        else defaults["depreciation_pct"],
        "nwc_change_pct": -nwc_change / revenue
        if nwc_change is not None
        else defaults["nwc_change_pct"],
    }


def get_fundamentals(ticker: str) -> dict:
    """Pull the handful of fundamentals the DCF needs.

    yfinance fields move around between releases, so every lookup is defensive:
    a missing field becomes None and the caller decides whether to fall back to
    a manual assumption.
    """
    tkr = yf.Ticker(ticker)
    info = tkr.info or {}

    def pick(*keys):
        for key in keys:
            v = info.get(key)
            if v is not None:
                return float(v)
        return None

    total_debt = pick("totalDebt") or 0.0
    cash = pick("totalCash") or 0.0
    revenue = pick("totalRevenue")

    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName"),
        "currency": info.get("currency"),
        "revenue": revenue,
        "ebit_margin": pick("operatingMargins"),
        "revenue_growth": pick("revenueGrowth"),
        "shares_outstanding": pick("sharesOutstanding", "impliedSharesOutstanding"),
        "net_debt": total_debt - cash,
        "current_price": pick("currentPrice", "regularMarketPrice", "previousClose"),
        "beta": pick("beta"),
        **_cashflow_ratios(tkr, revenue),
    }
