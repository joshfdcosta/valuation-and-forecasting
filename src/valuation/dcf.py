"""Discounted cash flow valuation.

Pure functions — no I/O, no globals. Everything here is unit-testable and
mirrors the standard two-stage DCF taught in corporate finance: an explicit
forecast period followed by a Gordon Growth terminal value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    revenue_base: float
    revenue_growth: float
    ebit_margin: float
    tax_rate: float
    wacc: float
    terminal_growth: float
    forecast_years: int = 5
    # Capex / D&A / working capital as a share of revenue.
    capex_pct: float = 0.05
    depreciation_pct: float = 0.04
    nwc_change_pct: float = 0.01

    def __post_init__(self) -> None:
        if self.wacc <= self.terminal_growth:
            raise ValueError(
                f"wacc ({self.wacc}) must exceed terminal_growth "
                f"({self.terminal_growth}); otherwise terminal value is undefined"
            )
        if self.forecast_years < 1:
            raise ValueError("forecast_years must be >= 1")


@dataclass
class Valuation:
    forecast: pd.DataFrame
    pv_explicit: float
    terminal_value: float
    pv_terminal: float
    enterprise_value: float
    equity_value: float | None = None
    fair_value_per_share: float | None = None
    inputs: Assumptions | None = field(default=None, repr=False)


def project_free_cash_flow(a: Assumptions) -> pd.DataFrame:
    """Build the explicit-period FCF schedule.

    FCF = EBIT * (1 - tax) + D&A - capex - change in net working capital.
    """
    years = np.arange(1, a.forecast_years + 1)
    revenue = a.revenue_base * (1 + a.revenue_growth) ** years
    ebit = revenue * a.ebit_margin
    nopat = ebit * (1 - a.tax_rate)
    depreciation = revenue * a.depreciation_pct
    capex = revenue * a.capex_pct
    nwc_change = revenue * a.nwc_change_pct
    fcf = nopat + depreciation - capex - nwc_change

    discount_factor = 1 / (1 + a.wacc) ** years

    return pd.DataFrame(
        {
            "year": years,
            "revenue": revenue,
            "ebit": ebit,
            "nopat": nopat,
            "fcf": fcf,
            "discount_factor": discount_factor,
            "pv_fcf": fcf * discount_factor,
        }
    )


def terminal_value(final_fcf: float, wacc: float, terminal_growth: float) -> float:
    """Gordon Growth terminal value as of the final explicit forecast year."""
    return final_fcf * (1 + terminal_growth) / (wacc - terminal_growth)


def value(
    a: Assumptions,
    net_debt: float | None = None,
    shares_outstanding: float | None = None,
) -> Valuation:
    forecast = project_free_cash_flow(a)

    tv = terminal_value(float(forecast["fcf"].iloc[-1]), a.wacc, a.terminal_growth)
    pv_tv = tv * float(forecast["discount_factor"].iloc[-1])
    pv_explicit = float(forecast["pv_fcf"].sum())
    ev = pv_explicit + pv_tv

    equity = None
    per_share = None
    if net_debt is not None:
        equity = ev - net_debt
        if shares_outstanding:
            per_share = equity / shares_outstanding

    return Valuation(
        forecast=forecast,
        pv_explicit=pv_explicit,
        terminal_value=tv,
        pv_terminal=pv_tv,
        enterprise_value=ev,
        equity_value=equity,
        fair_value_per_share=per_share,
        inputs=a,
    )


def sensitivity(
    a: Assumptions,
    wacc_range: list[float],
    growth_range: list[float],
    net_debt: float = 0.0,
    shares_outstanding: float = 1.0,
) -> pd.DataFrame:
    """Fair value per share across a WACC x terminal-growth grid."""
    rows = []
    for w in wacc_range:
        row = {}
        for g in growth_range:
            if w <= g:
                row[g] = np.nan
                continue
            trial = Assumptions(**{**a.__dict__, "wacc": w, "terminal_growth": g})
            row[g] = value(trial, net_debt, shares_outstanding).fair_value_per_share
        rows.append(row)
    return pd.DataFrame(rows, index=wacc_range)
