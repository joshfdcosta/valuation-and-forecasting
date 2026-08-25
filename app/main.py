"""FastAPI surface behind the wireframe.

Endpoints map one-to-one onto the panels:

    /valuation  -> assumptions, FCF schedule, EV, fair value, sensitivity
    /predict    -> candle forecast panel
    /metrics    -> MAE / direction accuracy / skill-vs-baseline cards
    /feedback   -> retrain status panel
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.fetch import get_candles, get_fundamentals
from src.pipeline import evaluate, predict, retrain, train
from src.storage.db import Prediction, make_engine
from src.valuation import dcf

app = FastAPI(title="DCF + candle forecasting", version="0.1.0")
CONFIG = train.load_config()

# Local dev only. A deployed build serves the frontend from the same origin,
# at which point this middleware comes out.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValuationRequest(BaseModel):
    ticker: str
    revenue_growth: float | None = Field(default=None, description="e.g. 0.08")
    ebit_margin: float | None = None
    tax_rate: float | None = None
    wacc: float | None = None
    terminal_growth: float | None = None
    forecast_years: int = 5
    capex_pct: float | None = None
    depreciation_pct: float | None = None
    nwc_change_pct: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/valuation")
def valuation(req: ValuationRequest) -> dict:
    f = get_fundamentals(req.ticker)
    v = CONFIG["valuation"]

    if not f.get("revenue"):
        raise HTTPException(422, f"no revenue data available for {req.ticker!r}")

    assumptions = dcf.Assumptions(
        revenue_base=f["revenue"],
        revenue_growth=req.revenue_growth
        if req.revenue_growth is not None
        else (f.get("revenue_growth") or 0.05),
        ebit_margin=req.ebit_margin
        if req.ebit_margin is not None
        else (f.get("ebit_margin") or 0.15),
        tax_rate=req.tax_rate if req.tax_rate is not None else v["default_tax_rate"],
        wacc=req.wacc if req.wacc is not None else v["default_wacc"],
        terminal_growth=req.terminal_growth
        if req.terminal_growth is not None
        else v["default_terminal_growth"],
        forecast_years=req.forecast_years,
        capex_pct=req.capex_pct if req.capex_pct is not None else f["capex_pct"],
        depreciation_pct=req.depreciation_pct
        if req.depreciation_pct is not None
        else f["depreciation_pct"],
        nwc_change_pct=req.nwc_change_pct
        if req.nwc_change_pct is not None
        else f["nwc_change_pct"],
    )

    try:
        result = dcf.value(assumptions, f.get("net_debt"), f.get("shares_outstanding"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    grid = dcf.sensitivity(
        assumptions,
        wacc_range=[assumptions.wacc - 0.01, assumptions.wacc, assumptions.wacc + 0.01],
        growth_range=[
            assumptions.terminal_growth - 0.005,
            assumptions.terminal_growth,
            assumptions.terminal_growth + 0.005,
        ],
        net_debt=f.get("net_debt") or 0.0,
        shares_outstanding=f.get("shares_outstanding") or 1.0,
    )

    upside = None
    if result.fair_value_per_share and f.get("current_price"):
        upside = (result.fair_value_per_share / f["current_price"] - 1) * 100

    return {
        "company": {k: f[k] for k in ("ticker", "name", "currency", "current_price")},
        "forecast": result.forecast.round(2).to_dict(orient="records"),
        "enterprise_value": round(result.enterprise_value, 2),
        "equity_value": round(result.equity_value, 2) if result.equity_value else None,
        "fair_value_per_share": round(result.fair_value_per_share, 2)
        if result.fair_value_per_share
        else None,
        "upside_pct": round(upside, 2) if upside is not None else None,
        "sensitivity": grid.round(2).to_dict(),
        "cash_flow_assumptions": {
            "capex_pct": round(assumptions.capex_pct, 4),
            "depreciation_pct": round(assumptions.depreciation_pct, 4),
            "nwc_change_pct": round(assumptions.nwc_change_pct, 4),
            "source": "cash_flow_statement"
            if req.capex_pct is None
            and req.depreciation_pct is None
            and req.nwc_change_pct is None
            else "override",
        },
    }


@app.post("/train/{ticker}")
def train_model(ticker: str, trigger: str = "manual") -> dict:
    return train.train_ticker(ticker, CONFIG, trigger=trigger)


@app.get("/predict/{ticker}")
def forecast(ticker: str) -> dict:
    try:
        return predict.predict_ticker(ticker, CONFIG)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/candles/{ticker}")
def candles(ticker: str, limit: int = 120) -> list[dict]:
    """Recent OHLCV history for the chart, oldest first."""
    df = get_candles(
        ticker,
        interval=CONFIG["data"]["interval"],
        period="1y",
        cache_dir=CONFIG["data"]["cache_dir"],
    ).tail(limit)

    return [
        {
            "time": ts.strftime("%Y-%m-%d"),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
        }
        for ts, row in df.iterrows()
    ]


@app.get("/predictions/{ticker}")
def prediction_log(ticker: str, limit: int = 60) -> list[dict]:
    """Logged predictions, resolved and unresolved, newest anchor last.

    The chart overlays these on the real candles — that side-by-side is the
    whole point of the project, so it gets its own endpoint rather than being
    derived on the client.
    """
    engine = make_engine(CONFIG["storage"]["url"])
    stmt = (
        select(Prediction)
        .where(Prediction.ticker == ticker.upper())
        .order_by(Prediction.target_ts.desc())
        .limit(limit)
    )
    with Session(engine) as session:
        rows = list(session.scalars(stmt))

    return [
        {
            "time": r.target_ts.strftime("%Y-%m-%d"),
            "anchor_ts": r.anchor_ts.strftime("%Y-%m-%d"),
            "step": r.step,
            "model_version": r.model_version,
            "open": round(r.pred_open, 2),
            "high": round(r.pred_high, 2),
            "low": round(r.pred_low, 2),
            "close": round(r.pred_close, 2),
            "close_std": round(r.pred_close_std, 2),
            "baseline_close": round(r.baseline_close, 2),
            "actual_close": round(r.actual_close, 2) if r.actual_close else None,
            "abs_error_close": round(r.abs_error_close, 2)
            if r.abs_error_close is not None
            else None,
            "resolved": r.actual_close is not None,
        }
        for r in reversed(rows)
    ]


@app.get("/metrics/{ticker}")
def model_metrics(ticker: str, last_n: int | None = None) -> dict:
    return evaluate.metrics(ticker, CONFIG, last_n=last_n)


@app.get("/metrics/{ticker}/trend")
def model_trend(ticker: str, freq: str = "W") -> list[dict]:
    return evaluate.error_trend(ticker, CONFIG, freq=freq).to_dict(orient="records")


@app.get("/feedback/{ticker}")
def feedback_status(ticker: str) -> dict:
    return retrain.check_drift(ticker, CONFIG)


@app.post("/feedback/{ticker}/run")
def feedback_run(ticker: str, force: bool = False) -> dict:
    return retrain.run_cycle(ticker, CONFIG, force=force)
