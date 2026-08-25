"""Prediction log and outcome reconciliation store.

Every prediction is written before its target candles exist. Later, once the
market prints those candles, `evaluate` fills in the actuals and the error.
That append-then-reconcile pattern is what makes the feedback loop honest — the
prediction cannot be edited after the fact.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One predicted candle. A 5-candle forecast writes 5 rows."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("ticker", "model_version", "anchor_ts", "step", name="uq_pred"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    anchor_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    target_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    step: Mapped[int] = mapped_column(Integer)  # 1..horizon
    anchor_close: Mapped[float] = mapped_column(Float)

    pred_open: Mapped[float] = mapped_column(Float)
    pred_high: Mapped[float] = mapped_column(Float)
    pred_low: Mapped[float] = mapped_column(Float)
    pred_close: Mapped[float] = mapped_column(Float)
    pred_close_std: Mapped[float] = mapped_column(Float, default=0.0)
    baseline_close: Mapped[float] = mapped_column(Float)

    # Filled in by the reconciliation pass once the candle actually closes.
    actual_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    abs_error_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_abs_error_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction_correct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrainingRun(Base):
    """Audit trail of every fit, so drift can be attributed to a version."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    trigger: Mapped[str] = mapped_column(String(32))  # initial | scheduled | drift
    window_days: Mapped[int] = mapped_column(Integer)
    n_samples: Mapped[int] = mapped_column(Integer)
    val_loss: Mapped[float] = mapped_column(Float)
    test_mae_close: Mapped[float] = mapped_column(Float)
    baseline_mae_close: Mapped[float] = mapped_column(Float)
    artifact_path: Mapped[str] = mapped_column(String(256))


def make_engine(url: str = "sqlite:///data/predictions.db"):
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


def open_predictions(session: Session, ticker: str) -> list[Prediction]:
    """Predictions whose target candle has not been reconciled yet."""
    stmt = (
        select(Prediction)
        .where(Prediction.ticker == ticker.upper())
        .where(Prediction.actual_close.is_(None))
        .order_by(Prediction.target_ts)
    )
    return list(session.scalars(stmt))


def latest_model_version(session: Session, ticker: str) -> str | None:
    stmt = (
        select(TrainingRun.model_version)
        .where(TrainingRun.ticker == ticker.upper())
        .order_by(TrainingRun.trained_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)
