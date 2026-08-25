/**
 * Hand-rolled SVG charts.
 *
 * Written rather than pulled from a library so each one can be annotated with
 * the point it is making, carry a real text description for screen readers, and
 * scale fluidly without a resize observer.
 */

import type { Candle, PredictionRow, Snapshot } from '../lib/snapshot'

const W = 900
const H = 320
const PAD = { top: 18, right: 54, bottom: 26, left: 10 }

function scaler(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  return (v: number) => r0 + ((v - d0) / span) * (r1 - r0)
}

function niceTicks(min: number, max: number, count = 4): number[] {
  const raw = (max - min) / count
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10
  const start = Math.ceil(min / step) * step
  const out: number[] = []
  for (let v = start; v <= max; v += step) out.push(Number(v.toFixed(6)))
  return out
}

/**
 * Price history with the model's forecasts overlaid against what actually
 * happened. Predictions are drawn as a thin line so the eye compares paths
 * rather than reading them as prints.
 */
export function ForecastChart({
  candles,
  predictions,
}: {
  candles: Candle[]
  predictions: PredictionRow[]
}) {
  const shown = candles.slice(-140)
  const times = shown.map((c) => c.time)
  const index = new Map(times.map((t, i) => [t, i]))

  // Only keep predictions whose target lands inside the visible window.
  const visible = predictions.filter((p) => index.has(p.time))

  const lows = shown.map((c) => c.low)
  const highs = shown.map((c) => c.high)
  const predVals = visible.flatMap((p) => [p.pred_close - p.pred_std, p.pred_close + p.pred_std])
  const min = Math.min(...lows, ...predVals)
  const max = Math.max(...highs, ...predVals)
  const pad = (max - min) * 0.08

  const x = scaler([0, shown.length - 1], [PAD.left, W - PAD.right])
  const y = scaler([min - pad, max + pad], [H - PAD.bottom, PAD.top])

  const actualPath = shown
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(c.close).toFixed(1)}`)
    .join('')

  // Group predictions by anchor so each forecast draws as its own short path.
  const byAnchor = new Map<string, PredictionRow[]>()
  for (const p of visible) {
    const list = byAnchor.get(p.anchor) ?? []
    list.push(p)
    byAnchor.set(p.anchor, list)
  }

  const forecasts = [...byAnchor.values()]
    .filter((g) => g.length > 1)
    .map((group) => {
      const sorted = [...group].sort((a, b) => a.step - b.step)
      return sorted
        .map((p, i) => {
          const xi = x(index.get(p.time)!)
          return `${i === 0 ? 'M' : 'L'}${xi.toFixed(1)},${y(p.pred_close).toFixed(1)}`
        })
        .join('')
    })

  const ticks = niceTicks(min, max)
  const first = times[0]
  const last = times[times.length - 1]

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Share price from ${first} to ${last}, with ${byAnchor.size} model forecasts drawn over the actual closing price. The forecast paths track close to the actual line without leading it.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(t)}
              y2={y(t)}
              stroke="#2b303c"
              strokeWidth="1"
            />
            <text
              x={W - PAD.right + 8}
              y={y(t) + 3.5}
              fill="#6b7280"
              fontSize="11"
              fontFamily="var(--mono)"
            >
              {t.toFixed(0)}
            </text>
          </g>
        ))}

        <path d={actualPath} fill="none" stroke="#e8e6e0" strokeWidth="1.6" />

        {forecasts.map((d, i) => (
          <path key={i} d={d} fill="none" stroke="#c2703f" strokeWidth="1.1" opacity="0.85" />
        ))}

        <text x={PAD.left} y={H - 8} fill="#6b7280" fontSize="11" fontFamily="var(--mono)">
          {first}
        </text>
        <text
          x={W - PAD.right}
          y={H - 8}
          fill="#6b7280"
          fontSize="11"
          fontFamily="var(--mono)"
          textAnchor="end"
        >
          {last}
        </text>
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#e8e6e0' }} />
          Actual close
        </span>
        <span>
          <i style={{ background: '#c2703f' }} />
          Model forecast, {byAnchor.size} five-day paths
        </span>
      </div>
    </>
  )
}

/**
 * Rolling error against the persistence baseline, with the refits marked. The
 * two lines sitting on top of each other is the finding.
 */
export function ErrorTrendChart({ snapshot }: { snapshot: Snapshot }) {
  const trend = snapshot.backtest.error_trend
  const runs = snapshot.backtest.training_runs
  const h = 220
  const pad = { top: 16, right: 54, bottom: 26, left: 10 }

  const vals = trend.flatMap((t) => [t.mae, t.baseline_mae])
  const min = 0
  const max = Math.max(...vals) * 1.1

  const x = scaler([0, trend.length - 1], [pad.left, W - pad.right])
  const y = scaler([min, max], [h - pad.bottom, pad.top])

  const line = (key: 'mae' | 'baseline_mae') =>
    trend
      .map((t, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(t[key]).toFixed(1)}`)
      .join('')

  const periods = trend.map((t) => t.period)
  const refitMarks = runs
    .filter((r) => r.trigger === 'drift')
    .map((r) => {
      // Nearest month bucket to the refit date.
      const idx = periods.findIndex((p) => p >= r.trained_at)
      return idx >= 0 ? { x: x(idx), version: r.version } : null
    })
    .filter((v): v is { x: number; version: string } => v !== null)

  const ticks = niceTicks(min, max, 3)

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Monthly mean absolute error for the model and the persistence baseline from ${periods[0]} to ${periods[periods.length - 1]}. The two lines track each other closely throughout, with ${refitMarks.length} retraining events marked.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)} stroke="#2b303c" />
            <text
              x={W - pad.right + 8}
              y={y(t) + 3.5}
              fill="#6b7280"
              fontSize="11"
              fontFamily="var(--mono)"
            >
              {t}
            </text>
          </g>
        ))}

        {refitMarks.map((m, i) => (
          <line
            key={i}
            x1={m.x}
            x2={m.x}
            y1={pad.top}
            y2={h - pad.bottom}
            stroke="#6c5342"
            strokeWidth="1"
            strokeDasharray="2 3"
          />
        ))}

        <path d={line('baseline_mae')} fill="none" stroke="#6b7280" strokeWidth="1.4" />
        <path d={line('mae')} fill="none" stroke="#c2703f" strokeWidth="1.6" />

        <text x={pad.left} y={h - 8} fill="#6b7280" fontSize="11" fontFamily="var(--mono)">
          {periods[0]}
        </text>
        <text
          x={W - pad.right}
          y={h - 8}
          fill="#6b7280"
          fontSize="11"
          fontFamily="var(--mono)"
          textAnchor="end"
        >
          {periods[periods.length - 1]}
        </text>
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#c2703f' }} />
          Model MAE
        </span>
        <span>
          <i style={{ background: '#6b7280' }} />
          Persistence baseline MAE
        </span>
        <span>
          <i style={{ background: '#6c5342' }} />
          Retrain triggered
        </span>
      </div>
    </>
  )
}

/** Error by forecast horizon — uncertainty compounds with distance. */
export function HorizonChart({ snapshot }: { snapshot: Snapshot }) {
  const rows = snapshot.backtest.by_horizon
  const h = 190
  const pad = { top: 16, right: 54, bottom: 30, left: 10 }
  const max = Math.max(...rows.flatMap((r) => [r.mae, r.baseline_mae])) * 1.15

  const bandW = (W - pad.left - pad.right) / rows.length
  const y = scaler([0, max], [h - pad.bottom, pad.top])
  const ticks = niceTicks(0, max, 3)

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${h}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Mean absolute error by forecast horizon. Error grows from ${rows[0]?.mae} at one day ahead to ${rows[rows.length - 1]?.mae} at ${rows.length} days ahead, tracking the baseline at every step.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={W - pad.right} y1={y(t)} y2={y(t)} stroke="#2b303c" />
            <text
              x={W - pad.right + 8}
              y={y(t) + 3.5}
              fill="#6b7280"
              fontSize="11"
              fontFamily="var(--mono)"
            >
              {t}
            </text>
          </g>
        ))}

        {rows.map((r, i) => {
          const cx = pad.left + bandW * i + bandW / 2
          const bw = Math.min(38, bandW * 0.34)
          return (
            <g key={r.step}>
              <rect
                x={cx - bw - 2}
                y={y(r.mae)}
                width={bw}
                height={h - pad.bottom - y(r.mae)}
                fill="#c2703f"
              />
              <rect
                x={cx + 2}
                y={y(r.baseline_mae)}
                width={bw}
                height={h - pad.bottom - y(r.baseline_mae)}
                fill="#4b5162"
              />
              <text
                x={cx}
                y={h - 10}
                fill="#6b7280"
                fontSize="11"
                fontFamily="var(--mono)"
                textAnchor="middle"
              >
                {r.step}d
              </text>
            </g>
          )
        })}
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#c2703f' }} />
          Model
        </span>
        <span>
          <i style={{ background: '#4b5162' }} />
          Baseline
        </span>
      </div>
    </>
  )
}
