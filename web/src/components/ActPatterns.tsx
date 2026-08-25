import { useState } from 'react'
import type { PatternResult, Snapshot } from '../lib/snapshot'
import { Explain, ReadThis } from './Explain'
import { intWithCommas, num, signedPct } from '../lib/format'

/**
 * Edge by pattern at the selected horizon, drawn as deviation from zero.
 * Everything sits inside the band of what chance produces, which is the point.
 */
function EdgeChart({
  rows,
  meta,
}: {
  rows: PatternResult[]
  meta: Snapshot['patterns']['meta']
}) {
  const W = 900
  const H = 260
  const pad = { top: 18, right: 54, bottom: 58, left: 10 }

  const edges = rows.map((r) => r.edge_pct ?? 0)
  const bound = Math.max(...edges.map(Math.abs), 1) * 1.25
  const band = (W - pad.left - pad.right) / rows.length

  const y = (v: number) =>
    pad.top + ((bound - v) / (2 * bound)) * (H - pad.top - pad.bottom)

  const ticks = [-bound, -bound / 2, 0, bound / 2, bound].map(
    (t) => Number(t.toFixed(2)),
  )

  return (
    <>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Average excess return after each candlestick pattern, compared with the typical return over the same period. All ${rows.length} patterns sit close to zero and none reaches statistical significance.`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={pad.left}
              x2={W - pad.right}
              y1={y(t)}
              y2={y(t)}
              stroke={t === 0 ? '#4b5162' : '#2b303c'}
            />
            <text
              x={W - pad.right + 8}
              y={y(t) + 3.5}
              fill="#6b7280"
              fontSize="11"
              fontFamily="var(--mono)"
            >
              {t > 0 ? `+${t}` : t}
            </text>
          </g>
        ))}

        {rows.map((r, i) => {
          const cx = pad.left + band * i + band / 2
          const bw = Math.min(30, band * 0.42)
          const edge = r.edge_pct ?? 0
          const top = Math.min(y(edge), y(0))
          const height = Math.abs(y(edge) - y(0))
          const label = meta[r.pattern]?.label ?? r.pattern
          return (
            <g key={r.pattern}>
              <rect
                x={cx - bw / 2}
                y={top}
                width={bw}
                height={Math.max(height, 1)}
                fill={r.significant ? '#c2703f' : '#4b5162'}
              />
              <text
                x={cx}
                y={H - pad.bottom + 16}
                fill="#6b7280"
                fontSize="10"
                fontFamily="var(--sans)"
                textAnchor="end"
                transform={`rotate(-40 ${cx} ${H - pad.bottom + 16})`}
              >
                {label}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: '#4b5162' }} />
          Not distinguishable from chance
        </span>
        <span>
          <i style={{ background: '#c2703f' }} />
          Survives correction for multiple testing
        </span>
      </div>
    </>
  )
}

export function ActPatterns({ snapshot }: { snapshot: Snapshot }) {
  const p = snapshot.patterns
  const [horizon, setHorizon] = useState(p.horizons.includes(5) ? 5 : p.horizons[0])

  const tested = p.results.filter((r) => !r.skipped)
  const atHorizon = tested
    .filter((r) => r.horizon === horizon)
    .sort((a, b) => (b.edge_pct ?? 0) - (a.edge_pct ?? 0))

  const skipped = p.results.filter((r) => r.skipped && r.horizon === horizon)
  const bestRaw = [...tested].sort((a, b) => (a.p_value ?? 1) - (b.p_value ?? 1))[0]
  const naivePasses = tested.filter((r) => (r.p_value ?? 1) < p.alpha).length

  return (
    <section id="patterns">
      <div className="shell">
        <div className="narrow">
          <p className="eyebrow">
            <span className="act-no">Act III</span> — Candlestick patterns
          </p>
          <h2>Do the shapes traders swear by actually predict anything?</h2>
          <p className="lede" style={{ marginTop: '1.5rem' }}>
            Technical analysis holds that certain candle shapes — a hammer, an
            engulfing candle, a morning star — signal what price does next. They have
            names, centuries of history, and enormous conviction behind them. They are
            also a testable claim, so I tested them.
          </p>

          <Explain term="What a candlestick actually shows">
            One candle is one day. The thick body spans the opening and closing price;
            the thin wicks reach to the high and low. A tall body means the price moved
            decisively; a long wick means it went somewhere and came back. Patterns are
            named combinations of one to three of these shapes.
          </Explain>

          <p>
            I detected {Object.keys(p.meta).length} classic patterns across{' '}
            {intWithCommas(p.n_sessions)} trading sessions, then measured what the price
            did over the following {p.horizons.join(', ')} days after each occurrence —
            and compared that with what the price did on a normal day over the same
            period. If a pattern predicts nothing, those two numbers should match.
          </p>

          <Explain term="Why testing eleven things at once is dangerous">
            Test enough ideas and one will look brilliant by pure luck. With{' '}
            {p.n_tests} tests, roughly {Math.round(p.n_tests * p.alpha)} should cross
            the usual "statistically significant" line by chance alone, even if every
            pattern is worthless. Reporting that winner is how technical-analysis edges
            get manufactured. The{' '}
            <strong>{p.correction.replace('-', '–')}</strong> correction raises the bar
            in proportion to how many things you tried, so luck stops qualifying.
          </Explain>

          <Explain term="Why not a standard significance test">
            The usual t-test assumes each measurement is independent. These are not — a
            five-day return starting today overlaps four days with tomorrow's, so the
            data repeats itself and a t-test reads that repetition as extra evidence.
            Instead I use a <strong>{p.bootstrap.method}</strong>: resampling
            contiguous blocks of real trading days {intWithCommas(p.bootstrap.n_resamples)}{' '}
            times to build a picture of what chance alone produces.
          </Explain>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Excess return after each pattern</span>
            <div className="horizon-switch" role="group" aria-label="Forecast horizon">
              {p.horizons.map((h) => (
                <button
                  key={h}
                  className={h === horizon ? 'active' : undefined}
                  onClick={() => setHorizon(h)}
                  aria-pressed={h === horizon}
                >
                  {h}d
                </button>
              ))}
            </div>
          </div>

          <EdgeChart rows={atHorizon} meta={p.meta} />

          <ReadThis>
            Each bar is how much better or worse the price did over the following{' '}
            {horizon} days after that pattern, compared with a typical {horizon}-day
            stretch. A real edge would be a tall bar that stays tall as you switch
            horizons. Bars near the zero line are noise wearing a name.
          </ReadThis>

          <div className="table-scroll">
            <table>
              <caption className="visually-hidden">
                Candlestick pattern results at {horizon} day horizon
              </caption>
              <thead>
                <tr>
                  <th scope="col">Pattern</th>
                  <th scope="col">Bias</th>
                  <th scope="col">Occurrences</th>
                  <th scope="col">Avg return</th>
                  <th scope="col">Normal day</th>
                  <th scope="col">Edge</th>
                  <th scope="col">Up rate</th>
                  <th scope="col">p</th>
                  <th scope="col">Survives</th>
                </tr>
              </thead>
              <tbody>
                {atHorizon.map((r) => {
                  const m = p.meta[r.pattern]
                  return (
                    <tr key={r.pattern}>
                      <td className="label">{m?.label ?? r.pattern}</td>
                      <td style={{ opacity: 0.7 }}>{m?.bias}</td>
                      <td>{r.n}</td>
                      <td>{signedPct(r.mean_return_pct, 2)}</td>
                      <td style={{ opacity: 0.7 }}>
                        {signedPct(r.baseline_return_pct, 2)}
                      </td>
                      <td className={(r.edge_pct ?? 0) >= 0 ? 'up' : 'down'}>
                        {signedPct(r.edge_pct, 2)}
                      </td>
                      <td>{num(r.hit_rate_pct, 1)}%</td>
                      <td style={{ opacity: 0.7 }}>{num(r.p_value, 3)}</td>
                      <td className={r.significant ? 'up' : 'down'}>
                        {r.significant ? 'yes' : 'no'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {skipped.length > 0 ? (
            <p style={{ marginTop: '1.1rem', fontSize: '0.8rem' }}>
              Excluded for having fewer than {p.min_occurrences} occurrences:{' '}
              {skipped.map((r) => p.meta[r.pattern]?.label ?? r.pattern).join(', ')}.
              Too rare to say anything about, so nothing is claimed.
            </p>
          ) : null}

          <div className="verdict">
            <p className="q">
              {p.n_significant === 0
                ? naivePasses > 0
                  ? `${p.n_tests} tests. ${naivePasses} would have passed a naive significance test. Not one survives once you account for having tried ${p.n_tests} of them.`
                  : `${p.n_tests} tests across ${Object.keys(p.meta).length} patterns, and not one of them beats chance.`
                : `${p.n_significant} of ${p.n_tests} tests survive correction. That is worth a second look, and a second dataset, before anyone calls it an edge.`}
            </p>
          </div>
        </div>

        <div className="narrow" style={{ marginTop: '2.5rem' }}>
          <p>
            {p.n_significant === 0 ? (
              <>
                Nothing here beats chance. The strongest single result was{' '}
                {p.meta[bestRaw?.pattern]?.label ?? bestRaw?.pattern} at{' '}
                {bestRaw?.horizon} days, with a p-value of {num(bestRaw?.p_value, 3)} —
                and even that is what you would expect to see somewhere among{' '}
                {p.n_tests} attempts on data with no signal at all.
              </>
            ) : (
              <>
                A result surviving correction is not the same as a tradeable edge. It
                would still need to hold on other tickers, in other periods, and after
                the costs of actually trading it.
              </>
            )}
          </p>
          <p>
            This is the same conclusion Act II reached by a completely different route.
            A neural network with sixty days of history could not beat assuming no
            change; centuries of chart-reading tradition cannot either. Two independent
            methods, one answer — which is a lot more convincing than either on its own.
          </p>
          <p>
            None of that makes candlesticks useless. They are a genuinely good way to
            read what <em>has</em> happened at a glance. The claim being tested here is
            narrower and much stronger: that the shape tells you what happens{' '}
            <em>next</em>. On this evidence, it does not.
          </p>
        </div>
      </div>
    </section>
  )
}
