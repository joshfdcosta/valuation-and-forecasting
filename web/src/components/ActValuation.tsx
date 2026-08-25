import { useMemo, useState } from 'react'
import type { Snapshot } from '../lib/snapshot'
import { impliedWacc, sensitivity, value, type Assumptions } from '../lib/dcf'
import { compactMoney, money, pct, signedPct } from '../lib/format'
import { Explain, ReadThis } from './Explain'

const WACC_STEPS = [-0.02, -0.01, 0, 0.01, 0.02]
const GROWTH_STEPS = [-0.01, -0.005, 0, 0.005, 0.01]

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format: (v: number) => string
}

function Slider({ label, value, min, max, step, onChange, format }: SliderProps) {
  const id = label.replace(/\s+/g, '-').toLowerCase()
  return (
    <div className="control">
      <label htmlFor={id}>
        {label}
        <span className="val">{format(value)}</span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export function ActValuation({ snapshot }: { snapshot: Snapshot }) {
  const f = snapshot.fundamentals
  const d = snapshot.defaults

  const base: Assumptions = useMemo(
    () => ({
      revenueBase: f.revenue ?? 0,
      revenueGrowth: f.revenue_growth ?? 0.05,
      ebitMargin: f.ebit_margin ?? 0.15,
      taxRate: d.tax_rate,
      wacc: d.wacc,
      terminalGrowth: d.terminal_growth,
      forecastYears: d.forecast_years,
      capexPct: f.capex_pct,
      depreciationPct: f.depreciation_pct,
      nwcChangePct: f.nwc_change_pct,
    }),
    [f, d],
  )

  const [a, setA] = useState<Assumptions>(base)
  const set = <K extends keyof Assumptions>(key: K) => (v: Assumptions[K]) =>
    setA((prev) => ({ ...prev, [key]: v }))

  const netDebt = f.net_debt ?? 0
  const shares = f.shares_outstanding ?? 1
  const price = snapshot.company.current_price ?? 0
  const currency = snapshot.company.currency

  const v = useMemo(() => value(a, netDebt, shares), [a, netDebt, shares])
  const upside = price ? (v.fairValuePerShare / price - 1) * 100 : null
  const implied = useMemo(
    () => impliedWacc(a, netDebt, shares, price),
    [a, netDebt, shares, price],
  )

  const waccRow = WACC_STEPS.map((s) => a.wacc + s).filter((w) => w > 0)
  const growthCol = GROWTH_STEPS.map((s) => a.terminalGrowth + s).filter((g) => g >= 0)
  const grid = useMemo(
    () => sensitivity(a, waccRow, growthCol, netDebt, shares),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [a, netDebt, shares],
  )

  const dirty = JSON.stringify(a) !== JSON.stringify(base)

  return (
    <section id="valuation">
      <div className="shell">
        <div className="narrow">
          <p className="eyebrow">
            <span className="act-no">Act I</span> — Valuation
          </p>
          <h2>What is a company actually worth?</h2>
          <p className="lede" style={{ marginTop: '1.5rem' }}>
            A discounted cash flow model says a business is worth all the cash it will
            ever produce, converted into what that cash is worth to you today. Below is
            that model, running live on {snapshot.company.name}'s real filings. Move a
            slider and every number updates.
          </p>

          <Explain term="Why discount future cash at all">
            A pound next year is worth less than a pound today — you could have
            invested today's pound, and next year's might not arrive. So we shrink each
            future year's cash by a rate that reflects both. That rate is the{' '}
            <strong>discount rate</strong>, or WACC. A high rate means you think the
            business is risky and you demand more for holding it. It is the single most
            powerful input in the model.
          </Explain>

          <p>
            The forecast only runs {a.forecastYears} years, but companies do not stop
            existing in year {a.forecastYears}. Everything after that gets bundled into
            one number called the <strong>terminal value</strong>, calculated by
            assuming the business grows steadily forever at some modest rate. That
            single assumption usually carries most of the valuation — for this company
            it is {pct(v.terminalShare, 0)} of the total, which the panel below states
            outright rather than burying.
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Assumptions</span>
            {dirty ? (
              <button className="reset" onClick={() => setA(base)}>
                Reset to filings
              </button>
            ) : (
              <span className="panel-note">Capex, D&amp;A and ΔNWC pulled from the cash flow statement</span>
            )}
          </div>

          <div className="controls">
            <Slider
              label="Revenue growth"
              value={a.revenueGrowth}
              min={-0.05}
              max={0.3}
              step={0.005}
              onChange={set('revenueGrowth')}
              format={(x) => pct(x, 1)}
            />
            <Slider
              label="EBIT margin"
              value={a.ebitMargin}
              min={0.02}
              max={0.6}
              step={0.005}
              onChange={set('ebitMargin')}
              format={(x) => pct(x, 1)}
            />
            <Slider
              label="Tax rate"
              value={a.taxRate}
              min={0}
              max={0.45}
              step={0.005}
              onChange={set('taxRate')}
              format={(x) => pct(x, 1)}
            />
            <Slider
              label="WACC (discount rate)"
              value={a.wacc}
              min={0.04}
              max={0.2}
              step={0.0025}
              onChange={set('wacc')}
              format={(x) => pct(x, 2)}
            />
            <Slider
              label="Terminal growth"
              value={a.terminalGrowth}
              min={0}
              max={Math.max(0.005, a.wacc - 0.005)}
              step={0.0025}
              onChange={set('terminalGrowth')}
              format={(x) => pct(x, 2)}
            />
            <Slider
              label="Forecast years"
              value={a.forecastYears}
              min={3}
              max={10}
              step={1}
              onChange={set('forecastYears')}
              format={(x) => `${x}`}
            />
          </div>

          <div className="figures">
            <div>
              <div className="fig">{compactMoney(v.enterpriseValue, currency)}</div>
              <div className="cap">Enterprise value</div>
            </div>
            <div>
              <div className="fig">{money(v.fairValuePerShare, currency)}</div>
              <div className="cap">Fair value per share</div>
            </div>
            <div>
              <div className="fig">{money(price, currency)}</div>
              <div className="cap">Market price</div>
            </div>
            <div>
              <div className={`fig ${upside && upside >= 0 ? 'up' : 'down'}`}>
                {signedPct(upside, 1)}
              </div>
              <div className="cap">Implied over/undervaluation</div>
            </div>
            <div>
              <div className="fig">{pct(v.terminalShare, 0)}</div>
              <div className="cap">Of value in the terminal assumption</div>
            </div>
          </div>

          <ReadThis>
            Each column is one forecast year. Revenue grows, a margin turns it into
            operating profit, tax comes off, then the non-cash and cash items that the
            income statement misses: depreciation added back, capital spending and
            working capital taken out. What survives is <strong>free cash flow</strong>{' '}
            — the cash actually available to whoever owns the business. The last row
            shrinks each year back to today's money.
          </ReadThis>

          <div className="table-scroll">
            <table>
              <caption className="visually-hidden">Free cash flow forecast by year</caption>
              <thead>
                <tr>
                  <th scope="col">Line</th>
                  {v.forecast.map((r) => (
                    <th key={r.year} scope="col">
                      Year {r.year}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(
                  [
                    ['Revenue', 'revenue'],
                    ['EBIT', 'ebit'],
                    ['NOPAT', 'nopat'],
                    ['+ D&A', 'depreciation'],
                    ['− Capex', 'capex'],
                    ['− ΔNWC', 'nwcChange'],
                    ['Free cash flow', 'fcf'],
                    ['PV of FCF', 'pvFcf'],
                  ] as const
                ).map(([label, key]) => (
                  <tr key={key}>
                    <td className="label">{label}</td>
                    {v.forecast.map((r) => (
                      <td key={r.year}>{compactMoney(r[key], currency)}</td>
                    ))}
                  </tr>
                ))}
                <tr>
                  <td className="label">Discount factor</td>
                  {v.forecast.map((r) => (
                    <td key={r.year}>{r.discountFactor.toFixed(3)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>

          <ReadThis>
            The same company valued under twenty-five different pairs of assumptions.
            Rows vary the discount rate, columns vary how fast you think the business
            grows forever. The highlighted cell is the one the sliders are currently
            set to. Notice how far the corners are from each other — that spread is the
            honest uncertainty in any valuation.
          </ReadThis>

          <div className="table-scroll">
            <table className="sens">
              <caption className="visually-hidden">
                Fair value per share across discount rate and terminal growth
              </caption>
              <thead>
                <tr>
                  <th scope="col">WACC ╲ g</th>
                  {growthCol.map((g) => (
                    <th key={g} scope="col">
                      {pct(g, 2)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {waccRow.map((w, i) => (
                  <tr key={w}>
                    <td className="label">{pct(w, 2)}</td>
                    {growthCol.map((g, j) => {
                      const cell = grid[i][j]
                      const hit = Math.abs(w - a.wacc) < 1e-9 && Math.abs(g - a.terminalGrowth) < 1e-9
                      return (
                        <td key={g} className={hit ? 'hit' : undefined}>
                          {cell === null ? '—' : money(cell, currency, 0)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="verdict">
            <p className="q">
              {implied
                ? `At today's price the market is pricing in a ${pct(implied, 1)} required return. My model says ${pct(a.wacc, 1)}. The gap between those two numbers is the entire argument.`
                : `The market price sits outside the range this model can reach at any plausible discount rate — which is itself the finding.`}
            </p>
          </div>
        </div>

        <div className="narrow" style={{ marginTop: '2rem' }}>
          <p>
            That sensitivity grid is the honest part. A quarter-point on the discount
            rate moves fair value more than most earnings surprises do. Anyone who
            quotes a single DCF number without showing the grid behind it is selling
            precision they do not have.
          </p>
        </div>
      </div>
    </section>
  )
}
