/**
 * Two-stage discounted cash flow — a direct port of src/valuation/dcf.py.
 *
 * It lives in the browser so the valuation is interactive with no server and no
 * latency. The Python original remains the reference implementation and is the
 * one covered by the test suite; tests/test_dcf.py pins the numbers this file
 * has to reproduce.
 */

export interface Assumptions {
  revenueBase: number
  revenueGrowth: number
  ebitMargin: number
  taxRate: number
  wacc: number
  terminalGrowth: number
  forecastYears: number
  capexPct: number
  depreciationPct: number
  nwcChangePct: number
}

export interface ForecastRow {
  year: number
  revenue: number
  ebit: number
  nopat: number
  depreciation: number
  capex: number
  nwcChange: number
  fcf: number
  discountFactor: number
  pvFcf: number
}

export interface Valuation {
  forecast: ForecastRow[]
  pvExplicit: number
  terminalValue: number
  pvTerminal: number
  enterpriseValue: number
  equityValue: number
  fairValuePerShare: number
  terminalShare: number
}

/** Gordon Growth terminal value as of the final explicit forecast year. */
export function terminalValue(
  finalFcf: number,
  wacc: number,
  terminalGrowth: number,
): number {
  return (finalFcf * (1 + terminalGrowth)) / (wacc - terminalGrowth)
}

export function projectFreeCashFlow(a: Assumptions): ForecastRow[] {
  const rows: ForecastRow[] = []
  for (let year = 1; year <= a.forecastYears; year++) {
    const revenue = a.revenueBase * Math.pow(1 + a.revenueGrowth, year)
    const ebit = revenue * a.ebitMargin
    const nopat = ebit * (1 - a.taxRate)
    const depreciation = revenue * a.depreciationPct
    const capex = revenue * a.capexPct
    const nwcChange = revenue * a.nwcChangePct
    // FCF = NOPAT + D&A - capex - change in net working capital.
    const fcf = nopat + depreciation - capex - nwcChange
    const discountFactor = 1 / Math.pow(1 + a.wacc, year)
    rows.push({
      year,
      revenue,
      ebit,
      nopat,
      depreciation,
      capex,
      nwcChange,
      fcf,
      discountFactor,
      pvFcf: fcf * discountFactor,
    })
  }
  return rows
}

export function value(
  a: Assumptions,
  netDebt: number,
  sharesOutstanding: number,
): Valuation {
  const forecast = projectFreeCashFlow(a)
  const last = forecast[forecast.length - 1]

  const tv = terminalValue(last.fcf, a.wacc, a.terminalGrowth)
  const pvTerminal = tv * last.discountFactor
  const pvExplicit = forecast.reduce((sum, r) => sum + r.pvFcf, 0)
  const enterpriseValue = pvExplicit + pvTerminal
  const equityValue = enterpriseValue - netDebt

  return {
    forecast,
    pvExplicit,
    terminalValue: tv,
    pvTerminal,
    enterpriseValue,
    equityValue,
    fairValuePerShare: equityValue / sharesOutstanding,
    // How much of the valuation rests on the terminal assumption rather than
    // the explicit forecast. Typically 60-80%, which is the honest caveat most
    // DCF write-ups leave out.
    terminalShare: pvTerminal / enterpriseValue,
  }
}

/** Fair value per share across a WACC x terminal-growth grid. */
export function sensitivity(
  a: Assumptions,
  waccRange: number[],
  growthRange: number[],
  netDebt: number,
  sharesOutstanding: number,
): (number | null)[][] {
  return waccRange.map((w) =>
    growthRange.map((g) => {
      // Gordon Growth is undefined once growth meets or exceeds the discount
      // rate; the cell is blank rather than a fabricated number.
      if (w <= g) return null
      return value({ ...a, wacc: w, terminalGrowth: g }, netDebt, sharesOutstanding)
        .fairValuePerShare
    }),
  )
}

/**
 * The discount rate at which the model's fair value equals the market price —
 * i.e. the return the market is implicitly demanding. Solved by bisection
 * because the closed form is unpleasant and this is exact enough.
 */
export function impliedWacc(
  a: Assumptions,
  netDebt: number,
  sharesOutstanding: number,
  marketPrice: number,
): number | null {
  let lo = a.terminalGrowth + 0.001
  let hi = 0.6

  const priceAt = (w: number) =>
    value({ ...a, wacc: w }, netDebt, sharesOutstanding).fairValuePerShare

  if (priceAt(lo) < marketPrice) return null // even the lowest rate undervalues it
  if (priceAt(hi) > marketPrice) return null

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2
    if (priceAt(mid) > marketPrice) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}
