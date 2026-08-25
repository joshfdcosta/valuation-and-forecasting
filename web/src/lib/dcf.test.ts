import { describe, expect, it } from 'vitest'
import { impliedWacc, projectFreeCashFlow, sensitivity, terminalValue, value, type Assumptions } from './dcf'

/**
 * These numbers are not invented. They are the output of the Python reference
 * implementation (src/valuation/dcf.py) run on the exact fundamentals stored in
 * public/snapshot.json. If this file and the Python suite ever disagree, the
 * browser is lying to visitors about the valuation and one of the two is wrong.
 */
const SNAPSHOT_INPUTS: Assumptions = {
  revenueBase: 466822987776.0,
  revenueGrowth: 0.164,
  ebitMargin: 0.32623002,
  taxRate: 0.21,
  wacc: 0.09,
  terminalGrowth: 0.025,
  forecastYears: 5,
  capexPct: 0.027237304787786406,
  depreciationPct: 0.025058748832679938,
  nwcChangePct: 0.05355348955522298,
}
const NET_DEBT = 21944995840.0
const SHARES = 14594180000.0

describe('parity with the Python reference implementation', () => {
  const v = value(SNAPSHOT_INPUTS, NET_DEBT, SHARES)

  it('reproduces enterprise value', () => {
    expect(v.enterpriseValue).toBeCloseTo(2641657706132.17, -3)
  })

  it('reproduces fair value per share', () => {
    expect(v.fairValuePerShare).toBeCloseTo(179.5039, 3)
  })

  it('reproduces the terminal value share', () => {
    expect(v.terminalShare).toBeCloseTo(0.7817, 4)
  })
})

describe('gordon growth', () => {
  it('matches a hand-computed terminal value', () => {
    // 100 * 1.025 / (0.09 - 0.025) = 1576.923...
    expect(terminalValue(100, 0.09, 0.025)).toBeCloseTo(1576.9231, 4)
  })

  it('blanks cells where growth meets or exceeds the discount rate', () => {
    const grid = sensitivity(SNAPSHOT_INPUTS, [0.03], [0.02, 0.03, 0.04], NET_DEBT, SHARES)
    expect(grid[0][0]).not.toBeNull()
    expect(grid[0][1]).toBeNull() // w === g
    expect(grid[0][2]).toBeNull() // g > w
  })
})

describe('forecast mechanics', () => {
  it('emits one row per forecast year with decaying discount factors', () => {
    const rows = projectFreeCashFlow(SNAPSHOT_INPUTS)
    expect(rows).toHaveLength(5)
    for (let i = 1; i < rows.length; i++) {
      expect(rows[i].discountFactor).toBeLessThan(rows[i - 1].discountFactor)
      expect(rows[i].revenue).toBeGreaterThan(rows[i - 1].revenue)
    }
  })

  it('builds free cash flow as NOPAT plus D&A less capex and working capital', () => {
    const r = projectFreeCashFlow(SNAPSHOT_INPUTS)[0]
    expect(r.fcf).toBeCloseTo(r.nopat + r.depreciation - r.capex - r.nwcChange, 6)
  })

  it('nets debt out of equity value', () => {
    const withDebt = value(SNAPSHOT_INPUTS, 1e9, 1e6)
    const without = value(SNAPSHOT_INPUTS, 0, 1e6)
    expect(withDebt.fairValuePerShare).toBeCloseTo(without.fairValuePerShare - 1000, 6)
  })
})

describe('sensitivity direction', () => {
  it('falls as the discount rate rises and rises as terminal growth rises', () => {
    const grid = sensitivity(
      SNAPSHOT_INPUTS,
      [0.08, 0.09, 0.1],
      [0.02, 0.025, 0.03],
      NET_DEBT,
      SHARES,
    )
    expect(grid[0][1]!).toBeGreaterThan(grid[1][1]!)
    expect(grid[1][1]!).toBeGreaterThan(grid[2][1]!)
    expect(grid[1][2]!).toBeGreaterThan(grid[1][1]!)
    expect(grid[1][1]!).toBeGreaterThan(grid[1][0]!)
  })
})

describe('implied discount rate', () => {
  it('recovers the rate that reproduces a given market price', () => {
    const target = value({ ...SNAPSHOT_INPUTS, wacc: 0.075 }, NET_DEBT, SHARES)
      .fairValuePerShare
    const solved = impliedWacc(SNAPSHOT_INPUTS, NET_DEBT, SHARES, target)
    expect(solved).toBeCloseTo(0.075, 4)
  })

  it('returns null when no rate in range can reach the price', () => {
    expect(impliedWacc(SNAPSHOT_INPUTS, NET_DEBT, SHARES, 1e9)).toBeNull()
  })
})
