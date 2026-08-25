export function money(v: number | null | undefined, currency = 'USD', digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(v)
}

/** Revenue lines run to twelve figures; compact notation keeps tables readable. */
export function compactMoney(v: number | null | undefined, currency = 'USD'): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(v)
}

/** Takes a decimal (0.09) and renders a percentage (9.0%). */
export function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

/** Takes an already-scaled percentage (-2.87) and keeps its sign visible. */
export function signedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`
}

export function num(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

export function intWithCommas(v: number): string {
  return new Intl.NumberFormat('en-US').format(v)
}
