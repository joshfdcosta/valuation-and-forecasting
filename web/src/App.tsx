import { useEffect, useState } from 'react'
import { loadSnapshot, type Snapshot } from './lib/snapshot'
import { Hero } from './components/Hero'
import { ActValuation } from './components/ActValuation'
import { ActPrediction } from './components/ActPrediction'
import { ActLoop } from './components/ActLoop'
import { Closing } from './components/Closing'

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSnapshot()
      .then(setSnapshot)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  if (error) {
    return (
      <main className="state">
        <p className="eyebrow">Snapshot unavailable</p>
        <p>{error}</p>
        <p style={{ fontSize: '0.85rem' }}>
          Regenerate it with <code>python -m scripts.build_snapshot AAPL</code>
        </p>
      </main>
    )
  }

  if (!snapshot) {
    return (
      <main className="state" aria-busy="true">
        <p className="eyebrow">Loading backtest</p>
      </main>
    )
  }

  return (
    <>
      <a className="skip" href="#valuation">
        Skip to the work
      </a>
      <Hero snapshot={snapshot} />
      <main id="main">
        <ActValuation snapshot={snapshot} />
        <ActPrediction snapshot={snapshot} />
        <ActLoop snapshot={snapshot} />
        <Closing snapshot={snapshot} />
      </main>
    </>
  )
}
