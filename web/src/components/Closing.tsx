import { profile } from '../profile'
import type { Snapshot } from '../lib/snapshot'

export function Closing({ snapshot }: { snapshot: Snapshot }) {
  const cfg = snapshot.config

  return (
    <>
      <section id="about">
        <div className="shell">
          <div className="narrow">
            <p className="eyebrow">Who built this</p>
            <h2>{profile.name}</h2>
            <div style={{ marginTop: '1.5rem' }}>
              {profile.about.map((para, i) => (
                <p key={i} className={i === 0 ? 'lede' : undefined}>
                  {para}
                </p>
              ))}
            </div>
          </div>

          <div className="skill-cols">
            {profile.skills.map((col) => (
              <div key={col.group}>
                <h3>
                  {col.group}
                  <span className="skill-note">{col.note}</span>
                </h3>
                <ul>
                  {col.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="method">
        <div className="shell">
          <div className="narrow">
            <p className="eyebrow">How it is built</p>
            <h2>The receipts</h2>
            <p className="lede" style={{ marginTop: '1.5rem' }}>
              Nothing on this page is a mockup. The valuation runs in your browser from
              the equations. The forecasting numbers come from a real backtest that
              takes several minutes to compute.
            </p>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Architecture</span>
              <span className="panel-note">Static front end, offline research pipeline</span>
            </div>

            <div className="timeline">
              <div className="tl-row">
                <span className="when">Valuation</span>
                <span className="what">
                  Two-stage DCF with Gordon Growth terminal value, ported to TypeScript
                  so it runs live with no server
                </span>
                <span className="why">dcf.ts</span>
              </div>
              <div className="tl-row">
                <span className="when">Model</span>
                <span className="what">
                  {cfg.lookback}-day LSTM forecasting {cfg.horizon} candles, PyTorch,
                  Huber loss, early stopping, Monte Carlo dropout for uncertainty
                </span>
                <span className="why">lstm.py</span>
              </div>
              <div className="tl-row">
                <span className="when">Features</span>
                <span className="what">
                  {cfg.indicators.join(', ')}, all trailing-window, all computed
                  strictly before the anchor
                </span>
                <span className="why">features.py</span>
              </div>
              <div className="tl-row">
                <span className="when">Patterns</span>
                <span className="what">
                  Eleven candlestick patterns detected by vectorised rules, then tested
                  with a stationary block bootstrap and Holm-Bonferroni correction
                </span>
                <span className="why">patterns/</span>
              </div>
              <div className="tl-row">
                <span className="when">Backtest</span>
                <span className="what">
                  Walk-forward over {cfg.train_span}-window training spans, refitting
                  whenever measured error drifts past {cfg.drift_threshold_pct}%
                </span>
                <span className="why">build_snapshot.py</span>
              </div>
              <div className="tl-row">
                <span className="when">Storage</span>
                <span className="what">
                  Append-then-reconcile prediction ledger in SQLAlchemy. Forecasts are
                  written before outcomes exist and cannot be edited afterwards
                </span>
                <span className="why">db.py</span>
              </div>
              <div className="tl-row">
                <span className="when">Serving</span>
                <span className="what">
                  FastAPI for the live pipeline, plus a static snapshot so this page
                  needs no backend at all
                </span>
                <span className="why">main.py</span>
              </div>
            </div>
          </div>

          <div className="narrow">
            <div className="stack-note">
              <strong>What I would fix next.</strong> The exchange calendar is
              approximated with business days, which overcounts by roughly nine
              holidays a year. The model predicts each candle leg independently and can
              return an incoherent bar, so coherence is clamped after the fact rather
              than constrained in the architecture. And this is one ticker. There are
              no cross-sectional or sector features. None of that is hidden in the
              repository either.
            </div>
          </div>

          <div className="narrow" style={{ marginTop: '2rem' }}>
            <p>
              The full source, including the test suite that pins the valuation
              arithmetic against hand-computed Gordon Growth values, is on{' '}
              <a href={profile.repoUrl}>GitHub</a>.
            </p>
          </div>
        </div>
      </section>

      <footer>
        <div className="shell" style={{ display: 'contents' }}>
          <span>
            {profile.name} · Built with Python, PyTorch, TypeScript and React
          </span>
          <span className="mono">
            {snapshot.ticker} snapshot generated {snapshot.generated_at} · not investment advice
          </span>
        </div>
      </footer>
    </>
  )
}
