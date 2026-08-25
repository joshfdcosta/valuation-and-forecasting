import { profile } from '../profile'
import type { Snapshot } from '../lib/snapshot'
import { intWithCommas, num, signedPct } from '../lib/format'

export function Hero({ snapshot }: { snapshot: Snapshot }) {
  const m = snapshot.backtest.metrics
  const p = snapshot.patterns
  const runs = snapshot.backtest.training_runs.length

  return (
    <header className="hero">
      <div className="shell">
        <p className="hero-name">
          <strong>{profile.name}</strong> · {profile.role}
        </p>

        <h1>
          I built a valuation model and a forecasting model, then tested whether
          either of them deserves to be believed.
        </h1>

        <p className="lede hero-thesis">{profile.thesis}</p>

        <p style={{ maxWidth: '46ch', marginTop: '1rem', fontSize: '0.95rem' }}>
          Everything below is live or measured — the valuation runs in your browser as
          you move it, and the forecasting numbers come from a real backtest. If you
          know roughly what a cash flow and a neural network are, you know enough; the
          rest is explained as it comes up.
        </p>

        <nav className="hero-links" aria-label="Contact">
          {profile.links.map((l) => (
            <a key={l.label} href={l.href}>
              {l.label}
            </a>
          ))}
        </nav>

        <div className="headline-figures">
          <div>
            <div className="fig">{intWithCommas(m.n)}</div>
            <div className="cap">Out-of-sample forecasts, walked forward</div>
          </div>
          <div>
            <div className="fig">{num(m.direction_accuracy_pct, 2)}%</div>
            <div className="cap">
              Direction called correctly — a coin flip pays {num(50, 0)}%
            </div>
          </div>
          <div>
            <div className={`fig ${m.skill_vs_baseline_pct > 0 ? 'up' : 'down'}`}>
              {signedPct(m.skill_vs_baseline_pct, 2)}
            </div>
            <div className="cap">Skill against a do-nothing baseline</div>
          </div>
          <div>
            <div className="fig">
              {p.n_significant}
              <span style={{ opacity: 0.45 }}>/{p.n_tests}</span>
            </div>
            <div className="cap">
              Candlestick patterns beating chance, once corrected
            </div>
          </div>
          <div>
            <div className="fig">{runs}</div>
            <div className="cap">Automatic retrains triggered by drift</div>
          </div>
        </div>
      </div>
    </header>
  )
}
