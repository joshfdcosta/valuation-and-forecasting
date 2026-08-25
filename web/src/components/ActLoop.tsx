import type { Snapshot } from '../lib/snapshot'
import { ErrorTrendChart } from './Charts'
import { money, num, signedPct } from '../lib/format'
import { Explain, ReadThis } from './Explain'

export function ActLoop({ snapshot }: { snapshot: Snapshot }) {
  const runs = snapshot.backtest.training_runs
  const cfg = snapshot.config
  const currency = snapshot.company.currency
  const drifts = runs.filter((r) => r.trigger === 'drift')
  // Show whole forecast runs rather than an arbitrary tail slice. Cutting
  // mid-anchor leaves the dates jumping backwards, which reads as a bug.
  const all = snapshot.backtest.predictions
  const anchors = [...new Set(all.map((p) => p.anchor))].slice(-3)
  const recent = all
    .filter((p) => anchors.includes(p.anchor))
    .sort((a, b) => a.anchor.localeCompare(b.anchor) || a.step - b.step)

  const span = snapshot.backtest.span
  const years = (
    (new Date(span.to).getTime() - new Date(span.from).getTime()) /
    (365.25 * 24 * 3600 * 1000)
  ).toFixed(1)

  return (
    <section id="loop">
      <div className="shell">
        <div className="narrow">
          <p className="eyebrow">
            <span className="act-no">Act IV</span> · The loop
          </p>
          <h2>A model that grades itself and starts over</h2>
          <p className="lede" style={{ marginTop: '1.5rem' }}>
            A forecast written down before the outcome exists is a testable claim. A
            forecast written afterwards is a story. The whole system is built around
            keeping the first kind and making the second kind impossible.
          </p>
          <p>
            Every prediction is committed to the database with an empty actuals column.
            When the day closes, a second pass fills in what happened and scores
            the error. Nothing can be edited afterwards.
          </p>

          <Explain term="Drift, and why models go stale">
            A model learns the patterns in the period it was trained on. Markets change:
            a calm year becomes a volatile one, interest rates move, the crowd changes
            its mind. When that happens the old patterns stop applying and the model
            quietly gets worse. That decay is called <strong>drift</strong>. This system
            watches its own recent error and, when it runs more than{' '}
            {cfg.drift_threshold_pct}% worse than it managed at training time, throws
            the model away and fits a new one on recent data.
          </Explain>

          <p>
            That is the part employers actually care about. Training a model once is a
            coursework exercise. Keeping one honest in production, noticing it has gone
            stale and doing something about it without a human watching, is the job.
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Error over time, with retraining events</span>
            <span className="panel-note">
              {runs.length} fits · {drifts.length} triggered by drift
            </span>
          </div>

          <ErrorTrendChart snapshot={snapshot} />

          <ReadThis>
            Both lines are average error per month: orange is the model, grey is the
            do-nothing baseline. Lower is better. The dotted verticals mark the moments
            drift crossed the threshold and the model was rebuilt. The two lines sitting
            on top of each other for two and a half years is the whole story.
          </ReadThis>

          <div className="verdict">
            <p className="q">
              {runs.length} fits across {years} years, and the model line never
              separates from the baseline. The loop works exactly as designed. What it
              keeps discovering is that there is no edge to hold on to.
            </p>
          </div>
        </div>

        <div className="narrow" style={{ marginTop: '2.5rem' }}>
          <p>
            I chose periodic refits on a rolling window over true online learning. Per
            sample weight updates on data this noisy drift into catastrophic forgetting,
            and there is no clean way to roll a bad update back. A refit is
            reproducible, auditable, and revertible to any previous artefact, which
            matters more than elegance when the thing you are building has to be
            trusted.
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Retraining history</span>
            <span className="panel-note">Every fit, and what retired it</span>
          </div>

          <div className="timeline">
            {runs.map((r) => (
              <div className="tl-row" key={r.version}>
                <span className="when">{r.trained_at}</span>
                <span className="what">
                  <span className={`tag ${r.trigger}`}>{r.trigger}</span>
                  {r.version} fitted on {r.n_samples} windows
                </span>
                <span className="why">
                  {r.retired_drift_pct
                    ? `retired · drift ${signedPct(r.retired_drift_pct, 0)}`
                    : 'active'}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Prediction ledger</span>
            <span className="panel-note">
              Last {anchors.length} forecast runs, scored against what printed
            </span>
          </div>

          <ReadThis>
            Each row is one forecast, written down before the outcome existed.{' '}
            <em>Forecast from</em> is the day it was made, <em>target</em> the day it
            was about. The last column asks the only question that matters: did the
            model land closer than simply assuming no change?
          </ReadThis>

          <div className="table-scroll">
            <table>
              <caption className="visually-hidden">
                Recent predictions with model forecast, baseline, actual close and error
              </caption>
              <thead>
                <tr>
                  <th scope="col">Forecast from</th>
                  <th scope="col">Target</th>
                  <th scope="col">Step</th>
                  <th scope="col">Model</th>
                  <th scope="col">Baseline</th>
                  <th scope="col">Actual</th>
                  <th scope="col">Model err</th>
                  <th scope="col">Base err</th>
                  <th scope="col">Won</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((p) => {
                  const won = p.abs_error < p.baseline_abs_error
                  return (
                    <tr key={`${p.anchor}-${p.step}`}>
                      <td className="label">{p.anchor}</td>
                      <td>{p.time}</td>
                      <td>{p.step}</td>
                      <td>
                        {num(p.pred_close)} <span style={{ opacity: 0.5 }}>±{num(p.pred_std)}</span>
                      </td>
                      <td style={{ opacity: 0.7 }}>{num(p.baseline_close)}</td>
                      <td>{num(p.actual_close)}</td>
                      <td className={won ? 'up' : 'down'}>{num(p.abs_error)}</td>
                      <td style={{ opacity: 0.7 }}>{num(p.baseline_abs_error)}</td>
                      <td className={won ? 'up' : 'down'}>{won ? 'yes' : 'no'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <p style={{ marginTop: '1.2rem', fontSize: '0.82rem' }}>
            Mean absolute error across the full backtest: {money(snapshot.backtest.metrics.mae, currency)}{' '}
            for the model, {money(snapshot.backtest.metrics.baseline_mae, currency)} for
            doing nothing at all.
          </p>
        </div>
      </div>
    </section>
  )
}
