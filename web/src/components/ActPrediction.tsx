import type { Snapshot } from '../lib/snapshot'
import { ForecastChart, HorizonChart } from './Charts'
import { intWithCommas, money, num, signedPct } from '../lib/format'
import { Explain, ReadThis } from './Explain'

export function ActPrediction({ snapshot }: { snapshot: Snapshot }) {
  const m = snapshot.backtest.metrics
  const span = snapshot.backtest.span
  const cfg = snapshot.config
  const currency = snapshot.company.currency
  const beat = m.skill_vs_baseline_pct > 0

  return (
    <section id="prediction">
      <div className="shell">
        <div className="narrow">
          <p className="eyebrow">
            <span className="act-no">Act II</span> — Prediction
          </p>
          <h2>Can a neural network forecast a share price?</h2>
          <p className="lede" style={{ marginTop: '1.5rem' }}>
            I trained a neural network on {cfg.lookback} days of price history and asked
            it to forecast the next {cfg.horizon}. Then I tested it the only way that
            counts, across {span.from} to {span.to}, never letting it see a single day
            beyond the one it was predicting from.
          </p>

          <Explain term="What an LSTM is">
            A type of neural network built for sequences. Ordinary networks look at one
            snapshot; an LSTM reads a run of days in order and carries a memory of what
            came earlier, so it can pick up on patterns that unfold over time. That
            makes it a reasonable choice for prices — and a fair test of whether such
            patterns exist at all.
          </Explain>

          <Explain term="Why you need a baseline">
            "The model was 97% accurate" means nothing on its own. Accurate compared to
            what? The honest comparison is <strong>persistence</strong>: just guess that
            tomorrow's price equals today's. It is free, needs no model, and on a liquid
            share it is very hard to beat, because prices move mostly on news nobody had
            yesterday. Any forecast that cannot clear that bar has learned nothing —
            even if its raw error looks small.
          </Explain>

          <Explain term="Why walk-forward testing">
            The tempting mistake is to train on all your data and then test on a slice
            of it. The model has already seen the answers, so it scores brilliantly and
            fails in real life. Walking forward means the model is only ever trained on
            days <em>before</em> the one it is predicting — the same information it
            would have had at the time. It is the difference between a real track record
            and marking your own homework.
          </Explain>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Walk-forward backtest · {snapshot.ticker}</span>
            <span className="panel-note">
              {intWithCommas(m.n)} out-of-sample predictions
            </span>
          </div>

          <ForecastChart
            candles={snapshot.candles}
            predictions={snapshot.backtest.predictions}
          />

          <ReadThis>
            The white line is what the share price actually did. Each short orange line
            is one five-day forecast, drawn starting from the day it was made. Look at
            where they point: they follow the white line rather than leading it. The
            model is describing where the price has just been, not where it is going.
          </ReadThis>

          <div className="figures">
            <div>
              <div className="fig">{num(m.direction_accuracy_pct, 2)}%</div>
              <div className="cap">Direction accuracy</div>
            </div>
            <div>
              <div className="fig">{money(m.mae, currency)}</div>
              <div className="cap">Model mean absolute error</div>
            </div>
            <div>
              <div className="fig">{money(m.baseline_mae, currency)}</div>
              <div className="cap">Persistence baseline error</div>
            </div>
            <div>
              <div className={`fig ${beat ? 'up' : 'down'}`}>
                {signedPct(m.skill_vs_baseline_pct, 2)}
              </div>
              <div className="cap">Skill versus baseline</div>
            </div>
            <div>
              <div className="fig">{num(m.mape_pct, 2)}%</div>
              <div className="cap">Mean absolute percentage error</div>
            </div>
          </div>

          <ReadThis>
            <strong>Direction accuracy</strong> is how often the model got up-or-down
            right — 50% is a coin flip.{' '}
            <strong>Mean absolute error</strong> is how far off it was in dollars, on
            average. <strong>Skill versus baseline</strong> is the only one that
            settles the argument: positive means the model beat just guessing today's
            price, negative means it did not.
          </ReadThis>

          <div className="verdict">
            <p className="q">
              {intWithCommas(m.n)} predictions. Direction called correctly{' '}
              {num(m.direction_accuracy_pct, 2)}% of the time. That is a coin flip,
              measured to two decimal places.
            </p>
          </div>
        </div>

        <div className="narrow" style={{ marginTop: '2.5rem' }}>
          <p>
            The model lost to persistence by {Math.abs(m.skill_vs_baseline_pct).toFixed(2)}%.
            I could have buried that. Instead it is the headline, because the finding is
            the point: on daily equity closes the efficient market hypothesis is not a
            textbook abstraction, it is a wall you hit at{' '}
            {num(m.direction_accuracy_pct, 2)}%.
          </p>
          <p>
            What the model <em>did</em> learn is the shape of volatility. Error grows
            with horizon exactly as it should — uncertainty compounds the further out
            you forecast — and it tracks the baseline at every step rather than
            diverging. That is a model correctly learning that there is little to learn.
          </p>
        </div>

        <div className="panel">
          <div className="panel-head">
            <span className="panel-title">Error by forecast horizon</span>
            <span className="panel-note">Uncertainty compounds with distance</span>
          </div>
          <HorizonChart snapshot={snapshot} />

          <ReadThis>
            Orange is the model, grey is the do-nothing baseline, grouped by how many
            days ahead the forecast was. Both bars grow together as you look further
            out, and the orange never gets meaningfully shorter than the grey. Growing
            error is correct behaviour — the future gets harder to see. Failing to beat
            grey at any horizon is the finding.
          </ReadThis>
        </div>

        <div className="narrow" style={{ marginTop: '2.5rem' }}>
          <p>
            Getting to a trustworthy negative result is harder than getting to a
            flattering positive one. Three decisions did the work. Targets are
            fractional offsets from the last close rather than raw prices, so the model
            cannot cheat by memorising price levels. Splits are strictly chronological,
            because shuffling a time series lets a model train on Thursday to predict
            Wednesday. And the scaler is fitted on training data only, so the test set's
            mean never leaks backwards.
          </p>
          <p>
            Remove any one of those and this page could have claimed 90% accuracy. It
            would have been fiction.
          </p>
        </div>
      </div>
    </section>
  )
}
