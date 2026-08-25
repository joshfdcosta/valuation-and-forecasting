/**
 * ─────────────────────────────────────────────────────────────────────────
 *  Personal details. This is the only file on the site that contains them.
 * ─────────────────────────────────────────────────────────────────────────
 */

export const profile = {
  name: "Joshua D'costa",
  role: 'Third year BSc Accounting and Finance · University of Birmingham Dubai',

  // The claim the whole page exists to support.
  thesis:
    'I build the finance and the machine learning myself, and I report what the numbers actually say, including when they say my model lost.',

  links: [
    { label: 'LinkedIn', href: 'https://www.linkedin.com/in/joshuafdcosta/' },
    { label: 'GitHub', href: 'https://github.com/joshfdcosta/valuation-and-forecasting' },
    { label: 'Email', href: 'mailto:joshfdcosta@gmail.com' },
  ],

  about: [
    'I am in my third year of a BSc in Accounting and Finance at the University of Birmingham Dubai. Corporate finance is where I am strongest and where my interest actually sits: how a business is valued, where its cash genuinely comes from, and what a discount rate is saying about risk. Management accounting is the other half of that, the internal numbers that decide what a company does next, rather than the ones it reports afterwards.',
    'Alongside that I follow markets, and I am interested in the statistics underneath them: trading, financial analysis, and what data can and cannot tell you about what happens next.',
    'This site is where those two interests meet. A discounted cash flow model I wrote from the equations rather than a template, and a machine learning system that forecasts share prices, scores itself against what actually happened, and retrains when it drifts. The valuation works. The forecasting does not beat a coin flip, and I have left that finding exactly where you can see it. Knowing which of the two you can trust is the skill worth having.',
  ],

  skills: [
    {
      group: 'Corporate finance',
      note: 'Where I am strongest',
      items: [
        'Discounted cash flow valuation',
        'Gordon Growth and terminal value',
        'Free cash flow built from the cash flow statement',
        'WACC, discount rates and sensitivity analysis',
        'Management accounting',
        'Financial statement analysis',
      ],
    },
    {
      group: 'AI and machine learning',
      note: 'Forecasting, and knowing when not to trust it',
      items: [
        'Time series forecasting with LSTMs (PyTorch)',
        'Walk-forward backtesting without lookahead',
        'Monte Carlo methods for uncertainty',
        'Baseline-relative evaluation: testing against doing nothing',
        'Leakage-free feature engineering',
        'Drift detection and scheduled retraining',
      ],
    },
    {
      group: 'Building things',
      note: 'Getting a model out of a notebook',
      items: [
        'Reproducible data pipelines',
        'Python, TypeScript, React',
        'Directing AI tools and verifying what they produce',
        'Automated testing as proof, not decoration',
      ],
    },
  ],

  repoUrl: 'https://github.com/joshfdcosta/valuation-and-forecasting',
} as const
