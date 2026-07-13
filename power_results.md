# power.py output — pre-registered (seed 20260713, 2026-07-14)

```
========================================================================
POWER ANALYSIS — spec v2.1, real folds, real 2012-2021 returns
========================================================================
  Pooled test days: 2517   folds: 10   non-overlapping PT samples: 504
  Null: Markov runs (mean 5d), 4000 sims per cell, seed 20260713
  Always-long benchmark: buy-and-hold, zero cost. E[alpha|null] < 0 by construction (drift drag).

------------------------------------------------------------------------
NULL DISTRIBUTIONS (zero skill on the real return path)
------------------------------------------------------------------------
  activity   E[IR]  sd(IR)  IR_min E[ann.alpha] P(folds>=8) P(PT<.10) P(joint)
       10%   -0.45    0.31    0.06      -1.70%        2.3%     10.4%    0.27%
       20%   -0.59    0.29   -0.11      -3.37%        0.3%     10.0%    0.05%
       30%   -0.68    0.27   -0.24      -4.97%        0.0%      9.8%    0.00%
  (IR_min = 95th pct of null IR. P(joint) = false-pass rate of the
   full criterion under zero skill — the bar the v1 spec failed.)

  IID-null sensitivity at 20% activity: sd(IR) 0.21 vs Markov 0.29 -> persistence widens the null; Markov (wider) kept.

------------------------------------------------------------------------
POWER OF THE JOINT CRITERION vs INJECTED SKILL
  (q = prob a prediction is replaced by the truth; MDE = delivered
   IR at the 80%-power crossing)
------------------------------------------------------------------------

  activity 10%  (IR_min = 0.06):
       q   power   E[IR] E[ann.alpha] lag-call hit
    0.00    0.5%   -0.44      -1.68%        29.9%
    0.03    2.1%   -0.27      -1.03%        36.0%
    0.06   10.5%   -0.10      -0.35%        41.3%
    0.09   28.9%    0.06       0.31%        46.1%
    0.12   48.9%    0.21       0.94%        50.4%
    0.15   70.6%    0.37       1.61%        54.3%
    0.18   82.7%    0.49       2.22%        57.7%
    0.21   91.8%    0.63       2.95%        60.8%
    0.24   97.1%    0.76       3.58%        64.0%

  activity 20%  (IR_min = -0.11):
       q   power   E[IR] E[ann.alpha] lag-call hit
    0.00    0.2%   -0.57      -3.25%        30.1%
    0.03    0.7%   -0.46      -2.65%        33.0%
    0.06    1.9%   -0.34      -1.95%        36.1%
    0.09    5.8%   -0.22      -1.21%        39.1%
    0.12   13.7%   -0.09      -0.42%        42.0%
    0.15   24.0%    0.02       0.21%        44.6%
    0.18   41.4%    0.15       0.95%        47.4%
    0.21   56.3%    0.26       1.63%        49.9%
    0.24   69.9%    0.37       2.32%        52.4%
    0.27   82.7%    0.49       3.04%        55.0%
    0.30   90.7%    0.59       3.71%        57.4%
    0.33   95.9%    0.70       4.46%        59.8%

  activity 30%  (IR_min = -0.24):
       q   power   E[IR] E[ann.alpha] lag-call hit
    0.00    0.0%   -0.67      -4.86%        30.2%
    0.03    0.3%   -0.58      -4.20%        32.1%
    0.06    0.4%   -0.49      -3.50%        34.1%
    0.09    1.6%   -0.38      -2.71%        36.3%
    0.12    2.6%   -0.29      -2.02%        38.3%
    0.15    6.1%   -0.18      -1.22%        40.5%
    0.18   13.3%   -0.07      -0.44%        42.6%
    0.21   21.8%    0.03       0.32%        44.6%
    0.24   33.7%    0.13       1.02%        46.7%
    0.27   51.9%    0.25       1.86%        48.9%
    0.30   67.9%    0.36       2.67%        51.1%
    0.33   78.3%    0.45       3.31%        53.0%
    0.36   88.3%    0.56       4.15%        55.3%
    0.39   96.3%    0.66       4.91%        57.4%

========================================================================
MDE TABLE — what each criterion can actually detect (80% power)
========================================================================
  activity  IR_min (luck bar)  MDE (true IR) ann.alpha @MDE  lag hit @MDE        verdict
       10%               0.06           0.46         2.09%         56.9%       marginal
       20%              -0.11           0.46         2.88%         54.4%       marginal
       30%              -0.24           0.47         3.45%         53.4%       marginal

  Pricing anchor: IR ~0.4 is the OPTIMISTIC edge of published
  5d index-timing results. MDE above ~0.8 = detectable but
  unachievable — v1's failure in a different costume.

  For the record, the superseded v1 criterion at 20% coverage:
  SE 6.8pp, 3pp bar = 0.44 sigma, zero-skill false-pass ~17%.
```
