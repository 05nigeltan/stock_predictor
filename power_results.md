# power.py output — pre-registered (seed 20260713, 2026-07-15)

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
       10%   -0.44    0.31    0.06      -1.68%        2.2%     10.4%    0.25%
       20%   -0.58    0.29   -0.11      -3.36%        0.4%     10.0%    0.10%
       30%   -0.68    0.27   -0.25      -4.95%        0.1%      9.8%    0.03%
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
    0.00    0.4%   -0.44      -1.68%        29.9%
    0.03    3.1%   -0.22      -0.83%        36.0%
    0.06   19.6%   -0.01       0.04%        41.3%
    0.09   50.5%    0.20       0.91%        46.1%
    0.12   75.7%    0.39       1.75%        50.4%
    0.15   91.0%    0.58       2.61%        54.3%
    0.18   97.5%    0.75       3.43%        57.7%

  activity 20%  (IR_min = -0.11):
       q   power   E[IR] E[ann.alpha] lag-call hit
    0.00    0.1%   -0.57      -3.33%        29.8%
    0.03    0.7%   -0.42      -2.40%        33.2%
    0.06    3.9%   -0.27      -1.54%        36.1%
    0.09   11.0%   -0.12      -0.63%        38.9%
    0.12   26.1%    0.04       0.29%        41.9%
    0.15   45.7%    0.19       1.20%        44.6%
    0.18   68.3%    0.35       2.21%        47.5%
    0.21   83.9%    0.48       3.02%        49.9%
    0.24   93.3%    0.63       3.98%        52.6%
    0.27   98.3%    0.76       4.84%        54.9%

  activity 30%  (IR_min = -0.25):
       q   power   E[IR] E[ann.alpha] lag-call hit
    0.00    0.0%   -0.69      -4.99%        30.0%
    0.03    0.2%   -0.55      -3.97%        32.2%
    0.06    1.1%   -0.42      -3.00%        34.3%
    0.09    2.6%   -0.29      -2.10%        36.3%
    0.12    6.2%   -0.17      -1.15%        38.4%
    0.15   16.0%   -0.03      -0.14%        40.5%
    0.18   29.0%    0.09       0.75%        42.5%
    0.21   48.8%    0.23       1.73%        44.7%
    0.24   66.9%    0.35       2.64%        46.7%
    0.27   84.7%    0.48       3.65%        48.9%
    0.30   93.1%    0.61       4.60%        51.1%
    0.33   98.0%    0.73       5.55%        53.0%

========================================================================
MDE TABLE — what each criterion can actually detect (80% power)
========================================================================
  activity  IR_min (luck bar)  MDE (true IR) ann.alpha @MDE  lag hit @MDE        verdict
       10%               0.06           0.44         1.99%         51.5%      plausible
       20%              -0.11           0.45         2.82%         49.3%      plausible
       30%              -0.25           0.45         3.38%         48.3%      plausible

  Pricing anchor: IR ~0.4 is the OPTIMISTIC edge of published
  5d index-timing results. MDE above ~0.8 = detectable but
  unachievable — v1's failure in a different costume.

  For the record, the superseded v1 criterion at 20% coverage:
  SE 6.8pp, 3pp bar = 0.44 sigma, zero-skill false-pass ~17%.
```
