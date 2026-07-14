# Phase 4 baselines — pre-registered (seed 20260713, 2026-07-15)

```
==============================================================================
PHASE 4 BASELINES — pooled walk-forward test folds 2012-2021
==============================================================================
  Test days: 2517   random-guess accuracy floor (sum p^2): 33.5%
  Gates: IR >= IR_min(activity)  AND  >=6/10 folds  AND  PT p < 0.1

  baseline                  acc  activ           IR [90% CI]   ann.a folds+   PT p  verdict
  --------------------------------------------------------------------------------------------
  always-long             34.9%   0.0%   0.00 [ 0.00, 0.00]   0.00%     0/10   0.50  anchor (IR=0 by definition)
  dummy most_frequent     34.9%   0.0%   0.00 [ 0.00, 0.00]   0.00%     0/10   0.50  below 10% activity floor
  dummy stratified        32.9%  35.4%  -0.89 [-1.41,-0.39]  -5.70%     1/10   0.23  fails (IR folds PT)
  persistence rule        33.1%  29.9%  -0.74 [-1.36,-0.26]  -7.39%     2/10   0.52  fails (IR folds PT)
  anti-persistence rule   36.0%  35.0%  -0.77 [-1.30,-0.29]  -4.60%     2/10   0.01  fails (IR folds)
  depth-3 tree            35.8%  34.1%  -0.81 [-1.39,-0.30]  -5.49%     3/10   0.02  fails (IR folds)

  Depth-3 tree, final fold (trained 2005-10 -> 2020-12) — the
  'simplest real model', printed because it can be:
    |--- ret_10d <= 0.01
    |   |--- sma20_to_sma200 <= 0.04
    |   |   |--- sma20_to_sma200 <= 0.02
    |   |   |   |--- class: 1.0
    |   |   |--- sma20_to_sma200 >  0.02
    |   |   |   |--- class: 1.0
    |   |--- sma20_to_sma200 >  0.04
    |   |   |--- month <= 10.50
    |   |   |   |--- class: -1.0
    |   |   |--- month >  10.50
    |   |   |   |--- class: 1.0
    |--- ret_10d >  0.01
    |   |--- vol_21d <= 0.01
    |   |   |--- ret_63d <= 0.09
    |   |   |   |--- class: 0.0
    |   |   |--- ret_63d >  0.09
    |   |   |   |--- class: -1.0
    |   |--- vol_21d >  0.01
    |   |   |--- ret_3d <= 0.01
    |   |   |   |--- class: 1.0
    |   |   |--- ret_3d >  0.01
    |   |   |   |--- class: 0.0
```
