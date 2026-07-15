# Locked holdout — the single evaluation (H1; seed 20260713, run 2026-07-15)

```
==============================================================================
LOCKED HOLDOUT — THE single evaluation  (protocol H1)
==============================================================================
  Window : 2022-01-03 -> 2026-07-02   (1128 days, ~225 independent 5d samples)
  Train boundary honored: fold-0 train ends 2021-12-21 (purged+embargoed)
  SPY buy-and-hold over the window: +66.7% total

  strategy                acc  activ           IR [90% CI]   ann.a  yrs+   PT p  lag prec
  ----------------------------------------------------------------------------------------
  always-long           36.2%   0.0%   0.00 [ 0.00, 0.00]   0.00%   0/5   0.50        --
  anti-persistence      33.9%  36.0%  -0.46 [-1.20, 0.27]  -3.05%   2/5   0.73     30.3%
  lightgbm (E3 cfg)     40.7%  18.4%  -0.15 [-0.93, 0.51]  -0.63%   2/5   0.04     39.1%

  Net active return by calendar year (annualized):
  strategy                 2022     2023     2024     2025     2026
  always-long             +0.0%    +0.0%    +0.0%    +0.0%    +0.0%
  anti-persistence        +7.9%    -9.8%    -9.1%    +1.3%    -8.1%
  lightgbm (E3 cfg)       +1.1%    -0.8%    -4.7%    -1.7%    +6.5%

  Comparison, pooled 2012-2021 test folds (for the record):
    anti-persistence: IR -0.77   lightgbm: IR -0.57

  THE HOLDOUT IS NOW SPENT. Per spec §0, any further evaluation
  of 2022+ data by any configuration is void.
```
