# E1 logistic — pre-registered (seed 20260713, 2026-07-15)

```
==============================================================================
E1 logistic — pooled walk-forward test folds 2012-2021
==============================================================================
  config                    acc  activ           IR [90% CI]   ann.a folds+   PT p  verdict
  --------------------------------------------------------------------------------------------
  E1 logistic (argmax)    39.0%  25.9%  -0.49 [-1.10, 0.03]  -2.79%     2/10   0.02  fails (IR folds)

  Final-fold coefficients, LAG class (top 8 by |coef|,
  standardized units — sign read: positive pushes toward lag):
              rsi_14: +0.199
             vol_21d: -0.136
         px_to_sma20: +0.131
         px_to_sma50: -0.129
             bb_pctb: -0.113
             ret_10d: -0.110
              ret_3d: -0.090
             ret_21d: -0.080
```
