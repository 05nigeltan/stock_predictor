# E2 random forest — pre-registered (seed 20260713, 2026-07-15)

```
==============================================================================
E2 random forest — pooled walk-forward test folds 2012-2021
==============================================================================
  config                    acc  activ           IR [90% CI]   ann.a folds+   PT p  verdict
  --------------------------------------------------------------------------------------------
  E2 rand forest (argmax)  39.8%  24.1%  -0.65 [-1.21,-0.09]  -2.54%     5/10   0.00  fails (IR folds)

  Final-fold impurity importances (top 8; record only —
  cluster-smeared, SHAP is the verdict):
     sma20_to_sma200: 0.092
             vol_21d: 0.077
             ret_10d: 0.072
         px_to_sma20: 0.062
            vix_lag1: 0.057
             bb_pctb: 0.057
             ret_63d: 0.056
              rsi_14: 0.054
```
