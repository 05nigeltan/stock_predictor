# SHAP falsification test (§1.5) — A1 (seed 20260713, 2026-07-15)

```
========================================================================
SHAP FALSIFICATION TEST (§1.5) — E3 models, out-of-sample, pooled
========================================================================
  Days explained: 2517 (2012-2021 test folds)

  LAG-class mean |SHAP| ranking (top 10 of 21):
     1.  sma20_to_sma200  0.0479
     2.          vol_21d  0.0293
     3.            month  0.0250
     4.       vix_chg_5d  0.0193
     5.           ret_2d  0.0126
     6.      days_to_eom  0.0107
     7.        vol_ratio  0.0100
     8.          ret_21d  0.0097
     9.      px_to_sma50  0.0095
    10.          ret_63d  0.0091
    ... ret_5d ranks 12 (0.0056)

  Direction (hypothesis: high recent return -> toward LAG):
           feature  corr w/ own LAG-SHAP  corr w/ own BEAT-SHAP
            ret_5d                  0.58                  -0.70
           ret_10d                 -0.30                  -0.40
       px_to_sma20                 -0.50                  -0.43
            rsi_14                  0.30                  -0.61
           bb_pctb                  0.00                  -0.65

  ret_5d LAG-SHAP, top vs bottom quartile of ret_5d: +0.0068 vs -0.0069
  CLUSTER (position/extension) summed LAG-SHAP vs extension score: corr +0.08

  Vol conditioning (slope of ret_5d LAG-SHAP on ret_5d):
    high-vol half: +0.174    low-vol half: +0.400    ratio 0.4x

  VERDICT (§1.5):
    Named feature (ret_5d): importance rank 12, direction REVERSAL (high ret -> lag) -> weak/dead on the letter
    Cluster substance: corr +0.08 -> REVERSAL-shaped: extension pushes toward lag
    Vol strengthening: NOT confirmed (predicted: steeper in high vol)

    Context from E1-E4: whatever survives here is the DIRECTIONAL
    signal (PT <= 0.02 everywhere) that could not be monetized.
    This test adjudicates the HYPOTHESIS's shape, not the P&L.
```
