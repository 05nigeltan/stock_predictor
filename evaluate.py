"""
evaluate.py — the strategy evaluation chain  (spec v2.1, items #1–#6)
=====================================================================
Single home for the machinery every consumer shares — power.py's Monte
Carlo, Phase 4's rule baselines, Phase 6's model evaluation. Extracted
from power.py so the tranche math exists in exactly one place, same
reasoning as the label-logic dedup (verified the same way: power.py
reproduces its pre-registered output byte-identically after the move).

The chain: ternary predictions -> long/flat tranche book -> daily net
active returns -> annualized IR + fold wins + m-class PT. All functions
are vectorized over strategies (rows); a single strategy is alpha[None].

Inference (spec #5): moving_block_bootstrap_ir is the PRIMARY inference
for a real strategy's IR — 10-day blocks, resampled with replacement,
handles both the tranche-induced overlap and alpha's non-normality.
"""

import numpy as np

from splitter import HORIZON

COST_ONEWAY = 2.5e-4                     # spec #4: one-way, 5bps round trip
ANN         = np.sqrt(252.0)
ALPHA_TEST  = 0.10                       # design alpha, two-sided

# LAG=0, FLAT=1, BEAT=2  (positions: LAG -> flat book, else long)
LAG, FLAT, BEAT = 0, 1, 2


# ---------------------------------------------------------------
# Tranche book -> daily net active returns  (spec #2, #3, #4)
# ---------------------------------------------------------------
def active_returns(preds, r):
    """alpha_t = (position_t - 1) * r_t - cost_t, per strategy row.

    signal s in {0,1}: 1 unless the prediction is LAG. The book earning
    day t's return holds tranches opened at closes t-5..t-1, so
    position_t = mean(s[t-5..t-1]); days before the test start are seeded
    long (s=1), the benchmark state. Each day exactly one tranche rolls:
    it trades iff its new signal differs from the one it replaces
    (s_t vs s_{t-5}), paying one-way COST on 1/5 of capital.
    """
    s = (preds != LAG).astype(np.float64)
    n = s.shape[1]
    pad = np.ones((s.shape[0], HORIZON))
    sp = np.concatenate([pad, s], axis=1)                # seeded history

    # pos_t = mean(sp[t .. t+4]) = mean(s[t-5 .. t-1]). NOT s[t-4 .. t]:
    # the signal formed at close t cannot earn day t's return — it had
    # already finished when the signal existed. (The original power.py
    # implementation had exactly that one-session lookahead; caught by
    # the hand-computed ground-truth test on extraction, spec A13.)
    c0 = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(sp, axis=1)],
                        axis=1)                          # c0[k] = sum sp[:k]
    pos = (c0[:, HORIZON:HORIZON + n] - c0[:, :n]) / HORIZON
    cost = (COST_ONEWAY / HORIZON) * np.abs(sp[:, HORIZON:] - sp[:, :-HORIZON])
    return (pos - 1.0) * r[None, :] - cost


def annualized_ir(alpha):
    mu, sd = alpha.mean(axis=1), alpha.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ir = np.where(sd > 0, mu / sd, 0.0) * ANN
    return ir, mu * 252.0                                # IR, ann. mean alpha


def folds_positive(alpha, fold_id, need=6):
    """Fold gate, >=6/10 per spec A12 (the 8/10 raise was calibrated
    against a coin-flip null; under the true drift-drag null, >=6/10
    already has ~2% false-pass alone and ~0.1% jointly)."""
    wins = np.zeros(alpha.shape[0], dtype=int)
    for k in range(fold_id.max() + 1):
        wins += alpha[:, fold_id == k].mean(axis=1) > 0
    return wins >= need


# ---------------------------------------------------------------
# m-class Pesaran-Timmermann, influence-function form,
# on non-overlapping (every 5th) test days. Vectorized over rows.
# ---------------------------------------------------------------
def pt_pvalue(preds, y):
    p = preds[:, ::HORIZON]                              # strategies x n
    t = y[::HORIZON]
    n = t.size
    n_rows = p.shape[0]

    hit = (p == t)
    P = hit.mean(axis=1)

    ym = np.bincount(t, minlength=3) / n                 # outcome margins
    pm = np.stack([(p == c).mean(axis=1) for c in range(3)], axis=1)
    Pstar = (pm * ym[None, :]).sum(axis=1)

    # influence: psi_k = 1(hit) - ym[pred_k] - pm[label_k], centered per row
    psi = hit - ym[p] - pm[np.arange(n_rows)[:, None], t[None, :]]
    V = psi.var(axis=1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(V > 0, (P - Pstar) / np.sqrt(V), 0.0)
    from scipy.stats import norm
    return 1.0 - norm.cdf(S)                             # one-sided: skill > chance


# ---------------------------------------------------------------
# Moving-block bootstrap on the IR  (spec #5 — PRIMARY inference)
# ---------------------------------------------------------------
def moving_block_bootstrap_ir(alpha, n_boot=5000, block=10, ci=0.90,
                              seed=20260713):
    """CI for one strategy's annualized IR from its 1-D daily alpha.

    Overlapping blocks of `block` days resampled with replacement to the
    original length; IR recomputed per replicate. Block = 10 (spec #5)
    covers the 5-day tranche overlap with margin. Returns
    (ir_point, ci_lo, ci_hi).
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    n = alpha.size
    if n <= block:
        raise ValueError(f"need more than {block} observations, got {n}")
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    boots = alpha[idx.reshape(n_boot, -1)[:, :n]]

    ir_b, _ = annualized_ir(boots)
    ir, _ = annualized_ir(alpha[None, :])
    lo, hi = np.quantile(ir_b, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(ir[0]), float(lo), float(hi)
