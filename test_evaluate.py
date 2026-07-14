"""
Tests for evaluate.py — the shared strategy-evaluation chain.

The extraction from power.py is already verified end-to-end (byte-
identical power_results.md reproduction). These tests add what that
can't: a hand-computed ground truth for the tranche book (previously
validated only via aggregate consistency), and calibration proofs for
the two inference tools — the bootstrap's CI coverage and PT's size.

Run:  pytest test_evaluate.py -v
"""

import numpy as np
import pytest

from evaluate import (ALPHA_TEST, BEAT, COST_ONEWAY, FLAT, LAG,
                      active_returns, annualized_ir, folds_positive,
                      moving_block_bootstrap_ir, pt_pvalue)
from splitter import HORIZON


# ---------------------------------------------------------------
# Tranche book — ground truth
# ---------------------------------------------------------------
def test_always_long_has_zero_alpha_and_zero_cost():
    preds = np.full((1, 30), BEAT, dtype=np.int8)
    r = np.random.default_rng(0).normal(0, 0.01, 30)
    assert np.all(active_returns(preds, r) == 0.0)


def test_single_flat_day_hand_computed():
    # One LAG signal at t=10, everything else long. Hand-derivation:
    #   the tranche opened at close 10 is flat for 5 sessions, so the
    #   book is 4/5 long on days 11..15: alpha_d = -(1/5) r_d there.
    #   Costs: that tranche trades twice — going flat at 10 and re-going
    #   long at 15 — each one-way on 1/5 capital: COST/5 charged at
    #   t=10 and t=15. All other days: alpha = 0.
    n, t = 30, 10
    preds = np.full((1, n), BEAT, dtype=np.int8)
    preds[0, t] = LAG
    r = np.random.default_rng(1).normal(0, 0.01, n)

    expected = np.zeros(n)
    expected[t + 1: t + 6] = -(1 / 5) * r[t + 1: t + 6]
    expected[t]     -= COST_ONEWAY / 5
    expected[t + 5] -= COST_ONEWAY / 5

    assert np.allclose(active_returns(preds, r)[0], expected, atol=1e-15)


def test_permanently_flat_book_exact_vector():
    # All-LAG, full hand-derivation. The book seeded long ramps down one
    # tranche per day; the signal at close t first earns day t+1:
    #   day 0      : pos = 1 (all pad)      alpha = -COST/5 (first roll)
    #   days 1..4  : pos = 1 - t/5          alpha = -(t/5) r_t - COST/5
    #   days >= 5  : pos = 0                alpha = -r_t, no more trades
    n = 30
    preds = np.full((1, n), LAG, dtype=np.int8)
    r = np.random.default_rng(2).normal(0, 0.01, n)

    expected = -r.copy()
    expected[0] = 0.0
    for t in range(1, 5):
        expected[t] = -(t / 5) * r[t]
    expected[:5] -= COST_ONEWAY / 5                  # one roll each day 0..4

    assert np.allclose(active_returns(preds, r)[0], expected, atol=1e-15)


def test_annualized_ir_degenerate_and_sign():
    flat = np.zeros((1, 100))
    ir, mu = annualized_ir(flat)
    assert ir[0] == 0.0 and mu[0] == 0.0        # sd=0 -> 0, not inf/nan
    up = np.full((1, 252), 1e-4)
    up[0, ::2] += 1e-5                           # tiny wiggle, positive mean
    assert annualized_ir(up)[0][0] > 0


def test_folds_positive_counting():
    fold_id = np.repeat(np.arange(10), 10)
    alpha = np.where(fold_id[None, :] < 7, 1e-4, -1e-4)   # 7 winning folds
    assert folds_positive(alpha, fold_id, need=6)[0]
    assert not folds_positive(alpha, fold_id, need=8)[0]


# ---------------------------------------------------------------
# Inference calibration
# ---------------------------------------------------------------
def test_pt_size_is_calibrated():
    # Predictions independent of labels -> rejection rate ~ ALPHA_TEST.
    rng = np.random.default_rng(3)
    y = rng.choice([LAG, FLAT, BEAT], size=2500, p=[0.31, 0.34, 0.35])
    preds = rng.choice([LAG, FLAT, BEAT], size=(4000, 2500),
                       p=[0.2, 0.3, 0.5]).astype(np.int8)
    rej = (pt_pvalue(preds, y) < ALPHA_TEST).mean()
    assert 0.07 < rej < 0.13


def test_bootstrap_ci_coverage_on_zero_skill():
    # 200 synthetic zero-mean strategies with 5-day autocorrelation
    # (block structure the bootstrap must respect): a nominal 90% CI
    # should cover the true IR (= 0) roughly 90% of the time.
    rng = np.random.default_rng(4)
    hits = 0
    for i in range(200):
        z = rng.normal(0, 1, (100, 5))           # 5-day correlated blocks
        alpha = (0.6 * z + 0.4 * rng.normal(0, 1, (100, 5))).ravel() * 1e-3
        _, lo, hi = moving_block_bootstrap_ir(alpha, n_boot=400, seed=i)
        hits += lo <= 0.0 <= hi
    assert 0.82 <= hits / 200 <= 0.97


def test_bootstrap_point_matches_annualized_ir():
    rng = np.random.default_rng(5)
    alpha = rng.normal(1e-4, 1e-3, 500)
    point, lo, hi = moving_block_bootstrap_ir(alpha, n_boot=300)
    assert point == pytest.approx(annualized_ir(alpha[None, :])[0][0])
    assert lo < point < hi


def test_bootstrap_rejects_tiny_samples():
    with pytest.raises(ValueError, match="more than"):
        moving_block_bootstrap_ir(np.zeros(8))
