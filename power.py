"""
power.py — pre-registered power analysis  (spec v2.1, build item 2)
===================================================================
Answers ONE question before any feature or model exists: on the real
pooled walk-forward test folds, what is the smallest true edge the v2.1
success criterion can detect at 80% power — and is that edge plausible?

Everything runs on the ACTUAL fold structure from splitter.py and the
ACTUAL 2012-2021 SPY returns. Nothing is an analytic approximation.

Mechanics under test (frozen in spec v2.1):
  position   : long/flat — beat/flat -> +1, lag -> 0
  tranches   : 5 overlapping, 1/5 capital, 5-day hold; the position
               earning day t's return is mean(signal[t-5 .. t-1]) — the
               signal formed at close t-1 is the newest one deployed
  costs      : one-way 2.5bps per tranche position change; benchmark free
  primary    : IR = mean(alpha_d)/std(alpha_d) * sqrt(252)
  gates      : IR >= IR_min  AND  mean net alpha > 0 in >= 6/10 folds
               AND m-class Pesaran-Timmermann p < 0.10
               (fold gate recalibrated 8 -> 6 per spec A12: the 8/10 raise
               was calibrated against a coin-flip null; under the true
               drift-drag null, per-fold P(alpha>0) is 6-49%, so >=6/10
               already has ~2% false-pass alone and ~0.1% jointly)

Null model (spec #8): zero-skill ternary predictions with persistent runs
(two-state Markov, mean deviation run 5 days), swept over activity
(P(pred=lag), the only state that moves the position) — run against the
real return path, so the drift drag E[alpha | no skill] < 0 is IN the
null, not assumed away. IID null reported as a sensitivity.

Skill model: with probability q the prediction equals the true label,
else it is drawn from the null process. q traces the power curve; the
delivered IR at the 80%-power crossing is the MDE, in units priceable
against the literature (IR ~0.4 is the optimistic edge for 5d SPY timing).

Two numbers come out, and they are not the same number:
  IR_min : the LUCK BAR — 95th pct of the null IR distribution
           (alpha=0.10, two-sided). Passing it means "not luck at 10%".
  MDE    : the true IR a model must actually possess for the JOINT
           criterion to fire at 80% power. Always > IR_min.
Verdict: the criterion is usable iff MDE <= a plausible effect (~0.4).

Run:  python power.py          (writes power_results.md next to it)
"""

import numpy as np
import pandas as pd

from splitter import EMBARGO, HORIZON, PurgedWalkForward

# --- label / strategy constants (phase 1, unchanged) -----------
K, VOL_SPAN, DRIFT_WIN = 0.4, 20, 252
COST_ONEWAY = 2.5e-4                     # spec #4
ANN         = np.sqrt(252.0)
ALPHA_TEST  = 0.10                       # two-sided -> 95th pct luck bar
TARGET_POW  = 0.80
RUN_LEN     = 5                          # mean deviation run length, days
N_SIMS      = 4000
SEED        = 20260713

# LAG=0, FLAT=1, BEAT=2  (positions: LAG -> flat book, else long)
LAG, FLAT, BEAT = 0, 1, 2


# ---------------------------------------------------------------
# 1. Real panel: returns, ternary labels, folds
# ---------------------------------------------------------------
def load_test_panel():
    raw = pd.read_parquet("data/raw/tiingo_spy_2005-01-01.parquet")
    close = raw["adjClose"]
    log_ret = np.log(close / close.shift(1))
    fwd     = np.log(close.shift(-HORIZON) / close)
    drift   = np.log(close / close.shift(HORIZON)).rolling(
                  DRIFT_WIN, min_periods=60).mean()
    excess  = fwd - drift
    sigma5  = log_ret.ewm(span=VOL_SPAN).std() * np.sqrt(HORIZON)
    tau     = K * sigma5

    label = pd.Series(FLAT, index=raw.index)
    label[excess >  tau] = BEAT
    label[excess < -tau] = LAG

    idx = raw.index[fwd.notna() & drift.notna() & sigma5.notna()]
    folds = PurgedWalkForward(raw.index).split(idx)

    test_dates = folds[0].test_dates.append([f.test_dates for f in folds[1:]])
    fold_id = np.concatenate(
        [np.full(len(f.test_dates), f.fold) for f in folds])

    return {
        "r":      log_ret.loc[test_dates].to_numpy(),      # daily returns
        "y":      label.loc[test_dates].to_numpy(),        # true ternary label
        "fold":   fold_id,
        "n_days": len(test_dates),
    }


# ---------------------------------------------------------------
# 2. Signal generation — vectorized across sims (rows), Markov in t
# ---------------------------------------------------------------
def null_predictions(rng, n_sims, n_days, activity, run_len=RUN_LEN,
                     p_beat_within=0.51):
    """Zero-skill ternary predictions with persistent deviation runs.

    Two-state Markov on {deviate=LAG, benchmark}: exit prob 1/run_len gives
    mean run length run_len; the entry prob is solved so the stationary
    deviation share equals `activity`. Non-deviating days split beat/flat
    at the training class ratio 35.4/34.0 (matters only to the PT gate).
    run_len=1 collapses to the IID null.
    """
    p_exit  = 1.0 / run_len
    p_enter = p_exit * activity / (1.0 - activity)
    u = rng.random((n_sims, n_days))

    dev = np.empty((n_sims, n_days), dtype=bool)
    dev[:, 0] = u[:, 0] < activity                       # start stationary
    for t in range(1, n_days):
        dev[:, t] = np.where(dev[:, t - 1],
                             u[:, t] >= p_exit,          # stay deviating
                             u[:, t] < p_enter)          # enter deviation

    preds = np.where(rng.random((n_sims, n_days)) < p_beat_within,
                     BEAT, FLAT).astype(np.int8)
    preds[dev] = LAG
    return preds


def inject_skill(rng, preds, y, q):
    """With prob q, the prediction becomes the true label."""
    if q == 0:
        return preds
    hit = rng.random(preds.shape) < q
    return np.where(hit, y[None, :], preds).astype(np.int8)


# ---------------------------------------------------------------
# 3. Tranche book -> daily net active returns  (spec #3, #4)
# ---------------------------------------------------------------
def active_returns(preds, r):
    """alpha_t = (position_t - 1) * r_t - cost_t, per sim.

    signal s in {0,1}: 1 unless the prediction is LAG. The book earning
    day t's return holds tranches opened at closes t-5..t-1, so
    position_t = mean(s[t-5..t-1]); days before the test start are seeded
    long (s=1), the benchmark state. Each day exactly one tranche rolls:
    it trades iff its new signal differs from the one it replaces
    (s_t vs s_{t-5}), paying one-way COST on 1/5 of capital.
    """
    s = (preds != LAG).astype(np.float64)
    pad = np.ones((s.shape[0], HORIZON))
    sp = np.concatenate([pad, s], axis=1)                # seeded history

    cs = np.cumsum(sp, axis=1)
    # mean of sp[t .. t+4] = mean(s[t-5 .. t-1]) in unpadded time
    pos = (cs[:, HORIZON:] - cs[:, :-HORIZON]) / HORIZON
    cost = (COST_ONEWAY / HORIZON) * np.abs(sp[:, HORIZON:] - sp[:, :-HORIZON])
    return (pos - 1.0) * r[None, :] - cost


def annualized_ir(alpha):
    mu, sd = alpha.mean(axis=1), alpha.std(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ir = np.where(sd > 0, mu / sd, 0.0) * ANN
    return ir, mu * 252.0                                # IR, ann. mean alpha


def folds_positive(alpha, fold_id, need=6):        # 6, not 8 — see A12 note above
    wins = np.zeros(alpha.shape[0], dtype=int)
    for k in range(fold_id.max() + 1):
        wins += alpha[:, fold_id == k].mean(axis=1) > 0
    return wins >= need


# ---------------------------------------------------------------
# 4. m-class Pesaran-Timmermann, influence-function form,
#    on non-overlapping (every 5th) test days. Vectorized over sims.
# ---------------------------------------------------------------
def pt_pvalue(preds, y):
    p = preds[:, ::HORIZON]                              # sims x n
    t = y[::HORIZON]
    n = t.size
    n_sims = p.shape[0]

    hit = (p == t)
    P = hit.mean(axis=1)

    ym = np.bincount(t, minlength=3) / n                 # outcome margins
    pm = np.stack([(p == c).mean(axis=1) for c in range(3)], axis=1)
    Pstar = (pm * ym[None, :]).sum(axis=1)

    # influence: psi_k = 1(hit) - ym[pred_k] - pm[label_k], centered per sim
    psi = hit - ym[p] - pm[np.arange(n_sims)[:, None], t[None, :]]
    V = psi.var(axis=1) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(V > 0, (P - Pstar) / np.sqrt(V), 0.0)
    from scipy.stats import norm
    return 1.0 - norm.cdf(S)                             # one-sided: skill > chance


# ---------------------------------------------------------------
# 5. The experiment
# ---------------------------------------------------------------
def run():
    panel = load_test_panel()
    r, y, fold_id = panel["r"], panel["y"], panel["fold"]
    n_days = panel["n_days"]
    rng = np.random.default_rng(SEED)

    lines = []                                           # report accumulator
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 72)
    emit("POWER ANALYSIS — spec v2.1, real folds, real 2012-2021 returns")
    emit("=" * 72)
    emit(f"  Pooled test days: {n_days}   folds: {fold_id.max() + 1}   "
         f"non-overlapping PT samples: {len(y[::HORIZON])}")
    emit(f"  Null: Markov runs (mean {RUN_LEN}d), {N_SIMS} sims per cell, "
         f"seed {SEED}")
    emit(f"  Always-long benchmark: buy-and-hold, zero cost. "
         f"E[alpha|null] < 0 by construction (drift drag).")

    # ---- NULL: luck bars per activity level -----------------------------
    activities = [0.10, 0.20, 0.30]
    null_stats = {}
    emit("\n" + "-" * 72)
    emit("NULL DISTRIBUTIONS (zero skill on the real return path)")
    emit("-" * 72)
    emit(f"  {'activity':>8} {'E[IR]':>7} {'sd(IR)':>7} {'IR_min':>7} "
         f"{'E[ann.alpha]':>12} {'P(folds>=8)':>11} {'P(PT<.10)':>9} "
         f"{'P(joint)':>8}")
    for a in activities:
        preds = null_predictions(rng, N_SIMS, n_days, a)
        alpha = active_returns(preds, r)
        ir, ann_mu = annualized_ir(alpha)
        fp = folds_positive(alpha, fold_id)
        pt = pt_pvalue(preds, y) < ALPHA_TEST
        ir_min = np.quantile(ir, 1 - ALPHA_TEST / 2)     # 95th pct luck bar
        joint = ((ir >= ir_min) & fp & pt).mean()
        null_stats[a] = {"ir_min": ir_min, "sd": ir.std(),
                         "sd_mu": ann_mu.std()}
        emit(f"  {a:>8.0%} {ir.mean():>7.2f} {ir.std():>7.2f} "
             f"{ir_min:>7.2f} {ann_mu.mean():>11.2%} {fp.mean():>11.1%} "
             f"{pt.mean():>9.1%} {joint:>8.2%}")
    emit("  (IR_min = 95th pct of null IR. P(joint) = false-pass rate of the")
    emit("   full criterion under zero skill — the bar the v1 spec failed.)")

    # IID sensitivity (spec #8)
    preds = null_predictions(rng, N_SIMS, n_days, 0.20, run_len=1)
    ir_iid, _ = annualized_ir(active_returns(preds, r))
    emit(f"\n  IID-null sensitivity at 20% activity: "
         f"sd(IR) {ir_iid.std():.2f} vs Markov {null_stats[0.20]['sd']:.2f} "
         f"-> persistence {'widens' if null_stats[0.20]['sd'] > ir_iid.std() else 'narrows'} the null; Markov (wider) kept.")

    # ---- POWER: skill sweep ---------------------------------------------
    emit("\n" + "-" * 72)
    emit("POWER OF THE JOINT CRITERION vs INJECTED SKILL")
    emit("  (q = prob a prediction is replaced by the truth; MDE = delivered")
    emit("   IR at the 80%-power crossing)")
    emit("-" * 72)
    qs = np.arange(0.0, 0.42, 0.03)
    mde = {}
    for a in activities:
        ir_min = null_stats[a]["ir_min"]
        emit(f"\n  activity {a:.0%}  (IR_min = {ir_min:.2f}):")
        emit(f"  {'q':>6} {'power':>7} {'E[IR]':>7} {'E[ann.alpha]':>12} "
             f"{'lag-call hit':>12}")
        prev = None
        for q in qs:
            base = null_predictions(rng, 1500, n_days, a)
            preds = inject_skill(rng, base, y, q)
            alpha = active_returns(preds, r)
            ir, ann_mu = annualized_ir(alpha)
            fp = folds_positive(alpha, fold_id)
            pt = pt_pvalue(preds, y) < ALPHA_TEST
            power = ((ir >= ir_min) & fp & pt).mean()
            lag_hit = ((preds == LAG) & (y[None, :] == LAG)).sum() \
                      / max((preds == LAG).sum(), 1)
            emit(f"  {q:>6.2f} {power:>7.1%} {ir.mean():>7.2f} "
                 f"{ann_mu.mean():>11.2%} {lag_hit:>12.1%}")
            if power >= TARGET_POW and a not in mde:
                if prev is None:
                    mde[a] = (q, ir.mean(), ann_mu.mean(), lag_hit)
                else:                                     # linear interp on q
                    q0, p0, ir0, mu0, lh0 = prev
                    w = (TARGET_POW - p0) / (power - p0)
                    mde[a] = (q0 + w * (q - q0), ir0 + w * (ir.mean() - ir0),
                              mu0 + w * (ann_mu.mean() - mu0),
                              lh0 + w * (lag_hit - lh0))
            prev = (q, power, ir.mean(), ann_mu.mean(), lag_hit)
            if power >= 0.95:
                break
        if a not in mde:
            mde[a] = None

    # ---- THE TABLE THE SPEC DEMANDS -------------------------------------
    emit("\n" + "=" * 72)
    emit("MDE TABLE — what each criterion can actually detect (80% power)")
    emit("=" * 72)
    emit(f"  {'activity':>8} {'IR_min (luck bar)':>18} {'MDE (true IR)':>14} "
         f"{'ann.alpha @MDE':>14} {'lag hit @MDE':>13} {'verdict':>14}")
    for a in activities:
        m = mde[a]
        if m is None:
            emit(f"  {a:>8.0%} {null_stats[a]['ir_min']:>18.2f} "
                 f"{'not reached':>14} {'-':>14} {'-':>13} {'UNDETECTABLE':>14}")
            continue
        q_, ir_, mu_, lh_ = m
        verdict = "plausible" if ir_ <= 0.45 else \
                  "IMPLAUSIBLE" if ir_ > 0.8 else "marginal"
        emit(f"  {a:>8.0%} {null_stats[a]['ir_min']:>18.2f} {ir_:>14.2f} "
             f"{mu_:>13.2%} {lh_:>13.1%} {verdict:>14}")
    emit("\n  Pricing anchor: IR ~0.4 is the OPTIMISTIC edge of published")
    emit("  5d index-timing results. MDE above ~0.8 = detectable but")
    emit("  unachievable — v1's failure in a different costume.")

    # v1 comparison row (from the gate check, kept for the record)
    emit("\n  For the record, the superseded v1 criterion at 20% coverage:")
    emit("  SE 6.8pp, 3pp bar = 0.44 sigma, zero-skill false-pass ~17%.")

    with open("power_results.md", "w") as f:
        f.write("# power.py output — pre-registered "
                "(seed %d, %s)\n\n```\n" % (SEED, pd.Timestamp.now().date()))
        f.write("\n".join(lines))
        f.write("\n```\n")
    print("\n[written] power_results.md")


if __name__ == "__main__":
    run()
