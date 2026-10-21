#!/usr/bin/env python3
"""Numerical verification of the closed-form claims of HeavyBridge (TMLR),
appendix "Numerical verification" (sec:Snumerics), items 1-8.

Each function `item_k` reproduces item k of the paper's enumerate list, prints
the measured error next to the tolerance and PASS/FAIL, and returns True/False.
Wherever the released implementation computes a quantity, the check calls it:
    experiments_navsim/runpod/heavybridge_mixing.py   (imported unchanged)
Closed forms that exist only in the paper (Student-t density, score, posterior
moments, composition identities, tempered-stable moments and transforms) are
written out here from the paper's equations.

Tolerances
  * Deterministic items: the paper's stated tolerance.
  * Monte-Carlo items: |estimate - closed form| <= 4 standard errors (SE
    estimated from the same samples); where the paper also states a magnitude
    (item 2) it is reported next to the measurement.

Usage:  python verify_all.py            # paper sample sizes (default)
        python verify_all.py --fast     # samples / 20, for a quick smoke run
        python verify_all.py --items 1 5 8
numpy + scipy only. All random streams are seeded (seed = 1000 + item).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.special import gammaln

# --- released implementation (unchanged copy) --------------------------------
_RUNPOD = Path(__file__).resolve().parent.parent / "experiments_navsim" / "runpod"
sys.path.insert(0, str(_RUNPOD))
import heavybridge_mixing as hm  # noqa: E402

# --- paper constants (sec:Snumerics / sec:Spractical) ------------------------
BETA_D, BETA_MIN = 2.0, 0.1          # VP-Karras schedule
T_END, N_GRID = 1.0, 20              # horizon T=1 (m_T=0.577, sigma_T^2=0.667), N=20
TS = np.linspace(0.0, T_END, N_GRID + 1)
MC_Z = 4.0                           # Monte-Carlo tolerance: 4 standard errors
MACHINE_TOL = 1e-14                  # "machine precision" (~45 ulp of 1.0)
QUAD_EPSREL = 1.2e-14                # tightest relative tolerance scipy.quad accepts
CHUNK = 500_000                      # MC chunk size (memory bound)


def B(t):
    return 0.5 * BETA_D * t ** 2 + BETA_MIN * t


def g2(t):
    return BETA_D * t + BETA_MIN


def mfun(t):
    return np.exp(-0.5 * B(t))


def q(f, a, b):
    return quad(f, a, b, epsabs=0.0, epsrel=QUAD_EPSREL, limit=400)[0]


class Item:
    """Collects sub-checks of one paper item and prints them uniformly."""

    def __init__(self, k, title):
        self.k, self.title, self.ok, self.t0 = k, title, True, time.time()
        print(f"\n=== Item {k}: {title} ===", flush=True)

    def check(self, label, err, tol, basis=""):
        passed = bool(np.all(np.isfinite(err))) and float(err) <= float(tol)
        self.ok &= passed
        b = f"  [{basis}]" if basis else ""
        print(f"  {'PASS' if passed else 'FAIL'}  {label:<58s} err={float(err):.3e}  tol={float(tol):.3e}{b}",
              flush=True)
        return passed

    def mc(self, label, est, ref, se):
        """Monte-Carlo check: |est-ref| <= 4 SE. Prints the error in SE units too."""
        err = abs(est - ref)
        tol = MC_Z * se
        passed = err <= tol
        self.ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {label:<58s} est={est:.6g} ref={ref:.6g} "
              f"err={err:.2e} ({err / se if se > 0 else np.inf:.2f} SE)  tol={tol:.2e}", flush=True)
        return passed

    def info(self, msg):
        print(f"  info  {msg}", flush=True)

    def done(self):
        dt = time.time() - self.t0
        print(f"--> Item {self.k}: {'PASS' if self.ok else 'FAIL'}   ({dt:.1f} s)", flush=True)
        return self.ok, dt


def table(alpha=1.5, ts=TS, K=8):
    return hm.precompute(ts, alpha=alpha, beta_d=BETA_D, beta_min=BETA_MIN, K=K)


def single_step_table(tab, c, u):
    """A one-step ScheduleTable carrying a given Laplace coefficient c, so that the
    released sampler `sample_profile` draws TS_{alpha/2}(c, kappa) / PS_{alpha/2}(c)."""
    return dataclasses.replace(tab, u=np.array([u]), c_lap=np.array([c]),
                               s_alpha=np.array([np.nan]), rho=np.array([np.nan]))


def nearest_index(ts, t):
    i = int(np.argmin(np.abs(ts - t)))
    assert abs(ts[i] - t) < 1e-12
    return i


# =============================================================================
# Item 1 - Lemma 2 (schedule invariance)
# =============================================================================
def item_1(args):
    it = Item(1, "Lemma 2: a_t, b_t invariant, (c_t^lambda)^2 = lambda c_t^2, t in {0.2,0.5,0.8}, lambda in {1,3.7}")
    tab = table()
    N = N_GRID
    idx = [nearest_index(TS, t) for t in (0.2, 0.5, 0.8)]
    pairs = [(idx[0], idx[1]), (idx[1], idx[2]), (idx[0], idx[2])]   # restricted a_{s|t}, b_{s|t}
    # Defining-integral reference: sigma^2_lambda(t) = m_t^2 int_0^t lambda g^2/m_s^2 ds
    I = np.array([0.0] + [q(lambda s: g2(s) / mfun(s) ** 2, 0.0, t) for t in TS[1:]])
    for lam in (1.0, 3.7):
        e_a = e_b = e_c = 0.0       # (a) released ScheduleTable.abc with sigma^2 -> lambda sigma^2
        p_a = p_b = p_c = 0.0       # (b) released profile_sig2/profile_abc, constant profile lambda
        r_a = r_b = r_c = 0.0       # (c) sigma^2_lambda from quadrature of the defining integral
        sig2_prof = hm.profile_sig2(tab, np.full((1, N), lam))
        sig2_quad = lam * tab.m ** 2 * I
        for i in idx:
            a0, b0, c0 = tab.abc(i, N)
            a1, b1, c1 = tab.abc(i, N, sig2=lam * tab.sig2)
            a2, b2, c2 = (x[0] for x in hm.profile_abc(tab, sig2_prof, i, N))
            a3, b3, c3 = tab.abc(i, N, sig2=sig2_quad)
            e_a, e_b, e_c = max(e_a, abs(a1 - a0) / a0), max(e_b, abs(b1 - b0) / b0), max(e_c, abs(c1 - lam * c0) / (lam * c0))
            p_a, p_b, p_c = max(p_a, abs(a2 - a0) / a0), max(p_b, abs(b2 - b0) / b0), max(p_c, abs(c2 - lam * c0) / (lam * c0))
            r_a, r_b, r_c = max(r_a, abs(a3 - a0) / a0), max(r_b, abs(b3 - b0) / b0), max(r_c, abs(c3 - lam * c0) / (lam * c0))
        for (i, j) in pairs:
            a0, b0, c0 = tab.abc(i, j)
            a1, b1, c1 = tab.abc(i, j, sig2=lam * tab.sig2)
            e_a, e_b, e_c = max(e_a, abs(a1 - a0) / a0), max(e_b, abs(b1 - b0) / b0), max(e_c, abs(c1 - lam * c0) / (lam * c0))
        tag = f"lambda={lam}"
        it.check(f"{tag} abc(): max rel |da|  (incl. a_s|t)", e_a, MACHINE_TOL, "machine precision")
        it.check(f"{tag} abc(): max rel |db|  (incl. b_s|t)", e_b, MACHINE_TOL, "machine precision")
        it.check(f"{tag} abc(): max rel |c^2_lam - lam c^2| (incl. s|t)", e_c, MACHINE_TOL, "machine precision")
        it.check(f"{tag} profile_sig2+profile_abc: max rel |da|", p_a, MACHINE_TOL, "machine precision")
        it.check(f"{tag} profile_sig2+profile_abc: max rel |db|", p_b, MACHINE_TOL, "machine precision")
        it.check(f"{tag} profile_sig2+profile_abc: max rel |dc^2|", p_c, MACHINE_TOL, "machine precision")
        it.check(f"{tag} quadrature of lambda*g^2: max rel |da|,|db|,|dc^2|", max(r_a, r_b, r_c), 1e-12,
                 "quadrature-limited reference")
    it.info(f"grid times used: {[float(TS[i]) for i in idx]}")
    return it.done()


# =============================================================================
# Student-t closed forms of Prop. 1 (written from the paper)
# =============================================================================
NU, D_T = 5.0, 4


def st_logpdf(x, mu, c2, nu=NU):
    D = x.shape[-1]
    r2 = np.sum((x - mu) ** 2, axis=-1) / c2
    return (gammaln((nu + D) / 2) - gammaln(nu / 2) - 0.5 * D * np.log(nu * np.pi)
            - 0.5 * D * np.log(c2) - 0.5 * (nu + D) * np.log1p(r2 / nu))


def st_score(x, mu, c2, nu=NU):
    D = x.shape[-1]
    r2 = np.sum((x - mu) ** 2) / c2
    return -(nu + D) / (nu + r2) * (x - mu) / c2


def bridge_setup(rng, t=0.5):
    tab = table()
    i = nearest_index(TS, t)
    a, b, c2 = tab.abc(i, N_GRID)
    x0 = rng.standard_normal(D_T)
    y = x0 + 0.5 * rng.standard_normal(D_T)
    mu = a * y + b * x0
    return mu, c2


R_GRID = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)


def eval_points(rng, mu, c2):
    pts = []
    for r in R_GRID:
        e = rng.standard_normal(D_T)
        e /= np.linalg.norm(e)
        pts.append(mu + math.sqrt(c2) * r * e)
    return pts


# =============================================================================
# Item 2 - Prop. 1(i): Student-t density vs Inverse-Gamma mixture
# =============================================================================
def item_2(args):
    it = Item(2, "Prop. 1(i): Student-t density vs 2e6-sample Inverse-Gamma mixture, D=4, nu=5")
    rng = np.random.default_rng(1002)
    n = args.n(2_000_000)
    mu, c2 = bridge_setup(rng)
    pts = eval_points(rng, mu, c2)
    # Formula vs scipy's multivariate t (independent implementation)
    mvt = stats.multivariate_t(loc=mu, shape=c2 * np.eye(D_T), df=NU)
    e_sc = max(abs(math.exp(st_logpdf(x, mu, c2)) / mvt.pdf(x) - 1) for x in pts)
    it.check("Eq. (Stdensity) vs scipy.stats.multivariate_t (rel)", e_sc, 1e-12, "exact formula")
    for strat in (False, True):
        lam = hm.sample_ig(NU, n, rng, stratified=strat)          # released IG(nu/2,nu/2) sampler
        maxrel = 0.0
        for x, r in zip(pts, R_GRID):
            r2 = np.sum((x - mu) ** 2) / c2
            k = np.exp(-0.5 * D_T * np.log(2 * np.pi * lam * c2) - 0.5 * r2 / lam)
            est, se = k.mean(), k.std(ddof=1) / math.sqrt(n)
            ref = math.exp(st_logpdf(x, mu, c2))
            maxrel = max(maxrel, abs(est - ref) / ref)
            if not strat:
                it.mc(f"iid IG mixture, r_t={r}: density (rel units)", est / ref, 1.0, se / ref)
        if strat:
            it.info(f"stratified IG mixture (sample_ig default), n={n}: max rel error {maxrel:.2e} "
                    f"(paper: ~1e-4; i.i.d. SE does not apply to stratified draws)")
        else:
            it.info(f"i.i.d. IG mixture, n={n}: max rel error {maxrel:.2e} (paper states ~1e-4)")
    return it.done()


# =============================================================================
# Item 3 - Prop. 1(iii): score vs central differences
# =============================================================================
def item_3(args):
    it = Item(3, "Prop. 1(iii): Student-t score vs central-difference differentiation")
    rng = np.random.default_rng(1003)
    mu, c2 = bridge_setup(rng)
    pts = eval_points(rng, mu, c2) + [mu + 2.0 * math.sqrt(c2) * rng.standard_normal(D_T) for _ in range(20)]
    h = 1e-5
    err_abs = err_rel = 0.0
    for x in pts:
        s = st_score(x, mu, c2)
        num = np.empty(D_T)
        for d in range(D_T):
            e = np.zeros(D_T); e[d] = h
            num[d] = (st_logpdf(x + e, mu, c2) - st_logpdf(x - e, mu, c2)) / (2 * h)
        err_abs = max(err_abs, np.max(np.abs(num - s)))
        if np.linalg.norm(s) > 0:
            err_rel = max(err_rel, np.linalg.norm(num - s) / np.linalg.norm(s))
    it.check(f"max abs |score - central diff| ({len(pts)} points, h={h:g})", err_abs, 1e-6, "paper: 1e-6")
    it.info(f"max relative error {err_rel:.2e}")
    return it.done()


# =============================================================================
# Item 4 - Prop. 1(ii): posterior moments vs SNIS
# =============================================================================
def item_4(args):
    it = Item(4, "Prop. 1(ii): E[1/lambda|.], E[lambda|.] vs self-normalized importance sampling")
    rng = np.random.default_rng(1004)
    n = args.n(2_000_000)          # sample size not stated in the paper; 2e6 as in item 2
    mu, c2 = bridge_setup(rng)
    lam = hm.sample_ig(NU, n, rng, stratified=False)              # prior proposals (i.i.d.)
    for x, r in zip(eval_points(rng, mu, c2), R_GRID):
        r2 = np.sum((x - mu) ** 2) / c2
        logw = -0.5 * D_T * np.log(lam) - 0.5 * r2 / lam
        w = np.exp(logw - logw.max())
        W = w.sum()
        for name, f, ref in (("E[1/lambda|.]", 1.0 / lam, (NU + D_T) / (NU + r2)),
                             ("E[lambda|.]", lam, (NU + r2) / (NU + D_T - 2))):
            est = np.dot(w, f) / W
            se = math.sqrt(np.dot(w ** 2, (f - est) ** 2)) / W      # delta-method SNIS SE
            it.mc(f"r_t={r}: {name}", est, ref, se)
        it.info(f"r_t={r}: ESS/n = {W ** 2 / np.dot(w, w) / n:.3f}")
    it.info(f"n = {n} prior draws per point")
    return it.done()


# =============================================================================
# Item 5 - Lemma S-composition at (s,t) = (0.3, 0.7)
# =============================================================================
def item_5(args):
    it = Item(5, "Lemma S-composition: a_s=a_s|t a_t, b_s=b_s|t+a_s|t b_t, c_s^2=c_s|t^2+a_s|t^2 c_t^2 at (s,t)=(0.3,0.7)")
    tab = table()
    i, j, N = nearest_index(TS, 0.3), nearest_index(TS, 0.7), N_GRID
    a_s, b_s, c_s = tab.abc(i, N)
    a_st, b_st, c_st = tab.abc(i, j)
    a_t, b_t, c_t = tab.abc(j, N)
    e1, e2, e3 = abs(a_s - a_st * a_t), abs(b_s - (b_st + a_st * b_t)), abs(c_s - (c_st + a_st ** 2 * c_t))
    it.check("|a_s - a_s|t a_t|", e1, 1e-15, "paper: 1e-15")
    it.check("|b_s - (b_s|t + a_s|t b_t)|", e2, 1e-15, "paper: 1e-15")
    it.check("|c_s^2 - (c_s|t^2 + a_s|t^2 c_t^2)|", e3, 1e-15, "paper: 1e-15")
    it.info(f"s={float(TS[i])!r}, t={float(TS[j])!r} (grid points); a_s={a_s:.6f} b_s={b_s:.6f} c_s^2={c_s:.6f}")
    # Extra (not listed in the paper): the same identities for profile coefficients
    # (sec:Sprofile says app:prelim applies verbatim with sigma^2 -> sigma^2_lambda).
    rng = np.random.default_rng(1005)
    lam = hm.sample_profile(tab, 1000, rng, kappa=1.3)
    s2 = hm.profile_sig2(tab, lam)
    A_s, B_s, C_s = hm.profile_abc(tab, s2, i, N)
    A_st, B_st, C_st = hm.profile_abc(tab, s2, i, j)
    A_t, B_t, C_t = hm.profile_abc(tab, s2, j, N)
    ex = max(np.max(np.abs(A_s - A_st * A_t)), np.max(np.abs(B_s - (B_st + A_st * B_t))),
             np.max(np.abs(C_s - (C_st + A_st ** 2 * C_t))))
    it.info(f"extra: profile coefficients, 1000 tempered profiles: max abs error {ex:.2e}")
    return it.done()


# =============================================================================
# Item 6 - Prop. 3(i): per-step stable mixing reproduces S-alpha-S increments
# =============================================================================
def item_6(args):
    alpha = 1.5
    it = Item(6, "Prop. 3(i): CF of sqrt(lambda_i u_i) eps_i and two-step closure, Kanter, 4e6 samples, alpha=1.5")
    rng = np.random.default_rng(1006)
    n = args.n(4_000_000)
    D = D_T
    tab = table(alpha)
    N = N_GRID
    s = tab.s_alpha ** (1 / alpha)
    ks = (0.5, 1.0, 2.0)                                  # ||xi|| * scale
    dirs = rng.standard_normal((len(ks), D)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # single steps: xi = k/s_i * dir
    acc1 = np.zeros((N, len(ks), 4))                       # sum cos, cos^2, sin, sin^2
    pairs = (0, 9, 18)
    drift = np.array([tab.m[i + 2] / tab.m[i + 1] for i in pairs])
    s2step = np.array([(drift[p] ** alpha * tab.s_alpha[i] + tab.s_alpha[i + 1]) ** (1 / alpha)
                       for p, i in enumerate(pairs)])
    acc2 = np.zeros((len(pairs), len(ks), 4))
    done = 0
    while done < n:
        nc = min(CHUNK, n - done)
        lam = hm.sample_profile(tab, nc, rng)              # released: lambda_i = c_i^{2/a} X_i (Kanter)
        incs = {}
        for i in range(N):
            eps = rng.standard_normal((nc, D))
            z = np.sqrt(lam[:, i] * tab.u[i])[:, None] * eps
            if i in pairs or (i - 1) in pairs:
                incs[i] = z
            for kk, k in enumerate(ks):
                ph = z @ (k / s[i] * dirs[kk])
                acc1[i, kk] += (np.cos(ph).sum(), (np.cos(ph) ** 2).sum(), np.sin(ph).sum(), (np.sin(ph) ** 2).sum())
        for p, i in enumerate(pairs):
            z2 = drift[p] * incs[i] + incs[i + 1]          # drift-propagate step i to t_{i+2}
            for kk, k in enumerate(ks):
                ph = z2 @ (k / s2step[p] * dirs[kk])
                acc2[p, kk] += (np.cos(ph).sum(), (np.cos(ph) ** 2).sum(), np.sin(ph).sum(), (np.sin(ph) ** 2).sum())
        done += nc

    def mc_cf(label, a, ref):
        mc_, ms = a[0] / n, a[2] / n
        se_c = math.sqrt(max(a[1] / n - mc_ ** 2, 0) / (n - 1))
        se_s = math.sqrt(max(a[3] / n - ms ** 2, 0) / (n - 1))
        p1 = it.mc(label + " Re", mc_, ref, se_c)
        p2 = it.mc(label + " Im", ms, 0.0, se_s)
        return max(abs(mc_ - ref) / se_c, abs(ms) / se_s)

    worst = 0.0
    for i in range(N):
        for kk, k in enumerate(ks):
            worst = max(worst, mc_cf(f"step {i:2d}, ||xi|| s_i={k}: E e^(i<xi,z>)", acc1[i, kk], math.exp(-k ** alpha)))
    for p, i in enumerate(pairs):
        for kk, k in enumerate(ks):
            worst = max(worst, mc_cf(f"steps {i},{i + 1} closure, ||xi|| s={k}", acc2[p, kk], math.exp(-k ** alpha)))
    it.info(f"n={n}, D={D}; worst deviation {worst:.2f} SE over {2 * (N + len(pairs)) * len(ks)} tests")
    return it.done()


# =============================================================================
# Item 7 - Corollary 1: tempered stable mixing by exponential-tilting rejection
# =============================================================================
def item_7(args):
    alpha, kappa = 1.5, 1.3
    beta = alpha / 2
    it = Item(7, "Corollary 1: tempered stable (alpha=1.5, kappa=1.3), tilt rejection, 1.5e6 samples")
    rng = np.random.default_rng(1007)
    n = args.n(1_500_000)
    D = D_T
    tab = table(alpha)
    N = N_GRID
    lam = np.concatenate([hm.sample_profile(tab, min(CHUNK, n - k), rng, kappa=kappa)
                          for k in range(0, n, CHUNK)])          # released tilt-rejection sampler
    thetas = (0.5, 2.0)
    ks = (0.5, 1.0, 2.0)
    worst = 0.0
    for i in range(N):
        c, u, L = tab.c_lap[i], tab.u[i], lam[:, i]
        # Laplace transform
        for th in thetas:
            v = np.exp(-th * L)
            ref = math.exp(-c * ((th + kappa) ** beta - kappa ** beta))
            se = v.std(ddof=1) / math.sqrt(n)
            it.mc(f"step {i:2d}: Laplace E e^(-{th} lambda)", v.mean(), ref, se); worst = max(worst, abs(v.mean() - ref) / se)
        # mean and variance
        m_ref = beta * c * kappa ** (beta - 1)
        v_ref = beta * (1 - beta) * c * kappa ** (beta - 2)
        mu = L.mean(); se_m = L.std(ddof=1) / math.sqrt(n)
        it.mc(f"step {i:2d}: E[lambda] = (a/2) c kappa^(a/2-1)", mu, m_ref, se_m); worst = max(worst, abs(mu - m_ref) / se_m)
        dev2 = (L - mu) ** 2
        var = dev2.sum() / (n - 1); se_v = dev2.std(ddof=1) / math.sqrt(n)
        it.mc(f"step {i:2d}: Var[lambda] = (a/2)(1-a/2) c kappa^(a/2-2)", var, v_ref, se_v); worst = max(worst, abs(var - v_ref) / se_v)
        # mixed increment: CF (Eq. Stempcf) and variance u E[lambda]
        eps = rng.standard_normal((n, D))
        z = np.sqrt(L * u)[:, None] * eps
        for k in ks:
            d = rng.standard_normal(D); d /= np.linalg.norm(d)
            xi = k / math.sqrt(u) * d
            ph = z @ xi
            ref = math.exp(-c * ((u * (xi @ xi) / 2 + kappa) ** beta - kappa ** beta))
            cs, sn = np.cos(ph), np.sin(ph)
            se_c, se_s = cs.std(ddof=1) / math.sqrt(n), sn.std(ddof=1) / math.sqrt(n)
            it.mc(f"step {i:2d}: increment CF, ||xi|| sqrt(u)={k} Re", cs.mean(), ref, se_c)
            it.mc(f"step {i:2d}: increment CF, ||xi|| sqrt(u)={k} Im", sn.mean(), 0.0, se_s)
            worst = max(worst, abs(cs.mean() - ref) / se_c, abs(sn.mean()) / se_s)
        zz = np.mean(z ** 2, axis=1)                      # per-sample mean of squared coordinates
        se_z = zz.std(ddof=1) / math.sqrt(n)
        it.mc(f"step {i:2d}: Var of increment coordinate = u E[lambda]", zz.mean(), u * m_ref, se_z)
        worst = max(worst, abs(zz.mean() - u * m_ref) / se_z)
    # scaling identity (Eq. Stsscaling), quantile match, at the middle step
    i = N // 2
    c, u = tab.c_lap[i], tab.u[i]
    probs = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    for h in (0.5, 2.0):
        lhs = h ** 2 * lam[:, i]                                          # h^2 lambda, lambda ~ TS(c, kappa)
        rhs = hm.sample_profile(single_step_table(tab, c * h ** alpha, u), n, rng,
                                kappa=kappa / h ** 2)[:, 0]               # TS(c h^a, kappa/h^2)
        rhs_sorted = np.sort(rhs)
        qrel = 0.0
        for p in probs:
            qa = np.quantile(lhs, p)
            Fb = np.searchsorted(rhs_sorted, qa, side="right") / n
            se = math.sqrt(2 * p * (1 - p) / n)
            it.mc(f"scaling h={h}: F_rhs(quantile_p(h^2 lambda)), p={p}", Fb, p, se)
            worst = max(worst, abs(Fb - p) / se)
            qrel = max(qrel, abs(qa / np.quantile(rhs, p) - 1))
        ks2 = stats.ks_2samp(lhs, rhs)
        it.info(f"scaling h={h}: max rel quantile difference {qrel:.2e}; two-sample KS p={ks2.pvalue:.3f}")
    it.info(f"n={n} per step, c_i in [{tab.c_lap.min():.3f},{tab.c_lap.max():.3f}]; worst deviation {worst:.2f} SE")
    return it.done()


# =============================================================================
# Item 8 - sec:Spractical: closed forms, Kanter, Gauss-Legendre, alpha=2, grid invariance
# =============================================================================
def s_alpha_quad(ts, alpha):
    out = np.empty(len(ts) - 1)
    for i in range(len(ts) - 1):
        hi = ts[i + 1]
        out[i] = q(lambda r: np.exp(0.5 * alpha * (B(r) - B(hi))) * g2(r) ** (0.5 * alpha), ts[i], hi)
    return out


def item_8(args):
    alpha = 1.5
    it = Item(8, "sec:Spractical: closed forms, Kanter + c_i^{2/alpha}, Gauss-Legendre s_i^alpha, alpha=2, grid invariance")
    tab = table(alpha)
    N = N_GRID
    # (a) closed forms vs quadrature of the defining integrals
    I = np.array([q(lambda s: g2(s) / mfun(s) ** 2, 0.0, t) for t in TS[1:]])  # int_0^t g^2/m^2
    e_sig = np.max(np.abs(tab.sig2[1:] - tab.m[1:] ** 2 * I) / tab.sig2[1:])
    e_g2 = np.max(np.abs(I - (np.exp(B(TS[1:])) - 1)) / (np.exp(B(TS[1:])) - 1))
    e_m = np.max(np.abs(tab.m[1:] - np.exp([q(lambda s: -0.5 * g2(s), 0.0, t) for t in TS[1:]])) / tab.m[1:])
    u_q = np.array([q(lambda r: (tab.m[i + 1] / mfun(r)) ** 2 * g2(r), TS[i], TS[i + 1]) for i in range(N)])
    e_u = np.max(np.abs(tab.u - u_q) / u_q)
    it.check("sigma_t^2 = 1-m_t^2 vs m_t^2 int_0^t g^2/m^2 (rel)", e_sig, 1e-10, "paper: <1e-10")
    it.check("g^2 = b_d t + b_min: int_0^t g^2/m^2 vs Karras (sigma/m)^2 = e^B-1", e_g2, 1e-10, "paper: <1e-10")
    it.check("m_t = e^{-B/2} vs exp(int_0^t f), f = -g^2/2 (rel)", e_m, 1e-10, "paper: <1e-10")
    it.check("integral-free u_i vs quadrature of Eq. (stepquants) (rel)", e_u, 1e-10, "paper: <1e-10")
    # (b) Kanter transform with c_i^{2/alpha} rescaling vs target Laplace transform
    rng = np.random.default_rng(1008)
    n = args.n(2_000_000)
    thetas = (0.1, 1.0, 10.0)
    accL = np.zeros((N, len(thetas), 2))
    accX = np.zeros((len(thetas), 2))
    done = 0
    while done < n:
        nc = min(CHUNK, n - done)
        lam = hm.sample_profile(tab, nc, rng)
        X = hm.sample_kanter(alpha / 2, nc, rng)
        for kk, th in enumerate(thetas):
            v = np.exp(-th * lam)
            accL[:, kk, 0] += v.sum(0); accL[:, kk, 1] += (v ** 2).sum(0)
            vx = np.exp(-th * X)
            accX[kk] += (vx.sum(), (vx ** 2).sum())
        done += nc

    def mcl(label, a, ref):
        m = a[0] / n
        se = math.sqrt(max(a[1] / n - m * m, 0) / (n - 1))
        it.mc(label, m, ref, se)
        return abs(m - ref) / se

    worst = 0.0
    for kk, th in enumerate(thetas):
        worst = max(worst, mcl(f"Kanter X: E e^(-{th} X) = e^(-{th}^(a/2))", accX[kk], math.exp(-th ** (alpha / 2))))
    for i in range(N):
        for kk, th in enumerate(thetas):
            worst = max(worst, mcl(f"step {i:2d}: E e^(-{th} lambda_i) = e^(-c_i {th}^(a/2))", accL[i, kk],
                                   math.exp(-tab.c_lap[i] * th ** (alpha / 2))))
    it.info(f"Kanter/Laplace: n={n}, worst deviation {worst:.2f} SE")
    # (c) Gauss-Legendre s_i^alpha, K = 2, 4, 8 (alpha=1.5, N=20)
    ref = s_alpha_quad(TS, alpha)
    ref64 = hm._s_alpha_only(TS, alpha, BETA_D, BETA_MIN, 64)
    it.info(f"reference s_i^alpha: scipy.quad vs 64-node GL agree to {np.max(np.abs(ref - ref64) / ref):.1e} (rel)")
    stated = {2: 2.6e-5, 4: 6.2e-9, 8: 1.5e-15}
    for K in (2, 4, 8):
        sK = table(alpha, K=K).s_alpha if K == 8 else hm._s_alpha_only(TS, alpha, BETA_D, BETA_MIN, K)
        e = np.max(np.abs(sK - ref) / ref)
        if K < 8:   # must reproduce the stated value to its two significant digits
            it.check(f"GL K={K}: max rel error; |measured/stated - 1| (stated {stated[K]:.1e}, meas {e:.3e})",
                     abs(e / stated[K] - 1), 0.05 / 2.6 if K == 2 else 0.05 / 6.2, "2 significant digits")
        else:       # stated value is at round-off level: require round-off level
            it.check(f"GL K=8 (precompute default): max rel error (stated {stated[K]:.1e})", e, 5e-15,
                     "round-off level")
    # (d) alpha = 2 degeneration s_i^2 = u_i
    tab2 = table(2.0)                                      # precompute also asserts this (rtol 1e-10)
    it.check("alpha=2: max rel |s_i^2 - u_i| (precompute, K=8)", np.max(np.abs(tab2.s_alpha - tab2.u) / tab2.u),
             1e-13, "paper: <1e-13")
    # (e) grid invariance of sum_i (m_T/m_{t_{i+1}})^alpha s_i^alpha over N in {20, 40, 80};
    #     the coarse grid N=5 is reported (paper: relative spread 2e-11 with the 8-node rule)
    s0T = q(lambda r: np.exp(0.5 * alpha * (B(r) - B(T_END))) * g2(r) ** (0.5 * alpha), 0.0, T_END)
    tots = {}
    for Ng in (5, 20, 40, 80):
        ts = np.linspace(0.0, T_END, Ng + 1)
        try:
            sa = table(alpha, ts=ts).s_alpha
            how = "precompute"
        except AssertionError as exc:
            it.info(f"N={Ng}: hm.precompute raised AssertionError('{exc}'); "
                    f"using hm._s_alpha_only (same K=8 rule) instead")
            sa = hm._s_alpha_only(ts, alpha, BETA_D, BETA_MIN, 8)
            how = "_s_alpha_only"
        m = mfun(ts)
        tots[Ng] = np.sum((m[-1] / m[1:]) ** alpha * sa)
        it.info(f"N={Ng:3d} ({how}, K=8): total = {float(tots[Ng])!r}; rel diff to s_0T^alpha (quad) "
                f"{abs(tots[Ng] / s0T - 1):.2e}")
    grid = (20, 40, 80)
    spread = max(abs(tots[a] - tots[b]) / tots[20] for a in grid for b in grid)
    it.check("grid invariance: max rel spread over N in {20,40,80} (K=8)", spread, 1e-15, "paper: <1e-15")
    d5 = abs(tots[5] - tots[20]) / tots[20]
    it.check("coarse grid N=5 vs N=20 (K=8)", d5, 5e-11, "paper: 2e-11")
    # diagnostic: the N=5 deviation is Gauss-Legendre resolution on the wider steps
    for K in (12, 16):
        t5 = np.linspace(0.0, T_END, 6)
        tot5 = np.sum((mfun(t5)[-1] / mfun(t5)[1:]) ** alpha * hm._s_alpha_only(t5, alpha, BETA_D, BETA_MIN, K))
        it.info(f"diagnostic: N=5 with K={K} nodes: rel diff to N=20 total {abs(tot5 - tots[20]) / tots[20]:.1e}")
    return it.done()


ITEMS = {1: item_1, 2: item_2, 3: item_3, 4: item_4, 5: item_5, 6: item_6, 7: item_7, 8: item_8}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="divide Monte-Carlo sample sizes by 20")
    ap.add_argument("--items", type=int, nargs="*", default=sorted(ITEMS))
    args = ap.parse_args()
    args.n = (lambda n: n // 20) if args.fast else (lambda n: n)
    print(f"HeavyBridge numerical verification  (numpy {np.__version__}, scipy {__import__('scipy').__version__})")
    print(f"mixing module: {hm.__file__}")
    print(f"schedule: VP-Karras beta_d={BETA_D}, beta_min={BETA_MIN}, T={T_END}, uniform grid N={N_GRID}; "
          f"MC tolerance = {MC_Z:g} SE; {'FAST (samples/20)' if args.fast else 'paper sample sizes'}")
    t0 = time.time()
    res = {k: ITEMS[k](args) for k in args.items}
    print("\n=== Summary ===")
    for k, (ok, dt) in res.items():
        print(f"  Item {k}: {'PASS' if ok else 'FAIL'}  ({dt:.1f} s)")
    print(f"  total runtime {time.time() - t0:.1f} s")
    return 0 if all(ok for ok, _ in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
