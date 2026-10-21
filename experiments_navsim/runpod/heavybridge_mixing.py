"""HeavyBridge mixing module — self-contained implementation of the quantities
in Eq. (16) and the boxed Eqs. (19)-(23) of the paper.

Framework-agnostic core (numpy); thin torch wrappers at the bottom.
Every closed form here was verified against quadrature / Monte Carlo
(supplement, Sec. F). The asserts in `precompute` are the two built-in
correctness checks of Algorithm 3 (Precompute).

Variants (flag `mixing`):
    "none"      : Gaussian bridge  == BridgeDrive        (Alg. 1, nu=inf)
    "ig"        : per-rollout lambda ~ IG(nu/2, nu/2)    (Alg. 1)
    "ig-learned": lambda ~ IG(A,B) from an amortized head (Alg. 2; head external)
    "stable"    : per-step positive alpha/2-stable        (Alg. 3)
    "tempered"  : per-step tempered stable, tempering kap  (Alg. 3 + Cor. 1)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Offline table  (Algorithm 3, Procedure Precompute; Eq. (16))
# ----------------------------------------------------------------------------

@dataclass
class ScheduleTable:
    ts: np.ndarray       # grid t_0..t_N
    m: np.ndarray        # m_{t_i}
    sig2: np.ndarray     # sigma^2_{t_i}
    u: np.ndarray        # u_i,      i=0..N-1
    s_alpha: np.ndarray  # s_i^alpha, i=0..N-1
    rho: np.ndarray      # varrho_i,  i=0..N-1
    c_lap: np.ndarray    # c_i = (2/u_i)^{a/2} s_i^a
    alpha: float
    beta_d: float
    beta_min: float

    # bridge coefficients at grid index i, horizon index j (i<=j), homogeneous
    def abc(self, i: int, j: int, sig2=None):
        sig2 = self.sig2 if sig2 is None else sig2
        g2 = (self.m[j] ** 2 * sig2[i]) / (self.m[i] ** 2 * sig2[j])
        a = self.m[i] * g2 / self.m[j]
        b = self.m[i] * (1.0 - g2)
        c2 = sig2[i] * (1.0 - g2)
        return a, b, c2


def precompute(ts: np.ndarray, alpha: float, beta_d: float = 2.0,
               beta_min: float = 0.1, K: int = 8) -> ScheduleTable:
    """Eq. (16): closed forms + K-point Gauss-Legendre for s_i^alpha."""
    B = lambda t: 0.5 * beta_d * t ** 2 + beta_min * t
    m = np.exp(-0.5 * B(ts))
    sig2 = 1.0 - m ** 2
    N = len(ts) - 1
    u = sig2[1:] - (m[1:] / m[:-1]) ** 2 * sig2[:-1]                 # integral-free
    xk, wk = np.polynomial.legendre.leggauss(K)                     # GL nodes/weights
    s_alpha = np.empty(N)
    for i in range(N):
        lo, hi = ts[i], ts[i + 1]
        r = 0.5 * (hi - lo) * xk + 0.5 * (hi + lo)
        integrand = np.exp(0.5 * alpha * (B(r) - B(ts[i + 1]))) * \
                    (beta_d * r + beta_min) ** (0.5 * alpha)
        s_alpha[i] = 0.5 * (hi - lo) * np.sum(wk * integrand)
    rho = (m[-1] / m[1:]) ** 2 * u
    c_lap = (2.0 / u) ** (0.5 * alpha) * s_alpha

    # --- Algorithm 3 asserts (built-in correctness checks) ---
    if abs(alpha - 2.0) < 1e-12:
        assert np.allclose(s_alpha, u, rtol=1e-10), "alpha=2 must give s_i^2 = u_i"
    tot = np.sum((m[-1] / m[1:]) ** alpha * s_alpha)
    ts2 = np.linspace(ts[0], ts[-1], 2 * N + 1)                     # refined grid
    tab2_salpha = _s_alpha_only(ts2, alpha, beta_d, beta_min, K)
    m2 = np.exp(-0.5 * B(ts2))
    tot2 = np.sum((m2[-1] / m2[1:]) ** alpha * tab2_salpha)
    assert abs(tot - tot2) / tot < 1e-12, "propagated sum must be grid-invariant"

    return ScheduleTable(ts, m, sig2, u, s_alpha, rho, c_lap, alpha, beta_d, beta_min)


def _s_alpha_only(ts, alpha, beta_d, beta_min, K):
    B = lambda t: 0.5 * beta_d * t ** 2 + beta_min * t
    xk, wk = np.polynomial.legendre.leggauss(K)
    out = np.empty(len(ts) - 1)
    for i in range(len(ts) - 1):
        lo, hi = ts[i], ts[i + 1]
        r = 0.5 * (hi - lo) * xk + 0.5 * (hi + lo)
        out[i] = 0.5 * (hi - lo) * np.sum(
            wk * np.exp(0.5 * alpha * (B(r) - B(ts[i + 1]))) *
            (beta_d * r + beta_min) ** (0.5 * alpha))
    return out

# ----------------------------------------------------------------------------
# Mixing-variable samplers
# ----------------------------------------------------------------------------

def sample_ig(nu: float, size, rng, stratified: bool = True):
    """lambda ~ IG(nu/2, nu/2). Stratified inverse-CDF over the batch axis."""
    n = int(np.prod(size))
    if stratified:
        qs = (np.arange(n) + rng.uniform(size=n)) / n
        rng.shuffle(qs)
        from scipy.stats import invgamma
        lam = invgamma.ppf(qs, a=nu / 2, scale=nu / 2)
    else:
        lam = 1.0 / rng.gamma(nu / 2, 2.0 / nu, size=n)
    return lam.reshape(size)


def sample_kanter(beta: float, size, rng):
    """X with E e^{-th X} = e^{-th^beta}  (Kanter; Eq. (16), last two lines)."""
    U = rng.uniform(0.0, np.pi, size=size)
    W = rng.exponential(1.0, size=size)
    return (np.sin(beta * U) / np.sin(U) ** (1.0 / beta)) * \
           (np.sin((1.0 - beta) * U) / W) ** ((1.0 - beta) / beta)


def sample_profile(tab: ScheduleTable, batch: int, rng, kappa: float | None = None,
                   max_tries: int = 64):
    """lambda_{1:N} per sample: stable (kappa=None) or tempered (Cor. 1).
    Returns (batch, N). Tempered: exponential-tilting rejection, exact."""
    beta = tab.alpha / 2.0
    N = len(tab.u)
    scale = tab.c_lap ** (1.0 / beta)                    # lambda_i = c_i^{2/alpha} X_i
    if kappa is None:
        return scale[None, :] * sample_kanter(beta, (batch, N), rng)
    lam = np.empty((batch, N))
    todo = np.ones((batch, N), dtype=bool)
    for _ in range(max_tries):
        n_todo = todo.sum()
        if n_todo == 0:
            break
        prop = (scale[None, :] * sample_kanter(beta, (batch, N), rng))[todo]
        acc = rng.uniform(size=n_todo) < np.exp(-kappa * prop)
        idx = np.argwhere(todo)
        lam[idx[acc, 0], idx[acc, 1]] = prop[acc]
        todo[idx[acc, 0], idx[acc, 1]] = False
    if todo.any():   # acceptance too low for (c_i, kappa): use Devroye (2009) instead
        raise RuntimeError("tilted-stable rejection: increase max_tries or use "
                           "a Devroye (2009) sampler for this (alpha, kappa)")
    return lam

# ----------------------------------------------------------------------------
# Per-sample profile schedule and coefficients (Eqs. (19)-(20))
# ----------------------------------------------------------------------------

def profile_sig2(tab: ScheduleTable, lam: np.ndarray) -> np.ndarray:
    """Eq. (19): sigma^2_{i,lambda}, recursion, O(N). lam: (batch, N) -> (batch, N+1)."""
    batch, N = lam.shape
    out = np.zeros((batch, N + 1))
    for i in range(N):
        out[:, i + 1] = (tab.m[i + 1] / tab.m[i]) ** 2 * out[:, i] + lam[:, i] * tab.u[i]
    return out


def profile_abc(tab: ScheduleTable, sig2_lam: np.ndarray, i: int, j: int):
    """Eq. (20) generalized to horizon index j: coefficients at t_i given endpoint t_j."""
    g2 = (tab.m[j] ** 2 * sig2_lam[:, i]) / (tab.m[i] ** 2 * sig2_lam[:, j])
    a = tab.m[i] * g2 / tab.m[j]
    b = tab.m[i] * (1.0 - g2)
    c2 = sig2_lam[:, i] * (1.0 - g2)
    return a, b, c2

# ----------------------------------------------------------------------------
# SNIS over the endpoint tilt (Eq. (21));  v(lambda) = sum_i rho_i lambda_i
# ----------------------------------------------------------------------------

def snis_resample(tab: ScheduleTable, x0: np.ndarray, y: np.ndarray,
                  M: int, rng, kappa: float | None = None):
    """Draw M prior profiles, weight by N(y; m_T x0, v(lam) I), resample one.
    x0, y: (D,). Returns (lam (N,), ess)."""
    lam = sample_profile(tab, M, rng, kappa=kappa)              # (M, N)
    v = lam @ tab.rho                                            # (M,)
    D = x0.shape[0]
    r2 = np.sum((y - tab.m[-1] * x0) ** 2)
    logw = -0.5 * D * np.log(2 * np.pi * v) - 0.5 * r2 / v
    logw -= logw.max()
    w = np.exp(logw); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    return lam[rng.choice(M, p=w)], ess

# ----------------------------------------------------------------------------
# DBIM step with churn  (Eq. (5) / Eq. (23))
# ----------------------------------------------------------------------------

def dbim_step(x_t, y, x0_hat, a_s, b_s, c2_s, a_t, b_t, c2_t, varsigma, rng):
    """One reverse step t -> s. All coefficient args are per-sample arrays.
    Global-mixing case: pass lambda-scaled c2's and varsigma (Prop. 3 note:
    the sqrt(lambda) cancels in the deterministic part automatically)."""
    eps_hat = (x_t - a_t[:, None] * y - b_t[:, None] * x0_hat) / np.sqrt(c2_t)[:, None]
    det = np.sqrt(np.clip(c2_s - varsigma ** 2, 0.0, None))[:, None] * eps_hat
    churn = varsigma[:, None] * rng.standard_normal(x_t.shape)
    return a_s[:, None] * y + b_s[:, None] * x0_hat + det + churn

# ----------------------------------------------------------------------------
# Torch adapters (used inside the training loop; thin on purpose)
# ----------------------------------------------------------------------------
try:
    import torch

    def torch_noise_level_embedding(lam_or_signal, dim: int = 64):
        """FiLM input: log lambda (Algs. 1-2) or sigma_{t,lambda} (Alg. 3)."""
        x = torch.log(lam_or_signal).unsqueeze(-1)
        half = dim // 2
        freqs = torch.exp(torch.linspace(0., 8., half, device=x.device))
        return torch.cat([torch.sin(x * freqs), torch.cos(x * freqs)], dim=-1)
except ImportError:      # numpy-only environments (e.g., this pack's self-test)
    pass

# ----------------------------------------------------------------------------
# Self-test: reproduces the paper's verified identities
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ts = np.linspace(0.0, 1.0, 21)

    tab2 = precompute(ts, alpha=2.0)                     # asserts alpha=2 check
    tab = precompute(ts, alpha=1.5)                      # asserts grid invariance
    print("Precompute asserts passed (alpha=2 degeneracy, grid invariance).")

    lam = sample_profile(tab, 200_000, rng)              # stable prior, Laplace check
    th = 0.7
    lhs = np.mean(np.exp(-th * lam[:, 3]))
    rhs = np.exp(-tab.c_lap[3] * th ** (tab.alpha / 2))
    assert abs(lhs - rhs) < 5e-3, (lhs, rhs)
    print(f"Stable Laplace check: MC={lhs:.5f} closed={rhs:.5f}")

    lamT = sample_profile(tab, 50_000, rng, kappa=1.3)   # tempered mean check
    beta = tab.alpha / 2
    assert np.allclose(lamT[:, 3].mean(),
                       beta * tab.c_lap[3] * 1.3 ** (beta - 1), rtol=2e-2)
    print("Tempered mean check passed.")

    x0 = rng.standard_normal(8); y = x0 + 0.3 * rng.standard_normal(8)
    lam_star, ess = snis_resample(tab, x0, y, M=512, rng=rng)
    print(f"SNIS ok, ESS/M = {ess/512:.2f}")

    s2 = profile_sig2(tab, np.ones((1, 20)))             # lambda==1 -> sigma_t^2
    assert np.allclose(s2[0], tab.sig2, rtol=1e-10)
    print("Profile schedule reduces to sigma_t^2 at lambda=1. All self-tests passed.")
