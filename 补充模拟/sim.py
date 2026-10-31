"""Small simulations for the talk.

textGrain here = the paper's eq.(1) with B = 2 blocks, fixed partition
(top token alone in block 1, everything else in block 2), beta = 1
(non-binding entropy budget, so pi is the plain LP optimal coupling).
For B = 2 the LP is a fractional knapsack: block-2 mass Delta is poured
into the columns with the largest Z_2j - Z_1j.
"""
import json
import numpy as np
from scipy.special import digamma
from scipy.stats import gamma as gamma_dist

rng = np.random.default_rng(20261031)
EG = np.euler_gamma


def gumbel(size):
    return -np.log(-np.log(rng.random(size)))


def FG(z):
    return np.exp(-np.exp(-z))


def textgrain_pivots(N, delta, m):
    """Return N watermarked pivots Y = F_G(Z_{g(W),J})."""
    Z = gumbel((N, 2, m))
    d = Z[:, 1, :] - Z[:, 0, :]
    order = np.argsort(-d, axis=1)
    k = int(np.floor(delta * m))
    rem = delta - k / m
    pi2_sorted = np.zeros(m)
    pi2_sorted[:k] = 1.0 / m
    if k < m:
        pi2_sorted[k] = rem
    pi2 = np.zeros((N, m))
    np.put_along_axis(pi2, order, np.broadcast_to(pi2_sorted, (N, m)), axis=1)
    J = rng.integers(0, m, N)
    p_block2 = m * pi2[np.arange(N), J]
    b = (rng.random(N) < p_block2).astype(int)
    return FG(Z[np.arange(N), b, J])


def textgrain_meanshift(delta, m, N=400_000):
    """E_1 h(Y) - E_0 h(Y), h(y) = -log(1-y), with the pi-weighted
    conditional expectation and the independent coupling as control variate."""
    out = []
    for _ in range(N // 100_000):
        n = 100_000
        Z = gumbel((n, 2, m))
        d = Z[:, 1, :] - Z[:, 0, :]
        order = np.argsort(-d, axis=1)
        k = int(np.floor(delta * m))
        rem = delta - k / m
        s = np.zeros(m)
        s[:k] = 1.0 / m
        if k < m:
            s[k] = rem
        pi2 = np.zeros((n, m))
        np.put_along_axis(pi2, order, np.broadcast_to(s, (n, m)), axis=1)
        pi1 = 1.0 / m - pi2
        h = -np.log1p(-FG(Z))
        indep2 = delta / m
        indep1 = (1 - delta) / m
        val = ((pi1 - indep1) * h[:, 0, :]).sum(1) + ((pi2 - indep2) * h[:, 1, :]).sum(1)
        out.append(val)
    v = np.concatenate(out)
    return v.mean(), v.std() / np.sqrt(len(v))


def gm_meanshift(delta):
    P = np.array([1 - delta, delta])
    return float(np.sum(P * (digamma(1 + 1 / P) + EG)) - 1.0)


def gm_pivots(N, delta):
    P1 = 1 - delta
    W2 = rng.random(N) < delta
    a = np.where(W2, delta, P1)
    return rng.random(N) ** a


# ---------- scan test of Algorithm 1 ----------
def grid_and_weights(n):
    Ln = max(2, int(np.ceil(np.log(n + 1))))
    dy = {2.0 ** (-j) for j in range(1, int(np.ceil(np.log2(max(n, 2)))) + 1)}
    mesh = {k / Ln for k in range(1, Ln)}
    U = np.array(sorted(dy | mesh))
    w = 1.0 / (2 * len(U)) + 0.5 * np.isclose(U, 0.5)
    return U, w


def kl_bern(a, u):
    a = np.clip(a, 1e-300, 1 - 1e-16)
    t1 = np.where(a > 0, a * np.log(a / u), 0.0)
    t2 = np.where(a < 1, (1 - a) * np.log((1 - a) / (1 - u)), 0.0)
    return t1 + t2


def scan_reject(p, alpha, U, w):
    n = p.shape[-1]
    F = (p[..., None] <= U).mean(axis=-2)
    stat = n * kl_bern(F, U) + np.log(w)
    stat = np.where(F > U, stat, -np.inf)
    return stat.max(-1) > np.log(1 / alpha)


def power(n, eps, sampler, reps=400, alpha=0.05):
    U, w = grid_and_weights(n)
    c = gamma_dist.ppf(1 - alpha, n)
    rs = rk = 0
    for _ in range(reps):
        Y = rng.random(n)
        mask = rng.random(n) < eps
        if mask.any():
            Y[mask] = sampler(mask.sum())
        T = -np.log1p(-Y).sum()
        rs += T > c
        rk += scan_reject(1 - Y, alpha, U, w)
    return rs / reps, rk / reps


if __name__ == "__main__":
    res = {}
    # 1. mean shift vs Delta (m = 2)
    deltas = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    tg, gm = [], []
    for dlt in deltas:
        mu, se = textgrain_meanshift(dlt, 2)
        tg.append((mu, se))
        gm.append(gm_meanshift(dlt))
        print(f"Delta={dlt:<6} textGrain={mu:.5f}±{se:.5f}  ratio/Delta={mu/dlt:.3f}  GM={gm[-1]:.5f}  GM/(D log 1/D)={gm[-1]/(dlt*np.log(1/dlt)):.3f}")
    res["deltas"] = deltas
    res["tg"] = tg
    res["gm"] = gm

    # 2. densities of Y at Delta = 0.3 (m = 2) and m = 64, Delta = 1/64
    Ytg = textgrain_pivots(2_000_000, 0.3, 2)
    Ygm = gm_pivots(2_000_000, 0.3)
    bins = np.linspace(0, 1, 41)
    res["dens_bins"] = bins.tolist()
    res["dens_tg"] = np.histogram(Ytg, bins, density=True)[0].tolist()
    res["dens_gm"] = np.histogram(Ygm, bins, density=True)[0].tolist()
    # p-value tail for growing columns
    Y2 = textgrain_pivots(1_000_000, 1 / 64, 2)
    Y64 = textgrain_pivots(1_000_000, 1 / 64, 64)
    pb = np.logspace(-4, 0, 25)
    res["tail_u"] = pb.tolist()
    res["tail_m2"] = [float(np.mean(1 - Y2 <= u)) for u in pb]
    res["tail_m64"] = [float(np.mean(1 - Y64 <= u)) for u in pb]
    print("mean shift m=2,Delta=1/64:", np.mean(-np.log1p(-Y2)) - 1, " m=64:", np.mean(-np.log1p(-Y64)) - 1)
    for u in [1e-3, 1e-2, 0.05, 0.5]:
        print(f"P(p<={u}) m=2: {np.mean(1-Y2<=u):.4f}  m=64: {np.mean(1-Y64<=u):.4f}")

    # 3. power: sum vs scan
    n = 2000
    eps_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    pw_a = [power(n, e, lambda k: textgrain_pivots(k, 0.2, 2)) for e in eps_grid]
    print("config A (m=2, Delta=0.2):", pw_a)
    eps_grid_b = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    pw_b = [power(n, e, lambda k: textgrain_pivots(k, 1 / 64, 64)) for e in eps_grid_b]
    print("config B (m=64, Delta=1/64):", pw_b)
    res["n"] = n
    res["eps_a"] = eps_grid
    res["pw_a"] = pw_a
    res["eps_b"] = eps_grid_b
    res["pw_b"] = pw_b
    with open("sim_results.json", "w") as f:
        json.dump(res, f)
