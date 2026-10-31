"""Fast exact sampler for textGrain with B=2, beta=1, Delta = 1/m (one block-2 column).
Block 2 is drawn iff J hits j* = argmax_j (Z_2j - Z_1j). Column J's diff is
compared with the max of the other m-1 iid logistic diffs, sampled by inversion."""
import json, numpy as np
from scipy.stats import gamma as gamma_dist
from sim import textgrain_pivots, grid_and_weights, scan_reject, rng, FG, gumbel

def fast_pivots(N, m):
    Z = gumbel((N, 2))
    U = rng.random(N)
    Mp = np.log(U**(1.0/(m-1)) / (1 - U**(1.0/(m-1))))   # logistic quantile
    hit = (Z[:, 1] - Z[:, 0]) > Mp
    return np.where(hit, FG(Z[:, 1]), FG(Z[:, 0]))

# validate against full-table sampler at m = 64
a = textgrain_pivots(1_000_000, 1/64, 64); b = fast_pivots(1_000_000, 64)
for u in [1e-3, 1e-2, 0.1, 0.5]:
    print(f"check u={u}: full {np.mean(1-a<=u):.4f} fast {np.mean(1-b<=u):.4f}")
print("mean h full", np.mean(-np.log1p(-a)), "fast", np.mean(-np.log1p(-b)))

def power(n, eps, sampler, reps, alpha=0.05):
    U, w = grid_and_weights(n); c = gamma_dist.ppf(1-alpha, n)
    rs = rk = 0
    for _ in range(reps):
        Y = rng.random(n); mask = rng.random(n) < eps
        if mask.any(): Y[mask] = sampler(mask.sum())
        rs += (-np.log1p(-Y).sum()) > c
        rk += scan_reject(1-Y, alpha, U, w)
    return float(rs/reps), float(rk/reps)

m = 4096; n = 300_000
epsB = [0.0, 0.25, 0.5, 0.75, 1.0]
pwB = [power(n, e, lambda k: fast_pivots(k, m), reps=400) for e in epsB]
print("config B (m=4096, Delta=1/4096, n=3e5):", pwB)
epsA = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
pwA = [power(2000, e, lambda k: textgrain_pivots(k, 0.2, 2), reps=1000) for e in epsA]
print("config A (m=2, Delta=0.2, n=2000):", pwA)
res = json.load(open("sim_results.json"))
res.update(epsA=epsA, pwA=pwA, epsB=epsB, pwB=pwB, nA=2000, nB=n, mB=m)
json.dump(res, open("sim_results.json", "w"))
