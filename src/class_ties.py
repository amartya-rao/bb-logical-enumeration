#!/usr/bin/env python3
"""
class_ties.py, the numbers quoted in the correction section of FLATNESS.md.

Everything here is exact. No Monte Carlo.

bb18
  The [[18,4,4]] bicycle code, l = m = 3, A = x + I + y^2, B = y + I + x^2.
  All 2^18 X-error patterns are enumerated and grouped into (syndrome, logical
  class) cells, so the optimal logical-class failure probability is an exact
  polynomial in the per-qubit rates.

  Shows: the frozen-decoder site derivatives are equal across all 18 qubits, so
  a site ranking carries no information; yet two arrangements of the same rate
  multiset have different one-sided derivatives under a decoder recalibrated to
  each arrangement. The gain is linear, not quadratic.

bb72
  The leading low-noise coefficient of the [[72,12,6]] code. With d_X = 2t and
  t = 3, F_ML(p*r) = p^3 C(r) + O(p^4) where

      C(r) = sum_s [ sum_l W_{s,l}(r) - max_l W_{s,l}(r) ]

  and W sums the product of r_i over weight-3 errors in each (syndrome, class).
  The max is what keeps the competition between logical explanations.

penalty
  The controlled test of whether reweighting helps when the objective is
  minimising weight. Two arms, penalty 0.0 and 0.30, same class and same
  forbidden set. Takes a couple of minutes.

Usage:
    python3 class_ties.py bb18
    python3 class_ties.py bb72 [--bounds]
    python3 class_ties.py penalty
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from bbcode import BBCode, rank
from distance import z_logical_basis


# --------------------------------------------------------------------------
# BB18
# --------------------------------------------------------------------------

def bb18_code():
    return BBCode(3, 3, [(1, 0), (0, 0), (0, 2)], [(0, 1), (0, 0), (2, 0)])


# Two arrangements of the same rate multiset within each half: four entries of
# -1, one 0, four of +1 per half.
#
# They cannot be related by any code automorphism, and the proof is the result
# itself: an automorphism would give them identical failure probabilities, and
# they differ. Checked directly for the 9 translations and the ZX duality, alone
# and composed, none of which relates them.
XA = np.array([-1, 1, 1, 1, 1, -1, -1, -1, 0,
               -1, -1, 1, 1, 0, 1, -1, 1, -1], float)
XB = np.array([-1, -1, 1, 0, -1, 1, 1, -1, 1,
                1, -1, -1, 1, -1, 0, 1, -1, 1], float)


def bb18(p0=0.01):
    c = bb18_code()
    n = c.n
    lz = z_logical_basis(c)

    print(f"  code: n={n} k={c.k} CSS={c.commutes()} "
          f"rank(H_X)={rank(c.HX)} H_X==H_Z={bool((c.HX == c.HZ).all())}")

    E = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    sid = (((E @ c.HZ.T.astype(np.int64)) % 2) * (1 << np.arange(9))).sum(1)
    lid = (((E @ lz.T.astype(np.int64)) % 2) * (1 << np.arange(lz.shape[0]))).sum(1)
    _, sidx = np.unique(sid, return_inverse=True)
    gid = sidx * 16 + lid
    w = E.sum(1)

    print(f"  {len(np.unique(sid))} syndromes x {len(np.unique(lid))} classes "
          f"= {len(np.unique(gid))} cells, {(1 << n) // len(np.unique(gid))} "
          f"errors each")

    # integer weight enumerator per cell, so ties can be checked exactly
    N = np.bincount(gid * 19 + w, minlength=2048 * 19).reshape(2048, 19)
    ws = np.arange(19)
    Z = (N * (p0 ** ws) * ((1 - p0) ** (n - ws))).sum(1).reshape(128, 16)
    mx = Z.max(1)
    T = [np.nonzero(Z[s] >= mx[s] - 1e-20)[0] for s in range(128)]

    print(f"\n  F_0 at p0={p0}        = {1 - mx.sum():.16f}")
    tied = sum(1 for t in T if len(t) > 1)
    N3 = N.reshape(128, 16, 19)
    exact = all((N3[s][t[0]] == N3[s][tj]).all()
                for s, t in enumerate(T) for tj in t)
    print(f"  syndromes with tied classes = {tied} of 128")
    print(f"  tied classes have identical integer weight enumerators: {exact}")
    print("    (so the ties are structural, not floating point)")

    # gradients of each cell probability at the uniform point
    P = ((p0 ** ws) * ((1 - p0) ** (n - ws)))[w]
    tot = np.bincount(gid, weights=P, minlength=2048)
    S = np.stack([np.bincount(gid, weights=P * E[:, i], minlength=2048)
                  for i in range(n)], 1)
    V = (S / p0 - (tot[:, None] - S) / (1 - p0)).reshape(128, 16, n)

    gF = np.zeros(n)
    for s, t in enumerate(T):
        gF -= V[s][t].mean(0)                 # decoder splits ties uniformly
    print(f"\n  frozen decoder dF/dp_i, all 18 qubits equal: "
          f"{np.allclose(gF, gF[0], rtol=0, atol=1e-14)}")
    print(f"    value = {gF[0]:.14f}")
    print("    a site ranking therefore carries no information here")

    for nm, x in (("A", XA), ("B", XB)):
        assert abs(x[:9].sum()) < 1e-9 and abs(x[9:].sum()) < 1e-9
        assert sorted(x[:9]) == sorted(x[9:])
    D = lambda x: -sum(max(V[s][l] @ x for l in T[s]) for s in range(128))
    dA, dB = D(XA), D(XB)
    print(f"\n  one-sided derivatives under a recalibrated optimal decoder")
    print(f"    F_A'(0+) = {dA:.13f}")
    print(f"    F_B'(0+) = {dB:.13f}")
    print(f"    B - A    = {dB - dA:.13f}  per unit eps, so the gap is LINEAR")

    def cells(p):
        pe = np.prod(np.where(E == 1, p, 1 - p), axis=1)
        return np.bincount(gid, weights=pe, minlength=2048).reshape(128, 16)
    Fstar = lambda p: 1 - cells(p).max(1).sum()
    Ffroz = lambda p: 1 - sum(cells(p)[s][t].mean() for s, t in enumerate(T))

    e = 0.001
    a, b = Fstar(p0 + e * XA), Fstar(p0 + e * XB)
    fa, fb = Ffroz(p0 + e * XA), Ffroz(p0 + e * XB)
    print(f"\n  at eps = {e}          A                  B")
    print(f"    frozen       {fa:.15f}  {fb:.15f}")
    print(f"    recalibrated {a:.15f}  {b:.15f}")
    print(f"    A is {100 * (b - a) / b:.4f}% better under recalibrated decoding")

    print(f"\n  scaling of |F_A - F_B|, which regime is each decoder in?")
    print(f"    {'eps':>9}  {'frozen':>12} {'exp':>6}   {'recalibrated':>12} {'exp':>6}")
    prev = None
    for eps in (4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5):
        df = abs(Ffroz(p0 + eps * XA) - Ffroz(p0 + eps * XB))
        dc = abs(Fstar(p0 + eps * XA) - Fstar(p0 + eps * XB))
        if prev:
            sf = np.log(df / prev[0]) / np.log(eps / prev[2])
            sc = np.log(dc / prev[1]) / np.log(eps / prev[2])
            print(f"    {eps:9.2e}  {df:12.4e} {sf:6.3f}   {dc:12.4e} {sc:6.3f}")
        else:
            print(f"    {eps:9.2e}  {df:12.4e} {'-':>6}   {dc:12.4e} {'-':>6}")
        prev = (df, dc, eps)
    print("    exponent 2 is quadratic, 1 is a kink. The frozen decoder is")
    print("    smooth at the symmetric point, the recalibrated one is not.")

    print(f"\n  the response is one-sided, not of the form -c|eps|:")
    for eps in (1e-3, -1e-3):
        print(f"    eps={eps:+.0e}  F*_A - F*_B = "
              f"{Fstar(p0 + eps * XA) - Fstar(p0 + eps * XB):+.6e}")


# --------------------------------------------------------------------------
# BB72
# --------------------------------------------------------------------------

# Two BB72 placements using the same rate multiset in each half: 18 sites at
# 0.000325 and 18 at 0.000675 per half. The listed indices are the low-rate ones.
PA = [2, 3, 5, 7, 8, 10, 16, 17, 18, 20, 23, 24, 28, 29, 30, 31, 34, 35,
      38, 39, 41, 42, 48, 50, 53, 57, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70]
PB = [1, 3, 4, 5, 8, 13, 14, 16, 17, 18, 19, 22, 24, 26, 28, 29, 31, 32,
      38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 54, 59, 64, 67, 69, 70, 71]


def bb72_bounds(c, lz, p=0.0005):
    """Certified bounds on the optimal failure probability, no sampling.

    Enumerate every X-error of weight <= 4 and aggregate by (syndrome, class).
    With L4 the optimal failure mass of that truncated distribution and D the
    probability of five or more errors,

        L4 <= F_ML <= L4 + D

    Upper: omitted mass can only raise each class, and max(a+d) >= max(a).
    Lower: max(a+d) <= max(a) + sum(d).
    """
    n = c.n
    hz = np.array([int(''.join(map(str, c.HZ[:, j][::-1])), 2) for j in range(n)])
    lc = np.array([int(''.join(map(str, lz[:, j][::-1])), 2) for j in range(n)])

    keys, idxs = [], []
    for w in range(5):
        if w == 0:
            keys.append(np.zeros(1, dtype=np.int64))
            idxs.append(np.zeros((1, 0), dtype=np.int64))
            continue
        idx = np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(range(n), w)), dtype=np.int64).reshape(-1, w)
        s = np.zeros(len(idx), dtype=np.int64)
        l = np.zeros(len(idx), dtype=np.int64)
        for j in range(w):
            s ^= hz[idx[:, j]]
            l ^= lc[idx[:, j]]
        keys.append(s * 4096 + l)
        idxs.append(idx)
    total = sum(len(k) for k in keys)
    u, inv = np.unique(np.concatenate(keys), return_inverse=True)
    syn = u >> 12
    o = np.argsort(syn, kind="stable")
    st = np.r_[0, np.nonzero(np.diff(syn[o]))[0] + 1]
    print(f"  enumerated {total} patterns of weight <= 4")

    def one(sites):
        pr = np.full(n, 27 * p / 20)      # 0.000675
        pr[list(sites)] = 13 * p / 20     # 0.000325
        Q = np.prod(1 - pr)
        od = pr / (1 - pr)
        vals = []
        for idx in idxs:
            v = np.full(len(idx), Q)
            for j in range(idx.shape[1]):
                v = v * od[idx[:, j]]
            vals.append(v)
        W = np.bincount(inv, weights=np.concatenate(vals), minlength=len(u))
        L4 = W.sum() - np.maximum.reduceat(W[o], st).sum()
        # tail: probability of five or more errors. Summed directly, because
        # 1 - Pr(<=4) loses about ten digits to cancellation at this size.
        dp = np.zeros(n + 1)
        dp[0] = 1.0
        for q in pr:
            nd = dp * (1 - q)
            nd[1:] += dp[:-1] * q
            dp = nd
        return L4, float(dp[5:].sum())

    la, tail = one(PA)
    lb, _ = one(PB)
    print(f"  omitted tail, Pr(>=5 errors)   {tail:.10e}")
    print(f"  A: [{la:.10e}, {la + tail:.10e}]")
    print(f"  B: [{lb:.10e}, {lb + tail:.10e}]")
    print(f"  intervals disjoint: {lb + tail < la}")
    if lb + tail < la:
        print(f"  B is at least {100 * (la - (lb + tail)) / la:.3f}% better")


def bb72(which="structure"):
    c = BBCode(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    n = c.n
    lz = z_logical_basis(c)
    print(f"  code: n={n} k={c.k} CSS={c.commutes()}")

    hz = np.array([int(''.join(map(str, c.HZ[:, j][::-1])), 2) for j in range(n)])
    lc = np.array([int(''.join(map(str, lz[:, j][::-1])), 2) for j in range(n)])
    i3 = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
    s3 = hz[i3[:, 0]] ^ hz[i3[:, 1]] ^ hz[i3[:, 2]]
    l3 = lc[i3[:, 0]] ^ lc[i3[:, 1]] ^ lc[i3[:, 2]]
    key = s3 * 4096 + l3
    u, inv = np.unique(key, return_inverse=True)
    syn = u >> 12

    from collections import defaultdict
    bys = defaultdict(dict)
    cnt = np.bincount(inv, minlength=len(u))
    for k, s in enumerate(syn):
        bys[s][int(u[k] & 4095)] = int(cnt[k])

    dist = defaultdict(int)
    C1 = comp = 0
    for s, d in bys.items():
        dist[len(d)] += 1
        if len(d) > 1:
            C1 += sum(d.values()) - max(d.values())
            comp += sum(d.values())
    print(f"  weight-3 patterns              : {len(i3)}")
    print(f"  syndromes by number of classes : "
          + ", ".join(f"{k}->{v}" for k, v in sorted(dist.items())))
    print(f"  competing weight-3 patterns    : {comp}")
    print(f"  C(1)                           : {C1}")

    def C(r):
        wgt = r[i3[:, 0]] * r[i3[:, 1]] * r[i3[:, 2]]
        W = np.bincount(inv, weights=wgt, minlength=len(u))
        o = np.argsort(syn, kind="stable")
        ss, Ws = syn[o], W[o]
        st = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
        return Ws.sum() - np.maximum.reduceat(Ws, st).sum()
    print(f"  C(1) recomputed from weights   : {C(np.ones(n)):.6f}")

    def rvec(sites):
        r = np.full(n, 1.35); r[list(sites)] = 0.65; return r
    print(f"\n  the two witness placements, same rate multiset per half:")
    print(f"    C(A) = {C(rvec(PA)):.4f}   C(B) = {C(rvec(PB)):.4f}")
    if which == "bounds":
        print()
        bb72_bounds(c, lz)


def penalty_test(seconds=70.0, tl=15.0, max_solves=12):
    """Does reweighting help when the objective is minimising weight?

    Same logical class, same forbidden set, penalty 0.0 against penalty 0.30,
    counting how many operators and how many orbits each arm finds. This is the
    controlled version of a claim that was first made from watching runs.
    """
    import time
    from scipy.optimize import milp
    from distance import _program
    from enumerate_weight import orbit, translations

    c = BBCode(12, 12, [(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)])
    lz = z_logical_basis(c)
    perms = translations(c)
    print(f"  two-gross [[{c.n},{c.k}]], class 0, {max_solves} solves max per arm")

    for penalty in (0.0, 0.30):
        found, supports, t0, n = set(), [], time.time(), 0
        while time.time() - t0 < seconds and n < max_solves:
            cc, cons, integ, bnds = _program(c, lz[0], forbid=supports,
                                             penalty=penalty)
            res = milp(c=cc, constraints=cons, integrality=integ, bounds=bnds,
                       options=dict(time_limit=tl, mip_rel_gap=0.0))
            n += 1
            if res.x is None:
                break
            v = np.round(res.x[:c.n]).astype(np.uint8)
            new = orbit(v, perms) - found
            found |= new
            for b in new:
                supports.append(tuple(np.nonzero(
                    np.frombuffer(b, dtype=np.uint8))[0]))
        orbs = {min(orbit(np.frombuffer(b, dtype=np.uint8), perms))
                for b in found}
        print(f"    penalty {penalty:<5} {n:2d} solves, {time.time()-t0:5.1f}s"
              f"  ->  {len(found):4d} operators, {len(orbs)} orbits")
    print("  If the two lines agree, reweighting changed nothing here.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["bb18", "bb72", "penalty"])
    ap.add_argument("--bounds", action="store_true",
                    help="bb72: also compute the certified failure bounds "
                         "(enumerates 1,091,059 patterns, takes a minute)")
    a = ap.parse_args()
    if a.code == "penalty":
        print("\nDoes reweighting help under a minimising objective?\n")
        penalty_test()
    elif a.code == "bb18":
        print("\n[[18,4,4]] bicycle code, exact\n")
        bb18()
    else:
        print("\n[[72,12,6]] bicycle code, leading order\n")
        bb72("bounds" if a.bounds else "structure")
    print()
