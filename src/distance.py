#!/usr/bin/env python3
"""
distance.py, minimum weight of a non-trivial X-logical, by integer program.

WHY AN INTEGER PROGRAM
----------------------
An X-type logical operator is a binary vector v with H_Z v = 0 (mod 2) that is
not a product of X stabilisers. For the two-gross code ker(H_Z) has dimension
150, so there are 2^150 candidates. Enumeration is not on the table and neither
is anything that walks the stabiliser group.

Posed as an optimisation it is small: 288 binary variables and 138 integer
slacks.

    minimise   sum_i v_i
    subject to H_Z v - 2s = 0,     v binary, s integer >= 0
               L_Z[j] . v - 2t = 1  for one chosen Z-logical j

The mod-2 constraints become linear by introducing an integer slack that
absorbs the multiple of two. The last constraint is what forces v to be a
*logical* rather than a stabiliser: a non-trivial X-logical must anticommute
with at least one Z-logical, so solving once per j and taking the minimum over
j gives the true distance. Restricting to one j at a time keeps each subproblem
a plain MILP; a disjunctive "at least one of k" constraint would need k extra
binaries and solves far more slowly.

HiGHS via scipy.optimize.milp. No commercial solver needed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, eye

from bbcode import build, nullspace, rref, row_space_contains


def z_logical_basis(code):
    """Basis for the Z-logical classes: ker(H_X) modulo rowspace(H_Z)."""
    ns = nullspace(code.HX)
    span = (code.HZ & 1).copy()
    R, piv = rref(span)
    chosen = []
    for v in ns:
        if not row_space_contains(R, piv, v):
            chosen.append(v.copy())
            span = np.vstack([span, v])
            R, piv = rref(span)
            if len(chosen) == code.k:
                break
    return np.array(chosen, dtype=np.uint8)


def _program(code, lz_row, forbid=None, penalty=0.0, ub=None):
    """Build the MILP.

    forbid : list of previously found supports. Each gets a constraint
             sum_{i in S} v_i <= |S| - 1, which is the classical no-good cut:
             the new solution must differ from that one somewhere.
    penalty: if > 0, qubits appearing in `forbid` supports get their objective
             coefficient raised by this much per appearance. This is the
             "dynamic support reweighting" idea, it biases the search away
             from ground already covered instead of merely forbidding exact
             repeats, which matters because exact-repeat cuts alone let the
             solver return near-duplicates one qubit at a time.
    """
    n, mz = code.n, code.HZ.shape[0]
    nslack = mz + 1

    # objective: weight of v, plus optional per-qubit penalty; slacks free
    c = np.ones(n, dtype=float)
    if penalty and forbid:
        hits = np.zeros(n, dtype=float)
        for S in forbid:
            hits[list(S)] += 1.0
        c = c + penalty * hits
    c = np.concatenate([c, np.zeros(nslack)])

    # H_Z v - 2 s = 0   and   lz . v - 2 t = 1
    top = hstack([csc_matrix(code.HZ.astype(float)),
                  -2.0 * eye(mz, nslack, format="csc")])
    bot = hstack([csc_matrix(lz_row.astype(float).reshape(1, -1)),
                  csc_matrix(([-2.0], ([0], [mz])), shape=(1, nslack))])
    A = csc_matrix(np.vstack([top.toarray(), bot.toarray()]))
    lo = np.concatenate([np.zeros(mz), [1.0]])
    cons = [LinearConstraint(A, lo, lo)]

    for S in forbid or []:
        row = np.zeros(n + nslack)
        row[list(S)] = 1.0
        cons.append(LinearConstraint(csc_matrix(row.reshape(1, -1)),
                                     -np.inf, len(S) - 1))

    if ub is not None:                      # cap the weight; prunes hard
        row = np.zeros(n + nslack)
        row[:n] = 1.0
        cons.append(LinearConstraint(csc_matrix(row.reshape(1, -1)),
                                     -np.inf, float(ub)))

    integrality = np.ones(n + nslack)
    bounds = Bounds(np.concatenate([np.zeros(n), np.zeros(nslack)]),
                    np.concatenate([np.ones(n), np.full(nslack, np.inf)]))
    return c, cons, integrality, bounds


def min_weight(code, lz, time_limit=60.0, verbose=True):
    """Minimum weight over all non-trivial X-logicals."""
    best, best_v = None, None
    for j, row in enumerate(lz):
        c, cons, integ, bnds = _program(code, row)
        t0 = time.time()
        res = milp(c=c, constraints=cons, integrality=integ, bounds=bnds,
                   options=dict(time_limit=time_limit, mip_rel_gap=0.0))
        if res.x is None:
            if verbose:
                print(f"    j={j:2d}  no solution ({res.message[:40]})")
            continue
        v = np.round(res.x[:code.n]).astype(np.uint8)
        w = int(v.sum())
        if verbose:
            print(f"    j={j:2d}  weight {w:3d}   [{time.time()-t0:5.1f}s]")
        if best is None or w < best:
            best, best_v = w, v
    return best, best_v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["gross", "two-gross"])
    ap.add_argument("--option", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=60.0)
    a = ap.parse_args()

    opt = a.option if a.option is not None else (0 if a.code == "gross" else 1)
    code = build(a.code, opt)
    print(f"{code}  CSS ok={code.commutes()}")
    lz = z_logical_basis(code)
    print(f"  Z-logical basis: {lz.shape}")
    d, v = min_weight(code, lz, time_limit=a.time_limit)
    print(f"\n  minimum X-logical weight = {d}")
