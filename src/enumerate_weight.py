#!/usr/bin/env python3
"""
enumerate_weight.py, count X-logicals of a fixed weight.

TARGET
------
Tour de gross (arXiv:2506.03094) quotes, from Appendix A.1 Figure 12, that the
two-gross [[288,12,18]] code has 336 weight-18 X-logicals. That number is the
correctness check for this whole pipeline.

WHY NOT ONE SOLVE PER OPERATOR
------------------------------
The obvious loop is: solve for a weight-w logical, forbid it, solve again,
repeat. That is 336+ integer programs on 288 binaries, and most of them
rediscover translates of something already found.

Bivariate bicycle codes are translation invariant: the l*m = 144 shifts
x^a y^b map the code to itself, so they map logical operators to logical
operators of the same weight. Orbits, not individuals, are the natural unit.
So: find one operator, generate its entire orbit for free by shifting, forbid
the whole orbit, and solve again. That turns hundreds of solves into a handful.

The no-good cuts and the objective reweighting are both applied, and the
reweighting is the part worth noting. Cuts alone forbid exact repeats, which
lets the solver hand back a near-duplicate differing in one qubit and makes no
progress. Raising the objective coefficient on every qubit already seen pushes
the search onto fresh support instead. That is the dynamic support reweighting
this work set out to try.

CHECKPOINTING
-------------
State is written to weight<W>_<code>.json after every orbit, so the run can be
killed and resumed. Long MILP sweeps need this.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, eye, hstack

from bbcode import build
from distance import z_logical_basis
from paths import data


# --------------------------------------------------------------------------
# translation symmetry
# --------------------------------------------------------------------------

def translations(code):
    """All l*m index permutations induced by the shifts x^a y^b.

    A vector is [left half | right half], each of length l*m, with cell (i, j)
    stored at i*m + j. A shift by (a, b) moves cell (i, j) to
    ((i+a) mod l, (j+b) mod m) in both halves identically.
    """
    l, m, half = code.l, code.m, code.l * code.m
    perms = []
    base_i, base_j = np.divmod(np.arange(half), m)
    for a in range(l):
        for b in range(m):
            dest = ((base_i + a) % l) * m + ((base_j + b) % m)
            p = np.empty(2 * half, dtype=np.int64)
            p[dest] = np.arange(half)              # left half
            p[dest + half] = np.arange(half) + half
            perms.append(p)
    return perms


def orbit(v, perms):
    """Distinct images of v under the translation group, as a set of bytes."""
    return {v[p].tobytes() for p in perms}


# --------------------------------------------------------------------------
# the integer program
# --------------------------------------------------------------------------

def solve_one(code, lz_row, weight, forbid_supports, seen_counts, penalty):
    n, mz = code.n, code.HZ.shape[0]
    nsl = mz + 1

    c = np.ones(n, dtype=float)
    if penalty:
        c = c + penalty * seen_counts
    c = np.concatenate([c, np.zeros(nsl)])

    top = hstack([csc_matrix(code.HZ.astype(float)),
                  -2.0 * eye(mz, nsl, format="csc")]).toarray()
    bot = np.zeros((1, n + nsl))
    bot[0, :n] = lz_row.astype(float)
    bot[0, n + mz] = -2.0
    wrow = np.zeros((1, n + nsl))
    wrow[0, :n] = 1.0

    A = csc_matrix(np.vstack([top, bot, wrow]))
    lo = np.concatenate([np.zeros(mz), [1.0], [float(weight)]])
    cons = [LinearConstraint(A, lo, lo)]        # weight pinned exactly

    if forbid_supports:
        rows = np.zeros((len(forbid_supports), n + nsl))
        ub = np.empty(len(forbid_supports))
        for r, S in enumerate(forbid_supports):
            rows[r, list(S)] = 1.0
            ub[r] = len(S) - 1
        cons.append(LinearConstraint(csc_matrix(rows), -np.inf, ub))

    res = milp(c=c, constraints=cons,
               integrality=np.ones(n + nsl),
               bounds=Bounds(np.zeros(n + nsl),
                             np.concatenate([np.ones(n), np.full(nsl, np.inf)])),
               options=dict(time_limit=TIME_LIMIT, mip_rel_gap=0.0))
    if res.x is None:
        # status 2 = proven infeasible. Anything else (1 = time limit) means
        # we simply did not find one in the budget, which is not the same
        # thing and must not close the class.
        return None, int(res.status)
    return np.round(res.x[:n]).astype(np.uint8), int(res.status)


TIME_LIMIT = 30.0


def run(code_name, weight, penalty, budget, out):
    code = build(code_name, 0 if code_name == "gross" else 1)
    lz = z_logical_basis(code)
    perms = translations(code)
    n = code.n

    found, exhausted, timed_out = set(), set(), set()
    if os.path.exists(out):
        st = json.load(open(out))
        found = {bytes.fromhex(h) for h in st["found"]}
        exhausted = set(st["exhausted"])
        print(f"  resumed: {len(found)} operators, "
              f"{len(exhausted)}/{len(lz)} classes closed")

    seen = np.zeros(n, dtype=float)
    for b in found:
        seen += np.frombuffer(b, dtype=np.uint8)

    supports = [tuple(np.nonzero(np.frombuffer(b, dtype=np.uint8))[0])
                for b in found]

    t_start = time.time()
    for j in range(len(lz)):
        if j in exhausted:
            continue
        while time.time() - t_start < budget:
            v, status = solve_one(code, lz[j], weight, supports, seen, penalty)
            if v is None:
                if status == 2:
                    exhausted.add(j)
                    print(f"  class {j:2d}: proven empty  "
                          f"(total {len(found)})")
                else:
                    print(f"  class {j:2d}: INCONCLUSIVE (status {status}, "
                          f"time limit) - not closed")
                    timed_out.add(j)
                break
            orb = orbit(v, perms)
            new = orb - found
            if not new:
                # Already have every translate; forbid this one explicitly so
                # the solver cannot return it again and stall the loop.
                supports.append(tuple(np.nonzero(v)[0]))
                continue
            found |= new
            for b in new:
                arr = np.frombuffer(b, dtype=np.uint8)
                seen += arr
                supports.append(tuple(np.nonzero(arr)[0]))
            print(f"  class {j:2d}: +{len(new):4d} (orbit {len(orb):3d})"
                  f"  total {len(found):5d}   [{time.time()-t_start:5.1f}s]")
            json.dump({"found": [b.hex() for b in found],
                       "exhausted": sorted(exhausted),
                       "weight": weight, "code": code_name},
                      open(out, "w"))
        else:
            break

    json.dump({"found": [b.hex() for b in found],
               "exhausted": sorted(exhausted),
               "weight": weight, "code": code_name}, open(out, "w"))
    if timed_out:
        print(f"  classes that timed out rather than closing: "
              f"{sorted(timed_out)} - raise --time-limit and resume")
    done = len(exhausted) == len(lz)
    print(f"\n  weight {weight}: {len(found)} operators "
          f"({'COMPLETE' if done else 'partial, resume to continue'})")
    return len(found), done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["gross", "two-gross"])
    ap.add_argument("--weight", type=int, required=True)
    ap.add_argument("--penalty", type=float, default=0.30,
                    help="objective bump per prior appearance of a qubit")
    ap.add_argument("--budget", type=float, default=140.0)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TIME_LIMIT = a.time_limit
    out = a.out or data(f"weight{a.weight}_{a.code}.json")
    run(a.code, a.weight, a.penalty, a.budget, out)
