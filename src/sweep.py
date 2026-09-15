#!/usr/bin/env python3
"""
sweep.py, the full logical-operator weight distribution (Figure 12).

Walks weights upward from the code distance and counts X-logicals at each,
using the same two passes that reproduced the 336:

  pass A, symmetric slices : impose invariance under a translation of prime
                             order, which ties variables into groups and makes
                             the program small. Finds the thin orbits an open
                             search rarely lands in. This is what took the
                             two-gross weight-18 count from 288 to 336.
  pass B, open search      : per logical class, solve for an operator of the
                             pinned weight subject to no-good cuts on
                             everything already found, then take its whole
                             translation orbit for free.

COMPLETE VERSUS LOWER BOUND
---------------------------
A weight's count is COMPLETE only when the integer program has *proven*
infeasibility (HiGHS status 2) for every one of the k logical classes with all
found operators forbidden. If any class merely ran out of time, the count is a
LOWER BOUND and is recorded as such.

An earlier version of this code read a time limit as a proof of emptiness and
reported 288 for a quantity whose true value is 336. Every number this script emits carries its status, and
plot_figure12.py draws bounds differently from complete counts.

Counts also grow fast with weight. --max-per-weight stops the search once a
weight has that many operators and marks it a lower bound, so one intractable
weight cannot eat the whole run.

USAGE
-----
    python3 sweep.py two-gross --from 18 --to 26 --budget 36000

Checkpoints to sweep_<code>.json after every weight, and each weight's
operators to weight<W>_<code>_sweep.json. Safe to kill and restart.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from bbcode import build, rref, row_space_contains
from distance import z_logical_basis
import enumerate_weight as EW
from enumerate_weight import orbit, solve_one, translations
from symmetric_search import solve_symmetric
from paths import data

# Generators of the prime-order subgroups of Z_l x Z_m worth slicing on.
# Order 2 and order 3 elements catch the orbits of size l*m/2 and l*m/3, which
# are the ones the open search misses.
SYM_GENS = [(4, 0), (0, 4), (4, 4), (4, 8), (8, 0), (0, 8), (8, 4), (8, 8),
            (6, 0), (0, 6), (6, 6)]


def weight_state(code_name, w):
    return data(f"weight{w}_{code_name}_sweep.json")


def load_weight(path):
    if os.path.exists(path):
        st = json.load(open(path))
        return ({bytes.fromhex(h) for h in st["found"]},
                set(st.get("proven_empty", [])))
    return set(), set()


def save_weight(path, found, proven, w, code_name):
    json.dump({"found": [b.hex() for b in found],
               "proven_empty": sorted(proven),
               "weight": w, "code": code_name}, open(path, "w"))


def do_weight(code, code_name, lz, perms, w, deadline, cap, solve_tl):
    """Count X-logicals of weight exactly w. Returns (count, complete)."""
    path = weight_state(code_name, w)
    found, proven = load_weight(path)
    if found or proven:
        print(f"    resumed weight {w}: {len(found)} ops, "
              f"{len(proven)}/{len(lz)} classes proven empty")
    if len(proven) == len(lz) and len(found) < cap:
        # Already closed on a previous run. Re-running the passes would spend
        # minutes rediscovering nothing.
        print(f"    weight {w} already complete, skipping search")
        return len(found), True, found

    # ---- pass A: symmetric slices ----
    for gen in SYM_GENS:
        if time.time() > deadline or len(found) >= cap:
            break
        for j in range(len(lz)):
            if time.time() > deadline:
                break
            v, _ = solve_symmetric(code, lz[j], gen, w,
                                   time_limit=min(solve_tl, 20.0))
            if v is None:
                continue
            new = orbit(v, perms) - found
            if new:
                found |= new
                print(f"    w={w} sym{gen} c{j:02d}: +{len(new):4d}"
                      f" -> {len(found)}")
                save_weight(path, found, proven, w, code_name)

    # ---- pass B: open search, weight pinned, no penalty ----
    # Penalty is deliberately 0 here. With the weight pinned by an equality
    # constraint this is a feasibility problem, and a modified objective only
    # gives the solver something pointless to optimise. Reweighting belongs in
    # the minimisation formulation, not this one.
    EW.TIME_LIMIT = solve_tl
    supports = [tuple(np.nonzero(np.frombuffer(b, np.uint8))[0]) for b in found]
    seen = np.zeros(code.n)

    for j in range(len(lz)):
        if j in proven:
            continue
        while time.time() < deadline and len(found) < cap:
            v, status = solve_one(code, lz[j], w, supports, seen, 0.0)
            if v is None:
                if status == 2:
                    proven.add(j)
                break
            new = orbit(v, perms) - found
            if not new:
                supports.append(tuple(np.nonzero(v)[0]))
                continue
            found |= new
            for b in new:
                supports.append(tuple(np.nonzero(np.frombuffer(b, np.uint8))[0]))
            print(f"    w={w} open c{j:02d}: +{len(new):4d} -> {len(found)}")
            save_weight(path, found, proven, w, code_name)

    save_weight(path, found, proven, w, code_name)
    complete = (len(proven) == len(lz)) and len(found) < cap
    return len(found), complete, found


def verify(code, found, w):
    """Independent re-check. A count nobody audited is not a result."""
    HXR, HXP = rref(code.HX)
    for b in found:
        v = np.frombuffer(b, np.uint8)
        if int(v.sum()) != w:
            return False, "wrong weight"
        if ((code.HZ @ v) & 1).any():
            return False, "not in ker(H_Z)"
        if row_space_contains(HXR, HXP, v):
            return False, "is a stabiliser, not a logical"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["gross", "two-gross"])
    ap.add_argument("--from", dest="w0", type=int, default=None)
    ap.add_argument("--to", dest="w1", type=int, default=None)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--budget", type=float, default=3600.0,
                    help="total wall seconds")
    ap.add_argument("--solve-time-limit", type=float, default=60.0)
    ap.add_argument("--max-per-weight", type=int, default=20000)
    a = ap.parse_args()

    code = build(a.code, 0 if a.code == "gross" else 1)
    d0 = 12 if a.code == "gross" else 18
    w0 = a.w0 if a.w0 is not None else d0
    w1 = a.w1 if a.w1 is not None else d0 + 8

    print(f"{code}  CSS ok={code.commutes()}")
    lz = z_logical_basis(code)
    perms = translations(code)

    out = data(f"sweep_{a.code}.json")
    results = json.load(open(out)) if os.path.exists(out) else {}
    deadline = time.time() + a.budget

    for w in range(w0, w1 + 1, a.step):
        if time.time() > deadline:
            print(f"\n  budget exhausted before weight {w}")
            break
        print(f"\n  --- weight {w} ---")
        t0 = time.time()
        cnt, complete, found = do_weight(code, a.code, lz, perms, w,
                                         deadline, a.max_per_weight,
                                         a.solve_time_limit)
        ok, why = verify(code, found, w)
        results[str(w)] = dict(count=cnt, complete=bool(complete),
                               verified=bool(ok), note=why,
                               seconds=round(time.time() - t0, 1))
        json.dump(results, open(out, "w"), indent=2)
        tag = "COMPLETE" if complete else "LOWER BOUND"
        print(f"  weight {w}: {cnt}  [{tag}]  verified={ok}"
              f"  ({time.time()-t0:.0f}s)")

    print(f"\n  === distribution so far ({a.code}) ===")
    for w in sorted(results, key=int):
        r = results[w]
        tag = "=" if r["complete"] else ">="
        print(f"    weight {w:>3}: {tag} {r['count']:>7}"
              f"   verified={r['verified']}")
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
