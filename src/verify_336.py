#!/usr/bin/env python3
"""
verify_336.py, reproduce the weight-18 X-logical count for the two-gross code.

Tour de gross (arXiv:2506.03094), Appendix A.1 Figure 12, quotes 336.

METHOD
------
Two passes, because the operators fall into orbits of two different sizes and
the cheap search only finds one kind.

Pass 1, symmetric search. Impose invariance under a translation of order 3.
That ties the 288 variables into 96 groups and the integer program solves in
seconds. It finds the small orbits, which the open search struggles to reach
because they sit in a thin slice of the space.

Pass 2, open enumeration with dynamic support reweighting. For each logical
class, solve for a weight-18 operator subject to no-good cuts on everything
already found, and with the objective coefficient of each already-seen qubit
raised. Generate the full translation orbit of each hit rather than solving
again for its 143 translates.

VERIFICATION
------------
Counting is not enough. Every operator is independently re-checked:
  - weight is exactly 18
  - H_Z v = 0 over GF(2), so it commutes with every Z check
  - v is NOT in rowspace(H_X), so it is a logical and not a stabiliser
  - the set is closed under all 144 translations
All four are checked, not just the count.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from bbcode import build, rref, row_space_contains
from distance import z_logical_basis
from enumerate_weight import orbit, solve_one, translations
import enumerate_weight as EW
from symmetric_search import solve_symmetric
from paths import data

STATE = data("weight18_two-gross_all.json")
W = 18


def load():
    if os.path.exists(STATE):
        st = json.load(open(STATE))
        return {bytes.fromhex(h) for h in st["found"]}
    return set()


def save(found):
    json.dump({"found": [b.hex() for b in found], "weight": W},
              open(STATE, "w"))


def main(budget=140.0):
    code = build("two-gross", 1)
    lz = z_logical_basis(code)
    perms = translations(code)
    found = load()
    print(f"  start: {len(found)} operators")
    t0 = time.time()

    # ---- pass 1: symmetric slices -------------------------------------
    for gen in [(4, 0), (0, 4), (4, 4), (4, 8), (8, 0), (0, 8),
                (6, 0), (0, 6), (6, 6)]:
        if time.time() - t0 > budget * 0.35:
            break
        for j in range(len(lz)):
            v, _ = solve_symmetric(code, lz[j], gen, W, time_limit=8.0)
            if v is None:
                continue
            new = orbit(v, perms) - found
            if new:
                found |= new
                print(f"  sym {gen} class {j:2d}: +{len(new):3d} "
                      f"-> {len(found)}")
    save(found)

    # ---- pass 2: open search with reweighting --------------------------
    EW.TIME_LIMIT = 20.0
    supports = [tuple(np.nonzero(np.frombuffer(b, np.uint8))[0])
                for b in found]
    seen = np.zeros(code.n)
    for b in found:
        seen += np.frombuffer(b, np.uint8)

    for j in range(len(lz)):
        while time.time() - t0 < budget:
            v, status = solve_one(code, lz[j], W, supports, seen, 0.30)
            if v is None:
                break
            new = orbit(v, perms) - found
            if not new:
                supports.append(tuple(np.nonzero(v)[0]))
                continue
            found |= new
            for b in new:
                arr = np.frombuffer(b, np.uint8)
                seen += arr
                supports.append(tuple(np.nonzero(arr)[0]))
            print(f"  open class {j:2d}: +{len(new):4d} -> {len(found)}"
                  f"   [{time.time()-t0:5.1f}s]")
            save(found)
        else:
            break
    save(found)

    # ---- verification --------------------------------------------------
    print(f"\n  === verifying {len(found)} operators ===")
    HXR, HXP = rref(code.HX)
    ok_w = ok_c = ok_l = 0
    for b in found:
        v = np.frombuffer(b, np.uint8)
        if v.sum() == W:
            ok_w += 1
        if not ((code.HZ @ v) & 1).any():
            ok_c += 1
        if not row_space_contains(HXR, HXP, v):
            ok_l += 1
    closed = all(orbit(np.frombuffer(b, np.uint8), perms) <= found
                 for b in found)

    print(f"  weight exactly {W}          : {ok_w}/{len(found)}")
    print(f"  in ker(H_Z)                : {ok_c}/{len(found)}")
    print(f"  not a stabiliser (logical) : {ok_l}/{len(found)}")
    print(f"  closed under translation   : {closed}")
    print(f"\n  COUNT = {len(found)}   (Tour de gross Figure 12 quotes 336)")
    return len(found)


if __name__ == "__main__":
    import sys
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 140.0)
