#!/usr/bin/env python3
"""
Why the vulnerability map is degenerate on bivariate bicycle codes.

Checks, for both gross and two-gross:

  1. the l*m translations preserve H_X and H_Z
  2. the action has 2 orbits on data qubits and 1 on each check type
  3. a syndrome round ordered by monomial is covariant under them, one ordered
     by qubit index is not
  4. the ZX duality, cell inversion composed with the half swap, maps H_X to H_Z

Together these force w-hat to be constant on each orbit, so it takes at most
four values: left data, right data, X-ancilla, Z-ancilla.

Read the correction in FLATNESS.md before drawing a conclusion from this.
It does not mean placement gains are second order.
"""

from __future__ import annotations

import numpy as np

from bbcode import build

CODES = (("gross", 0), ("two-gross", 1))


def _cells(code):
    half = code.l * code.m
    bi, bj = np.divmod(np.arange(half), code.m)
    return half, bi, bj


def translation_maps(code):
    """For each shift (a,b): the cell relabelling it induces."""
    half, bi, bj = _cells(code)
    for a in range(code.l):
        for b in range(code.m):
            yield a, b, ((bi + a) % code.l) * code.m + ((bj + b) % code.m)


def step1_checks_preserved(code):
    """Do the translations preserve H_X and H_Z?"""
    half = code.l * code.m
    for _, _, dq in translation_maps(code):
        pq = np.empty(code.n, dtype=np.int64)
        pq[dq] = np.arange(half)
        pq[dq + half] = np.arange(half) + half
        pc = np.empty(half, dtype=np.int64)
        pc[dq] = np.arange(half)
        if not (code.HX[pc][:, pq] == code.HX).all():
            return False
        if not (code.HZ[pc][:, pq] == code.HZ).all():
            return False
    return True


def step2_orbits(code):
    """Orbit counts on data qubits, X-checks, Z-checks."""
    half = code.l * code.m
    maps = [dq for _, _, dq in translation_maps(code)]

    def count(size, lift):
        seen, k = set(), 0
        for i in range(size):
            if i in seen:
                continue
            k += 1
            for dq in maps:
                seen.add(lift(dq, i))
        return k

    data = count(code.n, lambda dq, i:
                 int(dq[i]) if i < half else int(dq[i - half]) + half)
    chk = count(half, lambda dq, i: int(dq[i]))
    return data, chk, chk


def _schedule(code, by):
    """One syndrome round as a set of (time, control, target) triples."""
    half = code.l * code.m
    n = code.n
    _, bi, bj = _cells(code)

    def cell(i, j):
        return (i % code.l) * code.m + (j % code.m)

    G = set()
    for c in range(half):
        ci, cj = bi[c], bj[c]
        if by == "monomial":
            for k, (i, j) in enumerate(code.a_terms):
                G.add((k, n + c, cell(ci + i, cj + j)))
            for k, (i, j) in enumerate(code.b_terms):
                G.add((len(code.a_terms) + k, n + c,
                       half + cell(ci + i, cj + j)))
        else:                                   # naive: sorted by qubit index
            for k, q in enumerate(np.nonzero(code.HX[c])[0]):
                G.add((k, n + c, int(q)))
    return G


def step3_schedule_covariant(code, by):
    half, n = code.l * code.m, code.n
    G = _schedule(code, by)
    for _, _, dq in translation_maps(code):
        def mv(u):
            if u < half:
                return int(dq[u])
            if u < n:
                return int(dq[u - half]) + half
            return n + int(dq[u - n])
        if {(k, mv(u), mv(v)) for (k, u, v) in G} != G:
            return False
    return True


def zx_duality(code):
    """Cell inversion composed with the half swap: does it map H_X <-> H_Z?"""
    half, bi, bj = _cells(code)
    inv = ((-bi) % code.l) * code.m + ((-bj) % code.m)
    pq = np.concatenate([inv + half, inv])
    return ((code.HX[inv][:, pq] == code.HZ).all(),
            (code.HZ[inv][:, pq] == code.HX).all())


def main():
    for name, opt in CODES:
        code = build(name, opt)
        print(f"\n  {name}  [[{code.n},{code.k}]]   l={code.l} m={code.m}   "
              f"{code.l * code.m} translations")

        print(f"    1. translations preserve H_X and H_Z      "
              f"{step1_checks_preserved(code)}")

        d, x, z = step2_orbits(code)
        print(f"    2. orbits: data {d}, X-checks {x}, Z-checks {z}"
              f"   -> at most {d + x + z} distinct w values")

        for by in ("monomial", "sorted"):
            print(f"    3. schedule ordered by {by:9s} covariant  "
                  f"{step3_schedule_covariant(code, by)}")

        a, b = zx_duality(code)
        print(f"    +  ZX duality (inversion + half swap) H_X->H_Z {a}, "
              f"H_Z->H_X {b}")

    print("\n  Conclusion: w takes at most four values on these codes, so the"
          "\n  ranking my assignment method relies on degenerates to buckets."
          "\n"
          "\n  Do NOT conclude from this that placement gains are second order."
          "\n  That inference needs the failure probability to be differentiable"
          "\n  at the symmetric point, and it is not once the decoder recalibrates"
          "\n  -- optimal decoding has a kink where two logical classes tie. See"
          "\n  the CORRECTION section of FLATNESS.md for the exact"
          "\n  counterexample on the [[18,4,4]] bicycle code.")


if __name__ == "__main__":
    main()
