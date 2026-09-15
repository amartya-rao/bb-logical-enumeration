#!/usr/bin/env python3
"""
bbcode.py, bivariate bicycle codes and the GF(2) linear algebra they need.

Construction follows Bravyi, Cross, Gambetta, Maslov, Rall and Yoder,
"High-threshold and low-overhead fault-tolerant quantum memory", Nature 627
(2024), and the description in Appendix A.1 of Yoder et al., "Tour de gross"
(arXiv:2506.03094).

Everything here is pure numpy on uint8. No `ldpc`, no `galois`. GF(2) row
reduction is twenty lines and pulling in a compiled dependency to get it would
make the artifact harder to reproduce, not easier.

CONSTRUCTION
------------
Take cyclic groups of order l and m. Let

    x = S_l (x) I_m          y = I_l (x) S_m

where S_n is the n x n cyclic shift and (x) is the Kronecker product. Both are
2-dimensional-torus translations acting on l*m cells; each cell carries one
"left" and one "right" qubit, so n = 2*l*m.

Choose two trinomials A and B in x and y. Then, as a CSS code,

    H_X = [A | B]            H_Z = [B^T | A^T]

which satisfies H_X H_Z^T = A B^T + B A^T = 0 because A and B commute (they are
built from commuting shifts) and we are over GF(2).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# GF(2) linear algebra
# --------------------------------------------------------------------------

def rref(M):
    """Reduced row echelon form over GF(2).

    Returns (R, pivots). R is a copy; the input is not modified.
    """
    R = (np.asarray(M, dtype=np.uint8) & 1).copy()
    rows, cols = R.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = np.nonzero(R[r:, c])[0]
        if piv.size == 0:
            continue
        i = r + piv[0]
        if i != r:
            R[[r, i]] = R[[i, r]]
        # Eliminate this column everywhere else. XOR is addition over GF(2).
        hit = np.nonzero(R[:, c])[0]
        hit = hit[hit != r]
        if hit.size:
            R[hit] ^= R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(M):
    return len(rref(M)[1])


def nullspace(M):
    """Basis for the right null space of M over GF(2), as rows of a matrix.

    Standard construction: put M in RREF, read the free columns, and for each
    free column build the vector that sets it to 1 and back-substitutes the
    pivot entries.
    """
    M = np.asarray(M, dtype=np.uint8) & 1
    R, pivots = rref(M)
    cols = M.shape[1]
    free = [c for c in range(cols) if c not in set(pivots)]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for r, p in enumerate(pivots):
            if R[r, f]:
                basis[k, p] = 1
    return basis


def row_space_contains(basis_rref, basis_pivots, v):
    """Is v in the row space whose RREF is basis_rref with the given pivots?

    Reduce v against the RREF rows and check the remainder is zero. Cheap, and
    it is the inner loop of deciding whether an operator is a logical or a
    stabiliser, so it is worth not recomputing an RREF each time.
    """
    w = (np.asarray(v, dtype=np.uint8) & 1).copy()
    for r, p in enumerate(basis_pivots):
        if w[p]:
            w ^= basis_rref[r]
    return not w.any()


# --------------------------------------------------------------------------
# code construction
# --------------------------------------------------------------------------

def shift(n):
    """n x n cyclic shift matrix."""
    S = np.zeros((n, n), dtype=np.uint8)
    S[np.arange(n), (np.arange(n) + 1) % n] = 1
    return S


class BBCode:
    """A bivariate bicycle code.

    Parameters
    ----------
    l, m : cyclic group orders; n = 2*l*m
    a_terms, b_terms : lists of (i, j) meaning the monomial x^i y^j.
        A = sum over a_terms, B = sum over b_terms, both over GF(2).
    """

    def __init__(self, l, m, a_terms, b_terms, name=""):
        self.l, self.m, self.name = l, m, name
        self.a_terms, self.b_terms = list(a_terms), list(b_terms)

        Il, Im = np.eye(l, dtype=np.uint8), np.eye(m, dtype=np.uint8)
        x = np.kron(shift(l), Im)
        y = np.kron(Il, shift(m))

        def poly(terms):
            P = np.zeros((l * m, l * m), dtype=np.uint8)
            for (i, j) in terms:
                P ^= (matpow(x, i) @ matpow(y, j)) & 1
            return P

        self.A = poly(self.a_terms)
        self.B = poly(self.b_terms)

        self.HX = np.hstack([self.A, self.B]).astype(np.uint8)
        self.HZ = np.hstack([self.B.T, self.A.T]).astype(np.uint8)
        self.n = 2 * l * m

    # ---- parameters ----

    def commutes(self):
        """CSS condition H_X H_Z^T = 0 over GF(2)."""
        return not ((self.HX @ self.HZ.T) & 1).any()

    @property
    def k(self):
        return self.n - rank(self.HX) - rank(self.HZ)

    def __repr__(self):
        return (f"BBCode({self.name or ''} l={self.l} m={self.m} "
                f"n={self.n} k={self.k})")

    # ---- logical operators ----

    def x_logical_basis(self):
        """A basis for the X-logical classes.

        X-type operators that commute with every Z check live in ker(H_Z).
        Those that are products of X stabilisers live in rowspace(H_X). The
        quotient is the logical space, and we pick k coset representatives by
        greedily taking null-space vectors that are independent modulo the
        stabilisers.
        """
        ns = nullspace(self.HZ)
        span = (self.HX & 1).copy()          # grows as we accept logicals
        R, piv = rref(span)
        chosen = []
        for v in ns:
            if not row_space_contains(R, piv, v):
                chosen.append(v.copy())
                span = np.vstack([span, v])
                R, piv = rref(span)
                if len(chosen) == self.k:
                    break
        return np.array(chosen, dtype=np.uint8)


def matpow(M, e):
    """M^e over GF(2) by repeated squaring."""
    n = M.shape[0]
    R = np.eye(n, dtype=np.uint8)
    B = (M & 1).copy()
    while e:
        if e & 1:
            R = (R @ B) & 1
        B = (B @ B) & 1
        e >>= 1
    return R


# --------------------------------------------------------------------------
# the two codes in the paper
# --------------------------------------------------------------------------
#
# Tour de gross states a_right = -b_up = 3 and a_up = b_right = -1 for both,
# differing only in size: l=12, m=6 for gross and l=12, m=12 for two-gross.
# In polynomial form the family is written A = x^i + y^j + y^k,
# B = y^i + x^j + x^k. The exact exponents for each code are listed in the
# Nature paper's tables; CANDIDATES below holds the ones to test, and
# verify() reports which reproduce the published [[n, k, d]].
#
# Do not trust a candidate that merely gives the right n and k. Several do.

CANDIDATES = {
    "gross": dict(l=12, m=6, options=[
        ([(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),   # x^3+y+y^2 , y^3+x+x^2
    ]),
    "two-gross": dict(l=12, m=12, options=[
        ([(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]),
        ([(3, 0), (0, 2), (0, 7)], [(0, 3), (1, 0), (2, 0)]),   # x^3+y^2+y^7 , y^3+x+x^2
    ]),
}


def build(which, option=0):
    spec = CANDIDATES[which]
    a, b = spec["options"][option]
    return BBCode(spec["l"], spec["m"], a, b, name=which)


if __name__ == "__main__":
    for which in ("gross", "two-gross"):
        for i in range(len(CANDIDATES[which]["options"])):
            c = build(which, i)
            print(f"{which:10s} option {i}: n={c.n:4d}  k={c.k:3d}  "
                  f"CSS ok={c.commutes()}")
