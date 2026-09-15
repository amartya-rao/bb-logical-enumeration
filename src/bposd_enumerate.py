#!/usr/bin/env python3
"""
Figure 12 of Tour de gross (arXiv:2506.03094), done the way Appendix A.1 does it.

BP+OSD, with each qubit prior drawn from U[0.01, 0.99) every run, solving
[H_Z ; l] v = (0, 1) for a random non-trivial Z-logical l. Any solution is a
non-trivial X-logical, so no membership test is needed. Only weights d and d+2
are enumerated, which is all the paper reports.

  gross      [[144,12,12]]   1884 at w=12    19728 at w=14   BP 20, OSD 0
  two-gross  [[288,12,18]]    336 at w=18     1728 at w=20   BP 40, OSD 7

Prior modes: random (theirs), adaptive, balanced, decoy. The middle two build a
field out of the per-qubit support counts, which is constant on these codes, so
they do nothing. decoy uses a few specific found operators instead, which is not
translation invariant. Numbers in RESULTS.md.

Stopping rule is theirs and is a heuristic, not a proof. The weight-18 count is
proven separately by enumerate_weight.py.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bbcode import build
from distance import z_logical_basis
from enumerate_weight import orbit, translations
from paths import data


# Published counts, Appendix A.1. The point of the script is to hit these.
PUBLISHED = {
    "gross":     {12: 1884, 14: 19728},
    "two-gross": {18: 336,  20: 1728},
}

# Their decoder settings, same appendix.
SETTINGS = {
    "gross":     dict(max_iter=20, osd_order=0),
    "two-gross": dict(max_iter=40, osd_order=7),
}


# --------------------------------------------------------------------------
# ldpc adapter
# --------------------------------------------------------------------------
#
# The `ldpc` package changed API between v1 and v2 and both are in the wild.
# v2:  ldpc.BpOsdDecoder(H, error_channel=..., max_iter=, bp_method=,
#                        osd_method=, osd_order=)
# v1:  ldpc.bposd_decoder(H, channel_probs=..., max_iter=, bp_method=,
#                         osd_method=, osd_order=)
# Both expose .decode(syndrome). Detect once, then build per trial.

def _decoder_factory():
    try:
        from ldpc.bposd_decoder import BpOsdDecoder as _V2
    except Exception:
        _V2 = None
    if _V2 is None:
        try:
            from ldpc import BpOsdDecoder as _V2   # noqa: N806
        except Exception:
            _V2 = None

    if _V2 is not None:
        def make(H, priors, max_iter, osd_order):
            return _V2(H,
                       error_channel=list(map(float, priors)),
                       max_iter=int(max_iter),
                       bp_method="product_sum",
                       osd_method="osd_cs" if osd_order > 0 else "osd_0",
                       osd_order=int(osd_order))
        return make, "ldpc v2 (BpOsdDecoder)"

    try:
        from ldpc import bposd_decoder as _V1
    except Exception as exc:                        # pragma: no cover
        raise SystemExit(
            "The `ldpc` package is required and is not importable.\n"
            "  pip install ldpc\n"
            f"(import failed: {exc})") from exc

    def make(H, priors, max_iter, osd_order):
        return _V1(H,
                   channel_probs=list(map(float, priors)),
                   max_iter=int(max_iter),
                   bp_method="ps",
                   osd_method="osd_cs" if osd_order > 0 else "osd_0",
                   osd_order=int(osd_order))
    return make, "ldpc v1 (bposd_decoder)"


# --------------------------------------------------------------------------
# priors
# --------------------------------------------------------------------------

LO, HI = 0.01, 0.99          # IBM's interval, verbatim


def draw_priors_decoy(n, rng, supports, beta, k):
    """Reweight against k randomly chosen found operators, not the aggregate.

    WHY THIS EXISTS. The aggregate per-qubit count is useless in this code
    family, provably, see the docstring at the top of the file. A *single*
    operator's support is not translation invariant, so a field built from a
    random handful of found operators does carry position information. Each
    trial draws a different handful, which is what keeps the search moving
    instead of locking onto one direction.

    This is Prof. Zhang's suggestion in the only form that can do anything
    here: suppress the support of specific operators already in hand, rather
    than the marginal over all of them.
    """
    u = rng.uniform(LO, HI, size=n)
    if beta <= 0.0 or not supports:
        return u
    mask = np.zeros(n)
    for idx in rng.choice(len(supports), size=min(k, len(supports)),
                          replace=False):
        mask[list(supports[idx])] += 1.0
    top = mask.max()
    if top <= 0.0:
        return u
    p = u ** (1.0 + beta * (mask / top))
    return np.clip(p, LO, HI)


def draw_priors(n, rng, seen=None, beta=0.0, center=False):
    """One draw of per-qubit prior error probabilities.

    `seen[i]` counts how many found operators contain qubit i. With beta = 0
    this is exactly U[0.01, 0.99) per qubit and `seen` is ignored, which is
    IBM's method and the control arm of the comparison.

    `center=False` ("adaptive") only ever lowers priors, so the average qubit
    gets more expensive as the search proceeds and the decoder drifts to higher
    weight. `center=True` ("balanced") subtracts the mean first, so
    below-average qubits get cheaper by as much as above-average ones get dearer
    and the overall scale is left alone. Which of those is better is the
    experiment, not something to assume.
    """
    u = rng.uniform(LO, HI, size=n)
    if beta <= 0.0 or seen is None:
        return u
    top = float(seen.max())
    if top <= 0.0:
        return u
    f = seen / top                          # in [0, 1]
    if center:
        f = f - f.mean()                    # in [-1, 1], mean zero
    p = u ** (1.0 + beta * f)               # u in (0,1): f>0 lowers, f<0 raises
    return np.clip(p, LO, HI)


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

def run(code_name, mode, beta, seed, budget, max_trials, out, quiet=False,
        decoys=8):
    code = build(code_name, 0 if code_name == "gross" else 1)
    lz = z_logical_basis(code)
    perms = translations(code)
    n, mz = code.n, code.HZ.shape[0]
    d = min(PUBLISHED[code_name])
    targets = sorted(PUBLISHED[code_name])          # [d, d+2]

    cfg = SETTINGS[code_name]
    make, backend = _decoder_factory()
    rng = np.random.default_rng(seed)

    # [ H_Z ; l^T ] and the synthetic syndrome. Only the last row changes.
    M = np.zeros((mz + 1, n), dtype=np.uint8)
    M[:mz] = code.HZ & 1
    syn = np.zeros(mz + 1, dtype=np.uint8)
    syn[mz] = 1

    # found[w] : set of operator bytes.  uniq[w] : one canonical rep per orbit.
    found = {w: set() for w in targets}
    uniq = {w: set() for w in targets}
    since_new = {w: 0 for w in targets}
    # Trial index at which the published count was first reached. This is the
    # A/B metric: a decoder run costs the same whichever priors it was given,
    # so trials-to-target is the honest comparison and wall time is the noisy
    # proxy for it.
    hit_at = {w: None for w in targets}
    seen = np.zeros(n, dtype=float)
    supports = []               # one tuple per found operator, for decoy mode

    trials = bad = off_target = 0
    t0 = time.time()

    if not quiet:
        print(f"  {code_name}: n={n} k={code.k} d={d}   backend: {backend}")
        print(f"  priors: {mode}" + (f" (beta={beta})" if mode != "random" else "")
              + f"   BP iters {cfg['max_iter']}, OSD order {cfg['osd_order']}")
        print(f"  targets: " + ", ".join(
            f"w={w} -> {PUBLISHED[code_name][w]}" for w in targets))
        print()

    while trials < max_trials and time.time() - t0 < budget:
        trials += 1

        # A random non-trivial Z-logical. Coefficients over the k basis rows;
        # reject the zero vector. Adding Z-stabilisers to l would not change
        # l . v for any v in ker(H_Z), so the coset representative is free.
        while True:
            coef = rng.integers(0, 2, size=len(lz), dtype=np.uint8)
            if coef.any():
                break
        M[mz] = (coef @ lz) & 1

        if mode == "decoy":
            priors = draw_priors_decoy(n, rng, supports, beta, decoys)
        else:
            priors = draw_priors(n, rng, seen,
                                 0.0 if mode == "random" else beta,
                                 center=(mode == "balanced"))
        dec = make(M, priors, cfg["max_iter"], cfg["osd_order"])
        v = np.asarray(dec.decode(syn), dtype=np.uint8) & 1

        # OSD always returns something; it is not always a solution when BP has
        # not converged and the OSD sweep misses. Verify, do not assume.
        if ((M @ v) & 1 != syn).any():
            bad += 1
            continue

        w = int(v.sum())
        if w not in found:
            off_target += 1
            continue

        b = v.tobytes()
        if b in found[w]:
            since_new[w] += 1
            continue

        orb = orbit(v, perms)
        new = orb - found[w]
        is_new_class = not (uniq[w] & orb)

        found[w] |= new
        if is_new_class:
            uniq[w].add(min(orb))               # canonical rep = min bytes
            since_new[w] = 0
            if not quiet:
                print(f"  t{trials:6d}  w={w}  +{len(new):4d} "
                      f"(orbit {len(orb):3d})  total w={w}: {len(found[w]):5d}"
                      f"   [{time.time()-t0:6.1f}s]")
        else:
            since_new[w] += 1
        for nb in new:
            arr = np.frombuffer(nb, dtype=np.uint8)
            seen += arr
            supports.append(tuple(np.nonzero(arr)[0]))

        if hit_at[w] is None and len(found[w]) >= PUBLISHED[code_name][w]:
            hit_at[w] = trials

        # IBM's stopping rule, applied per weight and reported as heuristic.
        if all(len(uniq[w]) > 0 and since_new[w] >= 10 * len(uniq[w])
               for w in targets):
            if not quiet:
                print("\n  IBM stopping rule satisfied at every target weight.")
            break

    elapsed = time.time() - t0
    result = {
        "code": code_name, "mode": mode, "beta": beta, "seed": seed,
        "decoys": decoys if mode == "decoy" else None,
        "trials": trials, "elapsed_s": round(elapsed, 1),
        "failed_solves": bad, "off_target_weights": off_target,
        "backend": backend, "settings": cfg,
        "weights": {str(w): {"total": len(found[w]),
                             "shift_unique": len(uniq[w]),
                             "published": PUBLISHED[code_name][w],
                             "match": len(found[w]) == PUBLISHED[code_name][w],
                             "trials_to_target": hit_at[w]}
                    for w in targets},
    }

    if not quiet:
        print(f"\n  {trials} trials, {elapsed:.1f}s "
              f"({bad} non-solutions, {off_target} outside target weights)")
        print(f"  {'weight':>7} {'found':>7} {'published':>10} "
              f"{'shift-uniq':>11} {'trials':>9}")
        for w in targets:
            r = result["weights"][str(w)]
            flag = "MATCH" if r["match"] else (
                "over" if r["total"] > r["published"] else "short")
            ta = "-" if r["trials_to_target"] is None else r["trials_to_target"]
            print(f"  {w:>7} {r['total']:>7} {r['published']:>10} "
                  f"{r['shift_unique']:>11} {ta:>9}   {flag}")
        print("\n  Reminder: this is IBM's heuristic stopping rule, not a "
              "proof.\n  The weight-18 count IS proven, by enumerate_weight.py, "
              "at 336.")

    if out:
        json.dump(result, open(out, "w"), indent=2)
        if not quiet:
            print(f"\n  wrote {out}")
    return result


# --------------------------------------------------------------------------
# A/B: does adaptive reweighting beat random priors?
# --------------------------------------------------------------------------

def compare(code_name, arms, seeds, budget, max_trials, out, decoys=8):
    """Same budget, same seeds, random vs adaptive. Trials is the metric.

    The honest comparison is trials-to-reach-the-published-count, because a
    decoder run costs the same whichever priors it was given. Wall time is
    reported too but it is the noisier number.
    """
    ws = sorted(PUBLISHED[code_name])
    rows = []
    print(f"  A/B on {code_name}: trials to reach the published count at each "
          f"target weight.\n  The 'random' arm is IBM's method exactly. Same "
          f"seeds, same budget, same\n  decoder settings; only the priors "
          f"differ.\n")
    for mode, beta in arms:
        for s in seeds:
            r = run(code_name, mode, beta, s, budget, max_trials,
                    out=None, quiet=True, decoys=decoys)
            r["arm"] = f"{mode}/{beta}"
            rows.append(r)
            cells = []
            for w in ws:
                rw = r["weights"][str(w)]
                cells.append(f"w{w}: " + (
                    f"{rw['trials_to_target']:>7}" if rw["trials_to_target"]
                    else f"  ({rw['total']}/{rw['published']})"))
            print(f"  {mode:<9} beta={beta:<4} seed={s:<3} {r['trials']:>7} "
                  f"trials {r['elapsed_s']:>6.1f}s   " + "   ".join(cells))

    # Runs that never reached the target are censored, not zero. Report the
    # number reached and the mean over those only, and say so.
    print(f"\n  {'arm':>16} {'runs':>5}  " +
          "  ".join(f"{'w'+str(w)+' hit':>10} {'mean trials':>12}"
                    for w in ws))
    for mode, beta in arms:
        arm = f"{mode}/{beta}"
        grp = [r for r in rows if r.get("arm") == arm]
        line = f"  {arm:>16} {len(grp):>5}  "
        for w in ws:
            got = [r["weights"][str(w)]["trials_to_target"] for r in grp]
            got = [g for g in got if g is not None]
            mean = f"{np.mean(got):.0f}" if got else "-"
            line += f"{len(got):>4}/{len(grp):<5} {mean:>12}  "
        print(line)
    print("\n  Runs that never reached a target are excluded from that mean, "
          "so\n  compare the reached-counts first: a lower mean over fewer "
          "runs is not a win.")

    if out:
        json.dump(rows, open(out, "w"), indent=2)
        print(f"\n  wrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Reproduce Figure 12 with BP+OSD, IBM's method, and test "
                    "adaptive priors against it.")
    ap.add_argument("code", choices=["gross", "two-gross"])
    ap.add_argument("--mode",
                    choices=["random", "adaptive", "balanced", "decoy"],
                    default="random")
    ap.add_argument("--decoys", type=int, default=8,
                    help="decoy mode: how many found operators to suppress "
                         "per trial")
    ap.add_argument("--beta", type=float, default=2.0,
                    help="adaptive squeeze strength; 0 reduces to IBM's method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=1800.0,
                    help="wall-clock seconds; the paper reports under 1500")
    ap.add_argument("--max-trials", type=int, default=2_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", action="store_true",
                    help="A/B random vs adaptive vs balanced over seeds")
    ap.add_argument("--arms", nargs="+",
                    default=["random:0", "adaptive:2", "balanced:2"],
                    help="mode:beta pairs, e.g. random:0 balanced:1 balanced:4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    if a.compare:
        out = a.out or data(f"bposd_compare_{a.code}.json")
        arms = [(s.split(":")[0], float(s.split(":")[1])) for s in a.arms]
        compare(a.code, arms, a.seeds, a.budget, a.max_trials, out,
                decoys=a.decoys)
    else:
        out = a.out or data(f"bposd_{a.code}_{a.mode}.json")
        run(a.code, a.mode, a.beta, a.seed, a.budget, a.max_trials, out,
            decoys=a.decoys)
