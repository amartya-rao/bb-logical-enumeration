#!/usr/bin/env python3
"""
plot_figure12.py, draw the logical-operator weight distribution.

Reads sweep_<code>.json and plots count against weight on a log scale, which
is how Tour de gross presents it: the whole point of the figure is that the
counts span orders of magnitude.

Complete counts and lower bounds are drawn differently. A filled marker is a
proven count; an open marker with an upward arrow is a weight where the search
ran out of budget before proving no more exist. These searches can always be
given more time, so the two must not look the same.

The comparison line is the surface-code side of the figure: 875,178 weight-18
X-logicals in the d=18 rotated toric code against 336 in the two-gross code.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import data

TORIC_D18_WEIGHT18 = 875_178      # Tour de gross, section 2.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["gross", "two-gross"], nargs="?",
                    default="two-gross")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = data(f"sweep_{a.code}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}, run sweep.py first")
    res = json.load(open(path))
    if not res:
        raise SystemExit(f"{path} is empty")

    ws = sorted(int(w) for w in res)
    counts = [res[str(w)]["count"] for w in ws]
    done = [res[str(w)]["complete"] for w in ws]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.plot(ws, counts, "-", color="#1f4e79", lw=1.4, zorder=1)

    cw = [w for w, d in zip(ws, done) if d]
    cc = [c for c, d in zip(counts, done) if d]
    if cw:
        ax.plot(cw, cc, "o", ms=7, color="#1f4e79", zorder=3,
                label="complete (proven)")

    bw = [w for w, d in zip(ws, done) if not d]
    bc = [c for c, d in zip(counts, done) if not d]
    if bw:
        ax.plot(bw, bc, "o", ms=7, mfc="white", mec="#1f4e79", mew=1.6,
                zorder=3, label="lower bound (search incomplete)")
        for w, c in zip(bw, bc):
            ax.annotate("", xy=(w, c * 2.2), xytext=(w, c * 1.15),
                        arrowprops=dict(arrowstyle="->", color="#1f4e79",
                                        lw=1.2))

    if a.code == "two-gross" and 18 in ws:
        ax.axhline(TORIC_D18_WEIGHT18, ls="--", lw=1.2, color="#b03a2e")
        # Below the line, not above it: above collides with the title when
        # the y-range is short, which it is until several weights are done.
        ax.annotate(f"rotated toric code, d=18, weight 18: "
                    f"{TORIC_D18_WEIGHT18:,}",
                    xy=(ws[0], TORIC_D18_WEIGHT18), xytext=(4, -13),
                    textcoords="offset points", fontsize=8.5,
                    color="#b03a2e", va="top", ha="left")
        i = ws.index(18)
        ax.annotate(f"{counts[i]}", xy=(18, counts[i]), xytext=(6, -12),
                    textcoords="offset points", fontsize=9, color="#1f4e79")

    ax.set_yscale("log")
    ax.set_ylim(top=TORIC_D18_WEIGHT18 * 6)
    ax.set_xlabel("weight of X-logical operator")
    ax.set_ylabel("number of operators")
    title = {"gross": "[[144, 12, 12]] gross code",
             "two-gross": "[[288, 12, 18]] two-gross code"}[a.code]
    ax.set_title(f"Logical operator weight distribution\n{title}",
                 fontsize=10)
    ax.set_xticks(ws)
    ax.grid(alpha=0.25, which="both", ls=":")
    ax.legend(fontsize=8.5, frameon=False, loc="center left")
    fig.tight_layout()

    out = a.out or data(f"figure12_{a.code}.png")
    fig.savefig(out, dpi=200)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"  wrote {out} and {out.replace('.png', '.pdf')}")

    print("\n  distribution:")
    for w in ws:
        r = res[str(w)]
        print(f"    weight {w:>3}: {'=' if r['complete'] else '>='} "
              f"{r['count']:>8}   "
              f"{'proven complete' if r['complete'] else 'LOWER BOUND, not a count'}"
              f"   operators checked={r['verified']}")

    if not all(res[str(w)]["complete"] for w in ws):
        print("\n  WARNING: at least one weight above is a lower bound, not a")
        print("  count. A curve built from those measures how hard the search")
        print("  was, not the code. Do not quote or plot it as a distribution.")
        print("  See NOTES.md. Use bposd_enumerate.py for the real numbers.")


if __name__ == "__main__":
    main()
