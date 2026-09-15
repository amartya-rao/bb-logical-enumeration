# Figure 12 runbook

Working notes, roughly in the order things happened. RESULTS.md has the
final answers. This file is here so the wrong turns stay on the record.

## Where it ended up

- gross built and checked: [[144, 12, 12]], matches the paper
- two-gross built and checked: [[288, 12, 18]], matches
- weight 18 X-logicals of two-gross: 336, proven complete and verified
- full weight distribution: wrong target, see below

The 336 splits into three translation orbits, 144 + 144 + 48. The small one is
fixed by the order 3 shift (4,4).

## Files

```
src/bbcode.py            code construction and GF(2) linear algebra
src/distance.py          minimum weight logical by integer program
src/enumerate_weight.py  per class enumeration, no-good cuts, orbits
src/symmetric_search.py  restrict to a translation invariant slice
src/verify_336.py        the two passes plus verification
src/bposd_enumerate.py   the method the paper actually uses
src/class_ties.py        the numbers in FLATNESS.md
src/bb_flatness.py       symmetry checks
src/sweep.py             walks weights, see the warning below
src/plot_figure12.py     draws it
src/pass2.py             open search pass alone, for resuming
src/paths.py             where the json files live
```

## What is in data/

```
bposd_gross_random.json        the four counts, gross
bposd_tg_random.json           the four counts, two-gross
ab_fixed.json, ab_beta.json    the reweighting A/B, four seeds per arm
ab_random, ab_bal2, ab_decoy2  earlier A/B runs, kept for the record
weight18_two-gross_all.json    the 336 operators, from verify_336.py
weight18_two-gross_orbits.json the 48 from the symmetric search
weight18_two-gross_sweep.json  the 336 again, from sweep.py
weight20..28_*_sweep.json      lower bounds, not counts, see the warning above
sweep_two-gross.json           the sweep summary, with the complete flag
```

## The thing to be careful about

A count is only complete when every logical class has been proven empty. HiGHS
returns status 2 for proven infeasible and status 1 for ran out of time. They
are not the same. An earlier version of this code treated them as if they were,
and reported 288 for a number whose true value is 336.

sweep.py now records complete true or false for each weight, and
plot_figure12.py draws lower bounds differently from counts. Check that flag
before quoting a number.

Counts also grow fast with weight, so most weights above the distance come out
as lower bounds however long you run. That is a property of the problem, not a
failure. Say so rather than presenting a bound as a count.

## What worked

Symmetry slicing got the last 48. An open search stalls at 288 because the third
orbit sits in a thin invariant slice it almost never lands in. Imposing v = g(v)
for an order 3 translation ties the 288 variables into 96 groups and the program
solves in seconds. That is the generalisable trick here.

## What did not

Reweighting hurts when the weight is pinned, and does nothing when it is not.

Once the weight is pinned by an equality constraint the problem is pure
feasibility, and the penalty just gives the solver a pointless objective. Solves
that took 14 seconds started timing out at 22. sweep.py therefore sets the
penalty to zero in that path.

I had also written that it helps when the objective is minimising weight. That
was an impression from watching runs, never measured, and I stated it as a
finding. The controlled test, same logical class, same forbidden set, penalty
0.0 against penalty 0.30, five solves each, gives identical results: 720
operators and 5 orbits in both arms. Run `class_ties.py penalty` to reproduce
it. Together with the BP+OSD result, the honest summary is that reweighting has
not helped in any form I have tried.

---

# The sweep, 13 September 2026

It ran to completion, weights 18 to 28, about 2.6 hours.

| weight | count | classes proven empty |
|---|---|---|
| 18 | 336 | 12 of 12, complete |
| 20 | at least 144 | 0 of 12 |
| 22 | at least 432 | 0 of 12 |
| 24 | at least 2340 | 0 of 12 |
| 26 | at least 1440 | 0 of 12 |
| 28 | at least 3096 | 0 of 12 |

Only weight 18 is a result. The rest are unproven lower bounds and must not be
plotted as a distribution.

Two things settle it. No class was proven empty above weight 18, not one, so
every higher number is whatever the solver happened to find before timing out.
And the sequence is impossible: 336 at weight 18 falling to 144 at weight 20
cannot happen, because the number of logical operators grows with weight.

A confirming probe, forbidding the 144 and re-solving one class at weight 20
with a 130 second limit, returned status 1. Neither more operators nor a proof
there are none.

Integer programming per logical class is the right tool for proving a single
minimum weight count and the wrong one for a distribution. The 336 came out
clean precisely because weight 18 is the minimum, so the feasible set is small
enough to exhaust. The six point curve measures search difficulty, not the code.
Do not run sweep.py to build a distribution. Use bposd_enumerate.py.

## The right method, from the paper

Brouwer-Zimmermann was my guess and it was wrong. Appendix A.1 says what they
did, and it is neither integer programming nor BZ.

> "We can use the BP+OSD decoder to find low-weight logical operators [...] by
> solving a decoding problem that demands commutation with all stabilizers and
> anti-commutation with a random non-trivial logical operator. [...] However, we
> can introduce random priors to the BP+OSD decoder to coax it more often into
> finding higher weight logicals as well. Every decoding run, we pull each qubit
> prior error probability from the uniform distribution on [0.01, 0.99)."

Their settings: gross 20 BP iterations and OSD order 0, two-gross 40 and order 7
with combination sweep.

The scope is also much smaller than I assumed. They do weights d and d+2 only,
so for two-gross that is 18 and 20. Two numbers.

Their published counts, including all shifts:

| code | weight d | weight d+2 |
|---|---|---|
| gross [[144,12,12]] | 1884 at 12 | 19728 at 14 |
| two-gross [[288,12,18]] | 336 at 18 | 1728 at 20 |

Z counts are the same by the ZX duality.

This settles weight 20 at 1728. The sweep found 144, which is exactly one
translation orbit of the twelve. Shift-unique in their language is the same as
the orbit concept here.

Their stopping rule is a heuristic, not a proof. They run until they have seen
ten times the shift-unique count more operators of that weight without finding a
new one, and write that they strongly believe the weight 18 set is complete. The
336 here is proven, all twelve classes closed with status 2. That is a stronger
statement than the paper makes about the same number.

## Reweighting, in the setting it belongs to

The suggestion reads clearly once you know the enumeration is the decoder. It
means raising the prior on qubits already seen, so the decoder moves to fresh
support. RESULTS.md has why that cannot work on these codes, and what
happened to the one version that is not vacuous.

---

# Done, 13 September 2026

Everything above from the sweep onwards was the plan, and it was carried out.
See RESULTS.md.

- bposd_enumerate.py implements their method. All four counts reproduce.
- Weight 20 confirmed at 1728.
- Per qubit reweighting is provably vacuous here. Every one of the 288 qubits
  appears in exactly 21 of the 336 weight 18 operators.
- The one version that escapes the argument was tested at three strengths and is
  not an improvement: nothing at d+2, and 3 to 4 sigma worse at d.
- Artifacts in data/. Do not re-run the sweep. It is superseded for counting
  and kept only for the weight 18 completeness proof, which is still the
  strongest statement anyone has about that number.
