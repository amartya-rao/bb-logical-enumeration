# Figure 12 results

13 September 2026. Code in src/bposd_enumerate.py, output files in data/.

## The four counts

The method is BP+OSD with random priors. Not integer programming, and not
Brouwer-Zimmermann, which is what I tried first. Appendix A.1 says so plainly,
and it only covers weights d and d+2. Two numbers per code, not a distribution.

| code | weight | found | paper | orbits | trials |
|---|---|---|---|---|---|
| gross [[144,12,12]] | 12 | 1884 | 1884 | 30 | 4,176 |
| gross | 14 | 19728 | 19728 | 274 | 42,244 |
| two-gross [[288,12,18]] | 18 | 336 | 336 | 3 | 2,127 |
| two-gross | 20 | 1728 | 1728 | 12 | 22,079 |

Their settings: gross uses 20 BP iterations and OSD order 0, two-gross uses 40
and order 7 with combination sweep. Priors are drawn from U[0.01, 0.99) for each
qubit, fresh every run.

Both codes finish inside one 145 second run. The trials column is where the last
new operator turned up. The rest of the run is the stopping rule confirming
nothing else appears. No invalid solutions in either run.

Files: data/bposd_gross_random.json, data/bposd_tg_random.json.

## Two notes on these numbers

Weight 20 is 1728, and my earlier 144 was one orbit out of twelve. The integer
programming sweep gave 144 and that made the sequence look impossible, since 336
at weight 18 cannot fall to 144 at weight 20. It was a 12 times undercount.

The 336 is proven here, and only believed in the paper. Their rule is to keep
going until ten times the orbit count more operators turn up with nothing new,
and they write that they strongly believe the set is complete.
enumerate_weight.py closed all twelve logical classes as proven infeasible. That
is a stronger statement about the same number, and it is the one place the
integer program beats the decoder.

## Reweighting

The idea: make qubits that already appear in found operators expensive, so the
decoder is pushed onto new support. In BP the prior works the other way round
from a cost, since LLR = log((1-p)/p), so a high prior is a cheap qubit. Raising
the cost means lowering the prior. I used p_i = u_i^(1 + B f_i), with u_i drawn
from U[0.01, 0.99) and f_i the normalised count of how often qubit i has been
seen. B = 0 gives their method exactly.

### Why it cannot work

1. The code is invariant under its l*m translations, which act transitively on
   each half of the register. Translating a logical operator gives another
   logical operator of the same weight.
2. Orbits are free. Once you have one member you have all of them. So the set of
   operators found is closed under translation.
3. Any per qubit count of a translation closed set is therefore the same for
   every qubit in a half.

So the field you would reweight by is a constant. Measured on the proven 336
weight 18 operators of the two-gross code, every one of the 288 qubits appears
in exactly 21 of them. Each closed orbit is already uniform on its own: the
orbit of 48 puts every qubit in exactly 3 of its members.

This is why the first A/B gave identical trial counts for both arms on the same
seed. Every prior was being multiplied by the same thing.

### The version that is not constant

A single operator's support is not translation invariant. So each trial, pick 8
found operators at random and suppress only their support. That is --mode decoy.

Fixed budget of 20000 trials on the gross code, four seeds per arm, same seeds
across arms:

| priors | n | trials to reach 1884 at w=12 | classes at w=14, of 274 |
|---|---|---|---|
| random (theirs) | 4 | 3850 ± 182 | 261.25 ± 2.32 |
| decoy, B = 0.5 | 4 | 8392 ± 1052 | 261.25 ± 1.25 |
| decoy, B = 2.0 | 4 | 6823 ± 880 | 259.25 ± 1.31 |
| decoy, B = 8.0 | 4 | 5227 ± 1725 | 261.75 ± 1.31 |

Errors are s.e.m. over seeds.

At weight d it is significantly worse. B = 0.5 is 4.3 sigma slower than random
priors and B = 2.0 is 3.3 sigma slower. Ten of the twelve reweighted runs took
longer than any of the four random runs, and both exceptions are at B = 8. It
gets closer to parity as B grows, which makes sense: at large B the suppressed
qubits are driven to the floor and it behaves like a hard cut on those eight
operators rather than a soft push.

At weight d+2 it does nothing. Every arm lands near 261 of 274 classes, inside
one s.e.m.

### Why

93.0% of random prior trials return an operator outside weights 12 and 14. Under
decoy reweighting that is 93.2%, unchanged. The cost is not rediscovery, which
reweighting addresses. It is the decoder overshooting the target weights, which
reweighting does not touch, and pushing it off known support makes overshoot
more likely. At the minimum weight that is directly harmful, since minimum
weight solutions are what BP+OSD naturally returns.

Files: data/ab_fixed.json, data/ab_beta.json.

## Summary

- All four counts reproduce.
- Weight 18 is proven here, believed in the paper.
- Per qubit reweighting cannot work on these codes, by the transitivity
  argument, and the measurement confirms it exactly: 21 of 336, every qubit.
- The version that is not constant is no better at d+2 and worse at d, across
  three strengths.
- What it would need to beat is weight overshoot, not rediscovery. 93% of trials
  land outside the target weights and the priors do not change that.

I had also thought reweighting helped when the objective was minimising weight.
That was never measured. Running it properly, with the same logical class and
the same forbidden set, penalty 0.0 against penalty 0.30, five solves each,
gives identical results: 720 operators and 5 orbits in both arms. Run
`class_ties.py penalty` to reproduce it. So it has not helped in any form I
have tried.

What did find the orbit of 48 was symmetry slicing, not reweighting. Imposing
invariance under an order 3 translation ties the 288 variables into 96 groups
and the program solves in seconds. See symmetric_search.py.

## Running it

```bash
pip install numpy scipy ldpc
cd src
python3 bposd_enumerate.py gross
python3 bposd_enumerate.py two-gross
python3 bposd_enumerate.py gross --compare \
    --arms random:0 decoy:0.5 decoy:2 decoy:8 \
    --seeds 0 1 2 3 --max-trials 20000
python3 bposd_enumerate.py two-gross --mode adaptive --beta 2
```

The last one should do nothing, because the field is constant.

One environment note: ldpc 2.4.1 imports an optional sinter integration when the
package loads. If sinter is missing the import fails even though nothing here
uses it. A plain pip install ldpc pulls it in.
