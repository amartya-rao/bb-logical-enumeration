# The vulnerability map is flat on bivariate bicycle codes

14 September 2026, corrected 15 September.

Checks: src/bb_flatness.py for the symmetry result, src/class_ties.py for every
number in the correction section.

This started as a small observation about why the reweighting suggestion could
not work. Pushed further, it says something about the assignment method itself.

Read the correction before the conclusion. The symmetry result below is verified
and stands. An earlier conclusion drawn from it, that placement gains must be
second order, does not, and the correction section below replaces it.

## The claim

On a bivariate bicycle code, with the syndrome schedule the code's own structure
gives you, the per site vulnerability map

```
    w_i = d log p_L / d q_i        at uniform noise
```

takes at most four different values: left half data, right half data, X ancilla,
Z ancilla. Exactly four, by symmetry, not approximately.

This matters because the assignment method in my draft is a ranking. Sort sites
by w, sort physical qubits by quality, pair them up. On a surface code w changes
smoothly with radius and role, which is what makes the ranking useful. On a
bicycle code it collapses to four buckets, and inside a half all l*m data sites
tie exactly.

## The argument

Four steps, each checked numerically for both gross and two-gross.

1. The translations are code automorphisms. All l*m shifts preserve H_X and H_Z.
   They move data qubits within each half, X checks among themselves, Z checks
   among themselves. Checked for all 72 and all 144 shifts.

2. The action has four orbits: two on data qubits (the two halves), one on X
   checks, one on Z checks. It is transitive on each.

3. A syndrome round is covariant under them if its CNOTs are ordered by
   monomial. Checked.

   This step has a real condition. Order each check's CNOTs by qubit index
   instead and covariance fails outright, also checked. So the flatness depends
   on the schedule respecting the group, not on the code alone. IBM's schedule
   is monomial based, so it holds for the codes as built, but it is a hypothesis
   and it is the first thing a real deployment could break.

4. So w is constant on orbits. A translation invariant error model plus a
   translation equivariant decoder gives p_L(q) = p_L(q after shifting) for
   every shift. Differentiating at the uniform point gives w_i = w_j whenever i
   and j are in the same orbit. Four orbits, four values.

## A guess that was wrong

I expected some automorphism to merge the two data halves, which would collapse
data sites to one value. The obvious candidate is the torus flip (i,j) to (j,i)
combined with swapping the halves. It does not preserve H_X. Checked, false, for
two-gross. It only looks plausible because the gross polynomials A = x^3+y+y^2
and B = y^3+x+x^2 are exchanged by swapping x and y, and gross has l not equal
to m so the flip is not available there either.

The automorphism that does exchange the halves is the ZX duality: invert the
cell, (i,j) to (-i,-j), then swap the halves. Checked, it maps H_X to H_Z and
back, for both codes.

But it exchanges the X and Z sectors too. So it forces left half vulnerability
in one basis to equal right half vulnerability in the other basis. It does not
merge the halves within one basis, even though it looks like it should.

## Not established

This is a symmetry argument, not a measurement. I have not computed w on a
bicycle code and I do not know of anyone who has.

Whether the two data values differ is open. If they do differ there is still a
coarse first order signal even for a fixed decoder: put the good qubits in the
more vulnerable half.

Step 4 needs decoder equivariance, which real decoders do not have. BP+OSD with
different priors per qubit is not equivariant, and OSD breaks ties by column
order even with uniform priors. The idealisation is the right starting point.
The gap between it and a real decoder is where the work is.

---

## Correction, 15 September 2026

This file first concluded that if the two data values are equal then any benefit
is second order in the noise spread. That is wrong, and I can say exactly why.

Second order scaling needs the logical failure probability to be differentiable
at the symmetric point. It is not, as soon as the decoder is recalibrated to the
noise. Optimal decoding picks the most likely logical class for each syndrome. A
small uneven perturbation can flip which class wins. A maximum of smooth
functions has a kink where two of them tie, and Danskin's theorem gives the one
sided derivative there.

The premise survives, the conclusion does not. Flat site vulnerabilities and
linear placement gains can happen together.

### The counterexample

The [[18,4,4]] bicycle code, l = m = 3, A = x + I + y^2, B = y + I + x^2, under
independent X noise with perfect syndrome measurement and optimal decoding by
logical class. Run `class_ties.py bb18`.

| quantity | value |
|---|---|
| F0 at p0 = 0.01 | 0.0083131984078836 |
| syndromes where classes tie | 91 of 128 |
| tied classes share the same integer weight enumerator | yes, so the ties are structural |
| frozen decoder dF/dp_i | the same for all 18 qubits, 0.08870587057679 |

So a site ranking carries no information at all here. Yet take two arrangements
A and B that use the same set of error rates within each half, and their one
sided derivatives differ:

```
F_A'(0+) = -0.5162157705595
F_B'(0+) = -0.3176485597108
gap      =  0.1985672108487 per unit eps
```

That gap is linear. At eps = 0.001 it means A has a 2.8373% lower failure
probability under recalibrated decoding.

The scaling test says which regime each decoder is in:

| eps | frozen, gap | exponent | recalibrated, gap | exponent |
|---|---|---|---|---|
| 1.0e-4 | 3.38e-08 | 2.000 | 2.01e-05 | 1.020 |
| 5.0e-5 | 8.44e-09 | 2.000 | 1.00e-05 | 1.010 |
| 2.5e-5 | 2.11e-09 | 2.000 | 4.98e-06 | 1.005 |

Frozen decoder: exactly quadratic, which is what this file first claimed.
Recalibrated: exactly linear. The response is also genuinely one sided, since
the preferred arrangement flips for negative eps. So it is not of the form
-c|eps|.

### The same thing at BB72, to leading order

For [[72,12,6]] with t = 3, the exact leading coefficient of
F(p*r) = p^3 C(r) + higher terms is

```
C(r) = sum over syndromes of [ sum over classes W - max over classes W ]
```

where W adds up the product of r_i over weight 3 errors in each syndrome and
class. Run `class_ties.py bb72`:

- 552 syndromes with two competing classes, 96 with three
- 1,392 competing weight 3 error patterns
- C(1) = 744

This is the leading low noise behaviour, not an exact kink at a fixed nonzero
rate. The exact kink is only established at BB18.

### What it means for surrogate scores

A product score over minimum weight logicals, which is the family used by
calibration aware compilers, can rank two placements the opposite way round to
true failure probability. Checked on BB72 with certified bounds rather than
sampling: the surrogate prefers A, optimal decoding prefers B by at least
3.618%, and the two bound intervals do not overlap. The certificate is

```
L4 <= F <= L4 + Pr(five or more errors)
```

from enumerating all 1,091,059 patterns of weight 4 or less. Run
`class_ties.py bb72 --bounds`.

What this does not show: under a matched search both the class objective and the
product surrogate reach the same best leading coefficient, C = 436.644. A
surrogate that gets individual pairs wrong but still finds the same optimum
costs nothing at the job it is used for. The counterexample bounds how accurate
the surrogate is. It does not show that a better objective wins.

---

## What to test next

Computing w on BB72 would confirm the four value structure and settle whether
the two data values differ, but it no longer decides the placement question,
because the tie mechanism works either way.

What decides it is whether the ties survive a decoder anyone would actually run.
At BB18, 91 of 128 syndromes tie exactly, and that is a property of a tiny and
very degenerate code. At BB72 the leading order competition is 552 plus 96
syndromes out of a much larger space. Whether exact ties persist, or soften into
near ties that split at some crossover, is the open question. Exact optimal
decoding by logical class is #P-hard, so it is not the decoder to answer it
with.

So:

1. Compare frozen BP+OSD, recalibrated BP+OSD, and a decoder symmetrised in
   distribution, on the same BB18 arrangements, against the exact optimum as a
   reference. This says whether a practical decoder picks up any of the
   available gain.
2. Only if it does, check whether the arrangements that matter are reachable
   under real connectivity. If every allowed placement is a symmetry of the
   whole code, circuit and decoder together, the gain is zero at every order and
   none of this applies.

## Running the checks

```bash
cd src
python3 bb_flatness.py              # the symmetry checks
python3 class_ties.py bb18          # F0, the ties, the derivatives, the scaling
python3 class_ties.py bb72          # 552, 96, 1392, C(1) = 744, C(A), C(B)
python3 class_ties.py bb72 --bounds # the certified bounds
```
