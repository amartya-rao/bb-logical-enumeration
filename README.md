# BB code logical operator enumeration

Reproduction of Figure 12 from Tour de gross (arXiv:2506.03094).

Amartya Rao Ponugoti, EE, IIT Kanpur. With Prof. Hezi Zhang, UW-Madison.

## What it does

Appendix A.1 of that paper finds low weight logical operators with a BP+OSD
decoder, giving each qubit a random prior every run. It only does weights d and
d+2. All four counts come out right:

```
gross      w=12   1884     w=14  19728
two-gross  w=18    336     w=20   1728
```

Counts include all shifts. Both codes finish inside a 145 second run.

Weight 18 is also proven complete here. All twelve logical classes were closed
by integer programming. The paper only says it believes the set is complete.
The 336 operators form three translation orbits: 144 + 144 + 48.

## Reweighting

The idea was to make already used qubits expensive, so the search moves to new
ones. It cannot work on these codes.

Translations act transitively on each half of the qubit register. Once you find
one operator you get its whole orbit for free. So the set you have found is
closed under translation, and any per qubit count of it is the same for every
qubit. Measured: each of the 288 qubits appears in exactly 21 of the 336 weight
18 operators. There is nothing to reweight by.

I also tested the one version that escapes this, where you suppress a few
specific operators instead of the average over all of them. Three strengths,
four seeds, 20000 trials each. No effect at d+2, and worse at d. Tables are in
RESULTS.md.

## Layout

```
README.md     this file
RESULTS.md    the four counts and the reweighting result
FLATNESS.md   the symmetry result, and a correction to it
NOTES.md      working notes, what was tried and what was retracted
src/          the code
data/         every json file the scripts read or write
```

## Running

```
pip install numpy scipy ldpc matplotlib
cd src
python3 bposd_enumerate.py gross
python3 bposd_enumerate.py two-gross
python3 bb_flatness.py
python3 class_ties.py bb18
python3 class_ties.py bb72
python3 class_ties.py penalty
```

matplotlib is only needed for plot_figure12.py. Do not install ldpc with
--no-deps, it needs sinter at import time.

## Things to know

sweep.py was my earlier approach, using integer programming. Above weight 18 it
only gives lower bounds. Nothing was proven there, and the weight 20 number it
gave (144) is one orbit out of twelve. The real count is 1728. I kept the files
but the curve should not be plotted or quoted.

FLATNESS.md has a correction section. The symmetry result holds. An earlier
conclusion drawn from it does not, and the correction says why, with a worked
counterexample on a small code.
