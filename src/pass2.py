"""Pass 2 only: open search with reweighting, resuming from saved state."""
import json, os, sys, time
import numpy as np
from bbcode import build
from distance import z_logical_basis
from enumerate_weight import orbit, solve_one, translations
import enumerate_weight as EW
from paths import data

STATE, W = data("weight18_two-gross_all.json"), 18
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
EW.TIME_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 22.0

code = build("two-gross", 1)
lz = z_logical_basis(code)
perms = translations(code)
found = {bytes.fromhex(h) for h in json.load(open(STATE))["found"]} \
        if os.path.exists(STATE) else set()
print(f"  resumed: {len(found)}")

supports = [tuple(np.nonzero(np.frombuffer(b, np.uint8))[0]) for b in found]
seen = np.zeros(code.n)
for b in found:
    seen += np.frombuffer(b, np.uint8)

t0 = time.time()
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PEN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
for j in range(start, len(lz)):
    if time.time() - t0 > budget:
        print(f"  budget reached at class {j}")
        break
    while time.time() - t0 < budget:
        v, status = solve_one(code, lz[j], W, supports, seen, PEN)
        if v is None:
            print(f"  class {j:2d}: {'proven empty' if status==2 else 'timeout'}")
            break
        new = orbit(v, perms) - found
        if not new:
            supports.append(tuple(np.nonzero(v)[0])); continue
        found |= new
        for b in new:
            arr = np.frombuffer(b, np.uint8)
            seen += arr; supports.append(tuple(np.nonzero(arr)[0]))
        print(f"  class {j:2d}: +{len(new):4d} -> {len(found)}  [{time.time()-t0:5.1f}s]")
        json.dump({"found":[b.hex() for b in found],"weight":W}, open(STATE,"w"))
json.dump({"found":[b.hex() for b in found],"weight":W}, open(STATE,"w"))
print(f"  total now: {len(found)}")
