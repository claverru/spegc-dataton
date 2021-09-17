from pathlib import Path
from itertools import combinations
from collections import defaultdict


img_dir = 'data/ocean_v4_fondos'
# img_dir = 'data/ocean_elements_v4'

p = Path(img_dir)

d = defaultdict(set)

for path in p.rglob('*.jpg'):
    d[path.parent.name].add(path.name)

print({k: len(v) for k, v in d.items()})

for k1, k2 in combinations(d, 2):
    inter = d[k1].intersection(d[k2])
    print(k1, k2, len(inter))
    if inter:
        print(inter)
