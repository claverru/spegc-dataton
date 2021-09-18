from pathlib import Path
from itertools import combinations
from collections import defaultdict

from src.constants import NAME2ID


# img_dir = 'data/images/sea_floor'
img_dir = 'data/images/elements'

problem = img_dir.split('/')[-1]

p = Path(img_dir)

d = defaultdict(set)

for path in p.rglob('*.jpg'):
    d[path.parent.name].add(path.name)

print({k: len(d[k]) for k in NAME2ID[problem]})

for k1, k2 in combinations(d, 2):
    inter = d[k1].intersection(d[k2])
    print(k1, k2, len(inter))
    for repeated in inter:
        print(f'\t{repeated}')
