from pathlib import Path
from itertools import combinations
from argparse import ArgumentParser
from collections import defaultdict

from src.constants import NAME2ID


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'
    parser.add_argument('--img-dir', default='data/images/elements', type=str, help=h)
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    p = Path(args.img_dir)
    problem = p.name

    d = defaultdict(set)

    for path in p.rglob('*.jpg'):
        d[path.parent.name].add(path.name)

    print({k: len(d[k]) for k in NAME2ID[problem]})

    for k1, k2 in combinations(d, 2):
        inter = d[k1].intersection(d[k2])
        print(k1, k2, len(inter))
        for repeated in inter:
            print(f'\t{repeated}')
