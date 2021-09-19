from pathlib import Path
from argparse import ArgumentParser
from itertools import combinations
from collections import defaultdict

from src.constants import NAME2ID


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'
    parser.add_argument('--problem-img-dir', default='data/images/elements', help=h)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    path = Path(args.problem_img_dir)

    problem = path.name

    d = defaultdict(set)

    for path in path.rglob('*.jpg'):
        d[path.parent.name].add(path.name)

    print('Number of elemnts per dataset:', {k: len(d[k]) for k in NAME2ID[problem]})
    for n in range(2, len(NAME2ID[problem])):
        for k1, *k2 in combinations(d, n):
            inter = d[k1]
            for k in k2:
                inter = inter.intersection(d[k])
            print('NUmber of elemnts repeated between', k1, *k2, 'is', len(inter))
            for repeated in inter:
                print(f'\t{repeated}')
