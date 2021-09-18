import pickle
import functools
import multiprocessing
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import tqdm
from skimage import io


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'
    parser.add_argument('--img-dir', default='data', help=h)
    parser.add_argument('--save-path', default='data/cache_repeated.pkl', help=h)

    return parser


def compute_stats(image):
    h, w, _ = image.shape
    mean = image.mean()
    std = image.std()
    return h, w, mean, std


def process_label(img_path, cache: dict):
    try:
        image = io.imread(str(img_path))
    except Exception:
        return

    if image is None:
        return

    try:
        image_stats = compute_stats(image)

    except Exception as e:
        print(e)
        return

    if 0 in image_stats:
        return

    cache[str(img_path)] = image_stats


def clean_repeated(stats_dict: dict) -> set:
    groups = defaultdict(set)
    for image_path, stats in tqdm.tqdm(stats_dict.items()):
        groups[stats].add(image_path)

    return groups


if __name__ == '__main__':

    args = get_parser().parse_args()

    paths = [p for p in Path(args.img_dir).rglob('*') if p.is_file()]
    manager = multiprocessing.Manager()
    stats_dict = manager.dict()

    f = functools.partial(
        process_label,
        cache=stats_dict
    )
    with multiprocessing.Pool() as p:
        _ = list(tqdm.tqdm(p.imap_unordered(f, paths), total=len(paths)))

    dict_repeated = clean_repeated(stats_dict)

    pickle.dump(dict_repeated, open(args.save_path, 'wb'))
