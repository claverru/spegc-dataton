from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import cv2
from tqdm import tqdm
from albumentations import longest_max_size


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    parser.add_argument('--in-img-dir', default='data/ocean_v4_fondos/ocean_v2', help=h)
    parser.add_argument('--out-img-dir', default='data/images/sea_floor', help=h)

    parser.add_argument('--max-img-size', default=1024, type=int, help=h)

    return parser


def load_reduce_save_img(path):
    img = cv2.imread(str(path))
    img = longest_max_size(img, max_size=max_img_size, interpolation=cv2.INTER_LINEAR)
    out_path = path.replace(str(in_dir), out_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, img)


if __name__ == '__main__':

    args = get_parser().parse_args()

    global in_dir
    in_dir = Path(args.in_img_dir)
    global out_dir
    out_dir = str(Path(args.out_img_dir))
    global max_img_size
    max_img_size = args.max_img_size


    paths = [str(p) for p in in_dir.rglob('*.jpg')]
    print(paths[0])

    with ProcessPoolExecutor() as p:
        list(tqdm(p.map(load_reduce_save_img, paths), total=len(paths)))
