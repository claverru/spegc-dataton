import random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import torch

from src import utils
import albumentations as A
from src.model import MegaEnsemble
from src.constants import NAME2ID
from src.data_loading import get_basic_transforms


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    parser.add_argument('--img-dir', default='data/images', type=str, help=h)
    parser.add_argument('--elements-ckpt-dir', default='lightning_logs/resnet18/elements', type=str, help=h)
    parser.add_argument('--sea-floor-ckpt-dir', default='lightning_logs/mobilenet/sea_floor', type=str, help=h)

    parser.add_argument('--img-size', default=512, type=int, help=h)

    parser.add_argument('--plot-path', default='plots/grid.png', type=str, help=h)

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    sea_floor_id_to_name = {v: k for k, v in NAME2ID['sea_floor'].items()}

    model = MegaEnsemble({
        'elements': [utils.load_model(cp) for cp in Path(args.elements_ckpt_dir).rglob('*.ckpt')],
        'sea_floor': [utils.load_model(cp) for cp in Path(args.sea_floor_ckpt_dir).rglob('*.ckpt')]
    })
    model.eval()

    t = get_basic_transforms(img_size=512, grayscale=False)

    paths = [str(p) for p in Path(args.img_dir).rglob('*.jpg')]

    while True:
        path = random.choice(paths)
        print(path)
        img = cv2.imread(path)[..., ::-1]
        img = A.center_crop(img, img.shape[0], 600)
        X = t(image=img)['image'].unsqueeze(0)

        img = cv2.resize(img, (args.img_size, args.img_size))

        with torch.no_grad():
            preds = model(X)

        outs = preds['elements']['probas'].squeeze(0)
        heatmaps = preds['elements']['heatmap'].squeeze(0)
        sea_floor_probas = preds['sea_floor']['probas'].squeeze(0)

        sea_floor_label = sea_floor_id_to_name[sea_floor_probas.argmax().numpy().item()]

        outs = outs.numpy()
        heatmaps = heatmaps.numpy()

        utils.plot_grid(img, list(NAME2ID['elements']), outs, heatmaps, sea_floor_label, out_path=args.plot_path)

        input('Input a key to next prediction.')
