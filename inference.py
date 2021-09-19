import random
from pathlib import Path
from argparse import ArgumentParser

import cv2
import torch

from src import utils
import albumentations as A
from src.constants import NAME2ID, FAUNA_CLASSES
from src.agent import MegaEnsembleAgent, ClipAgent


def get_parser():
    parser = ArgumentParser()
    h = '%(type)s (default: %(default)s)'

    parser.add_argument('--img-dir', default='data/images', type=str, help=h)
    parser.add_argument('--elements-ckpt-dir', default='lightning_logs/elements', type=str, help=h)
    parser.add_argument('--sea-floor-ckpt-dir', default='lightning_logs/sea_floor', type=str, help=h)
    parser.add_argument('--fauna-model', default='ViT-B/32', type=str, help=h)
    parser.add_argument('--fauna-classes', nargs='+', default=FAUNA_CLASSES, type=str, help=h)

    parser.add_argument('--img-size', default=512, type=int, help=h)
    parser.add_argument('--center-crop', action='store_true', help=h)
    parser.add_argument('--element-th', default=0.5, type=int, help=h)

    parser.add_argument('--plot-path', default='grid.png', type=str, help=h)

    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    sea_floor_id_to_name = {v: k for k, v in NAME2ID['sea_floor'].items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = MegaEnsembleAgent(
        checkpoint_paths={
            'elements': [cp for cp in Path(args.elements_ckpt_dir).rglob('*.ckpt')],
            'sea_floor': [cp for cp in Path(args.sea_floor_ckpt_dir).rglob('*.ckpt')]
        },
        img_size=args.img_size,
        device = device
    )

    clip_agent = ClipAgent(args.fauna_classes, args.fauna_model, device=device)

    paths = [str(p) for p in Path(args.img_dir).rglob('*.jpg')]

    while True:

        path = random.choice(paths)
        print(path)

        img = cv2.imread(path)[..., ::-1]

        if args.center_crop:
            s = min(img.shape[:2])
            img = A.center_crop(img, s, s)

        resized_img = cv2.resize(img.copy(), (args.img_size, args.img_size))

        preds = agent(img)

        utils.plot_grid(
            resized_img,
            preds['elements']['filtered_labels'],
            preds['elements']['filtered_probas'],
            preds['elements']['filtered_heatmaps'],
            preds['sea_floor']['label'],
            preds['sea_floor']['proba'],
            utils.get_img_with_bboxes(
                img,
                resized_img,
                preds['elements']['filtered_labels'],
                preds['elements']['filtered_heatmaps'],
                clip_agent
            ),
            out_path=args.plot_path)

        input('Input a key to next prediction.')
