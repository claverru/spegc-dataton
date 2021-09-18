import random
from pathlib import Path

import cv2
import torch

from src import utils
import albumentations as A
from src.model import ElementsHeatMap
from src.constants import NAME2ID
from src.data_loading import get_basic_transforms


ckpt_dir = 'lightning_logs/resnet18'
models = [ElementsHeatMap(checkpoint_path=cp) for cp in Path(ckpt_dir).rglob('*.ckpt')]
[m.eval() for m in models]

img_size = models[0].hparams['img_size']

t = get_basic_transforms(img_size=img_size, grayscale=False)

paths = [str(p) for p in Path('data/images/elements').rglob('*.jpg')]
paths = ['data/images/elements/fauna/d79ede96662cb0e8e22489e6be6280d8.jpg']

while True:
    path = random.choice(paths)
    print(path)
    img = cv2.imread(path)[..., ::-1]
    img = A.center_crop(img, img.shape[0], 600)
    X = t(image=img)['image'].unsqueeze(0)

    img = cv2.resize(img, (img_size, img_size))

    outs = []
    heatmaps = []
    for m in models:
        with torch.no_grad():
            out, heatmap = models[0](X)
            outs.append(out)
            heatmaps.append(heatmap)

    heatmaps = sum(heatmaps)/len(heatmaps)
    outs = sum(outs)/len(outs)

    outs = outs.squeeze(0)
    heatmaps = heatmaps.squeeze(0)

    outs = outs.numpy()
    heatmaps = heatmaps.numpy()

    utils.plot_grid(img, list(NAME2ID['elements']), outs, heatmaps)

    input('Input a key to next prediction.')
