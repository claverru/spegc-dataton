import random
from pathlib import Path

import cv2
import torch

from src import utils
import albumentations as A
from src.model import MegaEnsemble
from src.constants import NAME2ID
from src.data_loading import get_basic_transforms

IMG_SIZE = 512


elements_ckpt_dir = 'lightning_logs/resnet18/elements'
sea_floor_ckpt_dir = 'lightning_logs/mobilenet/sea_floor'


model = MegaEnsemble({
    'elements': [str(cp) for cp in Path(elements_ckpt_dir).rglob('*.ckpt')],
    'sea_floor': [str(cp) for cp in Path(sea_floor_ckpt_dir).rglob('*.ckpt')]
})
model.eval()

t = get_basic_transforms(img_size=512, grayscale=False)

paths = [str(p) for p in Path('data/images').rglob('*.jpg')]


while True:
    path = random.choice(paths)
    print(path)
    img = cv2.imread(path)[..., ::-1]
    img = A.center_crop(img, img.shape[0], 600)
    X = t(image=img)['image'].unsqueeze(0)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    with torch.no_grad():
        preds = model(X)

    outs = preds['elements']['probas'].squeeze(0)
    heatmaps = preds['elements']['heatmap'].squeeze(0)

    outs = outs.numpy()
    heatmaps = heatmaps.numpy()

    utils.plot_grid(img, list(NAME2ID['elements']), outs, heatmaps)

    input('Input a key to next prediction.')
