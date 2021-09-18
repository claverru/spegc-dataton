import random
from pathlib import Path

import cv2
import numpy as np
import torch

from src import utils
from src.model import HeatMap
from src.constants import NAME2ID, COLORS
from src.data_loading import get_basic_transforms

id2name = {v: k for k, v in NAME2ID['elements'].items()}


ckpt_dir = 'lightning_logs/resnet18'
for ckpt_path in Path(ckpt_dir).rglob('*.ckpt'):
    print(ckpt_path)
    model = HeatMap(checkpoint_path=ckpt_path)
    break

img_size = model.hparams['img_size']

t = get_basic_transforms(img_size=img_size, grayscale=False)

paths = [str(p) for p in Path('data/images/elements').rglob('*.jpg')]


while True:
    path = random.choice(paths)
    print(path)
    img = cv2.imread(path)[..., ::-1]
    X = t(image=img)['image'].unsqueeze(0)

    img = cv2.resize(img, (img_size, img_size))

    with torch.no_grad():
        out, heatmap = model(X)

    out = out.squeeze(0)
    heatmap = heatmap.squeeze(0)

    pos_classes = [i.numpy().item() for i in torch.where(torch.sigmoid(out) > 0.5)[0]]

    out = out.numpy()
    heatmap = heatmap.numpy()

    binary_masks = np.stack([heatmap[c] for c in pos_classes], 0)

    for i, bn in enumerate(binary_masks):
        me = bn.mean()
        ma = bn.max()
        st = bn.std()
        print(bn.min(), ma, me, st)
        bn[(bn < me + st*2)] = 0
        bn /= ma
        binary_masks[i] = bn

    colors = [COLORS['elements'][id2name[c]] for c in pos_classes]
    result = utils.add_heatmap(img, binary_masks[0], colors[0])
    print(img.shape, img.dtype, result.shape, result.dtype)

    result = cv2.hconcat([img, result])[..., ::-1]
    cv2.imwrite('trash/test.png', result)
    input()
