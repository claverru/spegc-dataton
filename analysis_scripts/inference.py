import random
from pathlib import Path

import cv2
import timm
import torch
import matplotlib.pyplot as plt

from src.model import Model
from src.constants import NAME2ID
from src.data_loading import get_basic_transforms


class FeaturesModel(torch.nn.Module):

    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location='cpu').model
        # self.model = timm.create_model('resnet18', pretrained=True)
        self.model.eval()

    def forward(self, x):
        features = self.model.forward_features(x)
        pooled = self.model.global_pool(features)
        out = self.model.fc(pooled)
        return out, features, torch.tensor(next(self.model.fc.parameters()))


def plot_resnet_heatmap(img, heatmap, index_class):
    plt.clf()

    img = cv2.resize(img, heatmap.shape)
    pred_class = {v: k for k, v in NAME2ID['elements'].items()}[index_class]
    # plot image
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(10, 20)
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(img, alpha=0.5)
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].set_title(pred_class)
    ax[1].axis('off')

    plt.savefig('trash/test.png')


ckpt_dir = 'lightning_logs/resnet18'
for ckpt_path in Path(ckpt_dir).rglob('*.ckpt'):
    model = FeaturesModel(checkpoint_path=ckpt_path)
    break


t = get_basic_transforms(img_size=224, grayscale=False)

paths = [str(p) for p in Path('data/images/elements').rglob('*.jpg')]

while True:
    path = random.choice(paths)
    print(path)
    img = cv2.imread(path)[..., ::-1]
    X = t(image=img)['image'].unsqueeze(0)

    print(X.shape)

    with torch.no_grad():
        out, features, weights = model(X)

    inter_features = torch.nn.functional.interpolate(
        features, size=X.shape[-2:], scale_factor=None, mode='bilinear', align_corners=True)

    inter_features = inter_features.reshape(inter_features.shape[1], -1).numpy()

    classes = [i.numpy().item() for i in torch.where(torch.sigmoid(out.squeeze(0)) >= 0.5)[0]]

    # i = out[0].argmax()
    for i in classes:

        class_weights = weights[i].squeeze(0).numpy()

        heatmap =  class_weights @ inter_features
        heatmap = heatmap.reshape(*X.shape[-2:])

        plot_resnet_heatmap(img, heatmap, index_class=i)

        heatmap[heatmap < 0] = 0
        heatmap = heatmap - heatmap.min()
        heatmap = (heatmap / heatmap.max() * 255).astype('uint8')

        cv2.imwrite('trash/heatmap.png', heatmap)

        input('Another prediction:')
