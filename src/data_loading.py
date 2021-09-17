from collections import defaultdict
from pathlib import Path
from albumentations.augmentations.transforms import HorizontalFlip, ToGray

import cv2
import torch
import numpy as np
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2

from src.constants import NAME2ID


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dicts, transforms=None, tta=1):
        super().__init__()
        self.dicts = dicts * tta
        self.transforms = transforms

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, index):
        image_path = self.dicts[index]['image_path']
        target = torch.tensor(self.dicts[index]['label'], dtype=torch.long)

        img = cv2.imread(image_path)[..., ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, target


def get_aug_transforms(img_size, grayscale):
    return A.Compose([
        A.Rotate(limit=20, p=0.5),
        A.RandomResizedCrop(
            img_size,
            img_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33)
        ),
        A.FancyPCA(),
        A.HorizontalFlip(p=0.5),
        A.ToGray(p=0, always_apply=grayscale),
        A.Normalize(),
        ToTensorV2()
    ])


def get_basic_transforms(img_size, grayscale):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.ToGray(p=0, always_apply=grayscale),
        A.Normalize(),
        ToTensorV2()
    ])


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 train_dicts,
                 val_dicts,
                 test_dicts,
                 batch_size,
                 img_size,
                 grayscale,
                 tta=1,
                 num_workers=8):
        super().__init__()
        self.train_dicts = train_dicts
        self.val_dicts = val_dicts
        self.test_dicts = test_dicts
        self.batch_size = batch_size
        self.img_size = img_size
        self.grayscale = grayscale
        self.tta = tta
        self.num_workers = num_workers

    def setup(self, stage=None):

        train_T = get_aug_transforms(self.img_size, self.grayscale)
        if self.tta > 1:
            val_T = get_aug_transforms(self.img_size, self.grayscale)
        else:
            val_T = get_basic_transforms(self.img_size, self.grayscale)
        test_T = get_basic_transforms(self.img_size, self.grayscale)

        self.train_ds = Dataset(dicts=self.train_dicts, transforms=train_T)
        self.val_ds = Dataset(dicts=self.val_dicts, transforms=val_T)
        self.test_ds = Dataset(dicts=self.test_dicts, transforms=test_T)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )


def get_data(img_dir, problem):
    paths, labels = [], []

    if problem == 'sea_floor':
        for path in Path(img_dir).rglob('*.jpg'):
            paths.append(str(path))
            label = NAME2ID[problem][path.parent.name]
            labels.append(label)

    elif problem == 'elements':
        paths_dict = defaultdict(list)

        for path in Path(img_dir).rglob('*.jpg'):
            paths_dict[path.name].append(path)

        for _, ppaths in paths_dict.items():
            paths.append(str(ppaths[0]))
            label = [0 for _ in NAME2ID[problem]]
            for ppath in ppaths:
                c = ppath.parent.name
                label[NAME2ID[problem][c]] = 1
            labels.append(label)

    else:
        raise ValueError('sea_floor or elements')

    return paths, labels


def get_dicts(paths, labels, index):
    return [{
        'image_path': paths[i],
        'label': labels[i]
    } for i in index]


def get_loss_weight(labels, problem):

    if problem == 'sea_floor':

        counter = {c: 0 for c in NAME2ID[problem].values()}
        for label in labels:
            counter[label] += 1

        counter = {c: len(labels) / counter[c] for c in counter}
        s = sum(v for v in counter.values())
        result = [0 for _ in NAME2ID[problem]]

        for c, v in counter.items():
            result[c] = v / s

    elif problem == 'elements':
        # pos_weight = neg/pos
        result = [[0, 0] for _ in labels[0]]
        for label in labels:
            for i, value in enumerate(label):
                result[i][value] += 1

        result = [neg/pos for neg, pos in result]

    else:
        raise ValueError('sea_floor or elements')

    return result
