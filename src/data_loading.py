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

    def __init__(self, problem, dicts, transforms=None, tta=1):
        super().__init__()
        self.problem = problem
        self.dicts = dicts
        self.transforms = transforms

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, index):
        image_path = self.dicts[index]['image_path']
        label = self.dicts[index]['label']
        target = self.get_label_by_problem(label)

        img = cv2.imread(image_path)[:, :, ::-1]

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, target

    def get_label_by_problem(self, label):
        if self.problem == 'sea_floor':
            return label




def get_aug_transforms(img_size, grayscale):
    return A.Compose([
        A.RandomResizedCrop(
            img_size,
            img_size,
            scale=(0.75, 1.0),
            ratio=(0.75, 1.33)
        ),
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
                 problem,
                 train_dicts,
                 val_dicts,
                 test_dicts,
                 batch_size,
                 img_size,
                 grayscale,
                 tta=1,
                 num_workers=8):
        super().__init__()
        self.problem = problem
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

        self.train_ds = Dataset(problem=self.problem, dicts=self.train_dicts, transforms=train_T)
        self.val_ds = Dataset(problem=self.problem, dicts=self.val_dicts, transforms=val_T)
        self.test_ds = Dataset(problem=self.problem, dicts=self.test_dicts, transforms=test_T)

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


def get_data(img_dir):
    paths, labels = [], []

    for path in Path(img_dir).rglob('*.jpg'):
        paths.append(str(path))
        labels.append(path.parent.name)

    return paths, labels


def get_dicts(paths, labels, index, problem):
    return [{
        'image_path': paths[i],
        'label': NAME2ID[problem][labels[i]]
    } for i in index]





def get_loss_weight(labels, problem):
    counter = {c: 0 for c in NAME2ID[problem]}
    for label in labels:
        counter[label] += 1

    counter = {c: len(labels) / counter[c] for c in counter}
    s = sum(v for v in counter.values())
    result = [0 for _ in NAME2ID[problem]]

    for c, v in counter.items():
        result[NAME2ID[problem][c]] = v / s

    return result
