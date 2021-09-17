import cv2
import torch
import numpy as np
import albumentations as A
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, transforms=None, tta=1):
        super().__init__()
        self.image_paths = df['image_path'].to_numpy()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        sea_floor_label = None
        elements_label = None

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, {'sea_floor': sea_floor_label, 'elements': elements_label}


def get_aug_transforms(img_size, grayscale):
    return A.Compose([
        #
        A.Normalize(),
        ToTensorV2()
    ])


def get_basic_transforms(img_size, grayscale):
    return A.Compose([
        ##
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ])


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 train_df,
                 val_df,
                 test_df,
                 batch_size,
                 img_size,
                 grayscale,
                 tta=1,
                 num_workers=8):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.img_size = img_size
        self.grayscale = grayscale
        self.tta = tta
        self.num_workers = num_workers

    def setup(self, stage=None):

        train_T = get_aug_transforms(self.img_size)
        if self.tta > 1:
            val_T = get_aug_transforms(self.img_size, self.grayscale)
        else:
            val_T = get_basic_transforms(self.img_size, self.grayscale)
        test_T = get_basic_transforms(self.img_size, self.grayscale)

        self.train_ds = Dataset(df=self.train_df, transforms=train_T)
        self.val_ds = Dataset(df=self.val_df, transforms=val_T)
        self.test_ds = Dataset(df=self.test_df, transforms=test_T)

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
