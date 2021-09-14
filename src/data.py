import cv2
import torch
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
        label = None

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, label


def get_aug_transforms(img_size):
    return [
        # 
        A.Normalize(),
        ToTensorV2()
    ]


def get_basic_transforms(img_size):
    return [
        ## 
        A.Resize(img_size, img_size),
        A.Normalize(),
        ToTensorV2()
    ]


class DataModule(pl.LightningDataModule):

    def __init__(self, 
                 train_df, 
                 val_df, 
                 test_df, 
                 batch_size, 
                 img_size, 
                 tta=1,
                 num_workers=8):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.img_size = img_size
        self.tta = tta
        self.num_workers = num_workers

    def setup(self, stage=None):

        train_T = get_aug_transforms(self.img_size)
        val_T = get_basic_transforms(self.img_size)

        train_T = A.Compose(train_T)
        val_T = A.Compose(val_T)

        self.train_ds = Dataset(df=self.train_df, transforms=train_T)
        self.val_ds = Dataset(df=self.val_df, transforms=val_T)
        self.test_ds = Dataset(df=self.test_df, transforms=val_T)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
        )