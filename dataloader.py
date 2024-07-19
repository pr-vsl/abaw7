import os

import lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image
from pathlib import Path
import torch

import pandas as pd


class ABAW7Dataset(Dataset):
    def __init__(self, root_dir, split, transform=None, anno_file=None):
        if isinstance(root_dir, Path):
            self.root_dir = root_dir
        elif isinstance(root_dir, str):
            self.root_dir = Path(root_dir)
        else:
            raise ValueError('root_dir must be Path or str')

        self.transform = transform
        self.data = pd.read_csv(anno_file, skiprows=1, header=None).values
        self.split = split

        if self.split in ['train', 'val']:
            # print('Dropping samples')
            pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.root_dir / 'cropped_aligned' / self.data[idx][0]
        if self.split in ['train', 'val']:
            va = self.data[idx][1:3].astype(float)
            expr = np.zeros(8, dtype=float)
            if int(self.data[idx][3]) >= 0:
                expr[int(self.data[idx][3])] = 1
            else:
                expr = -1*np.ones(8)
            aus = self.data[idx][4:].astype(float)
        else:
            va = [-1, -1]
            expr = [-1] * 8
            aus = [-1] * 12
        try:
            image = read_image(file_path.__str__(), )
        except:
            print(file_path.__str__())

        if self.transform is not None:
            image = self.transform(image)

        return {'image': image, 'va': va, 'expr': expr, 'aus': aus, 'name': self.data[idx][0]}


class ABAW7DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", img_size=112, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = v2.Compose([
            # v2.RandomResizedCrop(img_size, antialias=True),
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_anno = self.data_dir / 'training_set_annotations.txt'
        self.val_anno = self.data_dir / 'validation_set_annotations.txt'
        self.test_anno = self.data_dir / 'MTL.txt'

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.abaw_train = ABAW7Dataset(self.data_dir, split='train', transform=self.train_transform,
                                           anno_file=self.train_anno)
            self.abaw_val = ABAW7Dataset(self.data_dir, split='val', transform=self.test_transform,
                                         anno_file=self.val_anno)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.abaw_val = ABAW7Dataset(self.data_dir, split='val', transform=self.test_transform,
                                         anno_file=self.val_anno)

        if stage == "predict":
            self.abaw_val = ABAW7Dataset(self.data_dir, split='val', transform=self.test_transform,
                                         anno_file=self.val_anno)
            self.abaw_test = ABAW7Dataset(self.data_dir, split='test', transform=self.test_transform,
                                          anno_file=self.test_anno)

    def train_dataloader(self):
        return DataLoader(self.abaw_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.abaw_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.abaw_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.abaw_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
