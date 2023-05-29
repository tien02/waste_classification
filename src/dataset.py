import os
import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from torchvision.datasets import ImageFolder
import src.config_training as config_training
from lightning.pytorch import LightningDataModule


def get_preprocess():
    if config_training.MODEL == "resnet-dino":
        return AutoImageProcessor.from_pretrained(config_training.RESNET_DINO)
    elif config_training.MODEL == "convnext":
        return AutoImageProcessor.from_pretrained(config_training.CONVNEXT)
    elif config_training.MODEL == "resnet-origin":
        return AutoImageProcessor.from_pretrained(config_training.RESTNET_ORIGIN)
    elif config_training.MODEL == "vit-mae":
        return AutoImageProcessor.from_pretrained(config_training.VIT_MAE)
    else:
        return AutoImageProcessor.from_pretrained(config_training.VIT)      

class WasteDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.preprocess = get_preprocess()
        self.trans = transform
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        sample = np.array(sample)
        
        if self.trans is not None:
            sample = self.trans(image=sample)['image']
        sample = self.preprocess(sample, return_tensors='pt')['pixel_values'][0]

        # return sample.pixel_values, target
        return sample, torch.tensor(target)


class TrashNet(LightningDataModule):
    def __init__(self, data_dir:str, batch_size:int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = A.Compose([
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.RandomBrightnessContrast(),
                        A.GaussianBlur(),
                        A.ISONoise(),
                        A.GlassBlur(),
                        A.RandomFog(),
                        A.SafeRotate()
                        ])

    def setup(self, stage: str):
        self.train_data = WasteDataset(root=os.path.join(self.data_dir, 'train'), transform=self.transform)
        self.val_data = WasteDataset(root=os.path.join(self.data_dir, 'val'), transform=self.transform)
        self.test_data = WasteDataset(root=os.path.join(self.data_dir, 'test'), transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
            return DataLoader(dataset=self.test_data, batch_size=self.batch_size)
