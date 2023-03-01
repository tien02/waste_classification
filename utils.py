from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import config
from models.vit import ViT
from models.convnext import ConvNext
from models.resnet_dino import ResNetDINO
from models.resnet_origin import ResNetOrigin
from dataset import WasteDataset

def get_model():
    if config.MODEL == "vit":
        model = ViT()
    elif config.MODEL == "convnext":
        model = ConvNext()
    elif config.MODEL == "resnet-origin":
        model = ResNetOrigin()
    else:
        model = ResNetDINO()
    return model

def get_train_val_dataloader():

    train_transform = get_train_transform()
    inference_transform = get_inference_transform()
    train_data = WasteDataset(config.TRAIN_PATH, transform=train_transform)
    val_data = WasteDataset(config.VAL_PATH, transform=inference_transform)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return train_dataloader, val_dataloader

def get_test_dataloader():
    inference_transform = get_inference_transform()

    test_data = WasteDataset(config.TEST_PATH, transform=inference_transform)

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return test_dataloader

def get_train_transform():

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5)
    ])

    return train_transform

def get_inference_transform():
    inference_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return inference_transform