import config
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.vit import ViT
from models.resnet_dino import ResNetDINO
from models.convnext import ConvNext
from dataset import WasteDataset

from trainer import WasteClassifier

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5)
    ])

inference_transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_model():
    if config.MODEL == "vit":
        model = ViT()
    elif config.MODEL == "convnext":
        model = ConvNext()
    else:
        model = ResNetDINO()
    return model

def get_dataloader():
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

if __name__ == "__main__":
    seed_everything(config.SEED)

    train_dataloader, val_dataloader = get_dataloader()
    model = get_model()
    system = WasteClassifier(model=model)

    checkpoint_callback = ModelCheckpoint(dirpath= config.CHECKPOINT_DIR, monitor="val_loss",
                                            save_top_k=3, mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = TensorBoardLogger(save_dir=config.TENSORBOARD["DIR"], name=config.TENSORBOARD["NAME"], version=config.TENSORBOARD["VERSION"])

    trainer = Trainer(accelerator=config.ACCELERATOR, check_val_every_n_epoch=config.VAL_EACH_EPOCH,
                    gradient_clip_val=1.0,max_epochs=config.EPOCHS,
                    enable_checkpointing=True, deterministic=True, default_root_dir=config.CHECKPOINT_DIR,
                    callbacks=[checkpoint_callback, early_stopping], logger=logger)

    trainer.fit(model=system, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader, 
            ckpt_path=config.CONTINUE_TRAINING)