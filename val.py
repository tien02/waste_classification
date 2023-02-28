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
    test_data = WasteDataset(config.TEST_PATH, transform=inference_transform)

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    return test_dataloader

if __name__ == "__main__":
    seed_everything(config.SEED)

    test_dataloader = get_dataloader()
    model = get_model()
    system = WasteClassifier(model=model)

    checkpoint_callback = ModelCheckpoint(dirpath= config.CHECKPOINT_DIR, monitor="val_loss",
                                            save_top_k=3, mode="min")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")

    logger = TensorBoardLogger(save_dir=config.TENSORBOARD["DIR"], name=config.TENSORBOARD["NAME"], version=config.TENSORBOARD["VERSION"])

    trainer = Trainer(accelerator=config.ACCELERATOR)

    trainer.fit(model=system, 
            ckpt_path=config.TEST_CKPT_PATH, 
            dataloaders=test_dataloader)