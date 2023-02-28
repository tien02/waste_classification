import config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from trainer import WasteClassifier
from utils import get_model, get_train_val_dataloader

if __name__ == "__main__":
    seed_everything(config.SEED)

    train_dataloader, val_dataloader = get_train_val_dataloader()
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