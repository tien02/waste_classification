import config
from pytorch_lightning import Trainer, seed_everything

from trainer import WasteClassifier
from utils import get_model, get_test_dataloader

if __name__ == "__main__":
    seed_everything(config.SEED)

    test_dataloader = get_test_dataloader()
    model = get_model()
    system = WasteClassifier(model=model)

    trainer = Trainer(accelerator=config.ACCELERATOR)

    trainer.test(model=system, 
            ckpt_path=config.TEST_CKPT_PATH, 
            dataloaders=test_dataloader)