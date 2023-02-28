import os

# MODEL CONFIG
VIT = "google/vit-base-patch16-224-in21k"
CONVNEXT = "facebook/convnext-tiny-224"
RESNET_DINO = "Ramos-Ramos/dino-resnet-50"

OUT_CHANNELS = 6

# DATA PATH
TRAIN_PATH = "split_data/train"
VAL_PATH = "split_data/val"
TEST_PATH = "split_data/test"

# DATALOADER
NUM_WORKERS = 2

# TRAINER
EPOCHS = 200
VAL_EACH_EPOCH = 4
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
ACCELERATOR = "gpu"

# OPTIMIZER
OPTIMIZER = 'AdamW'  # 'AdamW' 'Adam'

# MODEL
SEED = 42
MODEL = "resnet-dino"   # "resnet-dino", "convnext", "vit"

# TENSORBOARD LOG
TENSORBOARD = {
    "DIR": "",
    "NAME": f"{MODEL}_LR{LEARNING_RATE}",
    "VERSION": "0",
}

# CHECKPOINT
CHECKPOINT_DIR = os.path.join(TENSORBOARD["DIR"], TENSORBOARD["NAME"], TENSORBOARD["VERSION"], "CKPT")

# EVALUATE
TEST_CKPT_PATH = None

# KEEP_TRAINING PATH
CONTINUE_TRAINING = None