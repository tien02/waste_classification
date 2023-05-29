import os

# MODEL CONFIG
VIT = "google/vit-base-patch16-224-in21k"
VIT_MAE = "facebook/vit-mae-base"
CONVNEXT = "facebook/convnext-tiny-224"
RESNET_DINO = "Ramos-Ramos/dino-resnet-50"
RESTNET_ORIGIN = "microsoft/resnet-50"

OUT_CHANNELS = 6

# DATALOADER
NUM_WORKERS = 2

# TRAINER
LR = {
    'BACKBONE': 5e-5,
    'CLASSIFIER': 5e-4,
} 
ACCELERATOR = "gpu"

# OPTIMIZER
OPTIMIZER = 'AdamW'  # 'AdamW' 'Adam'

# MODEL
MODEL = "convnext"   # "resnet-dino", "resnet-origin", "convnext", "vit", "vit-mae"