import torch.nn as nn
import src.config_training as config_training
from transformers import AutoConfig, AutoModel

class ConvNext(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(config_training.CONVNEXT)
        else:
            model_config = AutoConfig.from_pretrained(config_training.CONVNEXT)
            self.backbone = AutoModel.from_config(model_config)

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(0.4),
            nn.Linear(768, config_training.OUT_CHANNELS),
        )

    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output    # (1, 768)
        x = self.classifier(x)
        return x
    

class ResNetDINO(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(config_training.RESNET_DINO)
        else:
            model_config = AutoConfig.from_pretrained(config_training.RESNET_DINO)
            self.backbone = AutoModel.from_config(model_config)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, config_training.OUT_CHANNELS),
        )

    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output    # (1,2048,1,1)
        x = x[:, :, 0, 0]   # out_channel: (1,2048)
        x = self.classifier(x)
        return x
    

class ResNetOrigin(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(config_training.RESTNET_ORIGIN)
        else:
            model_config = AutoConfig.from_pretrained(config_training.RESTNET_ORIGIN)
            self.backbone = AutoModel.from_config(model_config)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.4),
            nn.Linear(1024, config_training.OUT_CHANNELS),
        )

    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output    # out_channel: [1,2048,1,1]
        x = x[:, :, 0, 0]   # out_channel: (1,2048)
        x = self.classifier(x)
        return x


class ViT(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(config_training.VIT)
        else:
            model_config = AutoConfig.from_pretrained(config_training.VIT)
            self.backbone = AutoModel.from_config(model_config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features=768, out_features=config_training.OUT_CHANNELS),
        )
    
    def forward(self,x):
        x = self.backbone(pixel_values=x).pooler_output # (1, 768)
        x = self.classifier(x)
        return x
    

class ViT_MAE(nn.Module):
    def __init__(self, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = AutoModel.from_pretrained(config_training.VIT_MAE)
        else:
            model_config = AutoConfig.from_pretrained(config_training.VIT_MAE)
            self.backbone = AutoModel.from_config(model_config)
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features=768, out_features=config_training.OUT_CHANNELS),
        )
    
    def forward(self,x):
        x = self.backbone(pixel_values=x).last_hidden_state # (1,50,768)
        x = x[:,0,:]    # (1,768)
        x = self.classifier(x)
        return x


def get_model():
    if config_training.MODEL == "vit":
        model = ViT()
    elif config_training.MODEL == "convnext":
        model = ConvNext()
    elif config_training.MODEL == "resnet-origin":
        model = ResNetOrigin()
    elif config_training.MODEL == "vit-mae":
        model = ViT_MAE()
    else:
        model = ResNetDINO()
    return model