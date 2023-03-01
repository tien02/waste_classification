import config
import torch
import torch.nn as nn
from transformers import ResNetModel

from termcolor import colored

# ResNet50 origin from Microsoft
class ResNetOrigin(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(config.RESTNET_ORIGIN)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2048, 1024)
        self.classifier = nn.Linear(1024, config.OUT_CHANNELS)

    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output    # out_channel: [1,2048,1,1]
        x = x[:, :, 0, 0]   # out_channel: (1,2048)
        x = self.dropout(self.fc(x))
        x = self.classifier(x)
        return x


def test_ResNet():
    x = torch.rand(size=(1, 3, 224, 224))
    model = ResNetOrigin()

    with torch.no_grad():
        y = model(x)

    if y.size() == torch.Size([1,config.OUT_CHANNELS]):
        print(colored("=== PASS === ", 'green'))
    else:
        print(colored("=== ERROR ===", 'red'))

if __name__ == "__main__":
    test_ResNet()