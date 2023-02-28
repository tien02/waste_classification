import config
import torch
import torch.nn as nn
from transformers import ConvNextModel

from termcolor import colored

class ConvNext(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = ConvNextModel.from_pretrained(config.CONVNEXT)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, config.OUT_CHANNELS)

    def forward(self, x):
        x = self.backbone(pixel_values=x).pooler_output    # out channels=768
        x = self.dropout(self.fc(x))
        x = self.classifier(x)
        return x


def test_ConvNext():
    x = torch.rand(size=(1, 3, 256, 256))
    model = ConvNext()

    with torch.no_grad():
        y = model(x)

    if y.size() == torch.Size([1,config.OUT_CHANNELS]):
        print(colored("=== PASS === ", 'green'))
    else:
        print(colored("=== ERROR ===", 'red'))

if __name__ == "__main__":
    test_ConvNext()