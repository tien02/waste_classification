import config
import torch
import torch.nn as nn
from transformers import ViTModel

from termcolor import colored

class ViT(nn.Module):
    def __init__(self,):
        super().__init__()
        self.vit = ViTModel.from_pretrained(config.VIT)
        self.dropout = nn.Dropout(0.4)
        self.dense = nn.Linear(in_features=768, out_features=config.OUT_CHANNELS)
    
    def forward(self,x):
        x = self.vit(pixel_values=x).last_hidden_state # out_channel=(1,197,768)
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
def test_vit():
    x = torch.rand(size=(1, 3, 224, 224))
    model = ViT()

    with torch.no_grad():
        y = model(x)
    if y.size() == torch.Size([1, config.OUT_CHANNELS]):
        print(colored("=== PASS ===",'green'))
    else:
        print(colored("=== ERROR ===", "red"))

if __name__ == "__main__":
    test_vit()