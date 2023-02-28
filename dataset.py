import config
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from termcolor import colored
from transformers import ConvNextImageProcessor, ViTImageProcessor


def get_preprocess():
    if config.MODEL == "resnet-dino":
        return ConvNextImageProcessor.from_pretrained(config.RESNET_DINO)
    elif config.MODEL == "convnext":
        return ConvNextImageProcessor.from_pretrained(config.CONVNEXT)
    elif config.MODEL == "resnet-origin":
        return ConvNextImageProcessor.from_pretrained(config.RESTNET_ORIGIN)
    else:
        return ViTImageProcessor.from_pretrained(config.VIT)    

class WasteDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)
        self.preprocess = get_preprocess()
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample = self.preprocess(sample, return_tensors='pt')['pixel_values'][0,:,:,:]

        # return sample.pixel_values, target
        return sample, target
    
def test_dataset():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5)
    ])

    data = WasteDataset("split_data/train", train_transform)

    for img, label in data:
        fk_img = torch.zeros_like(img)
        assert torch.sum(~(fk_img == img)) != 0, "Error loading image"
        assert isinstance(img, torch.Tensor), "Expect image's data type is torch.Tensor"
        assert isinstance(label, int), "Expect label's data type in interger"
        break
    print(colored("=== PASS ===", "green"))

if __name__ == "__main__":
    test_dataset()