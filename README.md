# Waste classfication

## Problem

One of the major challenges in sorting waste at the source is most people are still not properly informed about waste segregation. Artificial intelligence has advanced to a level where I think it can assist humans in quickly sorting junk.

<p float="left">
  <img src="assets/cardboard143.jpg" height="150" />
  <img src="assets/glass123.jpg" height="150" /> 
  <img src="assets/metal214.jpg" height="150" />
</p>

<p float="left">
  <img src="assets/paper384.jpg" height="150" />
  <img src="assets/plastic151.jpg" height="150" /> 
  <img src="assets/trash54.jpg" height="150" />
</p>

## Experiments
1. **Data**

In order to recognize waste in wild, I train models on the dataset which similar to the real world scenario. [Trashnet](https://github.com/garythung/trashnet) is the dataset to be used to train models. It consists of 2527 images categorize into 6 classes: 

* 501 glass
* 594 paper
* 403 cardboard
* 482 plastic
* 410 metal
* 137 trash
 
For more details, please check out the repository.

In order to mimic the real world environment, I applied some augmentation technique like `RandomBrightnessContrast`, `ISONoise`, `Blur`, `RandomFog` with the help of [albumentation](https://github.com/albumentations-team/albumentations).

2. **Models**

I experiment with various Classification models: 

* ConvNeXt
* ResNet 50
* ResNet DINO
* ViT
* ViT-MAE

The accuracy on test set is impressive, these models get F1 score at 9x%. But when I test it on the real images, most of the cases it failed. I'm still working on finding a better solution.

## Training & Testing
1. Install dependecies
```
pip install -r requirements.txt
```

2. Check the [notebook](./main.ipynb) for more details

## Run the API

Testing on the real world images, I realize that `ConNeXT` although the F1 score on test set isn't as good as `ViT` but it recognize well on these images. So I create the API with `ConvNeXt` as the classifier instead of `ViT`. The API load the ONNX model stored on Google Drive, run the code and it will automatically download the model.

For your specific needs, checkout [helper.py](api/helper.py). Change `PRETRAINED_MODEL` to one of the [Hugging Face models](https://huggingface.co/models), the image preprocessing is based on this. `DRIVE_PATH` as the link to your ONNX model, train your own and put it on Google Drive.

1. Build the docker image
```
bash build.sh
```

2. Run the container
```
bash run.sh
```

3. Inside the container, execute the following command to run the api
```
bash run_api.sh
```