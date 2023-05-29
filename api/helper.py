import os
import gdown
import numpy as np
from PIL import Image
import onnxruntime as rt
from transformers import AutoImageProcessor

PRETRAINED_MODEL = "facebook/convnext-tiny-224"
DRIVE_PATH = "https://drive.google.com/file/d/1lQxIyuvbh2hBnR6ow4zVf-wH6SA6sipA/view?usp=sharing"

class_idx = {
    'cardboard': 0, 
    'glass': 1, 
    'metal': 2, 
    'paper': 3, 
    'plastic': 4, 
    'trash': 5
    }

idx_class = {v:k for k, v in class_idx.items()}

preprocessor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)


def load_classifier():
    ckpt_path = os.path.join("ckpt", "model.onnx")
    if not os.path.exists(ckpt_path):
        print(f"Can't not find model weights, attempt to download model.onnx...")

        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        
        print(f"Download to {ckpt_path}...")
        url = 'https://drive.google.com/uc?/export=download&id=' + DRIVE_PATH.split('/')[-2]
        gdown.download(url, ckpt_path, quiet=False)
    
    classifier = rt.InferenceSession(ckpt_path, providers=['CPUExecutionProvider'])
    input_name = classifier.get_inputs()[0].name
    return classifier, input_name


def read_image(img_file: str):
    img = Image.open(img_file)
    img = np.array(img)

    img = preprocessor(img, return_tensors='np')['pixel_values']

    return img


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)    