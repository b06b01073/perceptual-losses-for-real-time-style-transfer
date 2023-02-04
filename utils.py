import os
from PIL import Image

from torchvision import transforms
import torch
import numpy as np

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

def read_img(img_path, normalize=False):
    dir_path = os.path.dirname(__file__) 
    img_path = os.path.join(dir_path, img_path)

    img = Image.open(img_path).convert('RGB')

    preprocess = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]

    if normalize:
        preprocess.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))

    img = transforms.Compose(preprocess)(img)
    return img 

def gram_matrix(x, should_normalize=True):
    (b, c, h, w) = x.size()
    flattenned_x = x.view(b, c, w * h)
    flattenned_x_t = flattenned_x.transpose(1, 2)
    gram = flattenned_x.bmm(flattenned_x_t)

    if should_normalize:
        gram /= c * h * w
    return gram

def batch_norm(x):
    (b, c, h, w) = x.size()
    x /= c * h * w
    return x