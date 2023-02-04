import os
from PIL import Image

from torchvision import transforms
import torch

def read_img(img_path):
    dir_path = os.path.dirname(__file__) 
    img_path = os.path.join(dir_path, img_path)

    img = Image.open(img_path).convert('RGB')
    img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(img)

    
    return img 

def gram_matrix(x, should_normalize=True):
    (b, c, h, w) = x.size()
    flattenned_x = x.view(b, c, w * h)
    flattenned_x_t = flattenned_x.transpose(1, 2)
    gram = flattenned_x.bmm(flattenned_x_t)

    if should_normalize:
        gram /= c * h * w
    return gram