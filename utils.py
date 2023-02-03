import os
from PIL import Image

from torchvision import transforms

def read_img(img_path):
    dir_path = os.path.dirname(__file__) 
    img_path = os.path.join(dir_path, img_path)

    img = Image.open(img_path)
    img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(img)

    
    return img 