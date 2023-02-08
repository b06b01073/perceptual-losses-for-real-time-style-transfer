import os
from PIL import Image

from torchvision import transforms
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

import torch
import numpy as np


def get_path(path):
    dir_path = os.path.dirname(__file__) 
    path = os.path.join(dir_path, path)
    return path

def read_img(img_path):
    img_path = get_path(img_path)
    img = Image.open(img_path).convert('RGB')

    preprocess = [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
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

def load_model(model, model_path):
    model_path = get_path(model_path)
    model.load_state_dict(torch.load(model_path))


def read_video(video_path):
    video_path = get_path(video_path)
    video = torchvision.io.read_video(video_path)
    return video[0], video[2]['video_fps']


def save_gif(output, gif_path, fps):
    gif_path = get_path(gif_path)
    frames = [transforms.ToPILImage()(frame) for frame in tqdm(output, desc='frames')]
    first_frame = frames[0]

    duration = len(frames) / fps
    first_frame.save(gif_path, format="GIF", append_images=frames, save_all=True, duration=duration,  loop=0)