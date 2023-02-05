import os

import argparse
import torch
from torchvision.utils import save_image

import utils
from model import TransformNet_V2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    transform_net = TransformNet_V2().to(device)
    utils.load_model(transform_net, args.model_path)
    transform_net.eval()

    img = utils.read_img(args.input).to(device)

    output = transform_net(img)
    save_image(output, args.output)


    

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input', type=str, default='./input/input.jpg')
    parser.add_argument('--output', type=str, default='./output/result.jpg')

    parser.add_argument('--model_path', type=str, default='./model_params/night/epoch3.pth')
    args = parser.parse_args()


    main(args)