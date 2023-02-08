import os

import argparse
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import utils
from model import TransformNet_V2
from dataset import COCODataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    dataset = COCODataset(args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    transform_net = TransformNet_V2().to(device)
    utils.load_model(transform_net, args.model_path)
    transform_net.eval()

    batch = next(iter(data_loader)).to(device)

    with torch.no_grad():    
        output = transform_net(batch)

    save_image(torch.concat((batch, output)), args.output)


    

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--output', type=str, default='./output/images/result.jpg')
    parser.add_argument('--model_path', type=str, default='./model_params/night/epoch3.pth')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--data_path', '-d', type=str, default='./val2014')

    args = parser.parse_args()

    main(args)