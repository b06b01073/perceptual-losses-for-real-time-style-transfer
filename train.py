import os

from torch import optim
from torchvision.utils import save_image
from dataset import COCODataset
from torch.utils.data import DataLoader
import torch
import argparse

import utils
from model import TransformNet, LossNet

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def train(args):
    dataset = COCODataset(args.data_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    transform_net = TransformNet().to(device)
    loss_net = LossNet().to(device)

    optimizer = optim.Adam(transform_net.parameters(), lr=args.lr)

    style_img = utils.read_img(args.style_img).to(device)
    style_representation = loss_net(style_img)
    style_grams = [utils.gram_matrix(x) for x in style_representation] 

    for epoch in range(args.epochs):
        imgs = next(iter(data_loader)).to(device)
        outputs = transform_net(imgs)
        save_image(outputs, 'output.jpg')
        
        y = loss_net(outputs)
        content_representation = loss_net(imgs)
        content_loss = torch.nn.MSELoss()(y.relu1_2, content_representation.relu1_2)

        # style_loss = 0
        
        # for i in range(args.batch_size):
        #     y_grams = [utils.gram_matrix(x) for x in [y.relu1_2[i], y.relu2_2[i], y.relu3_3[i], y.relu4_3[i]]]
        #     for y_gram, s_gram in zip(y_grams, style_grams):
        #         style_loss += torch.nn.MSELoss()(y_gram, s_gram)

        # style_loss /= args.batch_size

        total_loss = content_loss
        print(f'epoch: {epoch}, loss: {total_loss.item()}')

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--style_img', '-s', type=str, default='./style_images/night.jpg')
    parser.add_argument('--data_path', '-d', type=str, default='./val2014')

    args = parser.parse_args()

    train(args)