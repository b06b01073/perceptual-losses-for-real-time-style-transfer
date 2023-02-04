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
    data_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    transform_net = TransformNet().to(device)
    loss_net = LossNet().to(device)

    optimizer = optim.Adam(transform_net.parameters(), lr=args.lr)

    style_img = utils.read_img(args.style_img).to(device).repeat((args.batch_size, 1, 1, 1))
    style_representation = loss_net(style_img)
    style_grams = [utils.gram_matrix(x) for x in style_representation] 

    for epoch in range(args.epochs):
        for index, batch in enumerate(data_loader):
            batch = batch.to(device)
            outputs = transform_net(batch)
            save_image(outputs, 'output.jpg')
            
            y = loss_net(outputs)
            content_representation = loss_net(batch)
            content_loss = torch.nn.MSELoss()(y.relu1_2, content_representation.relu1_2)

            
            style_loss = 0.0
            output_grams = [utils.gram_matrix(x) for x in y]
            for gram_gt, gram_hat in zip(style_grams, output_grams):
                style_loss += torch.nn.MSELoss()(gram_gt, gram_hat)

            style_loss /= args.batch_size



            optimizer.zero_grad()
            total_loss = content_loss + 5e5 * style_loss
            total_loss.backward()
            optimizer.step()

            print(f'epoch: {epoch}, batch: {index},loss: {total_loss.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--style_img', '-s', type=str, default='./style_images/mosaic.jpg')
    parser.add_argument('--data_path', '-d', type=str, default='./val2014')

    args = parser.parse_args()

    train(args)