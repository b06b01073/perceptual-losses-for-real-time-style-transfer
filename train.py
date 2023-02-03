import utils
from torchvision.utils import save_image
from dataset import COCODataset
from torch.utils.data import DataLoader
import torch

from model import TransformNet

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def train():
    dataset = COCODataset('./train2014')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    epochs = 10
    transform_net = TransformNet().to(device)

    for _ in range(epochs):
        imgs = next(iter(data_loader)).to(device)
        output = transform_net(imgs)
        print(output.shape)


if __name__ == '__main__':
    train()