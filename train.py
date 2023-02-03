import utils
from torchvision.utils import save_image
from dataset import COCODataset
from torch.utils.data import DataLoader

def train():
    dataset = COCODataset('./train2014')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    imgs = next(iter(data_loader))
    save_image(imgs[0], 'test.jpg')


if __name__ == '__main__':
    train()