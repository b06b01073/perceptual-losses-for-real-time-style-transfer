from torch import nn
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor 
from torchvision.utils import save_image
import utils
from collections import namedtuple

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_num = 5
        self.res_blocks = [ResBlock(in_channels=128, out_channels=128, kernel_size=3) for _ in range(self.block_num)]

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            self.res_blocks[0],
            nn.ReLU(),

            self.res_blocks[1],
            nn.ReLU(),

            self.res_blocks[2],
            nn.ReLU(),

            self.res_blocks[3],
            nn.ReLU(),

            self.res_blocks[4],
            nn.ReLU(),
            
            UpsampleLayer(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            UpsampleLayer(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4, padding_mode='reflect'),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )


    def forward(self, x): 
        
        return self.net(x)


class LossNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = vgg16(weights='IMAGENET1K_V1').features

        for param in self.parameters():
            param.requires_grad = False

        self.extractor = create_feature_extractor(self.net, {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
        })

    def forward(self, x):
        x = self.extractor(x)
        output = namedtuple('Extractor', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        loss_output = output(x['relu1_2'], x['relu2_2'], x['relu3_3'], x['relu4_3'])

        
        return loss_output

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels=128):
        super().__init__()

        self.padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        self.net = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=hidden_channels, padding=self.padding, stride=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=hidden_channels, out_channels=out_channels, padding=self.padding, stride=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        output = self.net(x)
        return output + x
    
class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        )

    def forward(self, x):
        return self.net(x)
