from torch import nn
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor 
from torchvision.utils import save_image
import utils

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.res_blocks = [ResBlock(in_channels=128, out_channels=128, kernel_size=3) for _ in range(5)]

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
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
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )


    def forward(self, x): 
        return self.net(x)


class LossNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = vgg16(weights='IMAGENET1K_V1').features
        self.style_extractor = create_feature_extractor(self.net, {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
        })

        self.content_extractor = create_feature_extractor(self.net, {
            '15': 'relu3_3',
        })


    def forward(self, x):
        return self.style_extractor(x), self.content_extractor(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=self.padding, stride=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=self.padding, stride=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        output = self.net(x)
        return output + x