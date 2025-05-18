from torch import nn
import torch
import torch.nn.functional as F
from src.utils import InputNormalize


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, output_size=32, transnorm=None):
        super(ResNet, self).__init__()
        self.transnorm = transnorm
        if self.transnorm is not None:
            data_mean = self.transnorm['mean']
            data_std =  self.transnorm['std']
            self.InputNormalize = InputNormalize(data_mean, data_std)
        
        self.inchannel = 64
        
        self.output_size = output_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        # self.pool = nn.AdaptiveAvgPool2d((1, 1)) # for cifar10 dataset artitecture
        # self.fc = nn.Linear(512, output_size) # for cifar10 dataset artitecture

        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # for cococ dataset artitecture
        self.fc = nn.Linear(8192, output_size) # for coco dataset artitecture


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.transnorm is not None:
            x = self.InputNormalize(x)
    
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = out.reshape(out.size(0), -1)

        out = self.fc(out)
        return out
