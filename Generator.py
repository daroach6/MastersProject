import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,nin, nout, kSize):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(nin, nout, kSize, padding = 1)
        self.lRelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(nin, nout, kSize, padding=1)
        self.lRelu2 = nn.LeakyReLU(0.2)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.lRelu1(self.conv1(x))
        x = self.lRelu2(self.conv2(x))
        return self.pool(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.down1 = ConvBlock(3, 16)
        self.down2 = ConvBlock(16, 32)
        self.down3 = ConvBlock(32, 64)
        self.down4 = ConvBlock(64, 128)
        self.down5 = ConvBlock(128, 256)
    def forward(self, x):
        return 1