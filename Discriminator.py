import torch.nn as nn
import numpy as np


#image_size=128, conv_dim=64, c_dim=5, repeat_num=6


class CNN(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1)
        self.lrelu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lrelu3 = nn.LeakyReLU(0.01)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.lrelu4 = nn.LeakyReLU(0.01)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.lrelu5 = nn.LeakyReLU(0.01)
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1)
        self.lrelu6 = nn.LeakyReLU(0.01)
        self.out1 = nn.Conv2d(2048, 1, kernel_size = 3, padding = 1, bias=False)
        self.out2 = nn.Conv2d(2048, 1, kernel_size = int(128/64), bias=False)

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.lrelu4(self.conv4(x))
        x = self.lrelu5(self.conv5(x))
        x = self.lrelu6(self.conv6(x))

        outSrc = self.out1(x)
        outCls = self.out2(x)
        return outSrc, outCls.view(outCls.size(0), outCls.size(1))