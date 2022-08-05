import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,nin, nout, kernal = 3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(nin,nout, kernel_size=kernal, padding=1)
        self.lrelu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(nout,nout, kernel_size=kernal, padding=1)
        self.lrelu2 = nn.LeakyReLU()
        self.pool =nn.AvgPool2d(2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return self.pool(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ConvBlock(16,32)
        self.block3 = ConvBlock(32,64)
        self.block4 = ConvBlock(64,128)

    def forward(self, canvas):
        x = self.block1(canvas)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(-1, 8192)

class agentRNN(nn.Module):
    def __init__(self):
        super(agentRNN, self).__init__()
        #
        self.targetCNN = CNN()
        self.drawCNN = CNN()
        #
        self.fc1 = nn.Linear(16384, 1024)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 256)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.fcBrush = nn.Linear(256, 4)
        self.fcCoords = nn.Linear(256, 6)

    def forward(self, target, canvas):

        target = self.targetCNN(target)
        x = self.drawCNN(canvas)
        x = torch.cat((target, x), 1)
        x = self.fc1(x)
        x = self.lrelu1(x)
        x = self.fc2(x)
        x = self.lrelu2(x)
        brush = torch.sigmoid(self.fcBrush(x))
        coords = torch.tanh(self.fcCoords(x))
        return torch.cat((coords,brush),1)