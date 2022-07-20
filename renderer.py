import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(10, 192))
        self.fc2 = (nn.Linear(192, 384))
        self.fc3 = (nn.Linear(384, 768))
        self.conv1 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv2 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(32, 64, 3, 1, 1))
        self.biUp = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 16,16,3)
        x = F.relu(self.conv1(x))
        x = self.biUp(x)
        x = F.relu(self.conv2(x))
        x = self.biUp(x)
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(x)
        return x.view(-1,128, 128, 3)