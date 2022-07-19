import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = (nn.Linear(, 1536))
        self.fc2 = (nn.Linear(1536, 3072))
        self.fc3 = (nn.Linear(3072, 6144))
        self.fc4 = (nn.Linear(6144, 12288))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 64, 3, 1, 1))
        self.conv3 = (nn.Conv2d(64, 128, 3, 1, 1))
        self.conv4 = (nn.Conv2d(128, 64, 3, 1, 1))
        self.conv5 = (nn.Conv2d(64, 32, 3, 1, 1))
        self.conv6 = (nn.Conv2d(32, 16, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        #//todo bi-linear up sampling
    def forward(self, x, batchSize = 1):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 128,128,3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(batchSize,128, 128, 3)