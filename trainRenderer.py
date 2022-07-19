import cv2
import numpy as np
import torch
import renderer
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

#Hyperparameters
learning_rate = 1e-6
epochs = 5
batchSize = 4
strokes = 10000
#
class BrushStrokeDataset(Dataset):
    """Brush Strokes dataset."""

    def __init__(self, rootDir = "data/BrushStrokes/stroke",csvDir = "data/strokeParameters.CSV",transform=None, strokes = 60000):
        """
        Args:
                        root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rootDir = rootDir
        self.strokeParameter = np.loadtxt(csvDir, delimiter=",").astype('float32')[:strokes]
        self.transform = transform

    def __len__(self):
        return len(self.strokeParameter)

    def __getitem__(self, idx):
        image = cv2.imread(self.rootDir + str(idx) +".png").astype('float32')
        image = torch.from_numpy(image)
        data = self.strokeParameter[idx]
        data = torch.from_numpy(data)
        sample = { 'data': data, 'image': image}

        if self.transform:
            sample = self.transform(sample)
        return sample

#
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batchNo, batch in enumerate(dataloader):
        # Compute prediction and loss
        x = batch["data"]
        y = batch["image"]
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchNo % 100 == 0:
            loss, current = loss.item(), batchNo * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"]
            y = batch["image"]
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
#
data = BrushStrokeDataset(strokes=strokes)
trainSize = int(0.7 * len(data))
testSize = len(data) - trainSize
train, test = torch.utils.data.random_split(data, [trainSize,testSize])
trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)
testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)
#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = renderer.FCN().to(device)
#
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainLoader, model, loss_fn, optimizer)
    test_loop(testLoader, model, loss_fn)
print("Done!")

img = model(train[1]["data"]).detach().numpy()
img = img.reshape(128,128,3)
#img = np.multiply(img,255)
print(img)
plt.imshow(img)
plt.show()
