import torch
import Renderer
from torch import nn
from torch.utils.data import Dataset, DataLoader
from BrushStrokeDataset import BrushStrokeDataset

#Hyperparameters
learning_rate = 1e-4
epochs = 20
batchSize = 64
strokes = 100000

#
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batchNo, batch in enumerate(dataloader):
        # Compute prediction and loss
        x = batch["data"][:,:7]
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
            x = batch["data"][:,:7]
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
model = Renderer.FCN().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#
print("Loading existing Model:")
model.load_state_dict((torch.load("data/renderer.pt")))
model.eval()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainLoader, model, loss_fn, optimizer)
    test_loop(testLoader, model, loss_fn)
    print("Saving current model:")
    torch.save(model.state_dict(), "data/renderer.pt")
print("Done!")