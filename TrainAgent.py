import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import Agent
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from tqdm import tqdm

#Hyperparameters
learning_rate = 1e-4
epochs = 20
batchSize = 64

#
def train_loop(dataloader1, dataloader2,model, loss_fn, optimizer):
    size = len(dataloader1.dataset)
    for batchNo, batch in enumerate(dataloader1):
        # Compute prediction and loss
        x = batch
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
            x = batch["image"]
            y = batch["data"]
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

#
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#supplying images into dataloaders for training
images = torchvision.datasets.CelebA(root = "data", transform = transform, download=True)
print()
smiling = []
notSmiling = []
for img in tqdm(images):
    if img[1][31] == 1:
        smiling.append(img[0])
    else:
        notSmiling.append(img[0])

loader1 = DataLoader(smiling)
loader0 = DataLoader(notSmiling)

#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = Agent.agentRNN().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




'''print("Loading existing Model:")

#model.load_state_dict((torch.load("data/agent.pt")))
#model.eval()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(trainLoader, model, loss_fn, optimizer)
    test_loop(testLoader, model, loss_fn)
    print("Saving current model:")
    torch.save(model.state_dict(), "data/agent.pt")
print("Done!")'''