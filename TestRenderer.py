import torch
from torch import nn
from torch.utils.data import DataLoader

import Renderer
import matplotlib.pyplot as plt
from BrushStrokeDataset import BrushStrokeDataset
import cv2
from utils import revertColour


#Hyperparameters
learning_rate = 1e-4
epochs = 1
batchSize = 64

#
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = Renderer.FCN().to(device)

print("Loading existing Model:")
model.load_state_dict((torch.load("data/renderer.pt")))
model.eval()

data = BrushStrokeDataset()
trainSize = int(0.7 * len(data))
testSize = len(data) - trainSize
train, data = torch.utils.data.random_split(data, [trainSize,testSize])
testLoader = DataLoader(data, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
fig = plt.figure(figsize=(10, 7))
rows = 6
columns = 3
for i in range(1,rows +1):
    d = data[i]["data"]
    b, g, r = d[-3:].detach().numpy()
    img = model(d[:7]).detach().numpy()
    img = img.reshape(128,128)
    fig.add_subplot(rows,columns,(i*3)-2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    if i ==1 : plt.title("Grey Prediction")
    fig.add_subplot(rows, columns, (i * 3) - 1)
    img = revertColour(img,b,g,r)
    plt.imshow(img)
    plt.axis('off')
    if i ==1 : plt.title("Colour Prediction")
    fig.add_subplot(rows,columns,i*3)
    img = data[i]["image"].detach().numpy()
    img = revertColour(img,b,g,r)
    plt.imshow(img)
    plt.axis('off')
    if i ==1 : plt.title("Truth")
plt.show()

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0
    low_loss = 999
    high_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"][:,:7]
            y = batch["image"]
            pred = model(x)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            if loss > high_loss:
                high_loss = loss
            elif loss < low_loss:
                low_loss = loss
    test_loss /= num_batches
    print(f"Test Error: \n Average loss: {test_loss:>8f} \n Lowest loss: {low_loss:>8f} \n Highest loss: {high_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    test_loop(testLoader, model, loss_fn)
print("Done!")