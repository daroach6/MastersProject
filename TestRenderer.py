import torch
import Renderer
import matplotlib.pyplot as plt
from BrushStrokeDataset import BrushStrokeDataset
import cv2
from utils import revertColour

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = Renderer.FCN().to(device)

print("Loading existing Model:")
model.load_state_dict((torch.load("data/renderer.pt")))
model.eval()

data = BrushStrokeDataset()
#trainSize = int(0.7 * len(data))
#testSize = len(data) - trainSize
#train, data = torch.utils.data.random_split(data, [trainSize,testSize])
fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 3
for i in range(1,5):
    d = data[i]["data"]
    b, g, r = d[-3:].detach().numpy()
    img = model(d[:7]).detach().numpy()
    img = img.reshape(128,128)
    fig.add_subplot(rows,columns,(i*3)-2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.title("Grey Prediction")
    fig.add_subplot(rows, columns, (i * 3) - 1)
    img = revertColour(img,b,g,r)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Colour Prediction")
    fig.add_subplot(rows,columns,i*3)
    img = data[i]["image"].detach().numpy()
    img = revertColour(img,b,g,r)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Truth")
plt.show()