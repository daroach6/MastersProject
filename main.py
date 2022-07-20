import torch
import renderer
import matplotlib.pyplot as plt
from BrushStrokeDataset import BrushStrokeDataset
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = renderer.FCN().to(device)

print("Loading existing Model:")
model.load_state_dict((torch.load("data/renderer.pt")))
model.eval()

data = BrushStrokeDataset()
fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 2
for i in range(1,5):
    img = model(data[i]["data"]).detach().numpy()
    img = img.reshape(128,128,3)
    fig.add_subplot(rows,columns,(i*2)-1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Prediction")
    fig.add_subplot(rows,columns,i*2)
    plt.imshow(data[i]["image"].detach().numpy())
    plt.axis('off')
    plt.title("Truth")
plt.show()
