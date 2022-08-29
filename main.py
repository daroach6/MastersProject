import torch as torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchviz import make_dot

import Renderer
import Generator
import Discriminator
from BrushStrokeDataset import BrushStrokeDataset
import netron

#Preparing the required modules to train the models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
generator = Generator.generatorNN().to(device)
discriminator = Discriminator.CNN().to(device)
renderer = Renderer.FCN().to(device)

d1 = BrushStrokeDataset()[0]["data"][:7]
# Transformations for the images in the dataset and
transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.ToTensor()
])
d2 = torchvision.datasets.CelebA(root = "data", transform = transform, split="test")[0][0]
d3 = torch.tensor([[1]])


r1 = renderer(d1)
r2 = discriminator(d2)
r3 = generator(d2,d3)

#torch.onnx.export(renderer, d1, 'test.onnx', input_names="Parameters", output_names="Image", verbose=True)
make_dot(r1, params=dict(list(renderer.named_parameters()))).render("C:/Users/Tom Jones/Documents/Masters/CodeBase/data/Results/Renderer", format="png")
make_dot(r2, params=dict(list(discriminator.named_parameters()))).render("C:/Users/Tom Jones/Documents/Masters/CodeBase/data/Results/Discriminator", format="png")
make_dot(r3, params=dict(list(generator.named_parameters()))).render("C:/Users/Tom Jones/Documents/Masters/CodeBase/data/Results/Generator", format="png")
