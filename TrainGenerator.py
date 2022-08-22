import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import Discriminator
import Generator
import Renderer

#Hyperparameters
import utils

learning_rate = 1e-4
epochs = 20
batchSize = 64
t2p = transforms.ToPILImage()
p2t = transforms.ToTensor()

#Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
def gradientPenalty(y, x, device):

    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

#Generate images with set of strokes
def generateImage(g, imgs, r, domains):
    gImgs = []
    strokes = g(imgs, domains)
    for set in zip(imgs, strokes):
        img = np.asarray(t2p(set[0]))
        img = utils.applyStrokes(img, set[1], r)
        gImgs.append(p2t(Image.fromarray(img)))
    return torch.stack(gImgs)

#Train the discriminator to be able to classify images and determine if an image is fake or not
def trainDiscriminator(d, g, r, imgs, labels, target):
    #Train for real images
    src, cls = d(imgs)
    lossReal = - torch.mean(src)
    lossCls = f.binary_cross_entropy_with_logits(cls, labels.float(), reduction='sum') / cls.size(0)

    #Train for Fake images
    gImgs = generateImage(g, imgs, r, target)
    src, cls = d(gImgs)
    lossFake = torch.mean(src)

    #Compute loss with gradient penalty
    alpha = torch.rand(imgs.size(0), 1, 1, 1)
    x_hat = (alpha * imgs.data + (1 - alpha) * gImgs.data).requires_grad_(True)
    src, _ = d(x_hat)
    losGradPen = gradientPenalty(src, x_hat, device)

    #Performing back propagation for the discriminator
    loss = lossReal + lossFake + 1 * lossCls + 10 * losGradPen
    genOptimizer.zero_grad()
    disOptimizer.zero_grad()
    loss.backward()
    disOptimizer.step()

    return loss

# Train the generator to be able to generate brush stroke parameters that are able to change the image to
# new domain and fool the discriminator
def trainGenerator(d, g, r, imgs, labels, target):
    #Convert Image to target domain
    gImgs = generateImage(g, imgs, r, target)
    src, cls = d(gImgs)
    fakeLoss = - torch.mean(src)
    lossCls = f.binary_cross_entropy_with_logits(cls, labels.float(), reduction='sum') / cls.size(0)

    #Revert image to original domain
    revImg = generateImage(g, imgs, r, labels)
    lossRev = torch.mean(torch.abs(imgs - revImg))  # L1 loss

    #Performing back propagation for the generator
    loss = fakeLoss + 10 * lossRev + 1 * lossCls
    genOptimizer.zero_grad()
    disOptimizer.zero_grad()
    loss.backward()
    genOptimizer.step()

    return loss

# training loop for both the discriminator & generator and prints out the loss for every 10th batch of training.
def trainLoop(loader, d, g, r):
    size = len(loader.dataset)
    for batchNo, batch in enumerate(loader):
        #
        imgs = batch[0]
        labels = batch[1][..., 31].view(-1, 1)
        target = torch.sub(1, labels)

        #
        dLoss = trainDiscriminator(d, g, r, imgs, labels, target)
        gLoss = trainGenerator(d, g, r, imgs, labels, target)
        if batchNo % 10 == 0:
            dLoss, current = dLoss.item(), batchNo * len(imgs)
            gLoss, current = gLoss.item(), batchNo * len(imgs)
            print(f"discriminator loss: {dLoss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"generator loss: {gLoss:>7f}  [{current:>5d}/{size:>5d}]")

#
transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

#supplying images into dataloaders for training
images = torchvision.datasets.CelebA(root = "data", transform = transform, split="train")
loader = DataLoader(images, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)

#Preparing the required modules to train the models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
generator = Generator.generatorNN().to(device)
discriminator = Discriminator.CNN().to(device)
renderer = Renderer.FCN().to(device)
genOptimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
disOptimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loads existing weights to our models
print("Loading existing Models:")
renderer.eval()
renderer.load_state_dict((torch.load("data/renderer.pt")))
discriminator.eval()
discriminator.load_state_dict((torch.load("data/discriminator.pt")))
generator.eval()
generator.load_state_dict((torch.load("data/generator.pt")))

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainLoop(loader, discriminator, generator, renderer)
    print("Saving current models:")
    torch.save(generator.state_dict(), "data/generator.pt")
    torch.save(discriminator.state_dict(), "data/discriminator.pt")
print("Done!")