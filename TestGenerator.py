import torch
import torch.nn.functional as f
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import Discriminator
import Generator
import Renderer
from utils import generateImage

#Hyperparamerters
learning_rate = 1e-4
epochs = 1
batchSize = 64

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

#Train the discriminator to be able to classify images and determine if an image is fake or not
def testDiscriminator(d, g, r, imgs, labels, target):
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
def testGenerator(d, g, r, imgs, labels, target):
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
def testLoop(loader, d, g, r):
    size = len(loader.dataset)
    num_batches = len(dataloader)
    dLoss = 0
    dLossLow = 999
    dLossHigh = -999
    gLoss = 0
    gLossLow = 999
    gLossHigh = -999
    for batchNo, batch in enumerate(tqdm(loader)):
        #
        imgs = batch[0]
        labels = batch[1][..., 31].view(-1, 1)
        target = torch.sub(1, labels)

        #
        loss = testDiscriminator(d, g, r, imgs, labels, target)
        dLoss += loss
        if loss > dLossHigh:
            dLossHigh = loss
        elif loss < dLossLow:
            dLossLow = loss
        loss = testGenerator(d, g, r, imgs, labels, target)
        gLoss += loss
        if loss > gLossHigh:
            gLossHigh = loss
        elif loss < gLossLow:
            gLossLow = loss

    gLoss /= num_batches
    dLoss /= num_batches
    print(f"Test Error: \n"
          f" Generator:"
          f"  Average generator loss: {gLoss:>8f} \n"
          f"  Lowest generator loss: {gLossLow:>8f} \n"
          f"  Highest generator loss: {gLossHigh:>8f} \n"
          f" Discriminator:"
          f"  Average discriminator loss: {dLoss:>8f} \n"
          f"  Lowest discriminator loss: {dLossLow:>8f} \n"
          f"  Highest discriminator loss: {dLossHigh:>8f} \n")

#Initiliasing the neural networks
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")
generator = Generator.generatorNN().to(device)
discriminator = Discriminator.CNN().to(device)
renderer = Renderer.FCN().to(device)
genOptimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
disOptimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loads existing weights to our models
print("Loading existing Models:\n")
renderer.eval()
renderer.load_state_dict((torch.load("data/renderer.pt")))
discriminator.eval()
discriminator.load_state_dict((torch.load("data/discriminator.pt")))
generator.eval()
generator.load_state_dict((torch.load("data/generator.pt")))

# Transformations for the images in the dataset and
transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
t2p = transforms.ToPILImage()
p2t = transforms.ToTensor()

#Preparing the data for testing with getting a small smaple smiling and nonSmiling faces.
print("Preparing data:\n")
data = torchvision.datasets.CelebA(root = "data", transform = transform, split="test")
dataloader = DataLoader(data, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)
smiling = []
notSmiling = []
for i in range(100):
    d = data[i]
    if d[1][31] == 1:
        smiling.append(d[0])
    else:
        notSmiling.append(d[0])

fig = plt.figure(figsize=(10, 7))
rows = 4
columns = 4
for i in range(1,5):

    img = smiling[i]
    fig.add_subplot(rows, columns, (i*columns) - 3)
    plt.imshow(t2p(img))
    plt.axis('off')
    if i ==1 : plt.title("Smilling")

    fig.add_subplot(rows, columns, (i * columns) - 2)
    img = img.view(-1,3,128,128)
    img = generateImage(generator, img, renderer, torch.tensor([[0]])).view(3,128,128)
    plt.imshow(t2p(img))
    plt.axis('off')
    if i ==1 : plt.title("Smilling to Not Smilling")

    img = notSmiling[i]
    fig.add_subplot(rows, columns, i*columns - 1)
    plt.imshow(t2p(img))
    plt.axis('off')
    if i ==1 :plt.title("Not Smilling")

    fig.add_subplot(rows, columns, (i * columns))
    img = img.view(-1, 3, 128, 128)
    img = generateImage(generator, img, renderer, torch.tensor([[1]])).view(3, 128, 128)
    plt.imshow(t2p(img))
    plt.axis('off')
    if i ==1 : plt.title("Not Smilling to Smilling")
plt.show()

testLoop(dataloader, discriminator, generator, renderer)

