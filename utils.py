import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
# Injects colour into a brush stroke by replacing all values below 0.9 with the given RGB value, while rounding up the
# other numbers to 1

t2p = transforms.ToPILImage()
p2t = transforms.ToTensor()

def revertColour(gimg, c1, c2, c3):
    C1 = gimg.copy()
    C2 = gimg.copy()
    C3 = gimg.copy()
    C1[C1 >= 0.9] = 1
    C2[C1 >= 0.9] = 1
    C3[C1 >= 0.9] = 1
    C1[C1 < 0.9] = c1
    C2[C1 < 0.9] = c2
    C3[C1 < 0.9] = c3
    return(cv2.merge([C3, C2, C1]))

def overlayLine(backImg, frontImg):
    alpha = frontImg.copy()[:, :,1]
    alpha[alpha < 1] = 0
    alpha1 = 1 - alpha
    newImg = backImg.copy()

    for c in range(0, 3):
        newImg[:,:, c] = ((alpha * backImg[:,:, c]) + (frontImg[:, :, c] * alpha1))
    return newImg

def applyStrokes(img, data, renderer):
    newImg = img.copy()
    strokes = renderer(data[:, :7])
    strokes = [revertColour(s.detach().numpy(), c[0], c[1], c[2]) for s, c in zip(strokes,data[:,:3].detach().numpy())]
    for s in strokes:
        newImg = overlayLine(newImg, s)
    return newImg

#Generate images with set of strokes
def generateImage(g, imgs, r, domains):
    gImgs = []
    strokes = g(imgs, domains)
    for set in zip(imgs, strokes):
        img = np.asarray(t2p(set[0]))
        img = applyStrokes(img, set[1], r)
        gImgs.append(p2t(Image.fromarray(img)))
    return torch.stack(gImgs)