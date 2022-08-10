import cv2

# Injects colour into a brush stroke by replacing all values below 0.9 with the given RGB value, while rounding up the
# other numbers to 1
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
    alpha = frontImg.copy()[:, :, 1]
    alpha[alpha < 1] = 0
    print(alpha)
    alpha1 = 1 - alpha
    newImg = backImg.copy()

    for c in range(0, 3):
        newImg[:,:, c] = ((alpha * backImg[:,:, c]) + (frontImg[:, :, c] * alpha1))
    return newImg