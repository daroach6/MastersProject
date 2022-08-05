import cv2

#
def revertColour(gimg, c1, c2, c3):
    C1 = gimg.copy()
    C2 = gimg.copy()
    C3 = gimg.copy()
    C1[C1 > 0.9] = 1
    C2[C1 > 0.9] = 1
    C3[C1 > 0.9] = 1
    C1[C1 < 0.9] = c1
    C2[C1 < 0.9] = c2
    C3[C1 < 0.9] = c3
    return(cv2.merge([C3, C2, C1]))