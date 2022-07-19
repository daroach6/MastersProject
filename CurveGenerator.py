import cv2
import numpy as np
import random
from Bezier import Bezier

x = 128
y = 128

def randPt(i):
    return random.randint(0, i)

img = np.zeros((x, y, 3), dtype=np.uint8)
parameters = []
print(parameters)
for i in range(60000):
    img[:, :] = (255, 255, 255)
    pos = np.array([(randPt(x), randPt(y)) for i in range(3)])
    pts = Bezier.Curve(np.arange(0, 1, 0.01), pos)
    pts = pts.reshape((-1, 1, 2))

    color = [random.randint(0,255) for i in range(3)]
    thickness = random.randint(1, 100)
    print()
    parameters.append(np.concatenate((pos.reshape((6)),[thickness],np.array(color))).tolist())

    cv2.polylines(img, np.int32([pts]), False,color, thickness)
    cv2.imshow('image', img)
    cv2.waitKey(1)
    cv2.imwrite("data/BrushStrokes/stroke"+ str(i) +".png", img)
print(parameters)

np.savetxt("data/strokeParameters.CSV",np.array(parameters),delimiter=",")
cv2.waitKey(0)