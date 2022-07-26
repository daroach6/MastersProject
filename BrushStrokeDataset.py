import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class BrushStrokeDataset(Dataset):
    """Brush Strokes dataset."""

    def __init__(self, rootDir = "data/BrushStrokes/stroke",csvDir = "data/strokeParameters.CSV",transform=None, strokes = 60000):
        """
        Args:
                        root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rootDir = rootDir
        self.strokeParameter = np.loadtxt(csvDir, delimiter=",").astype('float32')[:strokes]
        self.transform = transform

    def __len__(self):
        return len(self.strokeParameter)

    def __getitem__(self, idx):
        image = cv2.imread(self.rootDir + str(idx) +".png").astype('float32')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255
        image = torch.from_numpy(image)
        #print(image.size())
        data = self.strokeParameter[idx]

        data = torch.from_numpy(data)
        sample = { 'data': data, 'image': image}

        if self.transform:
            sample = self.transform(sample)
        return sample