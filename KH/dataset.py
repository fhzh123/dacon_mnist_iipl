import os
import time
import copy
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_path, data_path, transform=None, resize_pixel=32):
        self.image_path = image_path
        self.data_path = data_path

        data = pd.read_csv(data_path)
        self.label = list(data['label'])
        self.file = list(data['file'])

        self.num_data = len(data)
        self.transform = transform
        self.resize_pixel = resize_pixel

    def __getitem__(self, index):
        label = self.label[index]
        file = self.file[index]
        image_path = os.path.join(self.image_path, label)
        image = Image.open(os.path.join(image_path, file))
        image = image.resize((self.resize_pixel, self.resize_pixel), resample = Image.BILINEAR)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.num_data

## 미완