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

trans = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainset = torchvision.datasets.ImageFolder(root='../data/train/', transform=trans)
trainloader = DataLoader(trainset, batch_size)