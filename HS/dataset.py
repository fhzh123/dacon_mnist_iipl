from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, isTrain=True, transform=None):
        self.data = data
        self.isTrain = isTrain
        self.transform = transform

    def __getitem__(self, index):
        if self.isTrain:
            img = np.asarray(self.data.iloc[index][3:])\
                    .reshape(28,28).astype('uint8')
            img = Image.fromarray(img)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                
            label = str(self.data.iloc[index]['digit'])
            letter = pd.get_dummies(self.data['letter'])
            letter = np.asarray(letter)[index]
            
            return img, letter, label
        else:
            img = np.asarray(self.data.iloc[index][2:])\
                    .reshape(28,28).astype('uint8')
            img = Image.fromarray(img)
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                
            id = str(self.data.iloc[index]['id'])
            letter = pd.get_dummies(self.data['letter'])
            letter = np.asarray(letter)[index]
            
            return img, letter, id
        
    def __len__(self):
        return len(self.data.index)

def preprocess(train_dir, test_dir, split):
    train = pd.read_csv(train_dir)
    val = train.sample(frac=split)
    train = train.drop(val.index)
    test = pd.read_csv(test_dir)
    
    preprocess = {'train': train,
                 'val': val,
                 'test': test}
    return preprocess