import torch 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd 
import os


class CustomDataset(Dataset):
    
    def __init__(self, dataset_list, root_dir,transform=None, isTrain=True):

        self.original_csv=dataset_list
        self.root_dir=root_dir
        self.dataset_list=dataset_list
        self.transform=transform
        self.isTrain=isTrain
       

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx=idx.tolist()

        csv_content=self.original_csv.iloc[idx]

        #file_name
       
        file_name= csv_content['file_name']

        #digit
        digit= csv_content['digit']

        #letter
        letter= ord(csv_content['letter'])

        #letter->array로 변환 (tuple로 저장되는 문제가 있음)


        if self.isTrain:
        
            img_path=os.path.join(self.root_dir+'/'+str(digit)+'/', file_name )
        
        else:
            

            img_path=os.path.join(self.root_dir+'/', file_name )
            

        
        image=Image.open(img_path)

        image=image.convert('RGB')

        if self.transform is not None:
            image=self.transform(image)
    
        return image, digit, letter
        


    def __len__(self):
        return len(self.original_csv)
    