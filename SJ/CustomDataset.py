import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

class CustomDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):

        self.original_csv=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
       

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx=idx.tolist()

        csv_content=self.original_csv.iloc[idx]

        #file_name
       
        file_name= csv_content['file_name']

        #digit
        digit= csv_content['digit']

        #letter
        letter= csv_content['letter']

        img_path=os.path.join(self.root_dir+'/'+digit, file_name )
        
        image=io.imread(img_path)

        sample={'image':image, 'digit':digit, 'letter':letter}

        return sample


    def __len__(self):
            return len(self.original_csv)
    