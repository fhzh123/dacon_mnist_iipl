import os 
import argparse
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
from CustomDataset import CustomDataset
warnings.filterwarnings("ignore")
import torch 
from torchvision import transforms, models


class Net(nn.module):
    


def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms={
        'train': transforms.Compose([
                transforms.Resize((args.resize_pixel, args.resize_pixel)),
                transforms.RandomAffine(args.random_affine),
                transforms.ColorJitter(brightness=(0.5,2)),
                transforms.RandomResizedCrop((args.resize_pixel, args.resize_pixel), scale=(0.85,1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.RandomErasing(p=0.3, scale=(0.01, 0.05))

        ]),
        'valid':transforms.Compose([
            transforms.Resize((args.resize_pixel, args.resize_pixel)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

        ])
                 

    }
    whole_train_dataset=CustomDataset('train_dataset_list.csv',args.data_path+'/train',data_transforms)

    train_size=int(0.8*len(whole_train_dataset))
    validation_size=len(whole_train_dataset)-train_size

    train_set, val_set=torch.utils.data.random_split(whole_train_dataset, [train_size,validation_size])
    

    
#train모드 







if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Order_net argparser')
    parser.add_argument('--data_path',type=str, default='./data', help='Data path setting')
    parser.add_argument('--save_path',type=str, default='./save')
    #parser.add_argument('--letter_model_path') #사용용도를 아직 잘 모르겠어서 우선 skip
    
    #Image Setting
    parser.add_argument('--resize_pixel', type=int, default=360, help='Resize pixel')
    parser.add_argument('--random_affine',type=int, default=10, help='Random affine transformation ratio')

    #Model Setting 
    #parser.add_argument("--efficientnet_not_use",default=False, action="store_true",help="Do not use Efficientnet")
    #parser.add_argument("--efficientnet_model_number", type=str, default=7, help='Efficient model number ')

    #Training Setting
    #parser.add_argument('--num_epochs', type=int, default=300, help='The number of epoch')
    #parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    #parser.add_argument('--lr', type=float, default=le-2, help='Learning rate setting')
    #parser.add_argument('--lr_step_size', type=int, default=60, help='Learning rate scheduling step')
    #parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='Learning rate decay )
    #parser.add_argument('--weight_decay', type=float, default=le-4, help='Weight decay')
    #parser.add_argument('--max_grad_norm', type=int, default=5, help='Gradient clipping max norm')
    #parser.add_argument('--valid_ratio', type=float, default=0.1, help='Train/Valid split ratio')
    #parser.add_argument('--random_seed', type=int, default=42, help='Random state setting')
    #parser.add_argument('--num_workers', type=int, default=8, help='CPU worker setting')
    
    #입력받은 인자값을 args에 저장 (type: namespace)
    args=parser.parse_args()
    main(args)
    

