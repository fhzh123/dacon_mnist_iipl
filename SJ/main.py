import os 
import argparse
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings("ignore")
from CustomDataset import CustomDataset
from split_dataset import split_dataset_function
import torch 
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from skimage import io
from torch.utils.data import DataLoader
import torch.optim as optim


class CNNClassifier(nn.Module):

    #def __init__(self):
    #    super(CNNClassifier,self).__init__()
    #    self.conv1=nn.Conv2d(3,6,kernel_size=5)
    #    self.conv2=nn.Conv2d(6,16,kernel_size=5)
    #    self.fc1=nn.Linear(16,10)

    def __init__(self):
        super(CNNClassifier, self).__init__()
        conv1=nn.Conv2d(3,6,5,1)
        pool1=nn.MaxPool2d(2)
        conv2=nn.Conv2d(6,16,5,1)
        pool2=nn.MaxPool2d(2)


        self.conv_module=nn.Sequential(conv1, nn.ReLU(), pool1, conv2, nn.ReLU(), pool2)

        fc1=nn.Linear(16*4*4, 120)
        fc2=nn.Linear(120,84)
        fc3=nn.Linear(84,10)

        self.fc_module=nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU(), fc3)

    def forward(self,x):
        out=self.conv_module(x)
        dim=1
        for d in out.size()[1:]:
            dim=dim*d
        out=out.view(-1,dim)
        out=self.fc_module(out)
        return F.softmax(out,dim=1)

    #def forward(self,x):
    #    x=self.conv1(x)
    #    x=F.relu(x)
    #    x=F.max_pool2d(x,2)
    #    x=self.conv2(x)
    #    x=F.relu(x)
    #    x=F.max_pool2d(x,2)
    #    x=x.view(-1,16)
    #    x=self.fc1(x)
    #    output=F.log_softmax(x)
    #    return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (img, digit, letter) in enumerate(train_loader):
        
        #if torch.cuda.is_available:
            #img=img.cuda()
            #digit=digit.cuda()

        optimizer.zero_grad()
        output=model(img)
        loss=F.nll_loss(output, digit)
        loss.backward()
        optimizer.step()

    if batch_idx % args.batch_size==0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for img, digit in test_loader:
            if torch.cuda.is_available:
                img=img.cuda()
                digit=digit.cuda() 
            output=model(img)
            test_loss+=F.nll_loss(output, digit, reduction='sum').item() #sum up batch loss
            pred=output.argmax(dim=1, keepdim=True) # get the index of the max log-probability 
            correct+=pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


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

    train_data_list, validation_data_list=split_dataset_function(args.train_data_name)

    train_dataset=CustomDataset(train_data_list,args.data_path+'/train',data_transforms['train'])

    validation_dataset=CustomDataset(validation_data_list,args.data_path+'/train',data_transforms['valid'])

    
    train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader=DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    
    model=CNNClassifier()
    optimizer=optim.Adam(model.parameters() )
    
    #scheduler=StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.num_epochs+1):
        train(args,model,device,train_dataloader, optimizer, epoch)
        test(model, device, validation_dataloader)
        #scheduler.step()

    #if args.save_model:
        #torch.save()
    
#train모드 







if __name__=="__main__":

    parser=argparse.ArgumentParser(description='Order_net argparser')
    parser.add_argument('--train_data_name',type=str, default='train_dataset_list.csv', help='Data path setting')
    parser.add_argument('--data_path',type=str, default='./data', help='Data path setting')
    parser.add_argument('--save_path',type=str, default='./save')
    #parser.add_argument('--letter_model_path') #사용용도를 아직 잘 모르겠어서 우선 skip
    
    #Image Setting
    parser.add_argument('--resize_pixel', type=int, default=28, help='Resize pixel')
    parser.add_argument('--random_affine',type=int, default=10, help='Random affine transformation ratio')

    #Model Setting 
    #parser.add_argument("--efficientnet_not_use",default=False, action="store_true",help="Do not use Efficientnet")
    #parser.add_argument("--efficientnet_model_number", type=str, default=7, help='Efficient model number ')

    #Training Setting
    parser.add_argument('--num_epochs', type=int, default=200, help='The number of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    #parser.add_argument('--lr', type=float, default=le-2, help='Learning rate setting')
    #parser.add_argument('--lr_step_size', type=int, default=60, help='Learning rate scheduling step')
    #parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='Learning rate decay )
    #parser.add_argument('--weight_decay', type=float, default=le-4, help='Weight decay')
    #parser.add_argument('--max_grad_norm', type=int, default=5, help='Gradient clipping max norm')
    #parser.add_argument('--valid_ratio', type=float, default=0.1, help='Train/Valid split ratio')
    #parser.add_argument('--random_seed', type=int, default=42, help='Random state setting')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU worker setting')
    
    #입력받은 인자값을 args에 저장 (type: namespace)
    args=parser.parse_args()
    main(args)