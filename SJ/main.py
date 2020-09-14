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


#class CNNClassifier(nn.Module):

    #def __init__(self):
    #    super(CNNClassifier,self).__init__()
    #    self.conv1=nn.Conv2d(3,6,kernel_size=5)
    #    self.conv2=nn.Conv2d(6,16,kernel_size=5)
    #    self.fc1=nn.Linear(16,10)

    #def __init__(self):
    #    super(CNNClassifier, self).__init__()
    #    self.conv1=nn.Conv2d(3,6,5,1)
    #    self.conv2=nn.Conv2d(6,10,5,1)
    #    self.conv3=nn.Conv2d(10,16,3,1)
    #    self.fc1=nn.Linear(16,10)
    #    self.drop_out=nn.Dropout(p=0.1)
      

    #def forward(self,x):
    #    x=F.relu(F.max_pool2d(self.conv1(x),2)) 
        #x=self.drop_out(x)
    #    x=F.relu(F.max_pool2d(self.conv2(x),2))
        #x=self.drop_out(x)
    #    x=F.relu(F.max_pool2d(self.conv3(x),2))
    #    x=self.drop_out(x)

    #   x=x.view(-1,16)
    #    x=self.fc1(x)
    #   return F.log_softmax(x,dim=1)

train_Accuracy_array=[]
test_Accuracy_array=[]




def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss_for_average=0
    correct=0.0
    for batch_idx, (img, digit, letter) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            img=img.cuda()
            digit=digit.cuda()
        optimizer.zero_grad()
        output=model(img)
        loss=criterion(output, digit)
        loss.backward()
        optimizer.step()
        train_pred=output.argmax(dim=1,keepdim=True)
        correct+=train_pred.eq(digit.view_as(train_pred)).sum().item()
    
    accuracy=100.*correct/len(train_loader.dataset)
    train_loss_for_average+=loss/len(train_loader)
    
    print('\nTrain Epoch:{} , Accuracy: {:.0f}%\n'.format(epoch, accuracy))
    print('\nTrain Epoch: {}, Loss: {:.6f}\n'.format(epoch, train_loss_for_average ))
    return train_loss_for_average

#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_idx * len(img), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))         
#print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
    
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for (img, digit,letter) in test_loader:
            if torch.cuda.is_available():
                img=img.cuda()
                digit=digit.cuda()
            output=model(img)
            test_loss+=criterion(output, digit) #sum up batch loss
            pred=output.argmax(dim=1, keepdim=True) # get the index of the max log-probability 
            correct+=pred.eq(digit.view_as(pred)).sum().item()

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

    criterion=nn.CrossEntropyLoss()

    train_data_list, validation_data_list=split_dataset_function(args.train_data_name)

    train_dataset=CustomDataset(train_data_list,args.data_path+'/train',data_transforms['train'])

    validation_dataset=CustomDataset(validation_data_list,args.data_path+'/train',data_transforms['valid'])

    
    train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader=DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    
    model=models.mobilenet_v2(pretrained=False,num_classes=10).cuda()

    optimizer=optim.Adam(model.parameters() )
    total_train_loss=0
    #scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    for epoch in range(1, args.num_epochs+1):
        #scheduler.step()
        total_train_loss+=train(args,model,device,train_dataloader, optimizer, epoch, criterion)
        test(model, device, validation_dataloader,criterion)
    
    print('\n Average Train Loss: {:.6f}\n'.format(total_train_loss/args.num_epochs))
    
    #test(model, device, validation_dataloader,criterion)
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
    parser.add_argument('--num_epochs', type=int, default=300, help='The number of epoch')
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