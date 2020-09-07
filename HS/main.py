import argparse
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CustomDataset, preprocess
from train_teacher import train_teacher
from train_distiller import train_distiller
from model import Teacher, Student
from utils import test_model

parser = argparse.ArgumentParser(description='Order_net argparser')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--submit_dir', type=str, default='./submission.csv')
parser.add_argument('--split_ratio', type=float, default=.1, help='train-val data ratio')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

transform = transforms.Compose([
                    transforms.Resize((70,70)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5])
                    ])

preprocess = preprocess(train_dir=args.data_dir+'/train.csv',
                       test_dir=args.data_dir+'/test.csv',
                       split = args.split_ratio)
    
data = {'train': CustomDataset(preprocess['train'],
                               isTrain=True,
                               transform=transform),
        'val': CustomDataset(preprocess['val'],
                             isTrain=True,
                             transform=transform),
        'test': CustomDataset(preprocess['test'],
                              isTrain=False,
                              transform=transform)}
    
iter = {'train': DataLoader(data['train'],
                            batch_size = args.batch,
                            shuffle = True),
        'val': DataLoader(data['val'],
                          batch_size = args.batch,
                          shuffle = True),
        'test': DataLoader(data['test'],
                           batch_size = 1,
                           shuffle = False)}

teacher = train_teacher(epochs = args.epoch,
                        model = Teacher(),
                        iter = iter,
                        data = data,
                        step_size = args.step_size,
                        gamma = args.gamma,
                        lr = args.lr)

distiller = train_distiller(epochs = args.epoch,
                            model = Student(),
                            iter = iter,
                            data = data,
                            teacher = teacher,
                            step_size = args.step_size,
                            gamma = args.gamma,
                            lr= args.lr)

y_pred = test_model(model = distiller,
                   iter = iter)
submission = pd.read_csv(args.data_dir+'/submission.csv')
submission['digit'] = y_pred
submission.to_csv(args.submit_dir,
                  index = False)