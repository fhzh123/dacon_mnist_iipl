import argparse
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CustomDataset, preprocess, transform
from train_teacher import train_teacher
from train_distiller import train_distiller
from model import Teacher, Student
from utils import submission

parser = argparse.ArgumentParser(description='Order_net argparser')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--submit_dir', type=str, default='./submission.csv')
parser.add_argument('--split_ratio', type=float, default=.1, help='train-val data ratio')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--teacher_epoch', type=int, default=100)
parser.add_argument('--student_epoch', type=int, default=100)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--resize_pixel', type=int, default=100, help='Resize pixel')
parser.add_argument('--random_affine', type=int, default=10, help='Random affine transformation ratio')
parser.add_argument('--temp', type=int, default=10, help='Temperature for knowledge distillation')
parser.add_argument('--alpha', type=float, default=.1, help='Alpha for knowledge distillation')
args = parser.parse_args()

preprocess = preprocess(train_dir=args.data_dir+'/train.csv',
                       test_dir=args.data_dir+'/test.csv',
                       split=args.split_ratio)

transform = transform(pixel=args.resize_pixel,
                      affine=args.random_affine)
    
data = {'train': CustomDataset(preprocess['train'],
                               isTrain=True,
                               transform=transform['train']),
        'val': CustomDataset(preprocess['val'],
                             isTrain=True,
                             transform=transform['val']),
        'test': CustomDataset(preprocess['test'],
                              isTrain=False,
                              transform=transform['test'])}
    
iter = {'train': DataLoader(data['train'],
                            batch_size=args.batch,
                            shuffle=True),
        'val': DataLoader(data['val'],
                          batch_size=args.batch,
                          shuffle=True),
        'test': DataLoader(data['test'],
                           batch_size=1,
                           shuffle=False)}

teacher = train_teacher(epochs=args.teacher_epoch,
                        model=Teacher(),
                        iter=iter,
                        data=data,
                        step_size=args.step_size,
                        gamma=args.gamma,
                        lr=args.lr)

distiller = train_distiller(epochs=args.student_epoch,
                            model=Student(),
                            iter=iter,
                            data=data,
                            teacher=teacher,
                            step_size=args.step_size,
                            gamma=args.gamma,
                            lr=args.lr,
                            T=args.temp,
                            alpha=args.alpha)

y_preds = submission(model=distiller, iter=iter)
submission = pd.read_csv(args.data_dir+'/submission.csv')
submission['digit'] = y_preds
submission.to_csv(args.submit_dir, index=False)
print('------Job Finished!------')