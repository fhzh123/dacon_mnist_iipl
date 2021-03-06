# Import Modules
import os
import copy
import json
import time
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd

# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Import Custom Module
from dataset import CustomDataset
from utils import train_valid_split

def main(args):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation & augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.resize_pixel, args.resize_pixel)),
            transforms.RandomAffine(args.random_affine),
            transforms.ColorJitter(brightness=(0.5, 2)),
            transforms.RandomResizedCrop((args.resize_pixel, args.resize_pixel), 
                                         scale=(0.85, 1)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.3, scale=(0.01, 0.05))
        ]),
        'valid': transforms.Compose([
            transforms.Resize((args.resize_pixel, args.resize_pixel)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    # Train, valid data split (Check utils module)
    train_img_list, valid_img_list = train_valid_split(random_seed=args.random_seed, 
                                                       data_path=args.data_path,
                                                       valid_ratio=args.valid_ratio)

    # Custom dataset & dataloader setting
    image_datasets = {
        'train': CustomDataset(train_img_list, isTrain=True, 
                               transform=data_transforms['train']),
        'valid': CustomDataset(valid_img_list, isTrain=True, 
                               transform=data_transforms['valid'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
    }

    # Model Setting
    model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    lr_step_scheduler = lr_scheduler.StepLR(optimizer, 
                                            step_size=args.lr_step_size, gamma=args.lr_decay_gamma)
    model.to(device)

    ## Training
    # Initialize
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999999
    early_stop = False

    # Train
    for epoch in range(args.num_epochs):

        if early_stop:
            print('Early Stopping!!!')
            break

        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            # Iterate over data
            for inputs, _, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = torch.tensor([int(x) for x in labels]).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() 

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Epoch loss calculate
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            # If loss is best_loss, then save
            if phase == 'valid' and epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            # Early stop conditioning
            if phase == 'train' and epoch_loss < 0.001:
                early_stop = True
                
            # Print progress
            spend_time = (time.time() - start_time) / 60
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.3f}min'.format(phase, epoch_loss, epoch_acc, spend_time))
        # Learning rate scheduler
        lr_step_scheduler.step()

    # Model Saving
    model.load_state_dict(best_model_wts)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'digit')):
        os.mkdir(os.path.join(args.save_path, 'digit'))
    save_path_ = os.path.join(os.path.join(args.save_path, 'digit'), str(datetime.datetime.now())[:-4].replace(' ', '_'))
    os.mkdir(save_path_)
    print('Best validation loss: {:.4f} when epoch {}'.format(best_loss, best_epoch))
    with open(os.path.join(save_path_, 'hyperparameter.json'), 'w') as f:
        json.dump({
            'efficientnet_not_use': args.efficientnet_not_use,
            'efficientnet_model_number': args.efficientnet_model_number,
            'num_epochs': args.num_epochs,
            'resize_pixel': args.resize_pixel,
            'random_affine': args.random_affine,
            'lr': args.lr,
            'random_seed': args.random_seed,
            'best_loss': best_loss
        }, f)
    torch.save(model.state_dict(), os.path.join(save_path_, 'model.pt'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='argparser')
    # Path Setting
    parser.add_argument('--data_path', type=str, default='./data', help='Data path setting')
    parser.add_argument('--save_path', type=str, default='./KH/save')
    parser.add_argument('--letter_model_path', type=str, default='./KH/save/letter/2020-08-16_01:21:55.04/')
    # Image Setting
    parser.add_argument('--resize_pixel', type=int, default=360, help='Resize pixel')
    parser.add_argument('--random_affine', type=int, default=10, help='Random affine transformation ratio')
    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=300, help='The number of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate setting')
    parser.add_argument('--lr_step_size', type=int, default=60, help='Learning rate scheduling step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='Learning rate decay scheduling per lr_step_size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='Train / Valid split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random state setting')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU worker setting')
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print('All Done! {:.4f}min spend!'.format((time.time() - total_start_time)/60))