import os
import time
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Import Custom Module
from dataset import CustomDataset

def main(args):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation & augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.resize_pixel, args.resize_pixel)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.resize_pixel, args.resize_pixel)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    ## Train, valid data split
    # Image load
    np.random.seed(args.random_seed)
    total_train_img_list = glob(os.path.join(args.data_path, 'train/*/*.jpg'))
    test_img_list = glob(os.path.join(args.data_path, 'test/*.jpg'))
    # Data split
    valid_size = int(len(total_train_img_list)*args.valid_ratio)
    valid_img_list = list(np.random.choice(total_train_img_list, size=valid_size))
    train_img_list = list(set(total_train_img_list) - set(valid_img_list))

    # Custom dataset & dataloader setting
    image_datasets = {
        'train': CustomDataset(train_img_list, isTrain=True, transform=data_transforms['train']),
        'valid': CustomDataset(valid_img_list, isTrain=True, transform=data_transforms['test']),
        'test': CustomDataset(test_img_list, isTrain=False, transform=data_transforms['test'])
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=args.batch_size,
                            shuffle=False, num_workers=1)
    }

    # Model Setting
    model = models.mobilenet_v2(pretrained=False, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=args.lr)
    lr_step_scheduler = lr_scheduler.StepLR(optimizer, 
                                            step_size=20, gamma=0.1) # Decay LR by a factor of 0.1 every step_size
    model.to(device)

    ## Training
    # Initialize
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Train
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 100)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, letters, labels in tqdm(dataloaders[phase]):
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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            spend_time = (time.time() - start_time) / 60
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}min'.format(phase, epoch_loss, epoch_acc, spend_time))
        # Learning rate scheduler
        lr_step_scheduler.step()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Order_net argparser')
    parser.add_argument('--data_path', type=str, default='../data', help='Data Path Setting')
    parser.add_argument('--num_epochs', type=int, default=10, help='The Number of Epoch')
    parser.add_argument('--resize_pixel', type=int, default=64, help='Resize pixel')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate Setting')

    parser.add_argument('--valid_ratio', type=float, default=0.05, help='Train / Valid Split Ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random State Setting')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU Worker Setting')
    args = parser.parse_args()

    main(args)
    print('Done!')