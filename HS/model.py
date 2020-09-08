import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher,self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b4')
        self.fc1 = nn.Linear(1000+26,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self, image, letter):
        x1 = self.model(image)
        x2 = letter
        x = torch.cat((x1,x2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Student(nn.Module):
    def __init__(self):
        super(Student,self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.fc1 = nn.Linear(1000+26,100)
        self.fc2 = nn.Linear(100,10)

    def forward(self, image, letter):
        x1 = self.model(image)
        x2 = letter
        x = torch.cat((x1,x2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x