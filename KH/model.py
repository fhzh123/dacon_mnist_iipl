# Import Modules
import os
import numpy as np
from efficientnet_pytorch import EfficientNet

# Import PyTorch
import torch
import torchvision
from torch import nn

class conv_model(nn.Module):
    def __init__(self):
        super(conv_model, self).__init__()
        # self.efficient_net = EfficientNet.from_name('efficientnet-b0')
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        self.output_linear = nn.Linear(1000, 100)
        # self.output_norm = nn.GroupNorm(10, 100)
        self.output_linear2 = nn.Linear(100, 10)

    def forward(self, input_img):
        model_output = self.efficient_net(input_img)
        # model_output = mish(self.output_linear(model_output))
        # model_output = self.output_linear2(self.output_norm(model_output))
        model_output = self.output_linear(model_output)
        model_output = self.output_linear2(model_output)
        return model_output

def mish(x):
    soft_plus = nn.Softplus()
    tanh = nn.Tanh()
    return x * tanh(soft_plus(x))