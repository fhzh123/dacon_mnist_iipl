import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

def loss_ce(output, label):
    loss = F.cross_entropy(output, label)
    return loss

def loss_kd(t_output, s_output, label, alpha, T):
    t_output = F.softmax(t_output/T, dim=1)
    s_output = F.log_softmax(s_output/T, dim=1)
    
    criterion = F.cross_entropy(s_output, label)
    loss_kl = F.kl_div(s_output, t_output, reduction='batchmean')
    loss =  loss_kl * (alpha * T * T) + criterion * (1. - alpha)
    return loss

def submission(model, iter):
    y_preds = []
    device = torch.device('cuda:0')
    model.to(device)
    print('------Making submission file------')
    for image, letter, id in tqdm(iter['test']):
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        with torch.no_grad():
            y_pred = model.forward(image, letter)
            y_pred = torch.argmax(y_pred)
            y_preds = np.append(y_preds,
                                y_pred.cpu().numpy())
            y_preds = np.vstack(y_preds)
    return y_preds