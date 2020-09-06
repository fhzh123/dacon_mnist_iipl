import torch
import numpy as np
import pandas as pd
    
def test_model(model, iter):
    y_preds = []
    device = torch.device('cuda:0')
    with torch.no_grad():
        for input, letter, id in iter['test']:
            y_pred = model.forward(input.to(device),
                                   torch.tensor(letter).to(device))
            y_pred = torch.argmax(y_pred)
            y_preds = np.append(y_preds,
                                y_pred.cpu().numpy())
            y_preds = np.vstack(y_preds)
    return y_preds