import torch
import numpy as np
import pandas as pd
    
def submission(model, iter):
    y_preds = []
    device = torch.device('cuda:0')
    model.to(device)
    print('------Making submission file------')
    for image, letter, id in iter['test']:
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        with torch.no_grad():
            y_pred = model.forward(image, letter)
            y_pred = torch.argmax(y_pred)
            y_preds = np.append(y_preds,
                                y_pred.cpu().numpy())
            y_preds = np.vstack(y_preds)
    return y_preds