import torch
import numpy as np
import pandas as pd
    
def submission(model, iter):
    y_preds = []
    id_ = []
    device = torch.device('cuda:0')
    for image, letter, label in iter['train']:
        image = image.to(device)
        letter = torch.tensor(letter).to(device)
        with torch.no_grad():
            output = model(image, letter)
            y_pred = output.data.max(1, keepdim=True)[1]
            id_.append(id.tolist())
            y_preds.append(y_pred.tolist())
    
    submission = pd.DataFrame({
        'id': id_list,
        'digit': pred_list
    })
    submission = submission.sort_values(by=['id'])
    return submission