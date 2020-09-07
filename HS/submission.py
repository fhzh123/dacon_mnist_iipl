import torch
import numpy as np
import pandas as pd
    
def submission(model, iter):
    y_preds = []
    id_ = []
    device = torch.device('cuda:0')
    model.to(device)
    print('------Making submission file------')
    for image, letter, id in iter['train']:
        image = image.to(device)
        letter = letter.clone().detach().to(device)
        with torch.no_grad():
            output = model(image, letter)
            y_pred = output.data.max(1, keepdim=True)[1]
            id = [int(x) for x in id]
            id_.append(id)
            y_preds.append(y_pred.detach())
    
    submission = pd.DataFrame({
        'id': id_,
        'digit': y_preds
    })
    submission = submission.sort_values(by=['id'])
    return submission