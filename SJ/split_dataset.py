import numpy as np
import pandas as pd 


def split_dataset_function(target_csv):
    
    csv_content=pd.read_csv(target_csv)


    
    train_size=int(0.8*len(csv_content))

    
    train_data=csv_content.sample(n=train_size, random_state=47 )

    
    validation_data=csv_content[csv_content.join(train_data,how='left',rsuffix='_')['index_'].isnull()]
    





    return train_data, validation_data