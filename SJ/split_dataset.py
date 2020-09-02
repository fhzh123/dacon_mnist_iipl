import numpy as np
import pandas as pd 

#생각해보니까 이거 random sampling 형식으로 짜야할거 같은데 다시 짜보기 

def split_dataset_function(target_csv):
    
    csv_content=pd.read_csv(target_csv)


    
    train_size=int(0.8*len(csv_content))

    
    train_data=csv_content.sample(n=train_size)

    
    validation_data=csv_content[csv_content.join(train_data,how='left',rsuffix='_')['index_'].isnull()]
    





    return train_data, validation_data