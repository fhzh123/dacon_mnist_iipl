import numpy as np
import pandas as pd 

def split_dataset_function(target_csv):
    
    csv_content=pd.read_csv(target_csv)

    #csv_content_to_list=csv_content['file_name'].values.tolist()
    #csv_content_to_set=set(csv_content_to_list)
    
    train_size=int(0.8*len(csv_content))

    
    train_data=csv_content.sample(n=train_size)

    
    validation_data=csv_content[csv_content.join(train_data,how='left',rsuffix='_')['index_'].isnull()]
    



    
    #train_data_set=set(np.random.choice(csv_content_to_list,train_size))
    #validation_data_set=csv_content_to_set.difference(train_data_set)

    #train_data=list(train_data_set)
    #validation_data=list(validation_data_set)

    return train_data, validation_data