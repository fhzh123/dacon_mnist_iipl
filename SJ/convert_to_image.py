import os
import argparse
import pandas as pd 
import numpy as np
import csv 
from PIL import Image, ImageDraw 
from tqdm import tqdm


#이미지 전처리 파일 (따로  실행)


def img_save(dataset,isTrain=True): #main 함수에서 불러온 csv dataset을 img로 전처리 해주는 코드
    
    if isTrain:
        for_image_dataset=dataset.drop(columns=['id','letter','digit'])
        f=open('train_dataset_list'+'.csv','w',encoding='utf-8')
        wr=csv.writer(f)
        wr.writerow(['file_name','digit','letter'])
    else:
        for_image_dataset=dataset.drop(columns=['id','letter']) #Test의 경우 digit을 배열이 없음.  
        f=open('test_dataset_list'+'.csv','w',encoding='utf-8')
        wr=csv.writer(f)
        wr.writerow(['file_name','digit','letter'])
    for i in tqdm(range(len(dataset))): #tqdm: for문의 progress bar 출력 
        id_=dataset.iloc[i]['id']
        letter_=dataset.iloc[i]['letter']
        img_array=np.array(for_image_dataset.iloc[i]).reshape((28,28)).astype('uint8')
        img=Image.fromarray(img_array) #Image 객체로 바꿀 때는 fromarray 클래스 메서드 이용 
        if isTrain:
            digit_=str(dataset.iloc[i]['digit'])
            if not os.path.exists(os.path.join('./data/train/',digit_)):
                os.mkdir(os.path.join('./data/train/',digit_))
            img.save(os.path.join(f'./data/train/{digit_}/',f'{id_}_{letter_}.jpg'))
            wr.writerow([1,f'{id_}_{letter_}.jpg',digit_,letter_])
        else:
            img.save(os.path.join(f'./data/test/',f'{id_}_{letter_}.jpg'))
            wr.writerow([1,f'{id_}_{letter_}.jpg','',letter_])


def main(args):

#우선 해야하는게 이미지 파일 (csv)를 불러 오는 것이고 내가 고민하고 싶은건, 이 이미지에 있는 csv 파일을 한꺼번에 가져오는거임 
#(한꺼번에 가져오긴 어렵고 for문을 이용해야함)


    train_dataset=pd.read_csv(os.path.join(args.data_path,'train.csv')) #args에 있는 datapath에 있는 train.csv를 불러옴  
    

    test_dataset=pd.read_csv(os.path.join(args.data_path,'test.csv'))
    sample=pd.read_csv(os.path.join(args.data_path,'submission.csv'))

    print('Train data saving...')
    img_save(train_dataset,isTrain=True)

    print('Test data saving...')
    img_save(test_dataset,isTrain=False)





if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Parsing Method')
    args=parser.parse_args()
    parser.add_argument('--data_path',default='../data/',type=str,help='Default Data Path')
    args=parser.parse_args()

    main(args)
    