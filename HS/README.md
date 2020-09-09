# DACON Competition with Knowledge Distillation

This is the code for attending [DACON Computer Vision Competition](https://dacon.io/competitions/official/235626/overview/).

## How to execute?

First, you need to install libraries to set environment with this code:

    pip install -r requirements.txt

Then run main.py by this code to implement training:

    python3 main.py

In this code, there are several arguments to be put like
  - **data_dir**: directory where data is stored(default: ./data).
  - **split_ratio**: the ratio of training-validation split for train file(defalut: 0.1).
  - **batch**: the batch size(defalut: 64).
  - **epoch**: the time model trains data(default: 100).
  - **step_size**: the time how often learning rate scheduler updates the learning rate(default: 30).
  - **gamma**: while updating learning rate, gamma is multiplied to the existing learning rate(defalut: 0.1).
  - **lr**: the learning rate(default: 0.001).
  - **submit_dir**: directory where submission file is stored(default: './submission.csv').
  - **resize_pixel**: resize pixel for image(default=100).
  - **random_affine**: randomly rotate images in the angle set by this argument(default=10).

## Model Architecture
![model](https://user-images.githubusercontent.com/51365760/92325154-3fccf480-f083-11ea-82ab-8304af084212.JPG)
(Gou et al, 2020)

In this code, Resnet50 and MobilenetV2 is used as teacher network and student network, respectively(the code is imported from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html)).

For dealing with letter column in train and test file, the one-hot encoding is used(in python code, pandas.get_dummies()).
Therefore, one-hot for letter and image was put as input of the model.

## Result
- **2020.09.06**  
score: 0.81372,  rank: 215(public)
- **2020.09.08**  
score: 0.90686,  rank: 91(public)  
condition: teacher-resnet50, student-mobilenetV2  
           epochs-200, resize_pixel=200  

## Reference
  - [Knowledge Distillation: A Survey(Gou et al, 2020)](https://arxiv.org/abs/2006.05525)
  - [Revisit Knowledge Distillation: a Teacher-free Framework(Li et al, 2020)](https://arxiv.org/abs/1909.11723)
  - [Github - peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)

## Major Updates
- **2020.09.07**  
Largely modify image transformer: adding some components for train image transformer, changing means and standard deviations, etc.
