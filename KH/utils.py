# Import Modules
import os
import fcntl
import struct
import termios
import numpy as np
from glob import glob

def terminal_size():
    th, tw, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return tw

def train_valid_split(random_seed, data_path, valid_ratio):
    # Image load
    np.random.seed(random_seed)
    total_train_img_list = glob(os.path.join(data_path, 'train/*/*.jpg'))
    # Data split
    valid_size = int(len(total_train_img_list)*valid_ratio)
    valid_img_list = list(np.random.choice(total_train_img_list, size=valid_size))
    train_img_list = list(set(total_train_img_list) - set(valid_img_list))
    return train_img_list, valid_img_list