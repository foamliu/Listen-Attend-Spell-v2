import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
dim = 40  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT
cmvn = True  # apply CMVN on feature

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_train = 10000
num_test = 2495
DATA_DIR = 'data'
thchs30_folder = 'data/data_thchs30'
train_folder = os.path.join(thchs30_folder, 'train')
test_folder = os.path.join(thchs30_folder, 'test')
data_folder = os.path.join(thchs30_folder, 'data')
IMG_DIR = 'data/images'
pickle_file = 'data/thchs30.pickle'
