import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 40  # dimension of feature
window_size = 25  # window size for FFT (ms)
hidden_size = 512
embedding_dim = 512
stride = 10  # window stride for FFT
cmvn = True  # apply CMVN on feature
num_layers = 4

# Training parameters
batch_size = 32
lr = 1e-3
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 10  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
PAD_token = 0
SOS_token = 1
EOS_token = 2
num_samples = 141600
num_train = 10000
num_test = 2495
vocab_size = 2886

DATA_DIR = 'data'
aishell_folder = 'data/data_aishell'
wav_folder = os.path.join(aishell_folder, 'wav')
tran_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
IMG_DIR = 'data/images'
pickle_file = 'data/aishell.pickle'
