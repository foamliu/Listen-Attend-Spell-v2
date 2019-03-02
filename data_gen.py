import pickle

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import input_dim, window_size, stride, cmvn
from config import num_workers, pickle_file


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        f, trn = elem
        input_length = f.shape[0]
        input_dim = f.shape[1]
        # print('f.shape: ' + str(f.shape))
        feature = np.zeros((max_input_len, input_dim), dtype=np.float)
        feature[:f.shape[0], :f.shape[1]] = f
        trn = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=0)
        batch[i] = (feature, trn, input_length)
        # print('feature.shape: ' + str(feature.shape))
        # print('trn.shape: ' + str(trn.shape))

    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file):
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')


class Thchs30Dataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        if split == 'train':
            self.samples = data['train']
        else:
            self.samples = data['test']

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample['wave']
        trn = sample['trn']

        feature = extract_feature(wave)
        return feature, trn

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train_dataset = Thchs30Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)

    print(len(train_dataset))
    print(len(train_loader))
