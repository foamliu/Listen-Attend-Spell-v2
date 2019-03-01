import os
import pickle

import librosa
import numpy as np
from tqdm import tqdm

from config import dim, window_size, stride, cmvn
from config import pickle_file, train_folder, test_folder, data_folder


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file):
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')


def get_data(mode):
    print('getting {} data...'.format(mode))
    if mode == 'train':
        folder = train_folder
    else:
        folder = test_folder

    samples = []
    waves = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    for w in tqdm(waves):
        trn_path = os.path.join(data_folder, w + '.trn')
        with open(trn_path, 'r', encoding='utf-8') as file:
            trn = file.readline()
        trn = trn.strip().replace(' ', '')
        for token in trn:
            build_vocab(token)
        wave = os.path.join(folder, w)
        feature = extract_feature(wave)
        samples.append({'feature': feature, 'trn': trn})
    return samples


def build_vocab(token):
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


if __name__ == "__main__":
    VOCAB = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    IVOCAB = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}

    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['test'] = get_data('test')

    print('num_train: ' + str(len(data['train'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(VOCAB)))

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
