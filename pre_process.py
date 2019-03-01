import os
import pickle

from config import pickle_file, train_folder, test_folder, data_folder


def get_data(mode):
    print('getting {} data...'.format(mode))
    if mode == 'train':
        folder = train_folder
    else:
        folder = test_folder

    samples = []
    waves = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    for w in waves:
        trn_path = os.path.join(data_folder, w + '.trn')
        with open(trn_path, 'r', encoding='utf-8') as file:
            trn = file.readline()
        trn = trn.strip().replace(' ', '')
        for token in trn:
            build_vocab(token)
        samples.append({'wave': w, 'trn': trn})
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
