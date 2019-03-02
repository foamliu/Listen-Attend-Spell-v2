import os
import pickle

from tqdm import tqdm

from config import pickle_file, train_folder, test_folder, data_folder


def get_data(mode):
    print('getting {} data...'.format(mode))
    if mode == 'train':
        folder = train_folder
    else:
        folder = test_folder

    global VOCAB

    samples = []
    waves = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    for w in tqdm(waves):
        wave = os.path.join(folder, w)

        trn_path = os.path.join(data_folder, w + '.trn')
        with open(trn_path, 'r', encoding='utf-8') as file:
            trn = file.readline()
        trn = list(trn.strip().replace(' ', '')) + ['<EOS>']
        for token in trn:
            build_vocab(token)
        trn = [VOCAB[token] for token in trn]

        samples.append({'trn': trn, 'wave': wave})
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
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

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))
