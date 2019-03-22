import os
import pickle

from tqdm import tqdm

from config import wav_folder, tran_file, pickle_file


def get_data(mode):
    print('getting {} data...'.format(mode))

    with open(tran_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in tqdm(lines):
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    folder = os.path.join(wav_folder, mode)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in dirs:
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        for f in files:
            key = f.split('.')[0]
            trn = tran_dict[key]
            trn = list(trn.strip()) + ['<EOS>']
            for token in trn:
                build_vocab(token)
            file_name = os.path.join(dir, f)



    global VOCAB



    samples = []
    waves = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]
    for w in tqdm(waves):
        wave = os.path.join(folder, w)


        trn = list(trn.strip().replace(' ', '')) + ['<EOS>']

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
    data['dev'] = get_data('dev')
    data['test'] = get_data('test')

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_dev: ' + str(len(data['dev'])))
    print('num_test: ' + str(len(data['test'])))
    print('vocab_size: ' + str(len(data['VOCAB'])))
