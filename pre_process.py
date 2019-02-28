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
        samples.append({'wave': w, 'trn': trn})
    return samples


if __name__ == "__main__":
    data = dict()

    data['train'] = get_data('train')
    print('num_train: ' + str(len(data['train'])))
    data['test'] = get_data('test')
    print('num_test: ' + str(len(data['test'])))

    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)
