import os
import pickle
import random
from shutil import copyfile

import torch

from config import pickle_file, device
from data_gen import pad_collate
from models import Seq2Seq
from utils import ensure_folder
from utils import extract_feature


class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder'].module
    decoder = checkpoint['decoder'].module
    encoder.eval()
    decoder.eval()
    model = Seq2Seq(encoder, decoder)

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    VOCAB = data['VOCAB']
    char_list = list(VOCAB.keys())
    IVOCAB = data['IVOCAB']
    samples = data['test']

    samples = random.sample(samples, 10)

    ensure_folder('waves')

    args = adict()
    args.beam_size = 5
    args.nbest = 1
    args.decode_max_len = 40

    for i, sample in enumerate(samples):
        wave = sample['wave']
        dst = os.path.join('waves', '{}.wav'.format(i))
        copyfile(wave, dst)
        print(wave)

        feature = extract_feature(wave)
        trn = sample['trn']
        transcript = [IVOCAB[token_id] for token_id in trn]
        transcript = ''.join(transcript)
        print(transcript)

        batch = [(feature, trn)]
        data = pad_collate(batch)
        _features, _trns, _input_lengths = data
        _features = _features.float().to(device)
        _input_lengths = _input_lengths.long().to(device)

        nbest_hyps = model.recognize(_features, _input_lengths, char_list, args)

        nbest_hyps = ''.join([char_list[int(x)] for x in nbest_hyps[0]['yseq'][1:]])
        print('nbest_hyps: ' + nbest_hyps)
