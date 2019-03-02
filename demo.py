import pickle
import random

import cv2 as cv
import torch

from models import Seq2Seq
from config import pickle_file
from data_gen import pad_collate
from utils import ensure_folder

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder.eval()
    decoder.eval()
    model = Seq2Seq(encoder, decoder)

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    VOCAB = data['VOCAB']
    IVOCAB = data['IVOCAB']
    val = data['val']
    image_ids_val, questions_val, answers_val = val
    prefix = 'data/val2014/COCO_val2014_0000'

    num_val_samples = range(len(val[0]))
    _ids = random.sample(num_val_samples, 10)

    questions = []
    targets = []
    ensure_folder('images')

    batch = []

    for i, index in enumerate(_ids):
        image_id = int(image_ids_val[index])
        image_id = '{:08d}'.format(image_id)
        filename = prefix + image_id + '.jpg'
        img = cv.imread(filename)
        img = cv.resize(img, (im_size, im_size))
        filename = 'images/{}_img.png'.format(i)
        cv.imwrite(filename, img)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = (img - 127.5) / 128

        question = questions_val[index]
        answer = answers_val[index]

        questions.append(question)
        targets.append(answer)

        elem = img, question, answer
        batch.append(elem)

    data = pad_collate(batch)
    _imgs, _questions, _targets = data
    _imgs = _imgs.float().cuda()
    _questions = _questions.long().cuda()
    _targets = _targets.long().cuda()
    _mask = get_mask(_targets).cuda()
    outputs, loss = model.forward(_imgs, _questions, _targets, _mask)
    print('pred_ids.size(): ' + str(outputs.size()))
    outputs = list(outputs.cpu().numpy())
    print('len(_pred_ids): ' + str(outputs))

    for i in range(10):
        question = questions[i]
        question = ''.join([IVOCAB[id] for id in question]).replace('<EOS>', '')
        target = targets[i]
        target = ''.join([IVOCAB[id] for id in target]).replace('<EOS>', '')

        pred = outputs[i]
        pred = ''.join([IVOCAB[id] for id in pred]).replace('<EOS>', '')
        pred = pred.replace('<EOS>', '').replace('<PAD>', '')

        print('提问：' + question)
        print('标准答案：' + target)
        print('电脑抢答：' + pred)
        print()
