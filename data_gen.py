import pickle

import torch
from torch.utils.data import Dataset

from config import num_workers, pickle_file


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
        feature = sample['feature']
        trn = sample['trn']
        return feature, trn

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train_dataset = Thchs30Dataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    print(len(train_dataset))
    print(len(train_loader))
