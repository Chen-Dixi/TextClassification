import torch.utils.data as data

import numpy as np

import torch.nn as nn
import os.path
import re
import sys
import torch
from torchvision.datasets import MNIST




class Newsgroup(data.Dataset):
    training_file = 'train.npz'
    test_file = 'test.npz'

    def __init__(self,text_dir,train=True):
        
        self.text_dir = text_dir
        self.train = train
        #读文件，把文本保存起来，根据remove去掉文件内的无关信息
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if not self._check_if_exists():
            raise RuntimeError('Dataset not found.')

        texts = np.load(os.path.join(self.text_dir, data_file))

        data = texts['texts']
        labels = texts['labels']


        self.labels = torch.from_numpy(labels)
        

        self.data = torch.from_numpy(data)


    def _check_if_exists(self):
        return os.path.exists(os.path.join(self.text_dir, self.training_file)) and \
            os.path.exists(os.path.join(self.text_dir, self.test_file))


    def __getitem__(self, index):
        sample, target = self.data[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.labels)
