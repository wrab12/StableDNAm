import pickle
import random

import torch
import torch.utils.data as Data

import numpy as np
from util import util_file


class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode

        # label:
        self.train_label = None
        self.test_label = None
        # raw_data: ['ACTG','AACG']
        self.train_dataset = None
        self.test_dataset = None
        # iterator
        self.train_dataloader = None
        self.test_dataloader = None

    def load_traindata(self):
        self.train_dataset1, self.train_label1 = util_file.load_tsv_format_data(self.config.path_train_data1)
        # self.train_dataset2, self.train_label2 = util_file.load_tsv_format_data(self.config.path_train_data2)
        self.train_dataset, self.train_label = self.train_dataset1 , self.train_label1



        self.train_dataloader = self.construct_dataset(self.train_dataset, self.train_label, self.config.cuda,
                                                       self.config.batch_size)

    def load_testdata(self):
        self.test_dataset, self.test_label = util_file.load_tsv_format_data(self.config.path_test_data)


        self.test_dataloader = self.construct_dataset(self.test_dataset, self.test_label, self.config.cuda,
                                                      self.config.batch_size)

    def construct_dataset(self, sequences, labels, cuda, batch_size):
        if cuda:
            labels = torch.tensor(labels, dtype=torch.long, device='cuda')
        else:
            labels = torch.Tensor(labels, dtype=torch.long, device='cuda')
        dataset = MyDataSet(sequences, labels)
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      drop_last=True,
                                      shuffle=True)
        print('len(data_loader)', len(data_loader))
        return data_loader

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader

        return return_data


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
