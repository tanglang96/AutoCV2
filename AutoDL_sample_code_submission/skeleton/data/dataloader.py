# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging

import torch
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


# class TestDataLoader(DataLoader):
#     def __init__(self, dataset, steps=0, batch_size=1, shuffle=False, num_workers=4, pin_memory=False, drop_last=False,
#                  sampler=None):
#         super(TestDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, sampler=sampler,
#                                              num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
#         self.dataset = dataset
#
#     def __iter__(self):
#         for idx in range(len(self.dataset)):
#             data = self.dataset[idx]
#             # yield (torch.Tensor(data[0]).half(), torch.Tensor(data[1]).half())
#             yield (data[0], data[1])


class TestDataLoader:
    def __init__(self, dataset, steps=0, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
                 sampler=None):
        self.dataset = dataset
        self.sampler = torch.utils.data.SequentialSampler(dataset)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        self.steps = len(self.dataset) // batch_size + 1
        self.batch_size = batch_size
        self.data_len = len(self.dataset)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for data in self.dataloader:
            yield (data)


class FixedSizeDataLoader:
    def __init__(self, dataset, steps, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
                 sampler=None):
        sampler = InfiniteSampler(dataset, shuffle) if sampler is None else sampler
        self.batch_size = batch_size
        batch_size = 1 if batch_size is None else batch_size

        self.steps = steps
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _, data in zip(range(self.steps), self.dataloader):
            yield ([t[0] for t in data] if self.batch_size is None else data)


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        while True:
            index_list = torch.randperm(n).tolist() if self.shuffle else list(range(n))
            for idx in index_list:
                yield idx

    def __len__(self):
        return len(self.data_source)

class TestSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        while True:
            index_list = list(range(n))
            for idx in index_list:
                yield idx   #带有yield其实是一个迭代器

    def __len__(self):
        return len(self.data_source)