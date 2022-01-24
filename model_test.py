import torch
import torchaudio

from model import *
from args import get_args
from evals import evaluate
from utils import get_device
from data_utils import LibriMix

import os
from copy import deepcopy
from time import time


class Test:
    def __init__(self, config) -> None:
        self.config = config
        self.device = get_device()
        self.make_testdataset()
    
    def make_testdataset(self, config=None):
        task = 'sep_clean'
        if config is None:
            config = self.config
        if config.task != '':
            task = config.task + '_' + task

        self.testset = LibriMix(
            csv_dir=os.path.join(self.config.datapath, 'Libri2Mix/wav8k/min/test'),
            config=config,
            task=task,
            sample_rate=self.config.sr,
            n_src=self.config.speechnum,
            segment=None,
            return_id=True,
        )
        self.testset.df = self.testset.df[:1]

    def test_ConvTasNet_v1(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v1'
        self.make_testdataset(config)
        model = ConvTasNet_v1().to(self.device)
        for rev_sep, clean_sep, _, distance in self.testset:
            distance = torch.from_numpy(distance[None]).to(self.device)
            mix = rev_sep.sum(-1)[None].to(self.device)
            results = model(mix, distance)
            assert results.shape == clean_sep[None].shape

    def test_ConvTasNet_v2(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v2'
        self.make_testdataset(config)
        model = ConvTasNet_v2(reverse=True).to(self.device)
        for rev_sep, clean_sep, _, distance in self.testset:
            distance = torch.from_numpy(distance[None]).to(self.device)
            mix = rev_sep.sum(-1)[None].to(self.device)
            results = model(mix, distance)
            assert results.shape == clean_sep[None].shape

    
    def test_ConvTasNet_v3(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v3'
        self.make_testdataset(config)
        model = ConvTasNet_v3(distance=True).to(self.device)
        for rev_sep, clean_sep, _, distance in self.testset:
            distance = torch.from_numpy(distance[None]).to(self.device)
            mix = rev_sep.sum(-1)[None].to(self.device)
            results = model(mix, distance)
            assert results.shape == clean_sep[None].shape

    def run(self) -> None:
        functions = [i for i in dir(self) if 'test_' in i]
        test_st = time()
        print('Test start')
        for func in functions:
            st = time()
            print(func, 'testing...')
            getattr(self, func)()
            print(time() - st, 'seconds to test')
        print(f'All test time is {time() - test_st}')

if __name__ == '__main__':
    config = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.batch = 2
    test = Test(config)
    test.run()

