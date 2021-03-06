import torch
import torchaudio

from models import *
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

    def test_DPRNN(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'dprnn'
        self.make_testdataset(config)
        model = DPRNNTasNet(config.speechnum).to(self.device)
        for rev_sep, clean_sep, _ in self.testset:
            mix = rev_sep.sum(-1)[None].to(self.device)
            results = model(mix)
            assert results.shape == clean_sep[None].transpose(-2,-1).shape

    def test_T60_ConvTasNet_v1(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v2'
        config.test = False
        config.t60 = True
        self.make_testdataset(config)
        model = T60_ConvTasNet_v1(config).to(self.device)
        for rev_sep, clean_sep, _, t60 in self.testset:
            t60 = t60[None].to(self.device).repeat((2))
            mix = rev_sep.sum(-1)[None].to(self.device).repeat((2,1))

            dereverb_results = model(mix, t60=t60)
            assert dereverb_results.shape == clean_sep[None].transpose(-2,-1).repeat((2,1,1)).shape

    def test_T60_DPRNNTasNet_v1(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v2'
        config.test = False
        config.t60 = True
        self.make_testdataset(config)
        model = T60_DPRNNTasNet_v1(config).to(self.device)
        for rev_sep, clean_sep, _, t60 in self.testset:
            t60 = t60[None].to(self.device).repeat((2))
            mix = rev_sep.sum(-1)[None].to(self.device).repeat((2,1))

            dereverb_results = model(mix, t60=t60)
            assert dereverb_results.shape == clean_sep[None].transpose(-2,-1).repeat((2,1,1)).shape

    def test_T60_ConvTasNet_v2(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v2'
        config.test = False
        config.t60 = True
        self.make_testdataset(config)
        model = T60_ConvTasNet_v2(config).to(self.device)
        for rev_sep, clean_sep, _, t60 in self.testset:
            t60 = t60[None].to(self.device).repeat((2))
            mix = rev_sep.sum(-1)[None].to(self.device).repeat((2,1))

            dereverb_results = model(mix, t60=t60)
            assert dereverb_results.shape == clean_sep[None].transpose(-2,-1).repeat((2,1,1)).shape

    def test_T60_ConvTasNet_v3(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v2'
        config.test = False
        config.t60 = True
        self.make_testdataset(config)
        model = T60_ConvTasNet_v3(config).to(self.device)
        for rev_sep, clean_sep, _, t60 in self.testset:
            t60 = t60[None].to(self.device).repeat((2))
            mix = rev_sep.sum(-1)[None].to(self.device).repeat((2,1))

            dereverb_results = model(mix, t60=t60)
            assert dereverb_results.shape == clean_sep[None].transpose(-2,-1).repeat((2,1,1)).shape

    def test_Sepformer(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'Sepformer'
        self.make_testdataset(config)
        model = Sepformer(self.config).to(self.device)
        for rev_sep, clean_sep, _ in self.testset:
            mix = rev_sep.sum(-1)[None].to(self.device)
            results = model(mix)
            assert results.shape == clean_sep[None].transpose(-2,-1).shape

    def test_T60_Sepformer_v1(self):
        config = deepcopy(self.config)
        config.task = 'rir'
        config.model = 'v1'
        config.test = False
        config.t60 = True
        self.make_testdataset(config)
        model = T60_Sepformer_v1(config).to(self.device)
        for rev_sep, clean_sep, _, t60 in self.testset:
            t60 = t60[None].to(self.device).repeat((2))
            mix = rev_sep.sum(-1)[None].to(self.device).repeat((2,1))

            dereverb_results = model(mix, t60=t60)
            assert dereverb_results.shape == clean_sep[None].transpose(-2,-1).repeat((2,1,1)).shape

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

