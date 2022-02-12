import torch
import torchaudio

from models import ConvTasNet_v1
from args import get_args
from evals import evaluate
from utils import get_device
from data_utils import LibriMix

import os
from time import time


class Test:
    def __init__(self, config) -> None:
        self.config = config
        self.device = get_device()
        if config.model == '':
            self.model = torchaudio.models.ConvTasNet(msk_activate='relu')
        elif config.model == 'v1':
            self.model = ConvTasNet_v1()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.make_testdataset()
    
    def make_testdataset(self):
        self.testset = LibriMix(
            csv_dir=os.path.join(self.config.datapath, 'Libri2Mix/wav8k/min/test'),
            config=self.config,
            task='rir_sep_clean',
            sample_rate=self.config.sr,
            n_src=self.config.speechnum,
            segment=None,
            return_id=True,
        )
        self.testset.df = self.testset.df[:1]

    def test_evaluate(self):
        testset = self.testset
        si_sdri, si_snri = evaluate(self.config, self.model, testset, '', 1)

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
    test = Test(config)
    test.run()

