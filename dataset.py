from glob import glob
import os
import torchaudio
import torch
from torch.nn.functional import *


def get_dataset(config):
    if config.task == 'sep_clean':
        train_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/train-100/mix_clean')
        train_s1_dir = os.path.join('/'.join(train_dir.split('/')[:-1]), 's1')
        train_s2_dir = os.path.join('/'.join(train_dir.split('/')[:-1]), 's2')
        val_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/dev/mix_clean')
        val_s1_dir = os.path.join('/'.join(val_dir.split('/')[:-1]), 's1')
        val_s2_dir = os.path.join('/'.join(val_dir.split('/')[:-1]), 's2')
        test_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/test/mix_clean')
        test_s1_dir = os.path.join('/'.join(test_dir.split('/')[:-1]), 's1')
        test_s2_dir = os.path.join('/'.join(test_dir.split('/')[:-1]), 's2')

    trainset = Wave_dataset(config, train_dir, train_s1_dir, train_s2_dir)
    valset = Wave_dataset(config, val_dir, val_s1_dir, val_s2_dir)
    testset = Wave_dataset(config, test_dir, test_s1_dir, test_s2_dir)
    return trainset, valset, testset


class Wave_dataset(torch.utils.data.Dataset):
    def __init__(self, config, mix, s1, s2) -> None:
        super(Wave_dataset, self).__init__()
        self.config = config
        self.mix = sorted(glob(mix + '/*'))
        self.s1 = sorted(glob(s1 + '/*'))
        self.s2 = sorted(glob(s2 + '/*'))

    def __len__(self):
        return len(self.mix)
        
    def __getitem__(self, idx):
        mix = torchaudio.load(self.mix[idx])[0]
        s1 = torchaudio.load(self.s1[idx])[0]
        s2 = torchaudio.load(self.s2[idx])[0]

        if mix.shape[-1] - self.config.sr * self.config.segment < 0:
            idx = 0
        else:
            idx = torch.randint(0, mix.shape[-1] - self.config.sr * self.config.segment, ())
        mix = mix[..., idx: idx + self.config.sr * self.config.segment]
        s1 = s1[..., idx: idx + self.config.sr * self.config.segment]
        s2 = s2[..., idx: idx + self.config.sr * self.config.segment]
        mix = pad(mix, (0, self.config.sr * self.config.segment - mix.shape[-1]))
        s1 = pad(s1, (0, self.config.sr * self.config.segment - mix.shape[-1]))
        s2 = pad(s2, (0, self.config.sr * self.config.segment - mix.shape[-1]))
        return mix, (s1, s2)
        
