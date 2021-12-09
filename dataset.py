from glob import glob
import os
import torchaudio
import torch


def get_dataset(config):
    if config.task == 'sep_clean':
        train_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/train-100/mix_clean')
        val_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/dev/mix_clean')
        test_dir = os.path.join(config.datapath, f'Libri2Mix/wav{config.sr // 1000}k/min/test/mix_clean')

    trainset = Wave_dataset(config, train_dir)
    valset = Wave_dataset(config, val_dir)
    testset = Wave_dataset(config, test_dir)
    return trainset, valset, testset


class Wave_dataset(torch.utils.data.Dataset):
    def __init__(self, config, path) -> None:
        super(Wave_dataset, self).__init__()
        self.config = config
        self.path = sorted(glob(path + '/*'))

    def __len__(self):
        return len(self.path)
        
    def __getitem__(self, idx):
        wav = torchaudio.load(self.path[idx])[0]
        import pdb; pdb.set_trace()
        return wav
        
