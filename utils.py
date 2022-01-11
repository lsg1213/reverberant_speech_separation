from genericpath import exists
import os
import torch


def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)
        print(dirs)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
