import os
import torch


def makedir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print(dirs)


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
