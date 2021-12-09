import argparse
import os

import torch
import torchaudio
from tensorboardX import SummaryWriter
from torchsummary import summary

from args import get_args
from utils import *
from metrics import *
from dataset import get_dataset
from torch.utils.data import DataLoader

'''
    dataset: https://github.com/JorisCos/LibriMix.git
'''

def iterstep(model, writer, dataset, optimizer=None, mode='train'):
    if mode == 'train':
        pass


def main(config):
    tensorboard_path = makedir(config.tensorboard_path)

    writer = SummaryWriter(logdir=tensorboard_path)

    model = torchaudio.models.ConvTasNet(num_sources=config.speechnum)
    trainset, valset, testset = get_dataset(config)
    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=config.batch, num_workers=0)
    testloader = DataLoader(testset, batch_size=config.batch, num_workers=0)

    for x in trainloader:
        pass
    pass


if __name__ == '__main__':
    main(get_args())
