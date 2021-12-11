import argparse
import os

import torch
from torch.optim import lr_scheduler
import torchaudio
from tensorboardX import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from args import get_args
from utils import *
from metrics import *
from dataset import get_dataset
from torch.utils.data import DataLoader

'''
    dataset: https://github.com/JorisCos/LibriMix.git
'''

def iterstep(epoch, config, model, writer, pbar: tqdm, criterion, optimizer: torch.optim.Optimizer=None, mode='train'):
    losses = []
    for mix, (s1, s2) in pbar:
        if mode == 'train':
            optimizer.zero_grad()
        logits = model(mix)
        loss = criterion(torch.cat([s1, s2], 1), logits)
        if mode == 'train':
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        pbar.set_postfix({'epoch': epoch, 'mode': mode, 'sisdr': np.stack(losses).mean()})
    
    return np.stack(losses).mean()


def main(config):
    tensorboard_path = makedir(config.tensorboard_path)

    writer = SummaryWriter(logdir=tensorboard_path)

    model = torchaudio.models.ConvTasNet(num_sources=config.speechnum)
    summary(model, (1, 24000), device='cpu')
    trainset, valset, testset = get_dataset(config)
    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True, num_workers=0)
    valloader = DataLoader(valset, batch_size=config.batch, num_workers=0)
    testloader = DataLoader(testset, batch_size=config.batch, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    criterion = call_SISDR
    init_epoch = 0
    if config.resume:
        resume = torch.load(f'{config.name}.pt')
        init_epoch = resume['epoch']

    for epoch in range(init_epoch, config.epoch):
        model.train()
        with tqdm(trainloader) as pbar:
            train_loss = iterstep(epoch, config, model, writer, pbar, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            with tqdm(valloader) as pbar:
                val_loss = iterstep(epoch, config, model, writer, pbar, criterion, mode='val')
        
        lr_schedule.step(val_loss)
    pass


if __name__ == '__main__':
    main(get_args())
