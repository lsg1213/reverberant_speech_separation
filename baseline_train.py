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


def iterstep(epoch, config, model, writer, pbar: tqdm, criterion, optimizer: torch.optim.Optimizer=None, mode='train', device=torch.device('cpu')):
    losses = []
    for mix, (s1, s2) in pbar:
        mix = mix.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)

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
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.name += f'_{config.batch}'
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    tensorboard_path = makedir(config.tensorboard_path)
    device = get_device()

    writer = SummaryWriter(logdir=tensorboard_path)

    model = torchaudio.models.ConvTasNet(num_sources=config.speechnum)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # summary(model, (1, 24000))
    trainset, valset, testset = get_dataset(config)
    trainloader = DataLoader(trainset, batch_size=config.batch, shuffle=True, num_workers=10)
    valloader = DataLoader(valset, batch_size=config.batch, num_workers=10)
    testloader = DataLoader(testset, batch_size=config.batch, num_workers=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    criterion = call_SISDR
    init_epoch = 0
    patience = 0
    best_score = torch.inf
    if config.resume:
        resume = torch.load(f'{config.name}.pt')
        init_epoch = resume['epoch']
        best_score = resume['best_score']
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['opt'])
        lr_schedule.load_state_dict(resume['lr_schedule'])
        patience = resume['patience']

    for epoch in range(init_epoch, config.epoch):
        if patience == config.max_patience:
            print('EARLY STOPPING!')
            break
        model.train()
        with tqdm(trainloader) as pbar:
            train_loss = iterstep(epoch, config, model, writer, pbar, criterion, optimizer, device=device)
        writer.add_scalar('train/SISDR', - train_loss, epoch)

        model.eval()
        with torch.no_grad():
            with tqdm(valloader) as pbar:
                val_loss = iterstep(epoch, config, model, writer, pbar, criterion, mode='val', device=device)
        writer.add_scalar('val/SISDR', - val_loss, epoch)
        
        lr_schedule.step(val_loss)
        if best_score > val_loss:
            best_score = val_loss
            patience = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'patience': patience,
                'best_score': best_score,
                'lr_schedule': lr_schedule.state_dict()
            })
            with torch.no_grad():
                with tqdm(testloader) as pbar:
                    test_loss = iterstep(epoch, config, model, writer, pbar, criterion, mode='test', device=device)
                    writer.add_scalar('test/SISDR', - test_loss, epoch)
        else:
            patience += 1



if __name__ == '__main__':
    main(get_args())
