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
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
'''
    dataset: https://github.com/JorisCos/LibriMix.git
'''


def iterstep(epoch, config, model, writer, pbar: tqdm, criterion, optimizer: torch.optim.Optimizer=None, mode='train', device=torch.device('cpu')):
    max_value = 5.
    losses = []
    sisdris = []
    for mix, (s1, s2) in pbar:
        mix = mix.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        logits = model(mix)
        loss = criterion(logits, torch.cat([s1, s2], 1))
        
        sisdri = - loss.detach().cpu()
        # loss = loss.mean()
        # sisdri = get_sisdri(mix, torch.cat([s1, s2], 1), logits)
        # sisdri = torch.stack(list(map(cal_SDRi, torch.cat([s1, s2], 1)[-1:], logits[-1:], torch.cat([mix, mix], 1)[-1:])))
        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), max_value)
            optimizer.step()
        sisdris.append(sisdri.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        pbar.set_postfix({'epoch': epoch, 'mode': mode, 'loss': np.stack(losses).mean(), 'SI-SDRI': np.stack(sisdris).mean()})

    
    return np.stack(losses).mean(), np.stack(sisdris).mean()


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.name += f'_{config.batch}'
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    makedir(config.tensorboard_path)
    device = get_device()

    writer = SummaryWriter(logdir=config.tensorboard_path)

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
    lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    init_epoch = 0
    patience = 0
    early_patience = 0
    best_score = 0.
    if config.resume:
        resume = torch.load(f'{config.name}.pt')
        init_epoch = resume['epoch']
        best_score = resume['best_score']
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['opt'])
        lr_schedule.load_state_dict(resume['lr_schedule'])
        patience = resume['patience']
        early_patience = resume['early_patience']

    # import pytorch_lightning as pl
    # from asteroid.engine.system import System

    # system = System(
    #     model=model,
    #     loss_func=criterion,
    #     optimizer=optimizer,
    #     train_loader=trainloader,
    #     val_loader=valloader,
    #     scheduler=lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5),
    #     config=vars(config),
    # )

    # # Define callbacks
    # callbacks = []
    # # Don't ask GPU if they are not available.
    # gpus = -1 if torch.cuda.is_available() else None
    # distributed_backend = "ddp" if torch.cuda.is_available() else None

    # trainer = pl.Trainer(
    #     max_epochs=config.epoch,
    #     callbacks=callbacks,
    #     default_root_dir='',
    #     gpus=gpus,
    #     distributed_backend=distributed_backend,
    #     limit_train_batches=1.0,  # Useful for fast experiment
    #     gradient_clip_val=5.0,
    # )
    # trainer.fit(system)

    for epoch in range(init_epoch, config.epoch):
        if early_patience == config.max_patience:
            print('EARLY STOPPING!')
            break
        model.train()
        with tqdm(trainloader) as pbar:
            train_loss, score = iterstep(epoch, config, model, writer, pbar, criterion, optimizer, device=device)
        writer.add_scalar('train/SISDR', score, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)

        model.eval()
        with torch.no_grad():
            with tqdm(valloader) as pbar:
                val_loss, score = iterstep(epoch, config, model, writer, pbar, criterion, mode='val', device=device)
        writer.add_scalar('val/SISDR', score, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        
        lr_schedule.step(score)
        if best_score < score:
            best_score = score
            patience = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'patience': patience,
                'best_score': best_score,
                'lr_schedule': lr_schedule.state_dict(),
                'early_patience': early_patience
            }, f'{config.name}.pt')
            with torch.no_grad():
                with tqdm(testloader) as pbar:
                    test_loss, score = iterstep(epoch, config, model, writer, pbar, criterion, mode='test', device=device)
                    writer.add_scalar('test/SISDR', score, epoch)
                    writer.add_scalar('test/loss', train_loss, epoch)
        else:
            patience += 1
            early_patience += 1



if __name__ == '__main__':
    main(get_args())
