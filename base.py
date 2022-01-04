import os

from args import get_args
from multiprocessing import cpu_count
import json

import torch
import torchaudio
import numpy as np
from tensorboardX import SummaryWriter
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_utils import LibriMix
from utils import makedir, get_device
from callbacks import EarlyStopping, Checkpoint
from evals import evaluate


def iterloop(config, epoch, model, criterion, dataloader, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    scores = []
    with tqdm(dataloader) as pbar:
        for mix, clean in pbar:
            mix = mix.to(device)
            clean = clean.to(device)

            logits = model(mix)

            loss = criterion(logits, clean)

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            pbar.set_postfix({'mode': mode, 'loss': np.mean(losses)})
    if mode != 'test':
        return np.mean(losses)
    else:
        return np.mean(losses), np.mean(scores)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.name += f'_{config.batch}'
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    writer = SummaryWriter(os.path.join(config.tensorboard_path, config.name))
    savepath = os.path.join('save', config.name)
    device = get_device()
    makedir(config.tensorboard_path)
    makedir(savepath)

    init_epoch = 0
    final_epoch = 0
    
    gpu_num = torch.cuda.device_count()
    train_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/train-360'),
        task='sep_clean',
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )

    val_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/dev'),
        task='sep_clean',
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )
    
    test_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/test'),
        task='sep_clean',
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=None,
        return_id=True,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch,
        num_workers=gpu_num * (cpu_count() // 4),
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config.batch * 2,
        num_workers=gpu_num * (cpu_count() // 4),
    )

    model = torchaudio.models.ConvTasNet(msk_activate='relu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_loss', mode='min', verbose=True))
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    if config.resume:
        resume = torch.load(os.path.join(savepath, 'checkpoint.pt'))
        model.load_state_dict(resume['model'])
        model = model.to(device)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        init_epoch = resume['epoch']
        for callback in callbacks:
            state = resume.get(type(callback).__name__)
            if state is not None:
                callback.load_state_dict(state)

    model = model.to(device)
    for epoch in range(init_epoch, config.epoch):
        print(f'--------------- epoch: {epoch} ---------------')
        model.train()
        train_loss = iterloop(config, epoch, model, criterion, train_loader, optimizer, mode='train')
        writer.add_scalar('train/loss', train_loss, epoch)

        model.eval()
        with torch.no_grad():
            val_loss = iterloop(config, epoch, model, criterion, val_loader, mode='val')
        writer.add_scalar('val/loss', val_loss, epoch)
        scheduler.step(val_loss)
        final_epoch += 1
        for callback in callbacks:
            if type(callback).__name__ == 'Checkpoint':
                callback.elements.update({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                })
                for cb in callbacks:
                    if type(cb).__name__ != 'Checkpoint':
                        state = cb.state_dict()
                        callback.elements.update(state)
            if type(callback).__name__ == 'EarlyStopping':
                tag = callback(val_loss)
                if tag == False:
                    # model.load_state_dict(torch.load(os.path.join(savepath, 'checkpoint.pt'))['model'])
                    # model = model.to(device)
                    # score = evaluate(config, model, test_set, savepath, epoch)
                    # writer.add_scalar('test/SI-SNRI', score, final_epoch)
                    return
            else:
                callback(val_loss)
        print('---------------------------------------------')
    resume = torch.load(os.path.join(savepath, 'checkpoint.pt'))
    model.load_state_dict(resume['model'])
    model = model.to(device)
    score = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SNRI', score, resume['epoch'])
    

if __name__ == '__main__':
    main(get_args())
