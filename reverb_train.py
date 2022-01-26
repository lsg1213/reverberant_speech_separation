from argparse import ArgumentError
import os

from torch.nn.modules.loss import MSELoss

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
from model import ConvTasNet_v1, ConvTasNet_v2, ConvTasNet_v3


def minmaxnorm(data):
    ndim = data.ndim
    mindata = data.view(data.shape[0],-1).min(-1, keepdim=True)[0]
    maxdata = data.view(data.shape[0],-1).max(-1, keepdim=True)[0]
    
    while not (mindata.ndim == maxdata.ndim == ndim):
        mindata = mindata.unsqueeze(1)
        maxdata = maxdata.unsqueeze(1)
    data = (2 * (data - mindata) / (maxdata - mindata)) - 1.
    return data



def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    scores = []
    rev_losses = []
    clean_losses = []
    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            if config.model == '':
                mix, clean = inputs
            else:
                mix, clean, distance = inputs
                distance = distance.to(device)
            rev_sep = mix.to(device).transpose(1,2)
            clean_sep = clean.to(device).transpose(1,2)
            mix = rev_sep.sum(1)
            cleanmix = clean_sep.sum(1)

            if config.norm:
                mix_std = mix.std(-1, keepdim=True)
                mix_mean = mix.mean(-1, keepdim=True)
                clean_std = cleanmix.std(-1, keepdim=True)
                clean_mean = cleanmix.mean(-1, keepdim=True)
                mix = (mix - mix_mean) / mix_std
                cleanmix = (cleanmix - clean_mean) / clean_std
                mix_std = mix_std.unsqueeze(1)
                mix_mean = mix_mean.unsqueeze(1)
                clean_std = clean_std.unsqueeze(1)
                clean_mean = clean_mean.unsqueeze(1)

            if config.model == '':
                logits = model(mix)
                clean_logits = model(cleanmix)
            else:
                logits = model(mix, distance)
                clean_logits = model(cleanmix, torch.zeros_like(distance))
            if config.norm:
                logits = logits * mix_std + mix_mean
                clean_logits = clean_logits * clean_std + clean_mean
            rev_loss = criterion(logits, clean_sep)
            clean_loss = criterion(clean_logits, clean_sep)
            loss = rev_loss + clean_loss

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                optimizer.step()
            losses.append(loss.item())
            rev_losses.append(rev_loss.item())
            clean_losses.append(clean_loss.item())
            progress_bar_dict = {'mode': mode, 'loss': np.mean(losses), 'rev_loss': np.mean(rev_losses), 'clean_loss': np.mean(clean_losses)}

            writer.add_scalar(f'{mode}/loss', np.mean(losses), epoch)
            writer.add_scalar(f'{mode}/rev_loss', np.mean(rev_losses), epoch)
            writer.add_scalar(f'{mode}/clean_loss', np.mean(clean_losses), epoch)
            if mode == 'val':
                input_score = - metric(torch.stack([mix, mix], 1), clean_sep)
                output_score = - metric(logits, clean_sep)
                score = output_score - input_score
                scores.append(score.tolist())
                progress_bar_dict['input_score'] = np.mean(input_score.tolist())
                progress_bar_dict['output_score'] = np.mean(output_score.tolist())
                progress_bar_dict['score'] = np.mean(scores)
                writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
                writer.add_scalar(f'{mode}/input_SI-SNR', np.mean(input_score.tolist()), epoch)
                writer.add_scalar(f'{mode}/output_SI-SNR', np.mean(output_score.tolist()), epoch)
            pbar.set_postfix(progress_bar_dict)
    if mode == 'train':
        return np.mean(losses)
    else:
        return np.mean(losses), np.mean(scores)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    name = 'reverb_' + (config.model if config.model is not '' else 'baseline')
    name += f'_{config.batch}'
    if config.model != '' and config.task == '':
        raise ArgumentError('clean separation model should be baseline model')
    if 'rir' not in config.task:
        config.task = ''
        name += '_clean'
    else:
        name += '_' + config.task
        config.task += '_'
    if config.norm:
        name += '_norm'
    config.name = name + '_' + config.name if config.name is not '' else ''
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    writer = SummaryWriter(config.tensorboard_path)
    savepath = os.path.join('save', config.name)
    device = get_device()
    makedir(config.tensorboard_path)
    makedir(savepath)

    init_epoch = 0
    final_epoch = 0
    
    gpu_num = torch.cuda.device_count()
    train_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/train-360'),
        config=config,
        task=config.task + 'sep_clean',
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )

    val_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/dev'),
        config=config,
        task=config.task + 'sep_clean',
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )
    
    test_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/test'),
        config=config,
        task=config.task + 'sep_clean',
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

    if config.model == '':
        model = torchaudio.models.ConvTasNet(msk_activate='relu')
    elif config.model == 'v1':
        model = ConvTasNet_v1()
    elif config.model == 'v2':
        model = ConvTasNet_v2(reverse='reverse' in config.name)
    elif config.model == 'v3':
        model = ConvTasNet_v3(reverse='reverse' in config.name)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, verbose=True)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='max', verbose=True))
    metric = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    def mseloss():
        def _mseloss(logit, answer):
            return MSELoss(reduction='none')(logit, answer).mean(-1, keepdim=True)
        return _mseloss
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    if 'mse' in config.name:
        criterion = PITLossWrapper(mseloss(), pit_from="pw_mtx")
    
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
        train_loss = iterloop(config, writer, epoch, model, criterion, train_loader, metric, optimizer=optimizer, mode='train')

        model.eval()
        with torch.no_grad():
            val_loss, val_score = iterloop(config, writer, epoch, model, criterion, val_loader, metric, mode='val')

        results = {'train_loss': train_loss, 'val_loss': val_loss, 'val_score': val_score}

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
                        callback.elements[type(cb).__name__] = state
            if type(callback).__name__ == 'EarlyStopping':
                tag = callback(results)
                if tag == False:
                    resume = torch.load(os.path.join(savepath, 'best.pt'))
                    model.load_state_dict(resume['model'])
                    model = model.to(device)
                    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
                    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
                    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
                    return
            else:
                callback(results)
        print('---------------------------------------------')
    resume = torch.load(os.path.join(savepath, 'best.pt'))
    model.load_state_dict(resume['model'])
    model = model.to(device)
    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    main(get_args())
