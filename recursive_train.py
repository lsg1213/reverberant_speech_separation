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
from utils import makedir, get_device, no_distance_models
from callbacks import EarlyStopping, Checkpoint
from evals import evaluate
from models import *


def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    scores = {}
    losses = {}
    input_scores = {}
    output_scores = {}
    iternum = config.iternum
    for i in range(iternum):
        losses[f'loss{i}'] = []
        input_scores[i] = []
        output_scores[i] = []
        scores[i] = []

    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            if config.model in no_distance_models:
                mix, clean = inputs
            else:
                mix, clean, distance = inputs
                distance = distance.to(device)
            rev_sep = mix.to(device).transpose(1,2)
            clean_sep = clean.to(device).transpose(1,2)
            mix = rev_sep.sum(1)
            
            input_score = - metric(mix.unsqueeze(1).repeat((1,2,1)), clean_sep).item()

            mix_std = mix.std(-1, keepdim=True)
            mix_mean = mix.mean(-1, keepdim=True)
            mix = (mix - mix_mean) / mix_std

            for i in range(iternum):
                logits = model(mix)
                logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
                loss = criterion(logits, clean_sep)
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                    optimizer.step()

                loss = loss.item()
                losses[f'loss{i}'].append(loss)

                output_score = - loss
                input_scores[i].append((input_score))
                output_scores[i].append((output_score))
                scores[i].append((output_score - input_score))

                del mix
                mix = logits.clone().detach().sum(1)
                mix_std = mix.std(-1, keepdim=True)
                mix_mean = mix.mean(-1, keepdim=True)
                mix = (mix - mix_mean) / mix_std

            if np.isnan(losses['loss0'][-1]):
                print('nan is detected')
                exit()

            progress_bar_dict = {'mode': mode}
            
            for i in range(iternum):
                progress_bar_dict[f'loss{i}'] = np.mean(losses[f'loss{i}'])
                progress_bar_dict[f'score{i}'] = np.mean(scores[i])
            pbar.set_postfix(progress_bar_dict)
    
    for i in range(iternum):
        writer.add_scalar(f'{mode}/loss{i}', np.mean(losses[f'loss{i}']), epoch)
        writer.add_scalar(f'{mode}/scores{i}', np.mean(scores[i]), epoch)
        writer.add_scalar(f'{mode}/scores', np.mean(losses[i]), epoch)
    
    if mode == 'train':
        return np.mean(losses)
    else:
        writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
        writer.add_scalar(f'{mode}/input_SI-SNR', np.mean(input_scores), epoch)
        writer.add_scalar(f'{mode}/output_SI-SNR', np.mean(output_scores), epoch)
        return np.mean(losses), np.mean(scores[len(scores) - 1])


def get_model(config):
    if config.model == '':
        model = ConvTasNet(msk_activate='relu')
        if config.pretrain:
            pretrain = torch.load('/root/contrative_degree/save/reverb_baseline_24_rir_norm_sisdr/best.pt')['model']
            model.load_state_dict(pretrain)
    elif config.model == 'tas':
        model = TasNet()
    elif config.model == 'dprnn':
        model = DPRNNTasNet(config.speechnum, sample_rate=config.sr)
    return model


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    name = 'reverb_' + (config.model if config.model is not '' else 'baseline')
    name += f'_{config.batch}_iter{config.iternum}'
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
    if config.pretrain:
        config.name += '_pretrain'
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

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, verbose=True)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='max', verbose=True))
    metric = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
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

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
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
                if torch.cuda.device_count() > 1:
                    model_state = {}
                    for k, v in model.state_dict().items():
                        model_state[k[7:]] = v
                else:
                    model_state = model.state_dict()
                callback.elements.update({
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1
                })
                for cb in callbacks:
                    if type(cb).__name__ != 'Checkpoint':
                        state = cb.state_dict()
                        callback.elements[type(cb).__name__] = state
            if type(callback).__name__ == 'EarlyStopping':
                tag = callback(results)
                if tag == False:
                    resume = torch.load(os.path.join(savepath, 'best.pt'))
                    model = get_model(config)
                    model.load_state_dict(resume['model'])
                    if torch.cuda.device_count() > 1:
                        model = torch.nn.DataParallel(model)
                    model = model.to(device)
                    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
                    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
                    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
                    return
            else:
                callback(results)
        print('---------------------------------------------')
    resume = torch.load(os.path.join(savepath, 'best.pt'))
    model = get_model(config)
    model.load_state_dict(resume['model'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--iternum', type=int, default=3)
    args.add_argument('--pretrain', action='store_true')
    main(get_args(args))

