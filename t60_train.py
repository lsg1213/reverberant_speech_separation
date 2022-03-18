from argparse import ArgumentError
import argparse
from ast import Lambda
import os
from asteroid import DPRNNTasNet

import torchaudio

from args import get_args
from multiprocessing import cpu_count
import json

import torch
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch
torch.manual_seed(3000)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(3000)

from data_utils import LibriMix
from utils import makedir, get_device, no_distance_models
from callbacks import EarlyStopping, Checkpoint
from evals import evaluate
from torchaudio.models import ConvTasNet
from models import ConvBlock
import models
import joblib


class newPITLossWrapper(PITLossWrapper):
    def __init__(self, loss_func, pit_from="pw_mtx", perm_reduce=None, reduction=True):
        super(newPITLossWrapper, self).__init__(loss_func, pit_from, perm_reduce)
        self.reduction = reduction

    def forward(self, est_targets, targets, return_est=False, reduce_kwargs=None, **kwargs):
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == "pw_mtx":
            # Loss function already returns pairwise losses
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == "pw_pt":
            # Compute pairwise losses with a for loop.
            pw_losses = self.get_pw_losses(self.loss_func, est_targets, targets, **kwargs)
        elif self.pit_from == "perm_avg":
            # Cannot get pairwise losses from this type of loss.
            # Find best permutation directly.
            min_loss, batch_indices = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            # Take the mean over the batch
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, batch_indices)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, (
            "Something went wrong with the loss " "function, please read the docs."
        )
        assert pw_losses.shape[0] == targets.shape[0], "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, batch_indices = self.find_best_perm(
            pw_losses, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        if self.reduction:
            mean_loss = torch.mean(min_loss)
        else:
            mean_loss = min_loss
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, batch_indices)
        return mean_loss, reordered


def makelambda(name):
    def getlambda2(val):
        if not isinstance(val, torch.Tensor):
            val = torch.stack(val)
        return torch.e ** (val / 10)

    def getlambda3(val):
        if not isinstance(val, torch.Tensor):
            val = torch.stack(val)
        val = torch.e ** (1 - val / 10)
        val = val / (1 + val)
        return val

    if 'lambdaloss2' in name or 'lambda2' in name or 'lambda1' in name:
        return getlambda2
    elif 'lambdaloss3' in name:
        return getlambda3


def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    scores = []
    output_scores = []
    input_scores = []
    rev_losses = []
    clean_losses = []
    tmp = {}
    if 'lambdaloss2' in config.name or 'lambda2' in config.name or 'lambdaloss3' in config.name: # 조정 후
        meanstd = joblib.load('mean_std2.joblib')
    elif 'lambdaloss1' in config.name or 'lambda1' in config.name: # 조정 전
        meanstd = joblib.load('mean_std1.joblib')
    for i in meanstd:
        if int(i * 1000) not in tmp:
            tmp[int(i * 1000)] = {}
        for j in meanstd[i]:
            tmp[int(i * 1000)][j] = meanstd[i][j].to(device)
    meanstd = tmp
    calculate_lambda = makelambda(config.name)

    num = 0
    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            progress_bar_dict = {}
            if config.model == no_distance_models:
                mix, clean = inputs
            else:
                mix, clean, distance, t60 = inputs
                distance = distance.to(device)
                t60 = t60.to(device)
            rev_sep = mix.to(device).transpose(1,2)
            clean_sep = clean.to(device).transpose(1,2)
            cleanmix = clean_sep.sum(1)
            mix = rev_sep.sum(1)

            if 'lambda' in config.name:
                lambda_val = []
                time = torch.tensor(list(meanstd.keys())).unsqueeze(0).to(device)
                for i in time.squeeze()[torch.argmin(torch.abs(time - torch.round(t60 * 1000).int().unsqueeze(-1)), -1)].tolist():
                    lambda_val.append(torch.normal(meanstd[i]['mean'], meanstd[i]['std']))
                lambda_val = calculate_lambda(torch.stack(lambda_val))
            else:
                lambda_val = t60


            mix_std = mix.std(-1, keepdim=True)
            mix_mean = mix.mean(-1, keepdim=True)
            logits = model((mix - mix_mean) / mix_std, t60=lambda_val)
            logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
            
            if 'lambdaloss' in config.name:
                rev_loss = criterion(logits, clean_sep)
                cleanmix_mean = cleanmix.mean(-1, keepdim=True)
                cleanmix_std = cleanmix.std(-1, keepdim=True)
                clean_lambda_val = calculate_lambda(- criterion(cleanmix.unsqueeze(1).repeat((1,2,1)), clean_sep))
                clean_logits = model((cleanmix - cleanmix_mean) / cleanmix_std, t60=clean_lambda_val) 
                clean_logits = clean_logits * cleanmix_std.unsqueeze(1) + cleanmix_mean.unsqueeze(1)
                clean = criterion(clean_logits, clean_sep).mean()
                clean_losses.append(clean.item())
                
                loss = (rev_loss * lambda_val).mean() + clean
            elif 'lambda' in config.name:
                rev_loss = criterion(logits, clean_sep)
                loss = (rev_loss).mean()
            else:
                rev_loss = criterion(logits, clean_sep)
                loss = rev_loss
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                optimizer.step()

            if config.recursive or config.recursive2:
                logits = logits.detach()
                logits.requires_grad_(True)
                if config.recursive:
                    inputs = [mix, logits.sum(1)]
                elif config.recursive2:
                    inputs = [logits.sum(1)]
                for i in range(1, config.iternum):
                    mix = torch.stack(inputs).mean(0)
                    lambda_val = calculate_lambda(- criterion(mix.clone().detach().unsqueeze(1).repeat((1,2,1)), clean_sep))
                    mix_std = mix.std(-1, keepdim=True)
                    mix_mean = mix.mean(-1, keepdim=True)
                    logits = model((mix - mix_mean) / mix_std, t60=lambda_val)
                    logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
                    
                    if 'lambdaloss' in config.name:
                        rev_loss = criterion(logits, clean_sep)
                        clean_lambda_val = calculate_lambda(- criterion(cleanmix.unsqueeze(1).repeat((1,2,1)), clean_sep))
                        clean_logits = model((cleanmix - cleanmix_mean) / cleanmix_std, t60=clean_lambda_val) 
                        clean_logits = clean_logits * cleanmix_std.unsqueeze(1) + cleanmix_mean.unsqueeze(1)
                        clean = criterion(clean_logits, clean_sep).mean()
                        clean_losses.append(clean.item())
                        progress_bar_dict.update({'clean_loss': np.mean(clean_losses)})
                        loss = (rev_loss * lambda_val).mean() + clean
                    elif 'lambda' in config.name:
                        rev_loss = criterion(logits, clean_sep)
                        loss = (rev_loss).mean()
                    else:
                        rev_loss = criterion(logits, clean_sep)
                        loss = rev_loss
                    
                    if config.recursive:
                        logits = logits.detach()
                        logits.requires_grad_(True)
                        inputs.append(logits.clone().sum(1))
                    elif config.recursive2:
                        inputs = [logits.clone().sum(1)]
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                        optimizer.step()

            if torch.isnan(rev_loss).sum() != 0:
                print('nan is detected')
                exit()
            
            losses.append(loss.item())
            if isinstance(rev_loss.tolist(), float):
                rev_loss = rev_loss.unsqueeze(0)
            rev_losses += rev_loss.tolist()

            progress_bar_dict.update({'mode': mode, 'loss': np.mean(losses), 'rev_loss': np.mean(rev_losses)})
            if len(clean_losses) != 0:
                progress_bar_dict.update({'clean_loss': np.mean(clean_losses)})

            mixcat = rev_sep.sum(1, keepdim=True).repeat((1,2,1))
            output_score = - metric(logits, clean_sep).squeeze()
            input_score = - metric(mixcat, clean_sep).squeeze()
            if isinstance(output_score.tolist(), float):
                output_score = output_score.unsqueeze(0)
                input_score = input_score.unsqueeze(0)
            output_scores += output_score.tolist()
            input_scores += input_score.tolist()
            scores += (output_score - input_score).tolist()
            
            progress_bar_dict['input_SI_SNR'] = np.mean(input_scores)
            progress_bar_dict['out_SI_SNR'] = np.mean(output_scores)
            progress_bar_dict['SI_SNRI'] = np.mean(scores)

            pbar.set_postfix(progress_bar_dict)
            if mode == 'val' and (num == 1 or num == 10):
                sample_dir = f'sample/{config.name}'
                makedir(sample_dir)
                torchaudio.save(os.path.join(sample_dir, f'{num}_rev.wav'), mix[0,None].cpu(), 8000)
                torchaudio.save(os.path.join(sample_dir, f'{num}_clean_1.wav'), clean_sep[0,0,None].cpu(), 8000)
                torchaudio.save(os.path.join(sample_dir, f'{num}_result_1.wav'), logits[0,0,None].cpu(), 8000)
                if not config.test:
                    torchaudio.save(os.path.join(sample_dir, f'{num}_clean_2.wav'), clean_sep[0,1,None].cpu(), 8000)
                    torchaudio.save(os.path.join(sample_dir, f'{num}_result_2.wav'), logits[0,1,None].cpu(), 8000)
            num += 1


    writer.add_scalar(f'{mode}/loss', np.mean(losses), epoch)
    writer.add_scalar(f'{mode}/rev_loss', np.mean(rev_losses), epoch)
    if 'lambdaloss' in config.name:
        writer.add_scalar(f'{mode}/clean_loss', np.mean(clean_losses), epoch)
    if mode == 'train':
        return np.mean(losses)
    else:
        writer.add_scalar(f'{mode}/output_SI-SNR', np.mean(output_scores), epoch)
        writer.add_scalar(f'{mode}/input_SI-SNR', np.mean(input_scores), epoch)
        writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
        return np.mean(losses), np.mean(scores)


def get_model(config):
    splited_name = config.model.split('_')
    if config.model == '':
        model = torchaudio.models.ConvTasNet(msk_activate='relu')
    elif 'dprnn' in config.model:
        modelname = 'T60_DPRNNTasNet'
        if len(splited_name) > 2:
            modelname += '_' + splited_name[-1]
            model = getattr(models, modelname)(config, sample_rate=config.sr)
        else:
            model = DPRNNTasNet(config.speechnum, sample_rate=config.sr)
    elif 'tas' in config.model:
        modelname = 'T60_TasNet'
        if len(splited_name) > 2:
            modelname += '_' + splited_name[-1]
            model = getattr(models, modelname)(config)
    else:
        modelname = 'T60_ConvTasNet_' + splited_name[-1]
        model = getattr(models, modelname)(config)
        if config.recursive or config.recursive2:
            resume = torch.load('save/t60_T60_v1_16_rir_norm_sisdr_lambda2/best.pt')['model']
            model.load_state_dict(resume)
    return model


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    gpu_num = torch.cuda.device_count()
    config.batch *= max(gpu_num, 1)

    # v1: gru
    config.model = 'T60_' + config.model
    name = 't60_' + (config.model if config.model is not '' else 'baseline')
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
    if config.recursive:
        name += f'_recursive_{config.iternum}'
    if config.recursive2:
        name += f'_recursive2_{config.iternum}'
    config.name = name + '_' + config.name if config.name is not '' else ''
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    writer = SummaryWriter(config.tensorboard_path)
    savepath = os.path.join('save', config.name)
    device = get_device()
    makedir(config.tensorboard_path)
    makedir(savepath)

    init_epoch = 0
    final_epoch = 0
    
    train_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/train-360'),
        config=config,
        task=config.task[:-1],
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )

    val_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/dev'),
        config=config,
        task=config.task[:-1],
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=config.segment,
    )

    test_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/test'),
        config=config,
        task=config.task[:-1],
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
        batch_size=config.batch,
        num_workers=gpu_num * (cpu_count() // 4),
    )

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    if config.recursive or config.recursive2:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='max', verbose=True))
    metric = PITLossWrapper(pairwise_neg_sisdr)
    if 'lambda' in config.name:
        criterion = newPITLossWrapper(pairwise_neg_sisdr, reduction=False)
    else:
        criterion = PITLossWrapper(pairwise_neg_sisdr)
    if 'mse' in config.name:
        criterion = MSELoss()
    
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

        scheduler.step(val_score)
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
    model = model.to(device)
    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true')
    args.add_argument('--recursive', action='store_true')
    args.add_argument('--recursive2', action='store_true')
    args.add_argument('--t60', type=bool, default=True)
    args.add_argument('--iternum', type=int, default=2)
    main(get_args(args))
