from argparse import ArgumentError
import argparse
from audioop import bias
import os
from turtle import forward
from unicodedata import bidirectional
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

from data_utils import LibriMix
from utils import makedir, get_device, no_distance_models
from callbacks import EarlyStopping, Checkpoint
from evals import evaluate
from torchaudio.models import ConvTasNet
from models import ConvBlock
import models



# class dereverb_module(torch.nn.Module):
#     def __init__(self, input_channel, hidden_channel, kernel_size, layer_num=3) -> None:
#         super().__init__()
#         self.hidden_channel = hidden_channel
#         self.kernel_size = kernel_size
#         self.layer_num = layer_num

#         self.conv1d = torch.nn.Conv1d(512, self.hidden_channel, 1, bias=False)
#         self.prelu = torch.nn.PReLU()
#         # self.convs = torch.nn.Sequential(
#         #     torch.nn.Conv1d(512, self.hidden_channel, 3, 2),
#         #     torch.nn.PReLU(),
#         #     torch.nn.GroupNorm(1, self.hidden_channel),
#         #     torch.nn.Conv1d(self.hidden_channel, self.hidden_channel, 3, 2),
#         #     torch.nn.PReLU(),
#         #     torch.nn.GroupNorm(1, self.hidden_channel),
#         #     torch.nn.Conv1d(self.hidden_channel, self.hidden_channel, 3, 2),
#         #     torch.nn.PReLU(),
#         #     torch.nn.GroupNorm(1, self.hidden_channel),
#         #     torch.nn.Conv1d(self.hidden_channel, self.hidden_channel, 3, 2),
#         #     torch.nn.PReLU(),
#         #     torch.nn.GroupNorm(1, self.hidden_channel),
#         #     torch.nn.Conv1d(self.hidden_channel, self.hidden_channel, 3, 2),
#         #     torch.nn.PReLU(),
#         #     torch.nn.GroupNorm(1, self.hidden_channel)
#         # )
#         self.filter = torch.nn.GRU(self.hidden_channel, 512, num_layers=1, batch_first=True, dropout=0.1, bidirectional=True)
#         # self.chan = torch.nn.Conv1d(1, 512, 1)

#     def pad_for_dereverb_module(self, signal, filter_length):
#         return F.pad(signal, (filter_length - 1, 0))

#     def forward(self, input: torch.Tensor):
#         time = input.shape[-1]

#         # outputslice = self.pad_for_dereverb_module(output, layer.kernel_size[-1] * layer.dilation[-1])
#         # input = input.flip(-1).unsqueeze(1)
#         # out = self.convs(input)
#         out = input
#         out = self.conv1d(out)
#         out = self.prelu(out)
#         out, _ = self.filter(out.transpose(-2,-1))
#         out = out.transpose(-2,-1)
#         out = out[:,:out.shape[1]//2] + out[:,:out.shape[1]//2] # dimension reduction with summation


#         # input = self.pad_for_dereverb_module(input, out.shape[-1])
#         # out = F.conv2d(input.unsqueeze(0), out.unsqueeze(1), groups=input.shape[0]).squeeze(0).flip(-1)
#         # out = self.chan(out)
#         return out 
#         # return (out.flip(-1) * torch.softmax(out, 1)).sum(1)


def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    scores = []
    output_scores = []
    input_scores = []
    rev_losses = []
    sep_losses = []
    MSE = torch.nn.MSELoss()
    mses = []

    num = 0
    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            if config.model == no_distance_models:
                mix, clean = inputs
            else:
                mix, clean, distance = inputs
                distance = distance.to(device)
            rev_sep = mix.to(device).transpose(1,2)
            clean_sep = clean.to(device).transpose(1,2)
            if config.test:
                mix = rev_sep[:,0]
                clean_sep = clean_sep[:,:1]
            else:
                mix = rev_sep.sum(1)

            if config.norm:
                mix_std = mix.std(-1, keepdim=True)
                mix_mean = mix.mean(-1, keepdim=True)
                mix = (mix - mix_mean) / mix_std
                mix_std = mix_std
                mix_mean = mix_mean
            outs = model(mix)
            if config.test:
                logits = outs
            else:
                logits, sep_logits = outs
            

            if config.norm:
                mix = mix * mix_std + mix_mean
                logits = logits * mix_std[:,None] + mix_mean[:,None]
                if not config.test:
                    sep_logits = sep_logits * mix_std[:,None] + mix_mean[:,None]
            rev_loss = criterion(logits, clean_sep)
            if not config.test:
                sep_loss = criterion(sep_logits, rev_sep)

            if torch.isnan(rev_loss).sum() != 0:
                print('nan is detected')
                exit()

            loss = rev_loss
            if not config.test:
                loss += sep_loss

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                optimizer.step()
            losses.append(loss.item())
            if isinstance(rev_loss.tolist(), float):
                rev_loss = rev_loss.unsqueeze(0)
            if not config.test and isinstance(sep_loss.tolist(), float):
                sep_loss = sep_loss.unsqueeze(0)
            rev_losses += rev_loss.tolist()
            if not config.test:
                sep_losses += sep_loss.tolist()
            mse_score = MSE(logits, clean_sep)
            mses.append(mse_score.item())

            progress_bar_dict = {'mode': mode, 'loss': np.mean(losses), 'rev_loss': np.mean(rev_losses)}
            if not config.test:
                progress_bar_dict['sep_loss'] = np.mean(sep_losses)
            progress_bar_dict['mse_score'] = np.mean(mses)

            output_score = - metric(logits, clean_sep).squeeze()
            if config.test:
                mixcat = mix.unsqueeze(1)
            else:
                mixcat = torch.stack([mix, mix], 1)
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
    if not config.test:
        writer.add_scalar(f'{mode}/sep_loss', np.mean(sep_losses), epoch)
    writer.add_scalar(f'{mode}/mse_score', np.mean(mses), epoch)
    if mode == 'train':
        return np.mean(losses)
    else:
        writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
        writer.add_scalar(f'{mode}/MSE', np.mean(mses), epoch)
        return np.mean(losses), np.mean(scores)


def get_model(config):
    if config.model == '':
        model = torchaudio.models.ConvTasNet(msk_activate='relu')
    elif 'dprnn' in config.model:
        modelname = 'Dereverb_DPRNNTasNet'
        if len(config.model.split('_')) > 2:
            modelname += '_' + config.model.split('_')[-1]
            model = getattr(models, modelname)(config, sample_rate=config.sr)
        else:
            model = DPRNNTasNet(config.speechnum, sample_rate=config.sr)
    return model


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    gpu_num = torch.cuda.device_count()
    config.batch *= max(gpu_num, 1)

    # v1: gru
    if config.model not in ('dprnn_v1'):
        raise ValueError('model must be dprnn_v1')
    config.model = 'Dereverb_' + config.model
    name = 'derev_' + (config.model if config.model is not '' else 'baseline')
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
    if config.test:
        name += '_test'
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
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=1, verbose=True)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='max', verbose=True))
    metric = PITLossWrapper(pairwise_neg_sisdr)
    def mseloss():
        def _mseloss(logit, answer):
            return MSELoss(reduction='none')(logit, answer)
        return _mseloss
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
                    # resume = torch.load(os.path.join(savepath, 'best.pt'))
                    # model.load_state_dict(resume['model'])
                    # model = model.to(device)
                    # si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
                    # writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
                    # writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
                    return
            else:
                callback(results)
        print('---------------------------------------------')
    # resume = torch.load(os.path.join(savepath, 'best.pt'))
    # model.load_state_dict(resume['model'])
    # model = model.to(device)
    # si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    # writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    # writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true')
    main(get_args(args))
