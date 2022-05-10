from argparse import ArgumentError
import os

from args import get_args
from multiprocessing import cpu_count
import json

import torch
import numpy as np
torch.manual_seed(3000)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(3000)
from tensorboardX import SummaryWriter
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm
import speechbrain as sb

from data_utils import LibriMix
from utils import makedir, get_device, clip_grad_norm_
from callbacks import EarlyStopping, Checkpoint
from evals import evaluate
from models import *
from t60_utils import newPITLossWrapper


def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    if 'clean' in config.name:
        clean_losses = []
        rev_losses = []

    scores = []
    input_scores = []
    output_scores = []
    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            mix, clean = inputs
            rev_sep = mix.to(device).transpose(1,2)
            clean_sep = clean.to(device).transpose(1,2)
            mix = rev_sep.sum(1)
            cleanmix = clean_sep.sum(1)

            mix_std = mix.std(-1, keepdim=True)
            mix_mean = mix.mean(-1, keepdim=True)
            logits = model((mix - mix_mean) / mix_std)
            logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
            rev_loss = criterion(logits, clean_sep)

            if 'clean' in config.name:
                clean_std = cleanmix.std(-1, keepdim=True)
                clean_mean = cleanmix.mean(-1, keepdim=True)
                clean_logits = model((cleanmix - clean_mean) / clean_std)
                clean_logits = clean_logits * clean_std.unsqueeze(1) + clean_mean.unsqueeze(1)
                clean_loss = criterion(clean_logits, clean_sep)
                loss = rev_loss + clean_loss
            else:
                loss = rev_loss
            
            if torch.isnan(rev_loss).sum() != 0:
                print('nan is detected')
                exit()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), config.clip_val, error_if_nonfinite=True)
                optimizer.step()
            losses.append(loss.item())
            progress_bar_dict = {'mode': mode, 'loss': np.mean(losses)}
            if 'clean' in config.name:
                rev_losses.append(rev_loss.item())
                clean_losses.append(clean_loss.item())
                progress_bar_dict['rev_loss'] = np.mean(rev_losses)
                progress_bar_dict['clean_loss'] = np.mean(clean_losses)

            if mode == 'val':
                mixcat = rev_sep.sum(1, keepdim=True).repeat((1,2,1))
                input_score = - metric(mixcat, clean_sep).squeeze()
                output_score = - metric(logits, clean_sep).squeeze()
                score = output_score - input_score
                scores += score.tolist()
                input_scores += input_score.tolist()
                output_scores += output_score.tolist()
                progress_bar_dict['input_score'] = np.mean(input_scores)
                progress_bar_dict['output_score'] = np.mean(output_scores)
                progress_bar_dict['score'] = np.mean(scores)
            pbar.set_postfix(progress_bar_dict)

    writer.add_scalar(f'{mode}/loss', np.mean(losses), epoch)
    if 'clean' in config.name:
        writer.add_scalar(f'{mode}/rev_loss', np.mean(rev_losses), epoch)
        writer.add_scalar(f'{mode}/clean_loss', np.mean(clean_losses), epoch)

    if mode == 'train':
        return np.mean(losses)
    else:
        writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
        writer.add_scalar(f'{mode}/input_SI-SNR', np.mean(input_scores), epoch)
        writer.add_scalar(f'{mode}/output_SI-SNR', np.mean(output_scores), epoch)
        return np.mean(losses), np.mean(scores)


def get_model(config):
    if config.model == '':
        model = ConvTasNet()
    elif config.model == 'v1':
        model = ConvTasNet(msk_activate='relu', msk_num_layers=5)
    elif config.model == 'v2':
        model = ConvTasNet_v2(reverse='reverse' in config.name)
    # elif config.model == 'v3':
    #     model = ConvTasNet_v3(reverse='reverse' in config.name)
    elif config.model == 'tas':
        model = TasNet()
    elif config.model == 'dprnn':
        model = DPRNNTasNet(config.speechnum, sample_rate=config.sr, chunk_size=100)
    elif config.model == 'sepformer':
        model = Sepformer(config)
    return model


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

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, verbose=True)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []
    optimizer = Adam(model.parameters(), lr=config.lr)
    # if 'dprnn' in config.model:
    #     scheduler = StepLR(optimizer=optimizer, step_size=2, gamma=0.98, verbose=True)
    #     callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))
    # else:
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    callbacks.append(EarlyStopping(monitor="val_score", mode="max", patience=config.max_patience, verbose=True))

    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='max', verbose=True))
    metric = newPITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", reduction=False)
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

        # if 'dprnn' in config.name:
        #     scheduler.step()
        # else:
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
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    main(get_args())
