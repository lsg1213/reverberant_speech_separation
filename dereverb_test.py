from argparse import ArgumentError
import argparse
from audioop import bias
import os
from turtle import forward
from unicodedata import bidirectional

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
from model import ConvBlock


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.
        msk_activate (str): The activation function of the mask output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
        msk_activate: str,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_sources = num_sources

        self.input_norm = torch.nn.GroupNorm(
            num_groups=1, num_channels=input_dim, eps=1e-8
        )
        self.input_conv = torch.nn.Conv1d(
            in_channels=input_dim, out_channels=num_feats, kernel_size=1
        )

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += (
                    kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
                )
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats, out_channels=output_dim * num_sources, kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.output_dim, -1)


class Lfilter(torch.nn.Module):
    def __init__(self, config, input_channel, hidden_channel, kernel_size) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.config = config

        if 'v2' not in self.config.model:
            alpha = torch.rand((self.hidden_channel, 1), requires_grad=True)
            beta = torch.rand((self.hidden_channel, 1), requires_grad=True)
            self.kernel = []
            for i in range(self.hidden_channel):
                kernel = []
                for j in range(1):
                    kernel.append(alpha[i, j] * torch.arange(0, beta[i, j].item(), beta[i, j].item() / self.kernel_size, dtype=torch.float32, requires_grad=True)[:self.kernel_size])
                self.kernel.append(torch.stack(kernel, 0))
            self.kernel = torch.nn.Parameter(torch.stack(self.kernel, 0))
            self.kernel_weight = torch.nn.Parameter(torch.rand((self.hidden_channel, 1, self.kernel_size), requires_grad=True))
            self.prelu = torch.nn.PReLU()

        if 'v1' not in self.config.model:
            self.gru = torch.nn.GRU(self.input_channel, self.input_channel, batch_first=True, bidirectional=True)
        
    def pad_for_lfilter(self, signal, filter_length):
        return F.pad(signal, (filter_length - 1, 0))
            
    def forward(self, input: torch.Tensor):
        kernel = self.kernel * self.kernel_weight

        if 'v2' in self.config.model:
            out = input
        else:
            batch, feat, time = input.shape
            input = self.pad_for_lfilter(input, self.kernel_size)
            pad_time = input.shape[-1]
            input = input.reshape([-1, pad_time])
            out = F.conv1d(input.unsqueeze(1).flip(-1), kernel).flip(-1)
            out = out.sum(1).reshape((batch, feat, time))
            out = self.prelu(out)
        if 'v1' not in self.config.model:
            out = self.gru(out.transpose(-2, -1))[0].transpose(-2, -1)
            out = out[:, :out.shape[1] // 2] + out[:, out.shape[1] // 2:]
        return out


# class Lfilter(torch.nn.Module):
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

#     def pad_for_lfilter(self, signal, filter_length):
#         return F.pad(signal, (filter_length - 1, 0))

#     def forward(self, input: torch.Tensor):
#         time = input.shape[-1]

#         # outputslice = self.pad_for_lfilter(output, layer.kernel_size[-1] * layer.dilation[-1])
#         # input = input.flip(-1).unsqueeze(1)
#         # out = self.convs(input)
#         out = input
#         out = self.conv1d(out)
#         out = self.prelu(out)
#         out, _ = self.filter(out.transpose(-2,-1))
#         out = out.transpose(-2,-1)
#         out = out[:,:out.shape[1]//2] + out[:,:out.shape[1]//2] # dimension reduction with summation


#         # input = self.pad_for_lfilter(input, out.shape[-1])
#         # out = F.conv2d(input.unsqueeze(0), out.unsqueeze(1), groups=input.shape[0]).squeeze(0).flip(-1)
#         # out = self.chan(out)
#         return out 
#         # return (out.flip(-1) * torch.softmax(out, 1)).sum(1)


class DereverbModule(ConvTasNet):
    def __init__(
        self,
        config,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
    ):
        super(DereverbModule, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
        self.config = config
        self.num_sources = 1 if self.config.test else num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats, # embedding
            output_dim = enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )

        # 8000(sampling rate) * 0.5(T60 maximum value) / 8(encoded feature resolution)
        rir_func_length = np.ceil(8000 * 0.1 / 8).astype(np.int32)
        self.lfilter = Lfilter(config, enc_num_feats, 64, rir_func_length)

        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(1)
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M

        # separation module
        if not self.config.test:
            masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
            separated_feat = masked.view(
                batch_size * self.num_sources, self.enc_num_feats, -1 # + int(distance is not None)
            )  # B*S, F, M
            separated_feat = self.decoder(separated_feat)
            separated_feat = separated_feat.view(
                batch_size, self.num_sources, num_padded_frames
            )
            feats = separated_feat

        # dereberberation module
        dereverb_feats = self.lfilter(feats)

        masked = self.decoder(dereverb_feats)  # B*S, 1, L'
        output = masked.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
            if self.config.test:
                separated_feat = separated_feat[..., :-num_pads]
        if self.config.test:
            return output
        else:
            return output, separated_feat


def iterloop(config, writer, epoch, model: DereverbModule, criterion, dataloader, metric, optimizer=None, mode='train'):
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
                torchaudio.save(f'sample/{config.name}_sample{num}_rev.wav', mix[0,None].cpu(), 8000)
                torchaudio.save(f'sample/{config.name}_sample{num}_clean_1.wav', clean_sep[0,0,None].cpu(), 8000)
                torchaudio.save(f'sample/{config.name}_sample{num}_result_1.wav', logits[0,0,None].cpu(), 8000)
                if not config.test:
                    torchaudio.save(f'sample/{config.name}_sample{num}_clean_2.wav', clean_sep[0,1,None].cpu(), 8000)
                    torchaudio.save(f'sample/{config.name}_sample{num}_result_2.wav', logits[0,1,None].cpu(), 8000)
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


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    gpu_num = torch.cuda.device_count()
    config.batch *= max(gpu_num, 1)

    # v1: lfilter
    # v2: gru
    # v3: lfilter+gru
    if config.model not in ('v1','v2','v3'):
        raise ValueError('model must be v1, v2, v3')
    config.model = 'derev_' + config.model
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

    model = DereverbModule(config)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, verbose=True)

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
