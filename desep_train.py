from argparse import ArgumentError
import os

from args import get_args
from multiprocessing import cpu_count
import json

import torch
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F
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


def lfilter(signal, filter):
    '''
        signal (batch, signal_num, time),
        filter (batch, filter_num, filter)
    '''
    batch = signal.shape[0]
    feat = signal.shape[1]
    filter = filter.flip(-1)
    signal = F.pad(signal, (filter.shape[-1] - 1, 0))

    signal = signal.reshape((-1, signal.shape[-1]))
    filter = filter.reshape((-1, filter.shape[-1]))
    out = F.conv1d(signal.unsqueeze(0), filter.unsqueeze(1), groups=signal.shape[0]).squeeze(0)
    out = out.reshape((batch, feat, -1))
    return out
    

class DeSepNet(ConvTasNet):
    def __init__(
        self,
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
        super(DeSepNet, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

        self.num_sources = num_sources
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
            input_dim=enc_num_feats,
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
        rir_func_length = np.ceil(8000 * 0.5 / 8).astype(np.int32)
        self.rir_function_generator = torch.nn.ModuleList([
            torch.nn.Linear(3001, 1024),
            torch.nn.Linear(1024, rir_func_length * 2),
            torch.nn.Linear(rir_func_length * 2, rir_func_length),
        ])
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, test: bool = False) -> torch.Tensor:
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

        # rir predictor
        rirfeats = feats
        for layer in self.rir_function_generator:
            rirfeats = layer(rirfeats)
        dereverb_feats = lfilter(feats, rirfeats)

        mask_in_feats = dereverb_feats
        masked = self.mask_generator(mask_in_feats) * dereverb_feats.unsqueeze(1)  # B, S, F, M

        rev_masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1 # + int(distance is not None)
        )  # B*S, F, M 
        rev_masked = rev_masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1 # + int(distance is not None)
        )  # B*S, F, M 


        masked = self.decoder(masked)  # B*S, 1, L'
        masked = masked.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        rev_masked = self.decoder(rev_masked)  # B*S, 1, L'
        rev_masked = rev_masked.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            masked - masked[..., :-num_pads]
        if test:
            return masked
        return masked, rev_masked


def minmaxnorm(data):
    ndim = data.ndim
    mindata = data.view(data.shape[0],-1).min(-1, keepdim=True)[0]
    maxdata = data.view(data.shape[0],-1).max(-1, keepdim=True)[0]
    
    while not (mindata.ndim == maxdata.ndim == ndim):
        mindata = mindata.unsqueeze(1)
        maxdata = maxdata.unsqueeze(1)
    data = (2 * (data - mindata) / (maxdata - mindata)) - 1.
    return data



def iterloop(config, writer, epoch, model: DeSepNet, criterion, dataloader, metric, optimizer=None, mode='train'):
    device = get_device()
    losses = []
    scores = []
    rev_losses = []
    sep_losses = []
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

            if config.norm:
                mix_std = mix.std(-1, keepdim=True)
                mix_mean = mix.mean(-1, keepdim=True)
                mix = (mix - mix_mean) / mix_std
                mix_std = mix_std.unsqueeze(1)
                mix_mean = mix_mean.unsqueeze(1)

            logits, rev_logits = model(mix)

            if config.norm:
                logits = logits * mix_std + mix_mean
                rev_logits = rev_logits * mix_std + mix_mean
            rev_loss = criterion(logits, clean_sep)
            sep_loss = criterion(rev_logits, rev_sep)
            loss = rev_loss + sep_loss

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_val)
                optimizer.step()
            losses.append(loss.item())
            rev_losses.append(rev_loss.item())
            sep_losses.append(sep_loss.item())
            progress_bar_dict = {'mode': mode, 'loss': np.mean(losses), 'rev_loss': np.mean(rev_losses), 'sep_loss': np.mean(sep_losses)}

            writer.add_scalar(f'{mode}/loss', np.mean(losses), epoch)
            writer.add_scalar(f'{mode}/rev_loss', np.mean(rev_losses), epoch)
            writer.add_scalar(f'{mode}/sep_loss', np.mean(sep_losses), epoch)
            if mode == 'val':
                mix = rev_sep.sum(1)
                mixcat = torch.stack([mix, mix], 1)
                input_score = - metric(mixcat, clean_sep)
                output_score = - metric(logits, clean_sep)
                score = output_score - input_score
                scores.append(score.tolist())
                progress_bar_dict['input_score'] = np.mean(input_score.tolist())
                progress_bar_dict['output_score'] = np.mean(output_score.tolist())
                progress_bar_dict['score'] = np.mean(scores)
            pbar.set_postfix(progress_bar_dict)
    writer.add_scalar(f'{mode}/SI-SNRI', np.mean(scores), epoch)
    writer.add_scalar(f'{mode}/input_SI-SNR', np.mean(input_score.tolist()), epoch)
    writer.add_scalar(f'{mode}/output_SI-SNR', np.mean(output_score.tolist()), epoch)
    if mode == 'train':
        return np.mean(losses)
    else:
        return np.mean(losses), np.mean(scores)


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    config.model = 'desep'
    name = 'desep_' + (config.model if config.model is not '' else 'baseline')
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

    model = DeSepNet()

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
            return MSELoss(reduction='none')(logit, answer)
        return _mseloss
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
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
                        model_state[k[8:]] = v
                else:
                    model_state = model.state_dict()
                callback.elements.update({
                    'model': model_state,
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
