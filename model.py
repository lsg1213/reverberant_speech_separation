from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchaudio.models.conv_tasnet import ConvBlock, ConvTasNet
from asteroid.models import DPRNNTasNet


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
            num_groups=1, num_channels=output_dim, eps=1e-8
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

    def forward(self, input: torch.Tensor, dis=None) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.input_norm(input)
        if dis is not None:
            feats = torch.cat([feats, dis], 1)
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


# encoder 이후에 나온 feature에 distance 도입
class ConvTasNet_v1(ConvTasNet):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

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
        super(ConvTasNet_v1, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

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
            input_dim=enc_num_feats * 2, # embedding
            # input_dim=enc_num_feats + (int(distance) * 47), # onehot
            # input_dim=enc_num_feats,
            output_dim = enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.emb = torch.nn.Embedding(47, 512) # 47 = 벽까지거리(최대 2m) / (343 m/s / sr(=8000))

    def forward(self, input: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
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

        if distance is not None:
            dis = distance
            dis = torch.round(dis / 0.042875).long()
            # onehot
            # dis = F.one_hot(dis, 47)
            # dis = dis.unsqueeze(-1).expand(batch_size, 47, feats.shape[-1])

            # embedding
            dis = self.emb(dis) # (batch, 512)
            dis = dis.unsqueeze(-1).expand(batch_size, dis.shape[1], feats.shape[-1])

            if (distance != torch.zeros_like(distance)).sum() == 0:
                dis = torch.zeros_like(dis)
            # mask_in_feats = torch.cat([feats, dis], -2)
            mask_in_feats = feats
        else:
            mask_in_feats = feats
            
        masked = self.mask_generator(mask_in_feats, dis) * feats.unsqueeze(1)  # B, S, F, M

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1 # + int(distance is not None)
        )  # B*S, F, M  
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output


class SlimmableConv1dTranspose(torch.nn.ConvTranspose1d):
    def __init__(self, in_channel_max, out_channel_max,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(SlimmableConv1dTranspose, self).__init__(
            in_channel_max, out_channel_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.in_channels = in_channel_max
        self.out_channels = out_channel_max
        self.groups = groups

    def forward(self, input, channel_ratio, ratio_range):
        channel_range = torch.tensor(ratio_range, dtype=self.weight.dtype, device=self.weight.device) * self.weight.shape[0]
        min_chan = channel_range[0]
        max_chan = channel_range[1]
        channel_num = ((max_chan - min_chan) * channel_ratio).long() + min_chan
        channel_num = channel_num.unsqueeze(-1).expand(channel_num.shape[0], 2).reshape(-1)
        
        def conv_transpose_1d(idx): # idx: batch index
            number = channel_num[idx].int()
            weight = self.weight[:number, ...]
            if self.bias is not None:
                bias = self.bias[:number]
            else:
                bias = self.bias
            y = F.conv_transpose1d(
                input[idx:idx+1, :number], weight, bias, self.stride, self.padding,
                dilation=self.dilation, groups=self.groups)
            return y

        y = torch.cat(list(map(conv_transpose_1d, range(input.shape[0]))), 0)
        return y


# decoder를 slimmable network로 대체
class ConvTasNet_v2(ConvTasNet):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

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
        reverse: bool = True,
    ):
        super(ConvTasNet_v2, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.reverse = reverse

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
        self.decoder = SlimmableConv1dTranspose(
            in_channel_max=enc_num_feats,
            out_channel_max=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
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

        mask_in_feats = feats
            
        masked = self.mask_generator(mask_in_feats) * feats.unsqueeze(1)  # B, S, F, M

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M  

        dis = distance
        dis = torch.round(dis / 0.042875).long()

        # distance range 0.5 ~ 2.0 m
        dis_max = torch.round(torch.ones_like(dis) * 2 / 0.042875).long()

        # channel ratio 0.5 ~ 1.0: whole channel * channel ratio => used channel
        ratio_range = [0.5, 1.]

        # decoded = self.decoder(masked, dis / dis_max, ratio_range)  # B*S, 1, L'
        if self.reverse:
            decoded = self.decoder(masked, 1. - dis / dis_max, ratio_range)  # distance가 멀수록 적게 사용, reverse
        else:
            decoded = self.decoder(masked, dis / dis_max, ratio_range)  # distance가 멀수록 많이 사용

        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output


class SlimmableSeparableConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels_max, out_channels_max,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(SlimmableSeparableConv1d, self).__init__(
            in_channels_max, out_channels_max,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.in_channels_max = in_channels_max
        self.out_channels_max = out_channels_max
        self.groups = groups # no use
        self.pointwise_weight = Parameter(torch.empty((out_channels_max, out_channels_max, 1), device=self.weight.device, dtype=self.weight.dtype))
        torch.nn.init.kaiming_uniform_(self.pointwise_weight, a=5 ** 0.5)

    def forward(self, input, channel_ratio, ratio_range):
        input_chan = input.shape[1]
        channel_range = torch.tensor(ratio_range, dtype=self.weight.dtype, device=self.weight.device) * self.weight.shape[0]
        min_chan = channel_range[0]
        max_chan = channel_range[1]
        channel_num = ((max_chan - min_chan) * channel_ratio).int() + min_chan

        def conv_1d(idx): # idx: batch index
            number = channel_num[idx].int()
            weight = self.weight[:number, :input_chan, ...]
            if self.bias is not None:
                bias = self.bias[:number]
            else:
                bias = self.bias
            y = F.conv1d(input[idx:idx+1, :number], weight, None, self.stride, self.padding, dilation=self.dilation, groups=input.shape[1]) # separable
            y = F.conv1d(y, self.pointwise_weight[:, :number], bias) # pointwise
            return y
            
        y = torch.cat(list(map(conv_1d, range(input.shape[0]))), 0)
        return y


# encoder를 slimmable network로 대체
class ConvTasNet_v3(ConvTasNet):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

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
        reverse: bool = True,
    ):
        super(ConvTasNet_v3, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        self.reverse = reverse

        self.encoder = SlimmableSeparableConv1d(
            in_channels_max=1,
            out_channels_max=enc_num_feats,
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
        self.decoder = SlimmableConv1dTranspose(
            in_channel_max=enc_num_feats,
            out_channel_max=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
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

        # distance range 0.5 ~ 2.0 m
        dis = distance
        dis = torch.round(dis / 0.042875).long()
        dis_max = torch.round(torch.ones_like(dis) * 2 / 0.042875).long()
        dis_ratio = 1 - dis / dis_max if self.reverse else dis / dis_max

        # channel ratio 0.5 ~ 1.0: whole channel * channel ratio => used channel
        ratio_range = [0.5, 1.]

        feats = self.encoder(padded, 1 - dis / dis_max, ratio_range)  # B, F, M

        mask_in_feats = feats
            
        masked = self.mask_generator(mask_in_feats) * feats.unsqueeze(1)  # B, S, F, M

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M  


        decoded = self.decoder(masked, 1 - dis / dis_max, ratio_range)  # B*S, 1, L'
        # decoded = self.decoder(masked)

        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output


class ConvTasNet_feedback(ConvTasNet):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network
    *Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation*
    [:footcite:`Luo_2019`].

    Args:
        num_sources (int, optional): The number of sources to split.
        enc_kernel_size (int, optional): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int, optional): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int, optional): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int, optional): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int, optional): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int, optional): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int, optional): The numbr of conv blocks of the mask generator, <R>.
        msk_activate (str, optional): The activation function of the mask output (Default: ``sigmoid``).

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.
    """

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
        super(ConvTasNet_feedback, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

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
        
        mask_in_feats = feats
            
        masked = self.mask_generator(mask_in_feats) * feats.unsqueeze(1)  # B, S, F, M

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1 # + int(distance is not None)
        )  # B*S, F, M  
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output


import torch
from torch import nn

from asteroid import torch_utils
from asteroid import torch_utils
from asteroid_filterbanks import Encoder, Decoder, FreeFB
from asteroid.masknn.recurrent import SingleRNN
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn.norms import GlobLN


class TasNet(nn.Module):
    """Some kind of TasNet, but not the original one
    Differences:
        - Overlap-add support (strided convolutions)
        - No frame-wise normalization on the wavs
        - GlobLN as bottleneck layer.
        - No skip connection.
    Args:
        fb_conf (dict): see local/conf.yml
        mask_conf (dict): see local/conf.yml
    """

    def __init__(self):
        super().__init__()
        self.n_src = 2
        self.n_filters = 512
        # Create TasNet encoders and decoders (could use nn.Conv1D as well)
        self.encoder_sig = Encoder(FreeFB(512, 40, 20))
        self.encoder_relu = Encoder(FreeFB(512, 40, 20))
        self.decoder = Decoder(FreeFB(512, 40, 20))
        self.bn_layer = GlobLN(512)

        # Create TasNet masker
        self.masker = nn.Sequential(
            SingleRNN(
                "lstm",
                512,
                hidden_size=600,
                n_layers=4,
                bidirectional=True,
                dropout=0.3,
            ),
            nn.Linear(2 * 600, self.n_src * self.n_filters),
            nn.Sigmoid(),
        )

    def forward(self, x, distance):
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encode(x)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks
        return torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


if __name__ == '__main__':
    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TasNet().to(device)
    mix = torch.rand((2,24000), dtype=torch.float32, device=device)
    distance = torch.rand((2,1), dtype=torch.float32, device=device)
    aa = model(mix)
    print(aa.shape)
