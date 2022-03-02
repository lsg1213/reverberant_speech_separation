from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchaudio.models.conv_tasnet import ConvBlock, ConvTasNet
import torchaudio.models.conv_tasnet as conv_tasnet
from asteroid.models import DPRNNTasNet
from torch import nn

from asteroid import torch_utils
from asteroid import torch_utils
from asteroid_filterbanks import Encoder, Decoder, FreeFB
from asteroid.masknn.recurrent import SingleRNN
from asteroid.masknn.norms import GlobLN


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

    def forward(self, x, **kwargs):
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


@torch_utils.script_if_tracing
def _shape_reconstructed(reconstructed, size):
    """Reshape `reconstructed` to have same size as `size`

    Args:
        reconstructed (torch.Tensor): Reconstructed waveform
        size (torch.Tensor): Size of desired waveform

    Returns:
        torch.Tensor: Reshaped waveform

    """
    if len(size) == 1:
        return reconstructed.squeeze(0)
    return reconstructed


@torch_utils.script_if_tracing
def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


class Dereverb_module(torch.nn.Module):
    def __init__(self, config, input_channel, hidden_channel, kernel_size=None) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.config = config
        self.version = 'v2'

        if 'v2' not in self.version:
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

        if 'v1' not in self.version:
            self.gru = torch.nn.GRU(self.input_channel, self.input_channel, batch_first=True, bidirectional=True)
        
    def pad_for_dereverb_module(self, signal, filter_length):
        return F.pad(signal, (filter_length - 1, 0))
            
    def forward(self, input: torch.Tensor):
        if 'v2' in self.version:
            out = input
        else:
            kernel = self.kernel * self.kernel_weight
            batch, feat, time = input.shape
            input = self.pad_for_dereverb_module(input, self.kernel_size)
            pad_time = input.shape[-1]
            input = input.reshape([-1, pad_time])
            out = F.conv1d(input.unsqueeze(1).flip(-1), kernel).flip(-1)
            out = out.sum(1).reshape((batch, feat, time))
            out = self.prelu(out)
        if 'v1' not in self.version:
            out = self.gru(out.transpose(-2, -1))[0].transpose(-2, -1)
            out = out[:, :out.shape[1] // 2] + out[:, out.shape[1] // 2:]
        return out


class Dereverb_ConvTasNet_v1(ConvTasNet):
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
        super(Dereverb_ConvTasNet_v1, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
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
        
        self.mask_generator = conv_tasnet.MaskGenerator(
            input_dim=enc_num_feats,
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
        self.dereverb_module = Dereverb_module(config, enc_num_feats, 64, rir_func_length)

        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
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
            separated_output = self.decoder(separated_feat)
            separated_output = separated_output.view(
                batch_size, self.num_sources, num_padded_frames
            )
            feats = separated_feat

        # dereberberation module
        dereverb_feats = self.dereverb_module(feats)

        masked = self.decoder(dereverb_feats)  # B*S, 1, L'
        output = masked.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
            if not test:
                separated_output = separated_output[..., :-num_pads]
        if test:
            return output
        else:
            return output, separated_output


class Dereverb_DPRNNTasNet_v1(DPRNNTasNet):
    def __init__(self, config, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type="gLN", mask_act="sigmoid", bidirectional=True, rnn_type="LSTM", num_layers=1, dropout=0, in_chan=None, fb_name="free", kernel_size=16, n_filters=64, stride=8, encoder_activation=None, sample_rate=8000, use_mulcat=False, **fb_kwargs):
        super().__init__(config.speechnum, out_chan, bn_chan, hid_size, chunk_size, hop_size, n_repeats, norm_type, mask_act, bidirectional, rnn_type, num_layers, dropout, in_chan, fb_name, kernel_size, n_filters, stride, encoder_activation, sample_rate, use_mulcat, **fb_kwargs)
        self.config = config
        self.dereverb_module = Dereverb_module(config, input_channel=n_filters, hidden_channel=n_filters)
        
    def forward(self, wav, distance=None, test=False):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.

        Returns:
            torch.Tensor, of shape (batch, n_src, time) or (n_src, time).
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = torch_utils.jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)

        reshaped_masked_tf_rep = masked_tf_rep.reshape((-1, masked_tf_rep.shape[-2], masked_tf_rep.shape[-1]))
        dereverb_masked_tf_rep = self.dereverb_module(reshaped_masked_tf_rep)
        dereverb_masked_tf_rep = dereverb_masked_tf_rep.reshape((shape[0], -1, reshaped_masked_tf_rep.shape[-2], reshaped_masked_tf_rep.shape[-1]))
        dereverb_decoded = self.forward_decoder(dereverb_masked_tf_rep)
        dereverb_reconstructed = torch_utils.pad_x_to_y(dereverb_decoded, wav)
        dereverb_reconstructed = _shape_reconstructed(dereverb_reconstructed, shape)
        
        if test:
            return dereverb_reconstructed

        decoded = self.forward_decoder(masked_tf_rep)
        reconstructed = torch_utils.pad_x_to_y(decoded, wav)
        reconstructed = _shape_reconstructed(reconstructed, shape)
        return dereverb_reconstructed, reconstructed


class Dereverb_TasNet_v1(nn.Module):
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

    def __init__(self, config):
        super().__init__()
        self.config = config
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
        self.dereverb_module = Dereverb_module(config, 512, 512)

    def forward(self, x, test=False, **kwargs):
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encode(x)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks

        dereverb_masked_tf_rep = self.dereverb_module(masked_tf_rep.reshape((-1, masked_tf_rep.shape[-2], masked_tf_rep.shape[-1])))
        dereverb_masked_tf_rep = dereverb_masked_tf_rep.reshape((batch_size, -1, masked_tf_rep.shape[-2], masked_tf_rep.shape[-1]))
        dereverb_reconstructed = torch_utils.pad_x_to_y(self.decoder(dereverb_masked_tf_rep), x)
        if not test:
            reconstructed = torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)
            return dereverb_reconstructed, reconstructed
        return dereverb_reconstructed

    def encode(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


class Dereverb_T60_module(torch.nn.Module):
    def __init__(self, config, input_channel, hidden_channel, kernel_size=None, strides=16) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.kernel_size = kernel_size
        self.config = config
        self.strides = 16
        self.gru = torch.nn.GRU(1, 16, batch_first=True, bidirectional=True)
        
    def pad_for_dereverb_module(self, signal, filter_length):
        return F.pad(signal, (filter_length - 1, 0))
            
    def forward(self, input: torch.Tensor, t60s):
        # input = (batch * 2, num_feats, time)
        t60s = t60s.repeat((2)) # 1000 = 8000 / 8
        batch = input.shape[0]
        out = []
        for i in range(batch):
            inp = input[i]
            t60 = t60s[i]
            t60 = int(t60 * 1000)
            inp = inp.flip(-1)
            gruouts = []
            for j in range(int(ceil(inp.shape[-1] / self.strides))):
                gruinp = inp[..., j * self.strides: j * self.strides + t60].unsqueeze(-1)
                gruout = self.gru(gruinp)[0]
                gruout = gruout[..., -self.strides:, :gruout.shape[-1]//2] + gruout[..., -self.strides:, gruout.shape[-1]//2:]
                gruout = gruout.mean(-1) # global average pooling
                gruouts.append(gruout)
            gruouts = torch.cat(gruouts, -1)
            out.append(gruouts.flip(-1))
        return torch.stack(out)


class T60_ConvBlock(torch.nn.Module):
    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()
        self.padding=padding
        self.dilation=dilation
        
        self.prelu = torch.nn.PReLU()
        # self.conv1_weight = torch.nn.Parameter(torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1).weight)
        # self.conv1_bias = torch.nn.Parameter(torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1).bias)
        # self.gnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)

        # self.conv2_weight = torch.nn.Parameter(torch.nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels).weight)
        # self.conv2_bias = torch.nn.Parameter(torch.nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels).bias)
        # self.gnorm2 = torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)

        self.fc1_1 = torch.nn.Linear(1, hidden_channels)
        self.fc1_2 = torch.nn.Linear(1, hidden_channels)
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels, out_channels=hidden_channels, kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
        )
        self.gnorm = torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(
                in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
            )
        )
        self.skip_out = torch.nn.Conv1d(
            in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
        )

    def forward(
        self, input: torch.Tensor, t60
    ):
        out = self.convs(input)
        alpha = F.softplus(self.fc1_1(t60.unsqueeze(-1)))
        beta = self.fc1_2(t60.unsqueeze(-1))
        feature = alpha.unsqueeze(-1) * out + beta.unsqueeze(-1)
        feature = self.prelu(feature)
        feature = self.gnorm(feature)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class T60_MaskGenerator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
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
                    T60_ConvBlock(
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
            in_channels=num_feats, out_channels=input_dim * num_sources, kernel_size=1,
        )
        if msk_activate == "sigmoid":
            self.mask_activate = torch.nn.Sigmoid()
        elif msk_activate == "relu":
            self.mask_activate = torch.nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {msk_activate}")

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        t60 = kwargs.get('t60')

        batch_size = input.shape[0]
        feats = self.input_norm(input)
        feats = self.input_conv(feats)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats, t60=t60)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class Dereverb_T60_encoder(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=512, kernel_size=16, stride=8, padding=8, bias=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.kernel_weight = torch.nn.Parameter(torch.rand((41, out_channels, in_channels, kernel_size)))
        
    def forward(self, input: torch.Tensor, t60s):
        # input = (batch, 1, 24000)
        outs = []
        t60s = torch.round(t60s * 100).int() - 10
        for t60, inp in zip(t60s, input):
            outs.append(F.conv1d(inp.unsqueeze(0), self.kernel_weight[t60], stride=self.stride, padding=self.padding))
        return torch.cat(outs, 0)


class Dereverb_T60_decoder(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=512, kernel_size=16, stride=8, padding=8, bias=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.kernel_weight = torch.nn.Parameter(torch.rand((41, in_channels, out_channels, kernel_size)))
        
    def forward(self, input: torch.Tensor, t60s):
        # input = (batch, 1, 24000)
        outs = []
        t60s = torch.round(t60s * 100).int().repeat((2)) - 10
        for t60, inp in zip(t60s, input):
            outs.append(F.conv_transpose1d(inp.unsqueeze(0), self.kernel_weight[t60], stride=self.stride, padding=self.padding))
        return torch.cat(outs, 0)


class T60_ConvTasNet_v1(ConvTasNet):
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
        super(T60_ConvTasNet_v1, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
        self.config = config
        self.num_sources = 1 if self.config.test else num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2
        
        self.mask_generator = T60_MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
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

        t60 = kwargs.get('t60')
        # clean_sep = kwargs.get('clean')

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M

        # separation module
        mask = self.mask_generator(feats, t60=t60)
        
        masked_feature = mask * feats.unsqueeze(1)  # B, S, F, M
        
        masked = masked_feature.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        output = self.decoder(masked)
        output = output.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
        return output


class T60_v2_ConvBlock(T60_ConvBlock):
    def __init__(self, io_channels: int, hidden_channels: int, kernel_size: int, padding: int, dilation: int = 1, no_residual: bool = False):
        super(T60_v2_ConvBlock, self).__init__(io_channels, hidden_channels, kernel_size, padding, dilation, no_residual)
        self.fc1_1 = torch.nn.Linear(1, hidden_channels)
        self.fc1_2 = torch.nn.Linear(1, hidden_channels)

        conv = torch.nn.Conv1d(in_channels=io_channels, out_channels=hidden_channels, kernel_size=1)
        self.conv1_weight = conv.weight
        self.conv1_bias = conv.bias
        self.gnorm1 = torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)
        self.fc1_1 = torch.nn.Linear(1, hidden_channels)
        self.fc1_2 = torch.nn.Linear(1, hidden_channels)

        conv = torch.nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_channels)
        self.conv2_weight = conv.weight
        self.conv2_bias = conv.bias
        self.gnorm2 = torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08)
        self.fc2_1 = torch.nn.Linear(1, hidden_channels)
        self.fc2_2 = torch.nn.Linear(1, hidden_channels)
        self.padding = padding
        self.dilation = dilation


    def forward(
        self, input: torch.Tensor, t60
    ):
        alpha = F.softplus(self.fc1_1(t60.unsqueeze(-1)))
        beta = self.fc1_2(t60.unsqueeze(-1))
        feature = alpha.unsqueeze(-1) * F.conv1d(input, self.conv1_weight, self.conv1_bias) + beta.unsqueeze(-1)
        feature = self.prelu(feature)
        feature = self.gnorm1(feature)

        alpha = F.softplus(self.fc2_1(t60.unsqueeze(-1)))
        beta = self.fc2_2(t60.unsqueeze(-1))
        feature = alpha.unsqueeze(-1) * F.conv1d(feature, self.conv2_weight, self.conv2_bias, padding=self.padding, dilation=self.dilation, groups=feature.shape[1]) + beta.unsqueeze(-1)
        feature = self.prelu(feature)
        feature = self.gnorm2(feature)
        
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class T60_v2_MaskGenerator(T60_MaskGenerator):
    def __init__(self, input_dim: int, num_sources: int, kernel_size: int, num_feats: int, num_hidden: int, num_layers: int, num_stacks: int, msk_activate: str):
        super(T60_v2_MaskGenerator, self).__init__(input_dim, num_sources, kernel_size, num_feats, num_hidden, num_layers, num_stacks, msk_activate)

        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    T60_v2_ConvBlock(
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


class T60_ConvTasNet_v2(T60_ConvTasNet_v1):
    def __init__(self, config, num_sources: int = 2, enc_kernel_size: int = 16, enc_num_feats: int = 512, msk_kernel_size: int = 3, msk_num_feats: int = 128, msk_num_hidden_feats: int = 512, msk_num_layers: int = 8, msk_num_stacks: int = 3, msk_activate: str = "sigmoid"):
        super(T60_ConvTasNet_v2, self).__init__(config, num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
        self.mask_generator = T60_v2_MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )

    
class Dereverb_test_v1(ConvTasNet):
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
        super(Dereverb_test_v1, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
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

        self.auto_gru_encoder = torch.nn.ModuleList([
            torch.nn.GRU(512, 256, bidirectional=True, batch_first=True),
            torch.nn.GRU(256, 64, bidirectional=True, batch_first=True),
            torch.nn.GRU(64, 16, bidirectional=True, batch_first=True),
            torch.nn.GRU(16, 64, bidirectional=True, batch_first=True),
            torch.nn.GRU(64, 256, bidirectional=True, batch_first=True),
            torch.nn.GRU(256, 512 * num_sources, bidirectional=True, batch_first=True),
        ])
        self.autoencoder_dropout = torch.nn.Dropout(0.1)

        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
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
        derev_feats = feats.transpose(-2,-1)
        for layer in self.auto_gru_encoder:
            derev_feats = layer(derev_feats)[0]
            derev_feats = derev_feats[...,:derev_feats.shape[-1]//2] + derev_feats[...,derev_feats.shape[-1]//2:]
            derev_feats = self.autoencoder_dropout(derev_feats)
        derev_feats = derev_feats.transpose(-2,-1)
        masked = derev_feats.view(batch_size, self.num_sources, self.enc_num_feats, -1) * feats.unsqueeze(1)

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        output = self.decoder(masked)
        output = output.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
        return output

    
class Dereverb_test_v2(ConvTasNet):
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
        super(Dereverb_test_v2, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
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

        self.auto_gru_encoder = torch.nn.ModuleList([
            torch.nn.GRU(512, 128, bidirectional=True, batch_first=True),
            torch.nn.GRU(128, 16, bidirectional=True, batch_first=True),
            torch.nn.GRU(16, 128, bidirectional=True, batch_first=True),
            torch.nn.GRU(128, 512 * num_sources, bidirectional=True, batch_first=True),
        ])
        self.autoencoder_dropout = torch.nn.Dropout(0.1)

        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
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
        derev_feats = feats.transpose(-2,-1)
        for layer in self.auto_gru_encoder:
            derev_feats = layer(derev_feats)[0]
            derev_feats = derev_feats[...,:derev_feats.shape[-1]//2] + derev_feats[...,derev_feats.shape[-1]//2:]
            derev_feats = self.autoencoder_dropout(derev_feats)
        derev_feats = derev_feats.transpose(-2,-1)
        masked = derev_feats.view(batch_size, self.num_sources, self.enc_num_feats, -1) * feats.unsqueeze(1)

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        output = self.decoder(masked)
        output = output.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
        return output

    
class Dereverb_test_v3(ConvTasNet):
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
        super(Dereverb_test_v3, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
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

        self.auto_gru_encoder = torch.nn.ModuleList([
            torch.nn.GRU(512, 512, bidirectional=True, batch_first=True),
            torch.nn.GRU(512, 512, bidirectional=True, batch_first=True),
            torch.nn.GRU(512, 512, bidirectional=True, batch_first=True),
            torch.nn.GRU(512, 512 * num_sources, bidirectional=True, batch_first=True),
        ])
        self.autoencoder_dropout = torch.nn.Dropout(0.1)

        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
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
        derev_feats = feats.transpose(-2,-1)
        for layer in self.auto_gru_encoder:
            derev_feats = layer(derev_feats)[0]
            derev_feats = derev_feats[...,:derev_feats.shape[-1]//2] + derev_feats[...,derev_feats.shape[-1]//2:]
            derev_feats = self.autoencoder_dropout(derev_feats)
        derev_feats = derev_feats.transpose(-2,-1)
        masked = derev_feats.view(batch_size, self.num_sources, self.enc_num_feats, -1)

        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        output = self.decoder(masked)
        output = output.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'

        if num_pads > 0:
            output = output[..., :-num_pads]
        return output


class DualConvBlock(torch.nn.Module):
    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual = None,
    ):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels, out_channels=hidden_channels, kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.conv_layers2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels, out_channels=hidden_channels, kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = no_residual
        self.Linear = torch.nn.Linear(hidden_channels, io_channels)
        self.Linear_norm = GlobLN(io_channels)

        self.Linear2 = torch.nn.Linear(hidden_channels, io_channels)
        self.Linear_norm2 = GlobLN(io_channels)
        self.hidden_channels = hidden_channels

    def forward(self, input: torch.Tensor):
        B, N, K, L = input.shape
        output = input
        feats = input.permute(0, 3, 1, 2).reshape(B * L, N, K)
        feats = self.conv_layers(feats)
        feats = self.Linear(feats.transpose(-2,-1))
        feats = feats.reshape(B, L, K, N).transpose(1, 3)
        feats = self.Linear_norm(feats)
        output += feats
        feats = output.permute(0, 2, 1, 3).reshape(B * K, N, L)
        feats = self.conv_layers2(feats)
        feats = self.Linear2(feats.transpose(-2,-1))
        feats = feats.reshape((B, K, L, N)).permute((0, 3, 1, 2)).contiguous()
        feats = self.Linear_norm2(feats)
        output += feats

        if self.res_out is None:
            residual = None
        else:
            residual = input
        skip_out = output
        return residual, skip_out


class newMaskGenerator(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
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
        self.num_sources = num_sources
        self.chunk_size = 500
        self.hop_size = self.chunk_size // 2

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
                    DualConvBlock(
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
        self.mask_conv = torch.nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(num_feats, num_sources * num_feats, 1)
        )
        self.net_out = nn.Sequential(nn.Conv1d(num_feats, num_feats, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(num_feats, num_feats, 1), nn.Sigmoid())
        self.output_prelu = torch.nn.PReLU()
        self.output_conv = torch.nn.Conv1d(
            in_channels=num_feats, out_channels=input_dim, kernel_size=1,
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
        n_frames = input.shape[-1]
        feats = self.input_norm(input)
        if dis is not None:
            feats = torch.cat([feats, dis], 1)
        feats = self.input_conv(feats)
        feats = F.unfold(feats.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0), stride=(self.hop_size, 1),)
        n_chunks = feats.shape[-1]
        feats = feats.reshape(batch_size, -1, self.chunk_size, n_chunks)
        
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.mask_conv(output)
        to_unfold = output.shape[1] // 2 * output.shape[2]

        output = F.fold(output.reshape((batch_size * self.num_sources, to_unfold, n_chunks)),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),)

        output = output.reshape(batch_size * self.num_sources, output.shape[1], -1)
        output = self.net_out(output) * self.net_gate(output)

        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class ConvTasNet_v2(ConvTasNet):
    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 256,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
        msk_activate: str = "sigmoid",
        **kwargs
    ):
        super(ConvTasNet_v2, self).__init__(num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.mask_generator = newMaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        input = input.unsqueeze(1)
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output

