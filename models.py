import asteroid
import asteroid_filterbanks
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models.conv_tasnet import ConvBlock, ConvTasNet
from torch import nn
from hyperpyyaml import load_hyperpyyaml

from asteroid import torch_utils
from asteroid import torch_utils
from asteroid_filterbanks import Encoder, Decoder, FreeFB
from asteroid.masknn.recurrent import SingleRNN
from asteroid.masknn.norms import GlobLN
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


class T60_v1_MaskGenerator(T60_MaskGenerator):
    def __init__(self, input_dim: int, num_sources: int, kernel_size: int, num_feats: int, num_hidden: int, num_layers: int, num_stacks: int, msk_activate: str):
        super(T60_v1_MaskGenerator, self).__init__(input_dim, num_sources, kernel_size, num_feats, num_hidden, num_layers, num_stacks, msk_activate)

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

        self.encoder_fc1 = torch.nn.Linear(1, enc_num_feats)
        self.encoder_fc2 = torch.nn.Linear(1, enc_num_feats)

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
        input = input.unsqueeze(1)
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        t60 = kwargs.get('t60')

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        alpha = F.softplus(self.encoder_fc1(t60.unsqueeze(-1)))
        beta = self.encoder_fc2(t60.unsqueeze(-1))
        feats = alpha.unsqueeze(-1) * self.encoder(padded) + beta.unsqueeze(-1)  # B, F, M

        # separation module
        mask = self.mask_generator(feats)
        
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


class T60_DPRNNTasNet_v1_Encoder(asteroid_filterbanks.Encoder):
    def __init__(self, filterbank, is_pinv=False, as_conv1d=True, padding=0):
        super(T60_DPRNNTasNet_v1_Encoder, self).__init__(filterbank, is_pinv, as_conv1d, padding)
        self.fc_1 = torch.nn.Linear(1, self.n_feats_out)
        self.fc_2 = torch.nn.Linear(1, self.n_feats_out)

    def forward(self, waveform, t60):
        filters = self.get_filters()
        waveform = self.filterbank.pre_analysis(waveform)
        alpha = F.softplus(self.fc_1(t60))
        beta = self.fc_2(t60)
        spec = asteroid_filterbanks.enc_dec.multishape_conv1d(
            waveform,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            as_conv1d=self.as_conv1d,
        )
        spec = alpha.unsqueeze(-1) * spec + beta.unsqueeze(-1)
        return self.filterbank.post_analysis(spec)


# class T60_DPRNNTasNet_v1(asteroid.models.DPRNNTasNet):
#     def __init__(self, config, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type="gLN", mask_act="sigmoid", bidirectional=True, rnn_type="LSTM", num_layers=1, dropout=0, in_chan=None, fb_name="free", kernel_size=16, n_filters=64, stride=8, encoder_activation=None, sample_rate=8000, use_mulcat=False, **fb_kwargs):
#         self.config = config
#         n_src = self.config.speechnum
#         super().__init__(n_src, out_chan, bn_chan, hid_size, chunk_size, hop_size, n_repeats, norm_type, mask_act, bidirectional, rnn_type, num_layers, dropout, in_chan, fb_name, kernel_size, n_filters, stride, encoder_activation, sample_rate, use_mulcat, **fb_kwargs)

#         self.fc_1 = torch.nn.Linear(1, n_filters)
#         self.fc_2 = torch.nn.Linear(1, n_filters)

#     def forward(self, wav, **kwargs):
#         t60 = kwargs.get('t60').unsqueeze(-1)
#         # Remember shape to shape reconstruction, cast to Tensor for torchscript
#         shape = asteroid.models.base_models.jitable_shape(wav)
#         # Reshape to (batch, n_mix, time)
#         wav = _unsqueeze_to_3d(wav)

#         # Real forward
#         alpha = F.softplus(self.fc_1(t60))
#         beta = self.fc_2(t60)
#         tf_rep = alpha.unsqueeze(-1) * self.forward_encoder(wav) + beta.unsqueeze(-1)

#         est_masks = self.forward_masker(tf_rep)
#         masked_tf_rep = self.apply_masks(tf_rep, est_masks)
#         decoded = self.forward_decoder(masked_tf_rep)

#         reconstructed = asteroid.models.base_models.pad_x_to_y(decoded, wav)
#         return _shape_reconstructed(reconstructed, shape)


class T60_DPRNNTasNet_v1(asteroid.models.base_models.BaseEncoderMaskerDecoder):
    def __init__(
        self,
        config,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        use_mulcat=False,
        **fb_kwargs,
    ):
        self.config = config
        n_src = self.config.speechnum
        encoder, decoder = asteroid_filterbanks.make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = asteroid.models.dprnn_tasnet.DPRNN(
            n_feats,
            n_src,
            out_chan=out_chan,
            bn_chan=bn_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            mask_act=mask_act,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            use_mulcat=use_mulcat,
        )
        fb = asteroid_filterbanks.get(fb_name)(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **fb_kwargs)
        encoder = T60_DPRNNTasNet_v1_Encoder(fb, padding=fb_kwargs.get('padding', 0))
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)

    def forward(self, wav, **kwargs):
        t60 = kwargs.get('t60')
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = asteroid.models.base_models.jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav, t60.unsqueeze(1))
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = asteroid.models.base_models.pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)
        
    def forward_encoder(self, wav: torch.Tensor, t60) -> torch.Tensor:
        tf_rep = self.encoder(wav, t60)
        return self.enc_activation(tf_rep)


class T60_DPRNNTasNet_v2_Decoder(asteroid_filterbanks.Decoder):
    def __init__(self, filterbank, is_pinv=False, padding=0, output_padding=0):
        super(T60_DPRNNTasNet_v2_Decoder, self).__init__(filterbank, is_pinv, padding, output_padding)
        self.padding = padding
        self.output_padding = output_padding

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(1, 64)

    def forward(self, spec, t60, length = None) -> torch.Tensor:
        filters = self.get_filters()
        alpha = F.softplus(self.fc1(t60))
        beta = self.fc2(t60)
        spec = alpha.unsqueeze(-1).unsqueeze(1) * spec + beta.unsqueeze(-1).unsqueeze(1)
        spec = self.filterbank.pre_synthesis(spec)
        wav = asteroid_filterbanks.enc_dec.multishape_conv_transpose1d(
            spec,
            filters,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        wav = self.filterbank.post_synthesis(wav)
        if length is not None:
            length = min(length, wav.shape[-1])
            return wav[..., :length]
        return wav


class T60_DPRNNTasNet_v2(asteroid.models.DPRNNTasNet):
    def __init__(self, config, out_chan=None, bn_chan=128, hid_size=128, chunk_size=100, hop_size=None, n_repeats=6, norm_type="gLN", mask_act="sigmoid", bidirectional=True, rnn_type="LSTM", num_layers=1, dropout=0, in_chan=None, fb_name="free", kernel_size=16, n_filters=64, stride=8, encoder_activation=None, sample_rate=8000, use_mulcat=False, **fb_kwargs):
        self.config = config
        super().__init__(config.speechnum, out_chan, bn_chan, hid_size, chunk_size, hop_size, n_repeats, norm_type, mask_act, bidirectional, rnn_type, num_layers, dropout, in_chan, fb_name, kernel_size, n_filters, stride, encoder_activation, sample_rate, use_mulcat, **fb_kwargs)
        self.fc1 = nn.Linear(1, n_filters)
        self.fc2 = nn.Linear(1, n_filters)

    def forward(self, wav, **kwargs):
        t60 = kwargs.get('t60').unsqueeze(-1)
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = asteroid.models.base_models.jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        tf_rep = F.softplus(self.fc1(t60)).unsqueeze(-1) * tf_rep + self.fc2(t60).unsqueeze(-1)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = asteroid.models.base_models.pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)
        


class T60_DPRNNTasNet_v3(asteroid.models.base_models.BaseEncoderMaskerDecoder):
    def __init__(
        self,
        config,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        in_chan=None,
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        use_mulcat=False,
        **fb_kwargs,
    ):
        self.config = config
        n_src = self.config.speechnum
        encoder, decoder = asteroid_filterbanks.make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = asteroid.models.dprnn_tasnet.DPRNN(
            n_feats,
            n_src,
            out_chan=out_chan,
            bn_chan=bn_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            mask_act=mask_act,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
            use_mulcat=use_mulcat,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
        self.fc1 = nn.Linear(1, n_filters)
        self.fc2 = nn.Linear(1, n_filters)

    def forward(self, wav, **kwargs):
        t60 = kwargs.get('t60').unsqueeze(-1)
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = asteroid.models.base_models.jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        masked_tf_rep = F.softplus(self.fc1(t60)).unsqueeze(1).unsqueeze(-1) * masked_tf_rep + self.fc2(t60).unsqueeze(1).unsqueeze(-1)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = asteroid.models.base_models.pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)


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

        self.resout_fc1 = torch.nn.Linear(1, io_channels)
        self.resout_fc2 = torch.nn.Linear(1, io_channels)
        self.skipout_fc1 = torch.nn.Linear(1, io_channels)
        self.skipout_fc2 = torch.nn.Linear(1, io_channels)


    def forward(
        self, input: torch.Tensor, t60
    ):
        t60 = t60.unsqueeze(-1)
        alpha = F.softplus(self.fc1_1(t60))
        beta = self.fc1_2(t60)
        feature = alpha.unsqueeze(-1) * F.conv1d(input, self.conv1_weight, self.conv1_bias) + beta.unsqueeze(-1)
        feature = self.prelu(feature)
        feature = self.gnorm1(feature)

        feature = F.conv1d(feature, self.conv2_weight, self.conv2_bias, padding=self.padding, dilation=self.dilation, groups=feature.shape[1])
        feature = self.prelu(feature)
        feature = self.gnorm2(feature)
        
        if self.res_out is None:
            residual = None
        else:
            alpha = F.softplus(self.resout_fc1(t60))
            beta = self.resout_fc2(t60)
            residual = alpha.unsqueeze(-1) * self.res_out(feature) + beta.unsqueeze(-1)
        alpha = F.softplus(self.skipout_fc1(t60))
        beta = self.skipout_fc2(t60)
        skip_out = alpha.unsqueeze(-1) * self.skip_out(feature) + beta.unsqueeze(-1)
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
        self.encoder = Encoder(FreeFB(512, 40, 20))
        self.decoder = Decoder(FreeFB(512, 40, 20))
        self.bn_layer = GlobLN(512)

        # Create TasNet masker
        self.masker = nn.Sequential(
            SingleRNN(
                "lstm",
                512,
                hidden_size=500,
                n_layers=4,
                bidirectional=True,
            ),
            nn.Linear(2 * 500, self.n_src * self.n_filters),
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
        relu_out = torch.relu(self.encoder(x))
        sig_out = torch.sigmoid(self.encoder(x))
        return sig_out * relu_out


class T60_TasNet_v1(TasNet):
    def __init__(self):
        super().__init__()
        self.n_src = 2
        self.n_filters = 512
        # Create TasNet encoders and decoders (could use nn.Conv1D as well)
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(1, 512)
        self.encoder = Encoder(FreeFB(512, 40, 20))
        self.decoder = Decoder(FreeFB(512, 40, 20))
        self.bn_layer = GlobLN(512)

        # Create TasNet masker
        self.masker = nn.Sequential(
            SingleRNN(
                "lstm",
                512,
                hidden_size=500,
                n_layers=4,
                bidirectional=True,
            ),
            nn.Linear(2 * 500, self.n_src * self.n_filters),
            nn.Sigmoid(),
        )

    def forward(self, x, **kwargs):
        t60 = kwargs.get('t60')
        t60 = t60.unsqueeze(-1)
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        tf_rep = self.encode(x, t60)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks
        return torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x, t60):
        alpha = F.softplus(self.fc1(t60))
        beta = self.fc2(t60)
        out = alpha.unsqueeze(-1) * self.encoder(x) + beta.unsqueeze(-1)
        relu_out = torch.relu(out)
        sig_out = torch.sigmoid(out)
        return relu_out * sig_out


class T60_TasNet_v2(T60_TasNet_v1):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        t60 = kwargs.get('t60')
        t60 = t60.unsqueeze(-1)
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        raw_rep, tf_rep = self.encode(x, t60)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = raw_rep.unsqueeze(1) * est_masks
        return torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x, t60):
        alpha = F.softplus(self.fc1(t60))
        beta = self.fc2(t60)
        rawfeat = self.encoder(x)
        out = alpha.unsqueeze(-1) * rawfeat + beta.unsqueeze(-1)
        relu_out = torch.relu(out)
        sig_out = torch.sigmoid(out)
        rawrelu = torch.relu(rawfeat)
        rawsig = torch.sigmoid(rawfeat)
        return rawrelu * rawsig, relu_out * sig_out


class T60_ConvTasNet_v2(T60_ConvTasNet_v1):
    def __init__(self, config, num_sources: int = 2, enc_kernel_size: int = 16, enc_num_feats: int = 512, msk_kernel_size: int = 3, msk_num_feats: int = 128, msk_num_hidden_feats: int = 512, msk_num_layers: int = 8, msk_num_stacks: int = 3, msk_activate: str = "sigmoid"):
        super().__init__(config, num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
        input = input.unsqueeze(1)
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        t60 = kwargs.get('t60')

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        alpha = F.softplus(self.encoder_fc1(t60.unsqueeze(-1)))
        beta = self.encoder_fc2(t60.unsqueeze(-1))
        rawfeat = self.encoder(padded)
        feats = alpha.unsqueeze(-1) * rawfeat + beta.unsqueeze(-1)  # B, F, M

        # separation module
        mask = self.mask_generator(rawfeat)
        
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


class T60_ConvTasNet_v3(T60_ConvTasNet_v1):
    def __init__(self, config, num_sources: int = 2, enc_kernel_size: int = 16, enc_num_feats: int = 512, msk_kernel_size: int = 3, msk_num_feats: int = 128, msk_num_hidden_feats: int = 512, msk_num_layers: int = 8, msk_num_stacks: int = 3, msk_activate: str = "sigmoid"):
        super(T60_ConvTasNet_v3, self).__init__(config, num_sources, enc_kernel_size, enc_num_feats, msk_kernel_size, msk_num_feats, msk_num_hidden_feats, msk_num_layers, msk_num_stacks, msk_activate)
        self.mask_generator = torchaudio.models.conv_tasnet.MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
            msk_activate=msk_activate,
        )

        self.encoder_fc1 = torch.nn.Linear(1, enc_num_feats)
        self.encoder_fc2 = torch.nn.Linear(1, enc_num_feats)

    def forward(self, input: torch.Tensor, test=False, **kwargs) -> torch.Tensor:
        input = input.unsqueeze(1)
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        t60 = kwargs.get('t60')

        padded, num_pads = self._align_num_frames_with_strides(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        alpha = F.softplus(self.encoder_fc1(t60.unsqueeze(-1)))
        beta = self.encoder_fc2(t60.unsqueeze(-1))
        feats = alpha.unsqueeze(-1) * self.encoder(padded) + beta.unsqueeze(-1)  # B, F, M

        # separation module
        mask = self.mask_generator(feats)
        
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


class Sepformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        hparams_file = 'sepformer.yaml'
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin)
        modules = hparams['modules']
        self.config = config
        self.encoder = modules['encoder']
        self.decoder = modules['decoder']
        self.masknet = modules['masknet']
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mix_w = self.encoder(input)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.config.speechnum)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.config.speechnum)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source.transpose(-1,-2)
        