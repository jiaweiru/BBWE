import torch
import torch.nn.functional as F
import numpy as np
import interp_same as IS
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

from .hifigan_discriminator import MultiPeriodDiscriminator

LRELU_SLOPE = 0.1
EPSILON = torch.finfo(torch.float32).eps


class HarmonicSpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(
        self,
        fft_size=1024,
        hop_length=120,
        win_length=600,
        use_spectral_norm=False,
        anchor=7,
    ):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.discriminators = nn.ModuleList(
            [
                norm_f(
                    SingleLogHarmonicConv2d(
                        1, 32, kernel_size=(7, 7), padding=(3, 3), anchor=anchor
                    ),
                    "lowered_weight",
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
                ),
                norm_f(
                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                ),
            ]
        )

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):
        fmap = []
        y = torch.stft(
            y.squeeze(1),
            self.fft_size,
            self.hop_length,
            self.win_length,
            torch.hann_window(self.win_length).to(y.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        y = torch.sqrt(torch.clamp(y.real**2 + y.imag**2, EPSILON))
        y = y.unsqueeze(1).contiguous()
        for d in self.discriminators:
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(torch.nn.Module):
    def __init__(  # pylint: disable=dangerous-default-value
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                HarmonicSpecDiscriminator(
                    fft_sizes[0], hop_sizes[0], win_lengths[0], window
                ),
                HarmonicSpecDiscriminator(
                    fft_sizes[1], hop_sizes[1], win_lengths[1], window
                ),
                HarmonicSpecDiscriminator(
                    fft_sizes[2], hop_sizes[2], win_lengths[2], window
                ),
            ]
        )

    def forward(self, x):
        scores = []
        feats = []
        for d in self.discriminators:
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)

        return scores, feats


class HXDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiResSpecDiscriminator()

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_


class ShiftF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shift):
        assert input.dim() == 4, "this method suppose the dimension of input is 4"
        ctx.shift = shift

        output = torch.empty_like(input)
        IS.interp_shift(input, output, shift)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shift = -ctx.shift
        grad_input = grad_shift = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            grad_input = torch.empty_like(grad_output)
            IS.interp_shift(grad_output, grad_input, shift)
        return grad_input, grad_shift


ShiftFunctional = ShiftF.apply


class Shift(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, shift):
        # input [batch,channel,freq,time]
        if shift == int(shift):
            # special case
            if shift == 0:
                return input
            elif shift > 0:
                return F.pad(
                    input[:, :, : -int(shift), :], (0, 0, int(shift), 0), "constant", 0
                )
            else:
                return F.pad(
                    input[:, :, -int(shift) :, :], (0, 0, 0, -int(shift)), "constant", 0
                )
        else:
            return ShiftFunctional(input, shift)


def make_detail(k, n, device):
    index = torch.arange(n, dtype=torch.int32, device=device) * k / n
    weight = 1 - (torch.arange(n, dtype=torch.float32, device=device) * k / n - index)
    return index, weight


class ZoomF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, n):
        ctx.k = k
        ctx.n = n
        output = torch.empty_like(input)
        indexes, weights = make_detail(k, n, device=input.device)
        IS.interp_affine(input, output, indexes, weights, k, n)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_k = grad_n = None
        k = ctx.n
        n = ctx.k
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.contiguous()
            if n == 1:
                # stride case
                # [batch,channel,freq,time]
                freq_size = grad_output.shape[2]
                # get by stride
                strided_grad_output = grad_output[:, :, ::k, :]
                # 0-padding
                pad_shape = (0, 0, 0, freq_size - strided_grad_output.shape[2])
                grad_input = F.pad(strided_grad_output, pad_shape, "constant", 0)
            else:
                grad_input = torch.empty_like(grad_output)
                indexes, weights = make_detail(k, n, device=grad_output.device)
                IS.interp_affine(grad_output, grad_input, indexes, weights, k, n)
        return grad_input, grad_k, grad_n


ZoomFunctional = ZoomF.apply


class Zoom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, k, n):
        # special case
        if n == k:
            return input
        # interpolate same size by affine 1d interpolation
        return ZoomFunctional(input, k, n)


class BaseLowering(nn.Module):
    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
    ):
        super().__init__()
        self.anchor = anchor
        self.f_kernel_size = f_kernel_size
        self.in_channels = in_channels
        self.groups = groups

        # setting index rules of in channels
        self.channel_slice_func = lambda k_f: slice(
            k_f * self.in_channels, (k_f + 1) * self.in_channels
        )
        self.channel_type = "stack"
        if self.groups != 1:
            self.channel_slice_func = lambda k_f: slice(k_f, None, self.in_channels)
            self.channel_type = "textile"

        # make parallel streams
        self.parallel_streams = [torch.cuda.Stream() for _ in range(self.f_kernel_size)]

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        lowered_in_channels = in_channels * f_kernel_size
        return Tensor size(batch,lowered_in_channels,f,t)
        """
        # make empty lowered input
        batch, in_channels, in_freq_size, in_time_size = input.shape
        lowered_in_channels = in_channels * self.f_kernel_size
        lowered_shape = (batch, lowered_in_channels, in_freq_size, in_time_size)
        lowered_input = torch.empty(
            lowered_shape, dtype=input.dtype, device=input.device
        )
        # fill elements start
        current_stream = torch.cuda.current_stream()
        #   block current stream
        current_stream.synchronize()
        for fk in range(self.f_kernel_size):
            # start parallel
            with torch.cuda.stream(self.parallel_streams[fk]):
                lowered_input[:, self.channel_slice_func(fk)] = self.parallelized(
                    input, k=fk + 1
                )

        for s in self.parallel_streams:
            # block parallel streams
            s.synchronize()

        # fill elements end
        return lowered_input

    def parallelized(self, input, k):
        raise NotImplementedError

    def extra_repr(self):
        return f"channel_type={self.channel_type}"


class HarmonicLowering(BaseLowering):
    """
    Lowering input for normal convolution
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom = Zoom()

    def parallelized(self, input, k):
        return self.zoom(input, k, self.anchor)

    def extra_repr(self):
        return f"n={self.anchor}, K_f={self.f_kernel_size}, " + super().extra_repr()


class LogHarmonicLowering(BaseLowering):
    def __init__(
        self,
        anchor,
        f_kernel_size,
        in_channels,
        groups=1,
        out_log_scale=1000,
        in_log_scale=0.001,
        radix=None,
    ):
        super().__init__(anchor, f_kernel_size, in_channels, groups)
        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix
        self.shift = self.make_log_shift()
        self.Shifter = Shift()

    def make_log_shift(self):
        """
        compute log shift
        return ndarray size(f_kernel_size)
        """
        assert (
            1 <= self.anchor <= self.f_kernel_size
        ), f"invalid anchor={self.anchor}. anchor should be in [min=1,f_kernel_size={self.f_kernel_size}]"

        np_shift = (np.arange(self.f_kernel_size) + 1) / self.anchor
        if self.radix is None:
            log_shift = self.out_log_scale * np.log(self.in_log_scale * np_shift)
        else:
            log_shift = (
                self.out_log_scale
                * np.log(self.in_log_scale * np_shift)
                / np.log(self.radix)
            )
        target_index = self.anchor - 1
        log_shift -= log_shift[target_index]
        return -log_shift

    def parallelized(self, input, k):
        return self.Shifter(input, self.shift[k - 1])

    def extra_repr(self):
        radix = self.radix if self.radix is not None else "e"
        return (
            f"n={self.anchor}, K_f={self.f_kernel_size}, log_func(f)={self.out_log_scale}log_{radix} {self.in_log_scale}f, "
            + super().extra_repr()
        )


class BaseSingleHarmonicConv2d_ft(nn.Conv2d):
    """
    Base class for Harmonic Convolution
    """

    def __init__(self, *args, anchor=1, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(anchor, int):
            raise Exception("anchor should be integer")
        self.anchor = anchor

        if self.anchor < 1:
            raise Exception("anchor should be equal to or bigger than 1")
        if self.padding_mode != "zeros":
            raise NotImplementedError("only zero padding mode is implemented")

        if self.padding[0] != 0:
            self.padding = (0, self.padding[1])

        # transforming weight shape
        lowered_shape = (
            self.out_channels,
            self.in_channels * self.kernel_size[0],
            1,
            self.kernel_size[1],
        )
        self.lowered_weight = torch.nn.Parameter(self.weight.reshape(lowered_shape))
        self.weight = None

    def forward(self, input):
        # [batch, in_channel, f, t]
        raise NotImplementedError("overwrite forward method")


class SingleHarmonicConv2d(BaseSingleHarmonicConv2d_ft):
    """
    Harmonic Convolution by Harmonic Lowering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.HL = HarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.HL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return f"anchor={self.anchor}, " + super().extra_repr()


class SingleLogHarmonicConv2d(BaseSingleHarmonicConv2d_ft):
    def __init__(
        self, *args, out_log_scale=1000, in_log_scale=0.001, radix=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.out_log_scale = out_log_scale
        self.in_log_scale = in_log_scale
        self.radix = radix

        self.LHL = LogHarmonicLowering(
            anchor=self.anchor,
            f_kernel_size=self.kernel_size[0],
            in_channels=self.in_channels,
            groups=self.groups,
            out_log_scale=self.out_log_scale,
            in_log_scale=self.in_log_scale,
            radix=self.radix,
        )

    def forward(self, input):
        """
        input Tensor size(batch,in_channels,f,t)
        return Tensor size(batch,out_channels,f',t')
        """
        lowered_input = self.LHL(input)

        output = F.conv2d(
            input=lowered_input,
            weight=self.lowered_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return output

    def extra_repr(self):
        return (
            f"out_log_scale={self.out_log_scale}, in_log_scale={self.in_log_scale}, radix={self.radix}, "
            + super().extra_repr()
        )
