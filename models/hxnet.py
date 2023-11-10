import typing as tp

import torch
import torchaudio
from torch import nn
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: tp.Tuple[int]):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding="same",
                    ),
                ),
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding="same",
                    ),
                ),
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding="same",
                    ),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(channels, channels, kernel_size, 1, padding="same"),
                ),
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(channels, channels, kernel_size, 1, padding="same"),
                ),
                nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    WNConv1d(channels, channels, kernel_size, 1, padding="same"),
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = xt + x
        return x


class MRF(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: tp.Tuple[int],
        dilation: tp.Tuple[tp.Tuple[int]],
    ):
        super().__init__()
        self.mrf = nn.ModuleList(
            [ResBlock(channels, k, d) for k, d in zip(kernel_size, dilation)]
        )

    def forward(self, x):
        z = None
        for res_block in self.mrf:
            if z is None:
                z = res_block(x)
            else:
                z += res_block(x)
        return z / len(self.mrf)


class EncBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        stride: int,
        kernel_size: tp.Tuple[int],
        dilation: tp.Tuple[tp.Tuple[int]],
    ):
        super().__init__()

        self.mrf = MRF(out_channels // 2, kernel_size, dilation)
        self.conv = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            WNConv1d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
                # bias=False,
                padding_mode="reflect",
            ),  # only for in_channels=out_channels // 2
        )

    def forward(self, x):
        out = self.conv(self.mrf(x))
        return out


class DecBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        stride: int,
        kernel_size: tp.Tuple[int],
        dilation: tp.Tuple[tp.Tuple[int]],
    ):
        super().__init__()

        self.mrf = MRF(out_channels, kernel_size, dilation)
        self.conv_trans = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            WNConvTranspose1d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2,
                # bias=False,
            ),
        )

    def forward(self, x, encoder_output=None):
        if encoder_output is not None:
            x = x + encoder_output
        out = self.mrf(self.conv_trans(x))
        return out


class HXNet(nn.Module):
    def __init__(
        self,
        original_sr: int = 8000,
        target_sr: int = 16000,
        kernel_size: int = 7,
        block_channels: tp.Tuple[int] = (8, 16, 32, 64, 128),
        latent_channel: int = 128,
        block_strides: tp.Tuple[int] = (2, 2, 8, 8),
        mrf_kernel_size: tp.Tuple[int] = (3, 7, 11),
        dilation: tp.Tuple[tp.Tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        """U-net structure for time-domain bandwidth expansion network, where
        the decoder structure is derived from the same upsampling + MRF
        structure as in HiFiGAN, with encoder and decoder symmetry.

        Args:
            original_sr (int, optional): Original sample rate. Defaults to 8000.
            target_sr (int, optional): Target sample rate. Defaults to 16000.
            kernel_size (int, optional): The kernel size of the convolutional
                layers on both sides of the encoder and decoder. Defaults to 7.
            block_channels (tp.Tuple[int], optional): Number of channels in
                upsampling and downsampling. Defaults to (8, 16, 32, 64, 128).
            latent_channel (int, optional): Number of latent channel.
                Defaults to 128.
            block_strides (tp.Tuple[int], optional): Strides in upsampling and
                downsampling. Defaults to (2, 2, 8, 8).
            mrf_kernel_size (tp.Tuple[int], optional): The kernel size in the
                MRF. Defaults to (3, 7, 11).
            dilation (tp.Tuple[tp.Tuple[int]], optional):The dilation in the
                MRF. Defaults to ((1, 3, 5), (1, 3, 5), (1, 3, 5)).
        """
        super().__init__()
        self.original_sr = original_sr
        self.target_sr = target_sr

        self.first_conv = WNConv1d(
            in_channels=1,
            out_channels=block_channels[0],
            kernel_size=kernel_size,
            padding="same",
            # bias=False,
            padding_mode="reflect",
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncBlock(
                    out_channels=block_channels[i + 1],
                    stride=block_strides[i],
                    kernel_size=mrf_kernel_size,
                    dilation=dilation,
                )
                for i in range(len(block_strides))
            ]
        )

        self.latent_conv = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            WNConv1d(
                in_channels=block_channels[-1],
                out_channels=latent_channel,
                kernel_size=kernel_size,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(LRELU_SLOPE),
            # This activation function may need to be ignored in neural coding
            WNConv1d(
                in_channels=latent_channel,
                out_channels=block_channels[-1],
                kernel_size=kernel_size,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecBlock(
                    out_channels=block_channels[-i - 2],
                    stride=block_strides[-i - 1],
                    kernel_size=mrf_kernel_size,
                    dilation=dilation,
                )
                for i in range(len(block_strides))
            ]
        )

        self.last_conv = nn.Sequential(
            nn.LeakyReLU(LRELU_SLOPE),
            WNConv1d(
                in_channels=block_channels[0],
                out_channels=1,
                kernel_size=kernel_size,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
        )

        self.final_activation = nn.Tanh()

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of generator.
        Args:
            speech (torch.Tensor): low-resolution speech signal
        Returns:
            (torch.Tensor): estimated high-resolution speech signal
        """
        speech = torchaudio.functional.resample(
            speech, self.original_sr, self.target_sr
        )

        x = self.first_conv(speech)

        encoder_outputs = [x]

        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)

        x = self.latent_conv(x)

        for idx, block in enumerate(self.decoder_blocks):
            x = block(x, encoder_outputs[-idx - 1])

        x = x + encoder_outputs[0]
        x = self.last_conv(x)

        estimated_speech = self.final_activation(x + speech)  # residual

        return estimated_speech


if __name__ == "__main__":
    audio = torch.randn(1, 1, 12800)  # Batch, Channel, Length
    net = HXNet()
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    print(f"net_out.shape: {net_out.shape} \n")
