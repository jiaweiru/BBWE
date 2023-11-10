import typing as tp

import torch
import torchaudio
from torch import nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.ELU(),
            WNConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                dilation=dilation,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
        )
        self.conv2 = nn.Sequential(
            nn.ELU(),
            WNConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
        )

    def forward(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class EncBlock(nn.Module):
    def __init__(self, out_channels: int, stride: int):
        super().__init__()

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, dilation=1),
            ResidualUnit(channels=out_channels // 2, dilation=3),
            ResidualUnit(channels=out_channels // 2, dilation=9),
        )
        self.conv = nn.Sequential(
            nn.ELU(),
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
        out = self.conv(self.residuals(x))
        return out


class DecBlock(nn.Module):
    def __init__(self, out_channels: int, stride: int):
        super().__init__()

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, dilation=1),
            ResidualUnit(channels=out_channels, dilation=3),
            ResidualUnit(channels=out_channels, dilation=9),
        )
        self.conv_trans = nn.Sequential(
            nn.ELU(),
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
        out = self.residuals(self.conv_trans(x))
        return out


class SEANet(nn.Module):
    def __init__(
        self,
        original_sr: int = 8000,
        target_sr: int = 16000,
        kernel_size: int = 7,
        block_channels: tp.Tuple[int] = (32, 64, 128, 256, 512),
        latent_channel: int = 128,
        block_strides: tp.Tuple[int] = (2, 2, 8, 8),
    ) -> None:
        """SEANet for bandwidth extension.

        Args:
            original_sr (int, optional): Original sample rate. Defaults to 8000.
            target_sr (int, optional): Target sample rate. Defaults to 16000.
            kernel_size (int, optional): The kernel size of the convolutional
                layers on both sides of the encoder and decoder. Defaults to 7.
            block_channels (tp.Tuple[int], optional): Number of channels in
                upsampling and downsampling. Defaults to (32, 64, 128, 256, 512).
            latent_channel (int, optional): Number of latent channel.
                Defaults to 128.
            block_strides (tp.Tuple[int], optional): Strides in upsampling and
                downsampling. Defaults to (2, 2, 8, 8).
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
                )
                for i in range(len(block_strides))
            ]
        )

        self.latent_conv = nn.Sequential(
            nn.ELU(),
            WNConv1d(
                in_channels=block_channels[-1],
                out_channels=latent_channel,
                kernel_size=kernel_size,
                padding="same",
                # bias=False,
                padding_mode="reflect",
            ),
            nn.ELU(),
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
                )
                for i in range(len(block_strides))
            ]
        )

        self.last_conv = nn.Sequential(
            nn.ELU(),
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
    net = SEANet()
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    print(f"net_out.shape: {net_out.shape} \n")
