import typing as tp

import torch
import torchaudio
from torch import nn
from einops import rearrange


EPSILON = torch.finfo(torch.float32).eps


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRN(nn.Module):
    def __init__(
        self,
        original_sr: int = 8000,
        target_sr: int = 16000,
        win_len: int = 512,
        win_inc: int = 256,
        fft_len: int = 512,
        win_fn: tp.Callable[[int], torch.Tensor] = torch.hann_window,
        kernel_size: tp.Tuple[tp.Tuple[int, int], ...] = (((3, 3),) * 6),
        kernel_stride: tp.Tuple[tp.Tuple[int, int], ...] = (((2, 1),) * 6),
        conv_padding: tp.Tuple[tp.Tuple[int, int], ...] = (((1, 1),) * 6),
        kernel_num_encoder: tp.Tuple[int, ...] = (16, 32, 64, 128, 128, 128),
        kernel_num_decoder: tp.Tuple[int, ...] = (128, 128, 64, 32, 16, 16),
        rnn_num: int = 2,
        rnn_units: int = 640,
    ) -> None:
        super().__init__()

        self.original_sr = original_sr
        self.target_sr = target_sr

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = win_fn(self.win_len)

        kernel_num_encoder = (1,) + kernel_num_encoder
        self.encoder = nn.ModuleList(
            [
                Conv2dBlock(
                    kernel_num_encoder[i],
                    kernel_num_encoder[i + 1],
                    kernel_size[i],
                    kernel_stride[i],
                    conv_padding[i],
                )
                for i in range(len(kernel_stride))
            ]
        )
        self.latent_dim = kernel_num_encoder[-1]
        self.decoder = nn.ModuleList(
            [
                ConvTranspose2dBlock(
                    kernel_num_encoder[-i - 1] + kernel_num_decoder[i - 1]
                    if i != 0
                    else kernel_num_encoder[-i - 1] * 2,
                    kernel_num_decoder[i],
                    kernel_size[-i - 1],
                    kernel_stride[-i - 1],
                    conv_padding[-i - 1],
                )
                for i in range(len(kernel_stride))
            ]
        )

        rnn_input_dim = self.win_len // 2 + 1
        for k, s in zip(kernel_size, kernel_stride):
            rnn_input_dim = (rnn_input_dim + 2 * (k[0] // 2) - k[0]) // s[0] + 1
            # padding is k[0] // 2
        rnn_input_dim = rnn_input_dim * kernel_num_encoder[-1]

        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=rnn_units,
            num_layers=rnn_num,
            batch_first=True,
            bidirectional=False,  # Consistent with the paper
        )

        self.final_conv = nn.Conv2d(
            kernel_num_decoder[-1],
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        speech = torchaudio.functional.resample(
            speech, self.original_sr, self.target_sr
        )  # Default parameters for torchaudio upsampling
        speech = torch.squeeze(speech, dim=1)
        spec = torch.stft(
            speech,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(speech.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        spec = rearrange(spec, "(b c) f t -> b c f t", b=speech.shape[0])
        mag_comp = torch.log1p(
            torch.sqrt(torch.clamp(spec.real**2 + spec.imag**2, EPSILON))
        )

        fake_phase = torch.atan2(spec.imag, spec.real)

        # The original paper was mirrored with the axis of symmetry of the fft_len / 4
        # frequency point axis, where the DC component was considered
        # mirrored_phase = -torch.flip(fake_phase[:, :, : self.fft_len // 4], [2])

        # Phase inversion without considering the DC component
        mirrored_phase = -torch.flip(fake_phase[:, :, 1 : self.fft_len // 4 + 1], [2])
        est_phase = torch.cat(
            [fake_phase[:, :, : self.fft_len // 4 + 1], mirrored_phase], dim=2
        )

        # Forward pass
        x = mag_comp
        encoder_output = []
        for conv in self.encoder:
            x = conv(x)
            encoder_output.append(x)
        x = rearrange(x, "b c f t -> b t (f c)")
        x, _ = self.rnn(x)
        x = rearrange(x, "b t (f c) -> b c f t", c=self.latent_dim)
        for idx, transconv in enumerate(self.decoder):
            x = torch.cat([x, encoder_output[-idx - 1]], dim=1)
            x = transconv(x)
        x = self.final_conv(x)

        est_mag_comp = x
        est_mag = torch.expm1(est_mag_comp)
        est_spec = torch.complex(
            est_mag * torch.cos(est_phase), est_mag * torch.sin(est_phase)
        )

        est_spec = rearrange(est_spec, "b c f t -> (b c) f t", b=x.shape[0])
        est_speech = torch.istft(
            est_spec,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=self.window.to(x.device),
            center=True,
        )
        est_speech = torch.unsqueeze(est_speech, dim=1)
        est_speech = torch.clamp(est_speech, -1, 1)
        return est_speech


if __name__ == "__main__":
    audio = torch.randn(1, 1, 12800)  # Batch, Channel, Length
    net = CRN()
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    print(f"net_out.shape: {net_out.shape} \n")
