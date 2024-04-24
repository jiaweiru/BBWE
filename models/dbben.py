import torch
import torchaudio
from torch import nn
from einops import rearrange

EPSILON = torch.finfo(torch.float32).eps


class Encoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )
        self.Module_list = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

    def forward(self, x):
        sc_list = []
        for i in range(len(self.Module_list)):
            x = self.Module_list[i](x)
            sc_list.append(x)
            # print(x.shape)
        return x, sc_list


class TwoLSTM(nn.Module):
    def __init__(self, input_dim, rnn_units=1024):
        """
        input: [B, T, F * C]
        output: [B, T, F * C]
        """
        super().__init__()
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
        )
        # self.transform = nn.Linear(640, self.input_dim)

    def forward(self, x):

        batch_size, channels, lengths, dims = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, lengths, channels * dims)
        x, _ = self.lstm(x)
        # x = self.transform(x)
        x = x.view(batch_size, lengths, channels, dims).permute(0, 2, 1, 3).contiguous()

        return x


class Decoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1)
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        deconv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU(),
        )
        conv_last = nn.Conv2d(16, channel, kernel_size=(1, 1), stride=(1, 1))
        self.Module_list = nn.ModuleList(
            [deconv1, deconv2, deconv3, deconv4, deconv5, conv_last]
        )

    def forward(self, x, sc_list):
        for i in range(len(self.Module_list)):
            if i != (len(self.Module_list) - 1):
                x = self.Module_list[i](
                    torch.cat([x, sc_list[len(self.Module_list) - 2 - i]], 1)
                )
            else:
                x = self.Module_list[i](x)
            # print(x.shape)

        return x


def replace_denormals(x, eps):
    y = x.clone()
    y[torch.abs(x) < eps] = eps
    return y


def power_comp(x, alpha=0.5):
    # mag = torch.abs(x)
    mag = torch.sqrt(torch.clamp(x.real**2 + x.imag**2, EPSILON))
    # phase = torch.angle(x)
    phase = torch.arctan2(
        replace_denormals(x.imag, EPSILON), replace_denormals(x.real, EPSILON)
    )
    mag_comp = torch.pow(mag, alpha)
    # return torch.polar(mag_comp, phase)
    return torch.complex(mag_comp * torch.cos(phase), mag_comp * torch.sin(phase))


class DBBEN(nn.Module):
    def __init__(
        self,
        origin_sr=8000,
        new_sr=16000,
        fft_len=320,
        win_inc=160,
        win_len=320,
        alpha=0.5,
        res=True,
        dualbranch=True,
    ):
        super().__init__()
        self.origin_sr = origin_sr
        self.new_sr = new_sr
        self.fft_len = fft_len
        self.win_inc = win_inc
        self.win_len = win_len
        self.alpha = alpha
        self.res = res
        self.dualbranch = dualbranch

        self.magencoder = Encoder(channel=1)
        self.magdecoder = Decoder(channel=1)
        self.maglstm = TwoLSTM(1024)

        if self.dualbranch:
            self.complexencoder = Encoder(channel=2)
            self.comlpexdecoder = Decoder(channel=2)
            self.complexlstm = TwoLSTM(1024)

    def forward(self, speech):
        speech = torchaudio.functional.resample(
            speech, self.origin_sr, self.new_sr
        )  # Default parameters for torchaudio upsampling
        speech = torch.squeeze(speech, dim=1)
        spec = torch.stft(
            speech,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=torch.hann_window(320).to(speech.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        spec = rearrange(spec, "(b c) f t -> b c t f", b=speech.shape[0])
        spec_comp = power_comp(spec, self.alpha)
        x = torch.sqrt(torch.clamp(spec_comp.real**2 + spec_comp.imag**2, EPSILON))
        x_c = torch.cat([spec_comp.real, spec_comp.imag], dim=1)

        f, f_list = self.magencoder(x)
        f = self.maglstm(f)
        y = self.magdecoder(f, f_list)
        mags_m = torch.squeeze(y, dim=1)

        if self.dualbranch:
            f_c, f_c_list = self.complexencoder(x_c)
            f_c = self.complexlstm(f_c)
            y_c = self.comlpexdecoder(f_c, f_c_list)

            mags_c = torch.sqrt(torch.clamp(y_c[:, 0] ** 2 + y_c[:, 1] ** 2, EPSILON))
            phase_c = torch.arctan2(
                replace_denormals(y_c[:, 1], EPSILON),
                replace_denormals(y_c[:, 0], EPSILON),
            )
            est_mags_comp = (mags_m + mags_c) * 0.5
            est_phase = phase_c
            est_spec_comp = torch.complex(
                est_mags_comp * torch.cos(est_phase),
                est_mags_comp * torch.sin(est_phase),
            )
        else:
            est_phase = torch.arctan2(
                replace_denormals(spec_comp.imag, EPSILON),
                replace_denormals(spec_comp.real, EPSILON),
            )
            est_phase = torch.squeeze(est_phase, dim=1)
            est_mags_comp = mags_m
            est_spec_comp = torch.complex(
                est_mags_comp * torch.cos(est_phase),
                est_mags_comp * torch.sin(est_phase),
            )
        est_spec = power_comp(est_spec_comp, 1.0 / self.alpha)
        est_spec = torch.unsqueeze(est_spec, dim=1)
        est_spec = rearrange(est_spec, "b c t f -> (b c) f t", b=speech.shape[0])
        est_speech = torch.istft(
            est_spec,
            self.fft_len,
            self.win_inc,
            self.win_len,
            window=torch.hann_window(320).to(speech.device),
            center=True,
        )
        est_speech = est_speech + speech if self.res else est_speech
        est_speech = torch.unsqueeze(est_speech, dim=1)
        est_speech = torch.clamp(est_speech, -1, 1)

        return est_speech


if __name__ == "__main__":
    audio = torch.randn(1, 1, 12800)  # Batch, Channel, Length
    net = DBBEN(dualbranch=False)
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    print(f"net_out.shape: {net_out.shape} \n")
