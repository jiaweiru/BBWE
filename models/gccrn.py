import torch
import torchaudio
from torch import nn
from einops import rearrange

EPSILON = torch.finfo(torch.float32).eps


class GLSTM(nn.Module):
    def __init__(
        self,
        mid_features=None,
        hidden_size=1024,
        groups=2,
    ):
        super(GLSTM, self).__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )
        self.lstm_list2 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat(
            [self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, output_padding=(0, 0)
    ):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


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


class GCCRN(nn.Module):
    # The GCCRN parameters are consistent with the original paper
    # and therefore only support upsampling to 16khz
    def __init__(self, origin_sr=8000, new_sr=16000, alpha=0.5, res=True):
        super(GCCRN, self).__init__()

        self.origin_sr = origin_sr
        self.new_sr = new_sr
        self.alpha = alpha
        self.res = res

        self.conv1 = GluConv2d(
            in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv2 = GluConv2d(
            in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv3 = GluConv2d(
            in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv4 = GluConv2d(
            in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv5 = GluConv2d(
            in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2)
        )

        self.glstm = GLSTM(groups=2)

        self.conv5_t_1 = GluConvTranspose2d(
            in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv4_t_1 = GluConvTranspose2d(
            in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv3_t_1 = GluConvTranspose2d(
            in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv2_t_1 = GluConvTranspose2d(
            in_channels=64,
            out_channels=16,
            kernel_size=(1, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_1 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2)
        )

        self.conv5_t_2 = GluConvTranspose2d(
            in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv4_t_2 = GluConvTranspose2d(
            in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv3_t_2 = GluConvTranspose2d(
            in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2)
        )
        self.conv2_t_2 = GluConvTranspose2d(
            in_channels=64,
            out_channels=16,
            kernel_size=(1, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_2 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2)
        )

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.bn5_t_1 = nn.BatchNorm2d(128)
        self.bn4_t_1 = nn.BatchNorm2d(64)
        self.bn3_t_1 = nn.BatchNorm2d(32)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(1)

        self.bn5_t_2 = nn.BatchNorm2d(128)
        self.bn4_t_2 = nn.BatchNorm2d(64)
        self.bn3_t_2 = nn.BatchNorm2d(32)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)

        self.fc1 = nn.Linear(in_features=161, out_features=161)
        self.fc2 = nn.Linear(in_features=161, out_features=161)

    def forward(self, speech):
        # GCCRN For BWE
        speech = torchaudio.functional.resample(
            speech, self.origin_sr, self.new_sr
        )  # Default parameters for torchaudio upsampling
        speech = torch.squeeze(speech, dim=1)
        spec = torch.stft(
            speech,
            320,
            160,
            320,
            window=torch.hann_window(320).to(speech.device),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )

        spec = rearrange(spec, "(b c) f t -> b c t f", b=speech.shape[0])
        spec_comp = power_comp(spec, self.alpha)

        out = torch.cat([spec_comp.real, spec_comp.imag], dim=1)
        e1 = self.elu(self.bn1(self.conv1(out)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))

        out = e5

        out = self.glstm(out)

        out = torch.cat((out, e5), dim=1)

        d5_1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(out)), e4), dim=1))
        d4_1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(d5_1)), e3), dim=1))
        d3_1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(d4_1)), e2), dim=1))
        d2_1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(d3_1)), e1), dim=1))
        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))

        d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e4), dim=1))
        d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e3), dim=1))
        d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e2), dim=1))
        d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e1), dim=1))
        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))
        # print(e1.shape, e2.shape, e3.shape, e4.shape, e5.shape)
        # print(d5_1.shape, d4_1.shape, d3_1.shape, d2_1.shape, d1_1.shape)

        out1 = self.fc1(d1_1)
        out2 = self.fc2(d1_2)

        est_spec_comp = torch.complex(out1, out2)
        est_spec = power_comp(est_spec_comp, 1.0 / self.alpha)

        est_spec = rearrange(est_spec, "b c t f -> (b c) f t", b=speech.shape[0])
        est_speech = torch.istft(
            est_spec,
            320,
            160,
            320,
            window=torch.hann_window(320).to(speech.device),
            center=True,
        )
        est_speech = est_speech + speech if self.res else est_speech
        est_speech = torch.unsqueeze(est_speech, dim=1)
        est_speech = torch.clamp(est_speech, -1, 1)

        return est_speech


if __name__ == "__main__":
    audio = torch.randn(1, 1, 16000)  # Batch, Channel, Length
    net = GCCRN()
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    print(f"net_out.shape: {net_out.shape} \n")
