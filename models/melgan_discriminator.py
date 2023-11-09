import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm


class MelganDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=(5, 3),
        base_channels=16,
        max_channels=1024,
        downsample_factors=(4, 4, 4, 4),
        groups_denominator=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        layer_kernel_size = np.prod(kernel_sizes)
        layer_padding = (layer_kernel_size - 1) // 2

        # initial layer
        self.layers += [
            nn.Sequential(
                nn.ReflectionPad1d(layer_padding),
                weight_norm(
                    nn.Conv1d(in_channels, base_channels, layer_kernel_size, stride=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        # downsampling layers
        layer_in_channels = base_channels
        for downsample_factor in downsample_factors:
            layer_out_channels = min(
                layer_in_channels * downsample_factor, max_channels
            )
            layer_kernel_size = downsample_factor * 10 + 1
            layer_padding = (layer_kernel_size - 1) // 2
            layer_groups = layer_in_channels // groups_denominator
            self.layers += [
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            layer_in_channels,
                            layer_out_channels,
                            kernel_size=layer_kernel_size,
                            stride=downsample_factor,
                            padding=layer_padding,
                            groups=layer_groups,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            ]
            layer_in_channels = layer_out_channels

        # last 2 layers
        layer_padding1 = (kernel_sizes[0] - 1) // 2
        layer_padding2 = (kernel_sizes[1] - 1) // 2
        self.layers += [
            nn.Sequential(
                weight_norm(
                    nn.Conv1d(
                        layer_out_channels,
                        layer_out_channels,
                        kernel_size=kernel_sizes[0],
                        stride=1,
                        padding=layer_padding1,
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            weight_norm(
                nn.Conv1d(
                    layer_out_channels,
                    out_channels,
                    kernel_size=kernel_sizes[1],
                    stride=1,
                    padding=layer_padding2,
                )
            ),
        ]

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feats


class MelganMultiscaleDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_scales=3,
        kernel_sizes=(5, 3),
        base_channels=16,
        max_channels=1024,
        downsample_factors=(4, 4, 4),
        pooling_kernel_size=4,
        pooling_stride=2,
        pooling_padding=2,
        groups_denominator=4,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                MelganDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    base_channels=base_channels,
                    max_channels=max_channels,
                    downsample_factors=downsample_factors,
                    groups_denominator=groups_denominator,
                )
                for _ in range(num_scales)
            ]
        )

        self.pooling = nn.AvgPool1d(
            kernel_size=pooling_kernel_size,
            stride=pooling_stride,
            padding=pooling_padding,
            count_include_pad=False,
        )

    def forward(self, x):
        scores = []
        feats = []
        for disc in self.discriminators:
            score, feat = disc(x)
            scores.append(score)
            feats.append(feat)
            x = self.pooling(x)
        return scores, feats


if __name__ == "__main__":
    audio = torch.randn(16, 1, 16384)  # Batch, Channel, Length
    net = MelganMultiscaleDiscriminator()
    net_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"net_total_params: {net_total_params*1e-6:.1f}M")
    net_in = audio
    net_out = net(net_in)
    print(f"net_in.shape: {net_in.shape}")
    # print(f"net_out.shape: {net_out.shape} \n")
