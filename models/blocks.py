import numpy as np
import torch

from torch import nn



def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-3, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, field_size,):
        super().__init__()
        if field_size == 1:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=True,),
                nn.BatchNorm2d(out_channel),
            )
        elif field_size == 3:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=True, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=True,),
                nn.BatchNorm2d(out_channel),
            )
        elif field_size == 5:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=True, ),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=True,),
                nn.Conv2d(out_channel, out_channel, kernel_size=5, padding=5 // 2, bias=True,),
                nn.BatchNorm2d(out_channel),
            )
        elif field_size == 7:
            self.Conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias=True,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=3 // 2, bias=True,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=2, padding=5 // 2, bias=True,),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=3, padding=7 // 2, bias=True,),
                nn.BatchNorm2d(out_channel),
            )
        else:
            raise ValueError("The size of the input receptive field is not in the [1, 3, 5, 7] range")

    def forward(self, x):
        return self.Conv(x)


class DRF(nn.Module):
    def __init__(self, n_images, patch_sizes, out_channels, field_sizes=None, norm: str = None,):
        super().__init__()
        self.patch_sizes = patch_sizes
        self.hid_features = n_images * patch_sizes*patch_sizes

        self.field_sizes = field_sizes if field_sizes is not None else [3, 5]
        self.out_features = out_channels
        self.first_conv = nn.Conv2d(self.hid_features, self.out_features, 1)

        self.conv_blocks = nn.ModuleList(
            [
                conv_block(
                    self.out_features, self.out_features, self.field_sizes[i],
                )
                for i in range(len(self.field_sizes))
            ]
        )
        # self.last_conv = nn.Conv2d(int(self.out_features * len(self.field_sizes)), self.out_features, 1)

    def forward(self, x, ):

        b, t, c, h, w = x.shape
        x = (x.reshape(b, t, c, h // self.patch_sizes, self.patch_sizes,
                       w // self.patch_sizes, self.patch_sizes,).permute(0, 1, 2, 4, 6, 3, 5)
             .reshape(b*t, self.hid_features, h // self.patch_sizes, w // self.patch_sizes))
        x = self.first_conv(x)
        f_map = []
        for blk in self.conv_blocks:
            x1 = blk(x)
            f_map.append(x1)
        x = torch.cat(f_map, dim=1)
        # x = self.last_conv(x)
        # x = self.last_norm(x)
        return x

