import gc
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from util import get_clones
import torch.nn.functional as F

# 禁用矩阵乘法的 TF32
torch.backends.cuda.matmul.allow_tf32 = False
# 禁用 cuDNN 卷积的 TF32
torch.backends.cudnn.allow_tf32 = False


# class WaveAct(nn.Module):
#     def __init__(self):
#         super(WaveAct, self).__init__()
#         self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
#         self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
#
#     def forward(self, x):
#         return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class WaveAct(nn.Module):
    def __init__(self, act_type='wavelet'):
        """
        统一的激活函数模块，用于进行消融实验。
        参数:
            act_type (str): 可选 'wavelet', 'relu', 'tanh'
        """
        super(WaveAct, self).__init__()
        self.act_type = act_type.lower()

        # 根据选择初始化参数或标准激活函数
        if self.act_type == 'wavelet':
            self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
            self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)
        elif self.act_type == 'relu':
            self.act = nn.ReLU()
        elif self.act_type == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数类型: {act_type}。请选择 'relu', 'tanh', 或 'wavelet'。")

    def forward(self, x):
        if self.act_type == 'wavelet':
            return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)
        else:
            return self.act(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dt_rank=32, dim_inner=None, d_state=None):
        super().__init__()
        self.attn1 = LinearAttention(d_model=d_model, heads=heads)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        self.z_proj = nn.Linear(d_model, d_model)
        self.x_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        skip = x

        z = self.z_proj(x)
        z = self.act1(z)
        x = self.x_proj(x)
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(self.conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x1 = self.act2(x)
        x = self.attn1(x1, x1, x1)
        x = x * z
        x = self.out_proj(x)
        x = skip + x
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, 8, 32, 8), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)

        return self.act(x)


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer, hidden_d_ff=512, heads=2):
        super(Model, self).__init__()

        self.linear_emb = nn.Linear(in_dim, hidden_dim)

        self.encoder = Encoder(hidden_dim, num_layer, heads)
        self.linear_out = nn.Sequential(*[
            nn.Linear(hidden_dim, hidden_d_ff),
            WaveAct(),
            nn.Linear(hidden_d_ff, hidden_d_ff),
            WaveAct(),
            nn.Linear(hidden_d_ff, out_dim)
        ])

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)

        output = self.linear_out(e_outputs)
        return output


class LinearAttention(nn.Module):
    def __init__(self, d_model, heads=2):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.act = WaveAct()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(3 * self.head_dim, 3 * self.head_dim)

    def phase_feature_map(self, x):
        feature = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        features = self.act(self.proj(feature) + feature)
        out = F.softplus(features)
        return out

    def forward(self, q, k, v):
        B, L, D = q.shape
        H = self.heads

        q = self.q_proj(q).view(B, L, H, -1).transpose(1, 2)  # [B, H, L, d_h]
        k = self.k_proj(k).view(B, L, H, -1).transpose(1, 2)  # [B, H, L, d_h]
        v = self.v_proj(v).view(B, L, H, -1).transpose(1, 2)  # [B, H, L, d_h]

        q_mapped = self.phase_feature_map(q)  # [B, H, L, 3 * d_h]
        k_mapped = self.phase_feature_map(k)  # [B, H, L, 3 * d_h]

        kv_state = torch.einsum('bhld,bhlm->bhldm', k_mapped, v)

        kv_cumsum = torch.cumsum(kv_state, dim=2)  # [B, H, L, 3*d_h, d_h]

        num = torch.einsum('bhld,bhldm->bhlm', q_mapped, kv_cumsum)  # [B, H, L, d_h]

        k_cumsum = torch.cumsum(k_mapped, dim=2)  # [B, H, L, 3*d_h]

        denom = torch.einsum('bhld,bhld->bhl', q_mapped, k_cumsum)  # [B, H, L]

        out = num / (denom.unsqueeze(-1) + 1e-6)

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)
