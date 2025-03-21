import warnings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn import init

from cacti.models.builder import MODELS


########
# 5000消融实验
# 1.直接残差连接
# 2.加上FB
# 3.加上交叉注意力
###############
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Attention(nn.Module):
    def __init__(self, dim, length):
        super().__init__()
        self.pc_proj_q = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_q = nn.Parameter(torch.FloatTensor([1.]))
        self.pc_proj_k = nn.Linear(dim, 1, bias=False)
        self.bias_pc_proj_k = nn.Parameter(torch.FloatTensor([1.]))
        self.mlp1 = nn.Sequential(
            nn.Linear(length, 1, bias=False),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(length, length, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(length, 1, bias=False),
        )

    def forward(self, q, k):
        Sigma_q = self.pc_proj_q(q) + self.bias_pc_proj_q
        Sigma_k = self.pc_proj_k(k) + self.bias_pc_proj_k
        sim = einsum('b h B i d, b h B j d -> b h B i j', q, k)
        Sigma = einsum('b h B i d, b h B j d -> b h B i j', Sigma_q, Sigma_k)

        diag_sim = torch.diagonal(sim, dim1=-2, dim2=-1)
        sim_norm = sim - torch.diag_embed(diag_sim)
        theta = self.mlp1(sim_norm).squeeze(-1)
        theta = self.mlp2(theta).unsqueeze(-1)

        sim = sim * Sigma
        attn = sim.softmax(dim=-1) * (sim > theta)
        return attn


class FA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, shift=True):
        super().__init__()

        if sq_dim is None:
            self.rank = dim
        else:
            self.rank = sq_dim
        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head
        self.window_size = window_size
        self.shift = shift

        num_token = window_size[0] * window_size[1]
        self.cal_atten = Attention(dim_head, num_token)

        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_qk = nn.Linear(dim, self.rank * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def cal_attention(self, x):
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b n B (h d) -> b h B n d', h=self.heads_qk), (q, k))
        v = rearrange(v, 'b n B (h d) -> b h B n d', h=self.heads_v)
        attn = self.cal_atten(q, k)
        out = einsum('b h B i j, b h B j d -> b h B i d', attn, v)
        out = rearrange(out, 'b h B n d -> b n B (h d)')
        out = self.to_out(out)
        return out

    def forward(self, x):
        b, c, B, h, w = x.shape
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=4, dims=3).roll(shifts=4, dims=4)
        x_inp = rearrange(x, 'b c B (h b0) (w b1)-> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])
        out = self.cal_attention(x_inp)
        out = rearrange(out, '(b h w) (b0 b1) B c -> b c B (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        if self.shift:
            out = out.roll(shifts=-4, dims=3).roll(shifts=-4, dims=4)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out


class MPMLP(nn.Module):
    def __init__(self, dim, multi=4):
        super(MPMLP, self).__init__()
        self.multi = multi
        self.pwconv1 = nn.Sequential(
            nn.Conv3d(dim, dim * multi, 1, groups=dim, bias=False),
            GELU(),
        )
        self.groupconv = nn.Sequential(
            nn.Conv3d(dim * multi, dim * multi, 1, groups=multi, bias=False),
            GELU(),
        )
        self.pwconv2 = nn.Conv3d(dim * multi, dim, 1, groups=dim, bias=False)

    def forward(self, x):
        x = self.pwconv1(x.permute(0, 4, 1, 2, 3))
        x = rearrange(x, 'b (c m) B h w-> b (m c) B h w', m=self.multi)
        x = self.groupconv(x)
        x = rearrange(x, 'b (m c) B h w-> b (c m) B h w', m=self.multi)
        x = self.pwconv2(x)
        return x.permute(0, 2, 3, 4, 1)


class FAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        # self.pos_emb = nn.Conv3d(dim, dim, 5, 1, 2, bias=False, groups=dim)
        self.fa = PreNorm(dim, FA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift),
                          norm_type='ln')
        self.ffn = PreNorm(dim, FeedForward(dim=dim, mult=mult), norm_type='ln')

    def forward(self, x):
        # x = x + self.pos_emb(x)
        x = self.fa(x) + x
        x = self.ffn(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class TSAB(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        self.tsab = PreNorm(dim, TimesAttention3D(dim, num_head), norm_type='ln')
        self.ffn = PreNorm(dim, FeedForward(dim=dim), norm_type='ln')

    def forward(self, x):
        x = self.tsab(x) + x
        x = self.ffn(x) + x
        return x


class CrossPreNorm(nn.Module):
    def __init__(self, dim, fn, norm_type='ln'):
        super().__init__()
        self.fn = fn
        self.norm_type = norm_type
        if norm_type == 'ln':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.GroupNorm(dim, dim)
            self.norm2 = nn.GroupNorm(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, last, *args, **kwargs):
        if self.norm_type == 'ln':
            x = self.norm1(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
            last = self.norm2(last.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            x = self.norm1(x)
            last = self.norm2(last)
        return self.fn(x, last, *args, **kwargs)


class STSAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.FAB = FAB(dim=dim, sq_dim=sq_dim, window_size=window_size, dim_head=dim_head, mult=mult, shift=shift)
        self.TSAB = TSAB(dim, num_head=dim_head)

    def forward(self, x):
        x = self.FAB(x)
        x = self.TSAB(x)
        return x


class STConv3d(nn.Module):
    def __init__(self, dim, groups=1, bias=True):
        super().__init__()
        self.sw_conv = nn.Sequential(
            nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), groups=groups, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), groups=groups, bias=bias),
            nn.LeakyReLU(inplace=True),
        )
        self.tw_conv = nn.Conv3d(dim, dim, (3, 1, 1), 1, (1, 0, 0), groups=groups, bias=bias)

    def forward(self, x):
        x1 = self.sw_conv(x)
        x2 = self.tw_conv(x)
        y = x1 + x2
        return y


class StageInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.st_inter_enc = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.st_inter_dec = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.act_fn = nn.LeakyReLU()
        self.phi1 = nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim)
        # self.phi2 = nn.Conv3d(dim, dim, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim)
        self.gamma1 = nn.Conv3d(dim, dim, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim)
        # self.gamma2 = nn.Conv3d(dim, dim, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim)
        # self.dwconv = STConv3d(dim, groups=dim)
        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Conv3d(dim, dim, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(dim, dim, 1, padding=0),
        #     nn.Sigmoid(),
        # )

    def forward(self, inp, pre_enc, pre_dec):
        # pre_enc = inp
        out = self.st_inter_dec(pre_dec)
        skip = self.act_fn(out)
        phi1 = torch.sigmoid(self.phi1(skip))
        # phi2 = torch.sigmoid(self.phi2(skip))
        # skip2 = skip.permute(0, 2, 1, 3, 4)
        # phi = torch.sigmoid(self.phi1(skip) + self.phi2(skip2).permute(0, 2, 1, 3, 4))
        gamma1 = self.gamma1(skip)
        # gamma2 = self.gamma1(skip)
        # gamma = self.gamma1(skip) + self.gamma2(skip2).permute(0, 2, 1, 3, 4)

        # out = (phi1 * inp + gamma1) * phi2 + gamma2
        out = phi1 * inp + gamma1
        # out = phi1 * self.dwconv(inp) + gamma1
        # out = self.attention(skip) * out

        return out


class STT(nn.Module):
    def __init__(self, in_dim=8, out_dim=8, dim=8):
        super(STT, self).__init__()

        self.conv_in = nn.Conv3d(in_dim, dim, 3, 1, 1, bias=False)
        self.down1 = STSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=4)
        self.downsample1 = nn.Conv3d(dim, dim * 2, (3, 4, 4), (1, 2, 2), (1, 1, 1), bias=False)
        self.down2 = STSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4)
        self.downsample2 = nn.Conv3d(dim * 2, dim * 4, (3, 4, 4), (1, 2, 2), (1, 1, 1), bias=False)

        self.bottleneck_local = STSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4)
        self.bottleneck_swin = STSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        self.upsample2 = nn.ConvTranspose3d(dim * 4, dim * 2, (1, 2, 2), (1, 2, 2))
        self.fusion2 = nn.Conv3d(dim * 4, dim * 2, 1, 1, 0, bias=False)
        self.up2 = STSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        self.upsample1 = nn.ConvTranspose3d(dim * 2, dim, (1, 2, 2), (1, 2, 2))
        self.fusion1 = nn.Conv3d(dim * 2, dim, 1, 1, 0, bias=False)
        self.up1 = STSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        self.conv_out = nn.Conv3d(dim, out_dim, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
            x: [b,c,B,h,w]
            return out:[b,c,B,h,w]
        """
        b, c, B, h_inp, w_inp = x.shape
        x_in = x
        x = self.conv_in(x_in)
        x1 = self.down1(x)
        # stage_outs.append(x1)
        x = self.downsample1(x1)
        x2 = self.down2(x)

        x = self.downsample2(x2)
        # 通道 * 4
        x_local = self.bottleneck_local(x[:, :c * 2, :, :])

        x_swin = self.bottleneck_swin(x[:, c * 2:, :, :] + x_local)

        x = torch.cat([x_local, x_swin], dim=1)

        x = self.upsample2(x)
        x = x2 + self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)

        x = self.upsample1(x)
        x = x1 + self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        out = self.conv_out(x) + x_in

        return out, x


class ChanelAttention(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv3d(dim * 2, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.CA = nn.Sequential(
            # nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, last):
        x = self.dconv(x)
        x = self.ln(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        phi = self.CA(last)
        x1 = phi * x
        out = x1 + x
        return out


class FusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fusion = nn.Conv3d(in_dim * 2, in_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # self.SF = StageInteraction(dim=in_dim)
        self.CA = ChanelAttention(in_dim, out_dim)

    def forward(self, x, last):
        x = self.fusion(torch.cat([x, last], dim=1))
        out = self.CA(x, last)
        # x = self.SF(x, last, last)
        return out


class SCA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, shift=True):
        super().__init__()

        if sq_dim is None:
            self.rank = dim
        else:
            self.rank = sq_dim
        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head
        self.window_size = window_size
        self.shift = shift

        num_token = window_size[0] * window_size[1]
        self.cal_atten = Attention(dim_head // 2, num_token)

        self.to_q = nn.Conv3d(dim, dim // 2, 3, 1, padding=1, bias=False, groups=dim // 2)
        self.to_k = nn.Conv3d(dim, dim // 2, 3, 1, padding=1, bias=False, groups=dim // 2)
        self.to_v = nn.Conv3d(dim, dim // 2, 3, 1, padding=1, bias=False, groups=dim // 2)
        self.to_out = nn.Conv3d(dim // 2, dim, 1, 1, bias=False, groups=dim // 2)
        # self.to_out = nn.Linear(dim, dim)

    def cal_attention(self, x, last):
        w_size = self.window_size
        q = self.to_q(x)
        k = self.to_k(last)
        v = self.to_v(last)
        q = rearrange(q, 'b c B (h b0) (w b1) -> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])
        k = rearrange(k, 'b c B (h b0) (w b1) -> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])
        v = rearrange(v, 'b c B (h b0) (w b1) -> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])

        q, k = map(lambda t: rearrange(t, 'b n B (h d) -> b h B n d', h=self.heads_qk), (q, k))
        v = rearrange(v, 'b n B (h d) -> b h B n d', h=self.heads_v)
        attn = self.cal_atten(q, k)
        # attn = q @ k.transpose(-2, -1)
        out = einsum('b h B i j, b h B j d -> b h B i d', attn, v)
        out = rearrange(out, 'b h B n d -> b n B (h d)')
        return out

    def forward(self, x, last):
        b, c, B, h, w = x.shape
        w_size = self.window_size
        out = self.cal_attention(x, last)
        out = rearrange(out, '(b h w) (b0 b1) B c -> b c B (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        out = self.to_out(out)
        return out


class CrossSpatialFFN(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.x = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            nn.Conv3d(dim * mult, dim, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim),
        )
        self.last = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            nn.Conv3d(dim * mult, dim, (1, 3, 3), 1, (0, 1, 1), bias=False, groups=dim),
            GELU()
        )

    def forward(self, x, last):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.x(x) * self.last(last)
        return out


class SCAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.SCA = CrossPreNorm(dim,
                                SCA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift),
                                norm_type='crossln')
        self.ffn = CrossPreNorm(dim, CrossSpatialFFN(dim, mult=mult), norm_type='ln')

    def forward(self, x, last):
        x = self.SCA(x, last) + x
        x = self.ffn(x, last) + x
        return x


class TCA(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        head_dim = num_head
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1, bias=qkv_bias, groups=dim // 2)
        self.k = nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1, bias=qkv_bias, groups=dim // 2)
        self.v = nn.Conv3d(dim, dim // 2, kernel_size=3, padding=1, bias=qkv_bias, groups=dim // 2)

        self.proj = nn.Conv3d(dim // 2, dim, kernel_size=1, bias=qkv_bias, groups=dim // 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, last):
        # x = x.permute(0, 2, 1, 3, 4)
        _, _, _, h, w = x.shape

        q = self.q(x)
        k = self.k(last)
        v = self.v(last)
        # skip = self.act_fn(self.skip(last))
        q = rearrange(q, "b c B h w->(b h w) B c")
        k = rearrange(k, "b c B h w->(b h w) B c")
        v = rearrange(v, "b c B h w->(b h w) B c")
        # phi = self.phi(skip)
        # gamma = self.gamma(skip)

        n, B, C = q.shape
        q = q.reshape(n, B, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(n, B, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(n, B, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # C = C // 2

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # 无嵌入向量
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(n, B, C)
        x = rearrange(x, "(b h w) B c->b c B h w", h=h, w=w)
        x = self.proj(x)
        # x = x * phi + gamma
        # x = x.permute(0, 4, 1, 2, 3)
        return x


class CrossTimeFFN(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.x = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            nn.Conv3d(dim * mult, dim, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim),
        )
        self.last = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            nn.Conv3d(dim * mult, dim, (3, 1, 1), 1, (1, 0, 0), bias=False, groups=dim),
            GELU()
        )

    def forward(self, x, last):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.x(x) * self.last(last)
        return out


class TCAB(nn.Module):
    def __init__(self, dim, num_head):
        super().__init__()
        self.TCA = CrossPreNorm(dim, TCA(dim, num_head), norm_type='ln')
        self.ffn = CrossPreNorm(dim, CrossTimeFFN(dim), norm_type='ln')

    def forward(self, x, last):
        x = self.TCA(x, last) + x
        x = self.ffn(x, last) + x
        return x


class STCAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.SCAB = SCAB(dim=dim, sq_dim=sq_dim, window_size=window_size, dim_head=dim_head, mult=mult,
                         shift=shift)
        self.TCAB = TCAB(dim, num_head=dim_head)

    def forward(self, x, last):
        x = self.SCAB(x, last)
        x = self.TCAB(x, last)
        return x


class SA(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, shift=False):
        super().__init__()

        if sq_dim is None:
            self.rank = dim
        else:
            self.rank = sq_dim
        self.heads_qk = sq_dim // dim_head
        self.heads_v = dim // dim_head
        self.window_size = window_size
        self.shift = shift

        num_token = window_size[0] * window_size[1]
        self.cal_atten = Attention(dim_head // 2, num_token)

        self.to_v = nn.Linear(dim, dim // 2, bias=False)
        self.to_qk = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim // 2, dim)

    def cal_attention(self, x):
        q, k = self.to_qk(x).chunk(2, dim=-1)
        v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b n B (h d) -> b h B n d', h=self.heads_qk), (q, k))
        v = rearrange(v, 'b n B (h d) -> b h B n d', h=self.heads_v)
        attn = self.cal_atten(q, k)
        # attn = q @ k.transpose(-2, -1)
        out = einsum('b h B i j, b h B j d -> b h B i d', attn, v)
        out = rearrange(out, 'b h B n d -> b n B (h d)')
        out = self.to_out(out)
        return out

    def forward(self, x):
        b, c, B, h, w = x.shape
        w_size = self.window_size
        x_inp = rearrange(x, 'b c B (h b0) (w b1)-> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])
        out = self.cal_attention(x_inp)
        out = rearrange(out, '(b h w) (b0 b1) B c -> b c B (h b0) (w b1)', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        return out


class DSSCSAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.FA = PreNorm(dim, SA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift),
                          norm_type='ln')
        self.SCA = CrossPreNorm(dim,
                                SCA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim,
                                    shift=shift),
                                norm_type='ln')
        self.ffn = CrossPreNorm(dim, CrossSpatialFFN(dim, mult=mult), norm_type='ln')

    def forward(self, x, last):
        x = self.FA(x) + x
        x = self.SCA(x, last) + x
        x = self.ffn(x, last) + x
        return x


class DSSCAB(nn.Module):
    def __init__(self, dim, dim_head=28):
        super().__init__()
        self.DSSCSAB = DSSCSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=2)

        self.DSSCTAB = DSSCTAB(dim, num_head=dim_head, mult=2)

    def forward(self, x, last):
        x = self.DSSCSAB(x, last)
        x = self.DSSCTAB(x, last)
        return x


class TA(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        head_dim = num_head
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim // 2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim // 2, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        _, _, _, h, w = x.shape
        tsab_in = rearrange(x, "b c B h w->(b h w) B c")
        n, B, C = tsab_in.shape
        C = C // 2
        qkv = self.qkv(tsab_in)
        qkv = qkv.reshape(n, B, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # 无嵌入向量
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(n, B, C)
        x = self.proj(x)
        x = rearrange(x, "(b h w) B c->b c B h w", h=h, w=w)
        return x


class DSSCTAB(nn.Module):
    def __init__(self, dim, num_head, mult):
        super().__init__()
        self.tsab = PreNorm(dim, TA(dim, num_head), norm_type='ln')
        self.TCA = CrossPreNorm(dim,
                                TCA(dim, num_head), norm_type='ln')
        self.ffn = CrossPreNorm(dim, CrossTimeFFN(dim, mult), norm_type='ln')

    def forward(self, x, last):
        x = self.tsab(x) + x
        x = self.TCA(x, last) + x
        x = self.ffn(x, last) + x
        # x = self.CA(x, last)
        return x


class DSTCT(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, dim=4):
        super(DSTCT, self).__init__()

        # self.down1 = STSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=1)
        # self.fusion1 = nn.Conv3d(dim * 2, dim, 1, 1, 0, bias=False)
        # self.down1_action = StageInteraction(dim=dim)
        # self.up1_action = StageInteraction(dim=dim)
        self.FB = FusionBlock(dim, dim)
        # self.STCAB = STCAB(dim=dim, sq_dim=dim, dim_head=dim // 2, mult=2)
        self.reuse = DSSCAB(dim=dim, dim_head=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, stage_ins=None):
        """
            x: [b,c,B,h,w]
            return out:[b,c,B,h,w]
        """
        b, c, B, h_inp, w_inp = x.shape
        # if stage_outs is None:
        #     stage_outs = []
        x_in = x
        # x = self.down1(x)
        # x1 = self.down1_action(x1, stage_ins[0], stage_ins[1])
        # stage_outs.append(x1)
        x = self.FB(x, stage_ins)
        x = self.reuse(x, stage_ins)
        # x = x1 + self.fusion1(torch.cat([stage_ins[1], x1], dim=1))
        # x = self.up1_action(x, stage_ins[1], stage_ins[0])
        # stage_out = x

        out = x + x_in

        return out[:, :, :h_inp, :w_inp], x


class Mu_Estimator(nn.Module):
    def __init__(self, in_nc=8, out_nc=1, channel=8):
        super(Mu_Estimator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_nc, channel, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())

    def forward(self, x):
        x = self.conv(x)
        x = self.avpool(x)
        x = self.mlp(x) + 1e-6
        return x


class TimesAttention3D(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        head_dim = num_head
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        _, _, _, h, w = x.shape
        tsab_in = rearrange(x, "b c B h w->(b h w) B c")
        n, B, C = tsab_in.shape
        qkv = self.qkv(tsab_in)
        qkv = qkv.reshape(n, B, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # 无嵌入向量
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(n, B, C)
        x = self.proj(x)
        x = rearrange(x, "(b h w) B c->b c B h w", h=h, w=w)
        return x


class FEM(nn.Module):
    def __init__(self, in_dim=1, dim=3):
        super(FEM, self).__init__()
        self.fem = nn.Sequential(
            nn.Conv3d(in_dim, dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim * 2, dim * 4, kernel_size=3, stride=(1, 1, 1), padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.fem(x)


@MODELS.register_module
class NetVideo_base_reuse_2(torch.nn.Module):
    def __init__(self, opt):
        super(NetVideo_base_reuse_2, self).__init__()

        self.stage = opt.stage
        self.nC = opt.bands
        self.dim = opt.dim
        self.size = opt.size
        self.reuse = opt.reuse

        self.conv3d = nn.Conv3d(self.dim * 2, self.dim, 1, 1, 0)

        self.body_share_params = opt.body_share_params
        para_estimator = []
        for i in range(opt.stage):
            para_estimator.append(Mu_Estimator(in_nc=opt.dim))
        self.mu = nn.ModuleList(para_estimator)
        self.net = nn.ModuleList()
        for i in range(len(opt.reuse)):
            j = opt.reuse[i]
            if j == 1:
                self.net.append(nn.ModuleList([
                    STT(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
                ]))
            elif j == 0:
                self.net.append(nn.ModuleList([
                    DSTCT(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
                ]))

        self.fem = FEM(in_dim=1, dim=self.dim // 4)

        # 特征恢复块
        self.vrm = nn.Sequential(
            nn.Conv3d(opt.dim, opt.dim * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(opt.dim * 2, opt.dim, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(opt.dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def reverse(self, x, len_shift=2):
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :self.size]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, self.nC * 2 - 2, 0, 0], mode='constant', value=0)
        for i in range(self.nC):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_PhiTg(self, Phi_shift, g):
        temp_1 = g.repeat(1, Phi_shift.shape[1], 1, 1, 1).cuda()
        PhiTg = temp_1 * Phi_shift
        # PhiTg = self.reverse(PhiTg)
        return PhiTg

    def mul_Phif(self, Phi_shift, f):
        # f_shift = self.shift(f)
        Phif = Phi_shift * f
        Phif = torch.sum(Phif, 1)
        return Phif.unsqueeze(1)

    def forward(self, g, input_mask=None):
        # video传入的phi[b, 8, 128, 128](256)
        # video传入的phi_s[b, 1, 128, 128]
        # video传入的y[b, 1, 128, 128]
        Phi, PhiPhiT = input_mask
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)

        temp_g = temp_g.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]
        Phi = Phi.unsqueeze(4)  # 形状[bs, c=1, 8, h, w, c=1]
        PhiPhiT = PhiPhiT.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]
        g = g.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]

        temp_g = self.fem(temp_g.permute(0, 4, 1, 2, 3))  # 形状[bs, c 8, h, w]
        g = g.permute(0, 4, 1, 2, 3)  # 形状[bs, c 8, h, w]
        Phi = self.fem(Phi.permute(0, 4, 1, 2, 3))  # 形状[bs, c 8, h, w]
        PhiPhiT = PhiPhiT.permute(0, 4, 1, 2, 3)  # 形状[bs, c 8, h, w]

        f0 = temp_g
        f = self.conv3d(torch.cat([f0, Phi], dim=1))

        out = []

        stage_output = None
        for i in range(len(self.reuse)):
            j = self.reuse[i]
            mu = self.mu[i](f)
            if j != 0:
                z, stage_output = self.net[i][0](f)
            else:
                z, stage_output = self.net[i][0](f, stage_output)
            Phi_f = self.mul_Phif(Phi, z)
            f = z + self.mul_PhiTg(Phi, torch.div(g - Phi_f, mu + PhiPhiT))
            out_ = self.vrm(f).squeeze(1)
            out.append(out_)

        return out
