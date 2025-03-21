import warnings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn import init

from cacti.models.builder import MODELS


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


# class FA(nn.Module):
#     def __init__(self, dim, window_size=(8, 8), dim_head=28, sq_dim=None, shift=True):
#         super().__init__()
#
#         if sq_dim is None:
#             self.rank = dim
#         else:
#             self.rank = sq_dim
#         self.heads_qk = sq_dim // dim_head
#         self.heads_v = dim // dim_head
#         self.window_size = window_size
#         self.shift = shift
#
#         num_token = window_size[0] * window_size[1]
#         self.cal_atten = Attention(dim_head, num_token)
#
#         self.to_v = nn.Linear(dim, dim, bias=False)
#         self.to_qk = nn.Linear(dim, self.rank * 2, bias=False)
#         self.to_out = nn.Linear(dim, dim)
#
#     def cal_attention(self, x):
#         q, k = self.to_qk(x).chunk(2, dim=-1)
#         v = self.to_v(x)
#         q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_qk), (q, k))
#         v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads_v)
#         attn = self.cal_atten(q, k)
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out
#
#     def forward(self, x):
#
#         b, h, w, c = x.shape
#         w_size = self.window_size
#         if self.shift:
#             x = x.roll(shifts=4, dims=1).roll(shifts=4, dims=2)
#         x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
#         out = self.cal_attention(x_inp)
#         out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1], b0=w_size[0])
#         if self.shift:
#             out = out.roll(shifts=-4, dims=1).roll(shifts=-4, dims=2)
#         return out
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
        b, B, h, w, c = x.shape
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=4, dims=1).roll(shifts=4, dims=2)
        x_inp = rearrange(x, 'b B (h b0) (w b1) c -> (b h w) (b0 b1) B c', b0=w_size[0], b1=w_size[1])
        out = self.cal_attention(x_inp)
        out = rearrange(out, '(b h w) (b0 b1) B c -> b B (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                        b0=w_size[0])
        if self.shift:
            out = out.roll(shifts=-4, dims=1).roll(shifts=-4, dims=2)
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
        out = self.net(x.permute(0, 4, 1, 2, 3))
        return out.permute(0, 2, 3, 4, 1)


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


# class MPMLP(nn.Module):
#     def __init__(self, dim, multi=4):
#         super(MPMLP, self).__init__()
#
#         self.multi = multi
#         self.pwconv1 = nn.Sequential(
#             nn.Conv2d(dim, dim * multi, 1, groups=dim, bias=False),
#             GELU(),
#         )
#         self.groupconv = nn.Sequential(
#             nn.Conv2d(dim * multi, dim * multi, 1, groups=multi, bias=False),
#             GELU(),
#         )
#         self.pwconv2 = nn.Conv2d(dim * multi, dim, 1, groups=dim, bias=False)
#
#     def forward(self, x):
#         x = self.pwconv1(x.permute(0, 3, 1, 2))
#         x = rearrange(x, 'b (c m) h w -> b (m c) h w', m=self.multi)
#         x = self.groupconv(x)
#         x = rearrange(x, 'b (m c) h w -> b (c m) h w', m=self.multi)
#         x = self.pwconv2(x)
#         return x.permute(0, 2, 3, 1)


# class FAB(nn.Module):
#     def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
#         super().__init__()
#         self.pos_emb = nn.Conv3d(dim, dim, 5, 1, 2, bias=False, groups=dim)
#         self.norm1 = nn.LayerNorm(dim)
#         self.fa = FA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift)
#         # 用成串联
#         self.tsab = TimesAttention3D(dim, num_head=dim_head)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mpmlp = MPMLP(dim=dim, multi=mult)
#
#     def forward(self, x):
#         x = x + self.pos_emb(x)
#         # x = x.permute(0, 2, 3, 1)  # 把通道放在后面
#         x = x.permute(0, 2, 3, 4, 1)  # 把通道放在后面
#         x_ = self.norm1(x)
#         x = self.fa(x_) + x + self.tsab(x_)
#         x_ = self.norm2(x)
#         x = self.mpmlp(x_) + x
#         x = x.permute(0, 4, 1, 2, 3)
#         return x
class FAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.pos_emb = nn.Conv3d(dim, dim, 5, 1, 2, bias=False, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fa = FA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift)
        self.norm2 = nn.LayerNorm(dim)
        # self.mpmlp = MPMLP(dim=dim, multi=mult)
        self.ffn = FeedForward(dim=dim, mult=mult)

    def forward(self, x):
        x = x + self.pos_emb(x)
        # x = x.permute(0, 2, 3, 1)  # 把通道放在后面
        x = x.permute(0, 2, 3, 4, 1)  # 把通道放在后面
        x_ = self.norm1(x)
        x = self.fa(x_) + x
        x_ = self.norm2(x)
        x = self.ffn(x_) + x
        x = x.permute(0, 4, 1, 2, 3)
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
        self.norm2 = nn.LayerNorm(dim)
        # self.mpmlp = MPMLP(dim=dim, multi=4)
        self.ffn = FeedForward(dim=dim)

    def forward(self, x):
        x = self.tsab(x) + x
        x = x.permute(0, 2, 3, 4, 1)  # 把通道放在后面
        x = self.ffn(self.norm2(x)) + x
        x = x.permute(0, 4, 1, 2, 3)
        return x


class FAB_TSAB(nn.Module):
    def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
        super().__init__()
        self.FAB = FAB(dim=dim, sq_dim=sq_dim, window_size=window_size, dim_head=dim_head, mult=mult, shift=shift)
        self.TSAB = TSAB(dim, num_head=dim_head)

    def forward(self, x):
        x = self.FAB(x)
        x = self.TSAB(x)
        return x


# class FAB(nn.Module):
#     def __init__(self, dim, sq_dim, window_size=(8, 8), dim_head=28, mult=4, shift=False):
#         super().__init__()
#
#         # self.pos_emb = nn.Conv2d(dim, dim, 5, 1, 2, bias=False, groups=dim)
#         # 改动
#         self.pos_emb = nn.Conv3d(dim, dim, 5, 1, 2, bias=False, groups=dim)
#         self.norm1 = nn.LayerNorm(dim)
#         self.fa = FA(dim=dim, window_size=window_size, dim_head=dim_head, sq_dim=sq_dim, shift=shift)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mpmlp = MPMLP(dim=dim, multi=mult)
#
#     def forward(self, x):
#         x = x + self.pos_emb(x)
#         x = x.permute(0, 2, 3, 1)
#         x_ = self.norm1(x)
#         x = self.fa(x_) + x
#         x_ = self.norm2(x)
#         x = self.mpmlp(x_) + x
#         x = x.permute(0, 3, 1, 2)
#         return x
class StageInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.st_inter_enc = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.st_inter_dec = nn.Conv3d(dim, dim, 1, 1, 0, bias=False)
        self.act_fn = nn.LeakyReLU()
        self.phi = nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.gamma = nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def forward(self, inp, pre_enc, pre_dec):
        out = self.st_inter_enc(pre_enc) + self.st_inter_dec(pre_dec)
        skip = self.act_fn(out)
        phi = torch.sigmoid(self.phi(skip))
        gamma = self.gamma(skip)

        out = phi * inp + gamma

        return out


class IPB(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, dim=4):
        super(IPB, self).__init__()

        # self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(8, 1, 1, 1), torch.zeros(8, 1, 1, 1)], dim=1))
        # 改动
        # self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(dim, 1, 1, 1, 1), torch.zeros(dim, 1, 1, 1, 1)], dim=1))
        # self.conv_in = nn.Conv2d(in_dim, 28, 3, 1, 1, bias=False)
        # 改动
        self.conv_in = nn.Conv3d(in_dim, dim, 3, 1, 1, bias=False)
        self.down1 = FAB_TSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=4)
        # self.downsample1 = nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False)
        # 改动
        # self.downsample1 = nn.Conv3d(dim, dim * 2, 4, 2, 1, bias=False)
        # 改变下采样，帧数保持不变
        self.downsample1 = nn.Conv3d(dim, dim * 2, (3, 4, 4), (1, 2, 2), (1, 1, 1), bias=False)
        # self.downsample1 = nn.Sequential(
        #     nn.Conv2d(dim, dim//2, kernel_size=1, stride=1),
        #     nn.PixelUnshuffle(downscale_factor=2)
        # )
        self.down2 = FAB_TSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4)
        # self.downsample2 = nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False)
        # 改动
        # 改变下采样，帧数保持不变
        self.downsample2 = nn.Conv3d(dim * 2, dim * 4, (3, 4, 4), (1, 2, 2), (1, 1, 1), bias=False)
        # self.downsample2 = nn.Conv3d(dim * 2, dim * 4, 4, 2, 1, bias=False)

        self.bottleneck_local = FAB_TSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4)
        self.bottleneck_swin = FAB_TSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        # 改变上采样，帧数不变
        self.upsample2 = nn.ConvTranspose3d(dim * 4, dim * 2, (1, 2, 2), (1, 2, 2))
        # self.fusion2 = nn.Conv2d(dim * 4, dim * 2, 1, 1, 0, bias=False)
        # 改动
        self.fusion2 = nn.Conv3d(dim * 4, dim * 2, 1, 1, 0, bias=False)
        self.up2 = FAB_TSAB(dim=dim * 2, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        # 改变上采样，帧数不变
        self.upsample1 = nn.ConvTranspose3d(dim * 2, dim, (1, 2, 2), (1, 2, 2))
        # self.upsample1 = nn.ConvTranspose3d(dim * 2, dim, 2, 2)
        # self.fusion1 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=False)
        # 改动
        self.fusion1 = nn.Conv3d(dim * 2, dim, 1, 1, 0, bias=False)
        self.up1 = FAB_TSAB(dim=dim, sq_dim=dim, dim_head=dim, mult=4, shift=True)
        # self.conv_out = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        # 改动
        self.conv_out = nn.Conv3d(dim, out_dim, 3, 1, 1, bias=False)
        # 添加阶段融合模块
        # self.stage_interactions = nn.ModuleList([
        #     StageInteraction(dim=dim),
        #     StageInteraction(dim=dim * 2),
        #     StageInteraction(dim=dim * 2),
        #     StageInteraction(dim=dim * 2),
        #     StageInteraction(dim=dim * 2),
        #     StageInteraction(dim=dim)
        # ])
        self.down1_action = StageInteraction(dim=dim)
        self.down2_action = StageInteraction(dim=dim * 2)
        self.bottleneck_local_action = StageInteraction(dim=dim * 2)
        self.bottleneck_swin_action = StageInteraction(dim=dim * 2)
        self.up2_action = StageInteraction(dim=dim * 2)
        self.up1_action = StageInteraction(dim=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, stage_ins=None, stage_outs=None):
        """
            x: [b,c,B,h,w]
            return out:[b,c,B,h,w]
        """
        b, c, B, h_inp, w_inp = x.shape
        # hb, wb = 16, 16
        # pad_h = (hb - h_inp % hb) % hb
        # pad_w = (wb - w_inp % wb) % wb
        # x = F.pad(x, [0, pad_w, 0, pad_h, 0, B], mode='reflect')
        if stage_outs is None:
            stage_outs = []
        x = rearrange(x, 'b (n c) B h w -> b (c n) B h w', n=2)
        # x_in = F.conv2d(x, self.shuffle_conv, groups=8)
        # 改动
        x_in = x

        x = self.conv_in(x)
        x1 = self.down1(x)
        # 阶段融合
        if stage_ins:
            stage_outs.append(self.down1_action(x1, stage_ins[0], stage_ins[5]))
        else:
            stage_outs.append(self.down1_action(x1, x1, x1))

        # x = rearrange(stage_outs[0], 'b c B h w -> (b B) c h w')
        x = self.downsample1(x)
        # x = rearrange(x, '(b B) c h w -> b c B h w', B=8)

        x2 = self.down2(x)
        if stage_ins:
            stage_outs.append(self.down2_action(x2, stage_ins[1], stage_ins[4]))
        else:
            stage_outs.append(self.down2_action(x2, x2, x2))
        x = self.downsample2(stage_outs[1])
        # 通道 * 4
        x_local = self.bottleneck_local(x[:, :2*c, :, :])
        if stage_ins:
            stage_outs.append(self.bottleneck_local_action(x_local, stage_ins[2], stage_ins[3]))
        else:
            stage_outs.append(self.bottleneck_local_action(x_local, x_local, x_local))
        x_swin = self.bottleneck_swin(x[:, 2*c:, :, :] + stage_outs[2])
        if stage_ins:
            stage_outs.append(self.bottleneck_swin_action(x_swin, stage_ins[3], stage_ins[2]))
        else:
            stage_outs.append(self.bottleneck_swin_action(x_swin, x_swin, x_swin))
        x = torch.cat([stage_outs[2], stage_outs[3]], dim=1)

        x = self.upsample2(x)
        x = x2 + self.fusion2(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        if stage_ins:
            stage_outs.append(self.up2_action(x, stage_ins[4], stage_ins[1]))
        else:
            stage_outs.append(self.up2_action(x, x, x))
        x = self.upsample1(stage_outs[4])
        x = x1 + self.fusion1(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        if stage_ins:
            stage_outs.append(self.up1_action(x, stage_ins[5], stage_ins[0]))
        else:
            stage_outs.append(self.up1_action(x, x, x))
        out = self.conv_out(stage_outs[5]) + x_in

        return out[:, :, :h_inp, :w_inp], stage_outs


class Mu_Estimator(nn.Module):
    def __init__(self, in_nc=8, out_nc=1, channel=8):
        super(Mu_Estimator, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True),
            # 改动
            nn.Conv3d(in_nc, channel, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            # nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            # 改动
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            # 改动
            nn.Conv3d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            # 改动
            nn.Conv3d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus())

    def forward(self, x):
        x = self.conv(x)
        x = self.avpool(x)
        x = self.mlp(x) + 1e-6
        return x


class DPB(nn.Module):
    def __init__(self, in_dim=8, dim=4):
        super().__init__()
        self.norm_n = nn.LayerNorm(in_dim)
        self.norm_mask = nn.LayerNorm(in_dim)
        self.fusion = nn.Sequential(
            # 改动
            # nn.Conv2d(in_dim, 8, 1, 1, 0, bias=False),
            nn.Conv3d(in_dim, dim, 1, 1, 0, bias=False),
            GELU(),
        )
        self.weight = nn.Sequential(
            # 改动
            # nn.Conv2d(16, 8, 1, 1, 0, bias=False),
            nn.Conv3d(in_dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            # 改动
            nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
            # nn.Conv2d(8, 8, 1, 1, 0, bias=False),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight.data, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, Phi=None, Phi_compre=None):

        x = self.norm_n(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = self.fusion(x)
        mask = self.norm_mask(torch.cat([Phi, Phi_compre], dim=1).permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        weight = self.weight(mask)
        return self.out(x * weight)


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
        x = x.permute(0, 2, 1, 3, 4)
        _, _, _, h, w = x.shape
        tsab_in = rearrange(x, "b B c h w->(b h w) B c")
        n, B, C = tsab_in.shape
        qkv = self.qkv(tsab_in)
        qkv = qkv.reshape(n, B, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # C = C // 2
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # 无嵌入向量
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(n, B, C)
        x = rearrange(x, "(b h w) B c->b B h w c", h=h, w=w)
        x = self.proj(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


# class FB(nn.Module):
#     def __init__(self, in_dim=16):
#         super().__init__()
#         self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(8, 1, 1, 1), torch.zeros(8, 1, 1, 1)], dim=1))
#         self.out = nn.Sequential(
#             nn.Conv2d(in_dim, 28, 1, 1, 0),
#             nn.GroupNorm(28, 28),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#             nn.Conv2d(28, 8, 1, 1, 0),
#             nn.GroupNorm(8, 8),
#             nn.LeakyReLU(negative_slope=0.1, inplace=True),
#         )
#         self.apply(self.init_weights)
#
#     def init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             init.normal_(m.weight.data, mean=0.0, std=0.01)
#         elif isinstance(m, nn.GroupNorm):
#             init.normal_(m.weight.data, 0.0, 0.01)
#             init.normal_(m.bias.data, 0.0, 0.01)
#
#     def forward(self, f1, f2):
#         f = torch.cat([f1, f2], dim=1)
#         f = rearrange(f, 'b (n c) h w -> b (c n) h w', n=2)
#         out = F.conv2d(f, self.shuffle_conv, groups=8) + self.out(f)
#         return out
class FB(nn.Module):
    def __init__(self, in_dim=32, dim=16):
        super().__init__()
        self.shuffle_conv = nn.Parameter(torch.cat([torch.ones(dim, 1, 1, 1, 1), torch.zeros(dim, 1, 1, 1, 1)], dim=1))
        self.out = nn.Sequential(
            nn.Conv3d(in_dim, in_dim, 1, 1, 0),
            nn.GroupNorm(in_dim, in_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_dim, dim, 1, 1, 0),
            nn.GroupNorm(dim, dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.apply(self.init_weights)
        self.dim = dim

    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            init.normal_(m.weight.data, mean=0.0, std=0.01)
        elif isinstance(m, nn.GroupNorm):
            init.normal_(m.weight.data, 0.0, 0.01)
            init.normal_(m.bias.data, 0.0, 0.01)

    def forward(self, f1, f2):
        f = torch.cat([f1, f2], dim=1)
        f = rearrange(f, 'b (n c) B h w -> b (c n) B h w', n=2)
        out = F.conv3d(f, self.shuffle_conv, groups=self.dim) + self.out(f)
        return out


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


class GlobalMaxPoolingModel(nn.Module):
    def __init__(self):
        super(GlobalMaxPoolingModel, self).__init__()

    def forward(self, x):
        for i in range(5):
            for j in range(6):
                x[i][j] = torch.max(x[i][j], dim=1, keepdim=False)[0].squeeze(0)
        return x


class GlobalMaxPoolingModel_f(nn.Module):
    def __init__(self):
        super(GlobalMaxPoolingModel_f, self).__init__()

    def forward(self, x):
        for i in range(5):
            for j in range(2):
                x[i][j] = torch.max(x[i][j], dim=1, keepdim=False)[0].squeeze(0)
        return x


@MODELS.register_module
class NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y_reduceGPU(torch.nn.Module):
    def __init__(self, opt):
        super(NetVideo_conv3d_TSAB_shareBody_action_changeSample_seriesConnection_withoutDP_keep_y_reduceGPU, self).__init__()

        self.stage = opt.stage
        self.nC = opt.bands
        self.dim = opt.dim
        self.size = opt.size
        # self.conv = nn.Conv2d(self.nC * 2, self.nC, 1, 1, 0)
        # 改动
        self.conv3d = nn.Conv3d(self.dim * 2, self.dim, 1, 1, 0)
        self.body_share_params = opt.body_share_params
        para_estimator = []
        for i in range(opt.stage):
            para_estimator.append(Mu_Estimator(in_nc=opt.dim))
        self.mu = nn.ModuleList(para_estimator)
        self.net_stage_head = nn.ModuleList(nn.ModuleList([
            IPB(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
            # DPB(in_dim=opt.dim * 2, dim=opt.dim),
            FB(in_dim=opt.dim * 2, dim=opt.dim)
        ]))
        self.net_stage_body = nn.ModuleList([
            nn.ModuleList([
                IPB(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
                # DPB(in_dim=opt.dim * 2, dim=opt.dim),
                FB(in_dim=opt.dim * 2, dim=opt.dim)
            ]) for _ in range(opt.stage - 2)
        ]) if not opt.body_share_params else nn.ModuleList([
            IPB(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
            # DPB(in_dim=opt.dim * 2, dim=opt.dim),
            FB(in_dim=opt.dim * 2, dim=opt.dim)
        ])
        self.net_stage_tail = nn.ModuleList(nn.ModuleList([
            IPB(in_dim=opt.dim, out_dim=opt.dim, dim=opt.dim),
            # DPB(in_dim=opt.dim * 2, dim=opt.dim),
            FB(in_dim=opt.dim * 2, dim=opt.dim)
        ]))

        # 改动：添加一个特征提取块
        self.fem = FEM(in_dim=1, dim=self.dim // 4)

        # 特征恢复块
        self.vrm = nn.Sequential(
            nn.Conv3d(opt.dim, opt.dim * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(opt.dim * 2, opt.dim, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(opt.dim, 1, kernel_size=3, stride=1, padding=1),
        )
        # self.pool_stage_outputs = GlobalMaxPoolingModel()
        # self.pool_f_outputs = GlobalMaxPoolingModel_f()

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
        f_shift = f
        Phif = Phi_shift * f_shift
        Phif = torch.sum(Phif, 1)
        return Phif.unsqueeze(1)

    def forward(self, g, input_mask=None):
        # video传入的phi[b, 8, 128, 128](256)
        # video传入的phi_s[b, 1, 128, 128]
        # video传入的y[b, 1, 128, 128]
        Phi, PhiPhiT = input_mask
        # Phi_shift = self.shift(Phi, len_shift=2)
        # Phi_compressive = torch.sum(Phi, dim=1, keepdim=True)
        # Phi_compressive = Phi_compressive / self.nC * 2
        # Phi_compressive = self.reverse(Phi_compressive.repeat(1, self.nC, 1, 1))
        # Phi_compressive = self.reverse(Phi_compressive.repeat(1, self.nC, 1, 1))  # video sci mask无shift操作
        g_normal = g / self.nC * 2
        temp_g = g_normal.repeat(1, self.nC, 1, 1)
        # f0 = self.reverse(temp_g)

        # efficient sci [bs, c=1, 8, h, w]
        # efficient sci经过一个初始的Feature Extraction Module块 [bs, c=256, 8, h, w]

        temp_g = temp_g.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]
        Phi = Phi.unsqueeze(4)  # 形状[bs, c=1, 8, h, w, c=1]
        PhiPhiT = PhiPhiT.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]
        # Phi_compressive = Phi_compressive.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]
        g = g.unsqueeze(4)  # 形状[bs, 8, h, w, c=1]

        # 直接特征提取
        temp_g = self.fem(temp_g.permute(0, 4, 1, 2, 3))  # 形状[bs, c 8, h, w]
        g = g.permute(0, 4, 1, 2, 3)  # 形状[bs, c 8, h, w]
        Phi = self.fem(Phi.permute(0, 4, 1, 2, 3))  # 形状[bs, c 8, h, w]
        PhiPhiT = PhiPhiT.permute(0, 4, 1, 2, 3)  # 形状[bs, c 8, h, w]
        # Phi_compressive = self.fem(Phi_compressive.permute(0, 4, 1, 2, 3))  # 形状[bs, c 8, h, w]

        # temp_g = temp_g.permute(0, 2, 3, 4, 1)

        f0 = temp_g
        # f = self.conv(torch.cat([f0, Phi], dim=1))
        f = self.conv3d(torch.cat([f0, Phi], dim=1))
        # f = self.fem(f.permute(0, 4, 1, 2, 3))  # 形状[bs, c=16 8, h, w]
        # z_ori = f
        y = 0
        # r = 0

        out = []
        mu = self.mu[0](f)
        z, stage_output = self.net_stage_head[0](f)

        # r = self.net_stage_head[1](torch.cat([z_ori - y / mu - f, f], dim=1), Phi, Phi_compressive)
        Phi_f = self.mul_Phif(Phi, z - y / mu)
        f = z - y / mu + self.mul_PhiTg(Phi, torch.div(g - Phi_f, mu + PhiPhiT))  # 有个reverse操作需要去掉
        f = self.net_stage_head[1](f, z)
        # z_ori = z
        y = y + mu * (f - z)
        out_ = self.vrm(f).squeeze(1)
        out.append(out_)

        if not self.body_share_params:
            for i in range(self.stage - 2):
                mu = self.mu[i + 1](f)
                z, stage_output = self.net_stage_body[i][0](f, stage_output)
                # r = self.net_stage_body[i][1](torch.cat([z_ori - y / mu - f, f], dim=1), Phi, Phi_compressive)
                Phi_f = self.mul_Phif(Phi, z - y / mu)
                f = z - y / mu + self.mul_PhiTg(Phi, torch.div(g - Phi_f, mu + PhiPhiT))  # 有个reverse操作需要去掉
                f = self.net_stage_body[i][1](f, z)
                # z_ori = z
                y = y + mu * (f - z)
                out_ = self.vrm(f).squeeze(1)
                out.append(out_)
        else:
            for i in range(self.stage - 2):
                mu = self.mu[i + 1](f)
                z, stage_output = self.net_stage_body[0](f, stage_output)
                # r = self.net_stage_body[1](torch.cat([z_ori - y / mu - f, f], dim=1), Phi, Phi_compressive)
                Phi_f = self.mul_Phif(Phi, z - y / mu)
                f = z - y / mu + self.mul_PhiTg(Phi, torch.div(g - Phi_f, mu + PhiPhiT))  # 有个reverse操作需要去掉
                f = self.net_stage_body[1](f, z)
                # z_ori = z
                y = y + mu * (f - z)
                out_ = self.vrm(f).squeeze(1)
                out.append(out_)

        mu = self.mu[self.stage - 1](f)
        z, stage_output = self.net_stage_tail[0](f, stage_output)
        # r = self.net_stage_tail[1](torch.cat([z_ori - y / mu - f, f], dim=1), Phi, Phi_compressive)
        Phi_f = self.mul_Phif(Phi, z - y / mu)
        f = z - y / mu + self.mul_PhiTg(Phi, torch.div(g - Phi_f, mu + PhiPhiT))  # 有个reverse操作需要去掉
        f = self.net_stage_tail[1](f, z)
        # z_ori = z
        # y = y + mu * (f - z + r)
        out_ = self.vrm(f).squeeze(1)
        out.append(out_)
        return out
