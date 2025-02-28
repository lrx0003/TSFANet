#!/usr/bin/python3
#coding=utf-8
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.PVT import pvt_v2_b1, PyramidVisionTransformerV2
from functools import partial
from src.Gate_Fold_ASPP import GFASPP
from src.AFAM import AFAM


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Unfold, nn.Sigmoid, nn.AdaptiveAvgPool2d,
                            nn.Softmax, nn.Dropout2d, nn.Upsample)):
            pass
        else:
            weight_init(m)



try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)

class SEM(nn.Module):

    def __init__(self, c_in, c_feat, c_atten):
        super(SEM, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)

    def forward(self, input: torch.Tensor):
        b, c, h, w = input.size()
        feat = self.conv_feat(input).view(b, self.c_feat, -1)
        atten = self.conv_atten(input).view(b, self.c_atten, -1)
        atten = F.softmax(atten, dim=-1)
        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))

        return descriptors
    def initialize(self):
        weight_init(self)

class SDM(nn.Module):

    def __init__(self, c_atten, c_de):
        super(SDM, self).__init__()
        self.c_atten = c_atten
        self.c_de = c_de
        self.conv_de = nn.Conv2d(c_atten, c_atten // 4, kernel_size=1)
        self.out_conv = nn.Conv2d(c_atten, c_de, kernel_size=1)

    def forward(self, descriptors: torch.Tensor, input_de: torch.Tensor):
        b, c, h, w = input_de.size()
        atten_vectors = F.softmax(self.conv_de(input_de), dim=1)
        output = descriptors.matmul(atten_vectors.view(b, self.c_atten // 4, -1)).view(b, -1, h, w)

        return self.out_conv(output)
    def initialize(self):
        weight_init(self)


class LRM(nn.Module):

    def __init__(self, c_en, c_de):
        super(LRM, self).__init__()
        self.c_en = c_en
        self.c_de = c_de
        self.conv_1 =  nn.Conv2d(c_de, c_en, kernel_size=1, bias=False)
    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)

        energy = self.conv_1(input_de).view(b, self.c_en, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)

        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)

        return input_en
    def initialize(self):
        weight_init(self)

class IFM(nn.Module):

    def __init__(self, fpn_dim, c_atten):
        super(IFM, self).__init__()
        self.fqn_dim = fpn_dim
        self.c_atten = c_atten
        self.sdm = SDM(c_atten, fpn_dim)
        self.lrm = LRM(fpn_dim, c_atten)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(c_atten, fpn_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_en, input_de, global_descripitors):
        feat_global = self.sdm(global_descripitors, input_de)
        feat_local = self.lrm(input_en, input_de, feat_global)
        feat_local = self.gamma * feat_local + input_en

        return self.conv_fusion(self.conv(input_de) + self.alpha * feat_global + self.beta * feat_local)
    def initialize(self):
        weight_init(self)


class SGAFF(nn.Module):
    def __init__(self, in_channel_1=None, in_channel_2=None, out_channel=None):
        super(SGAFF, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channel_2, in_channel_1, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channel_1)

        self.sem = SEM(in_channel_1, in_channel_1, in_channel_1 // 4)
        self.ifm = IFM(in_channel_1, out_channel)


    def forward(self, x, y):

        z = F.relu(self.bn(self.conv(y)))
        att = self.sem(z)
        out = self.ifm(x, self.upsample(z), att)

        return out

    def initialize(self):
        weight_init(self)


def antidiagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor.device) - shift) % W

    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def antidiagonal_scatter(tensor_flat, original_shape):

    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)

    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def diagonal_scatter(tensor_flat, original_shape):
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W

    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)

    return result_tensor

def diagonal_gather(tensor):
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)
    index = (shift + torch.arange(W, device=tensor.device)) % W

    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))

        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W

        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)

        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)


        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)

        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)

        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape

        xs = x.new_empty((B, 8, C, L))


        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)

class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        to_dtype=True,
        force_fp32=False,
        nrows=-1,
        backnrows=-1,
        ssoflex=True,
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R, _ = dt_projs_weight.shape
    L = H * W
    x_proj_weight = x_proj_weight.squeeze(-1)
    _, C, D = x_proj_weight.shape

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)
    # x_dbl = torch.einsum("b k d l, k c d n -> b k c l", xs, x_proj_weight)

    xs_reshaped = xs.permute(0, 1, 3, 2).reshape(B * K, L, D)
    x_proj_weight_t = x_proj_weight.transpose(1, 2)
    x_proj_weight_expanded = x_proj_weight_t.unsqueeze(0).expand(B, -1, -1, -1)
    x_proj_weight_expanded = x_proj_weight_expanded.reshape(B * K, D, C)
    x_dbl = torch.bmm(xs_reshaped, x_proj_weight_expanded)
    x_dbl = x_dbl.view(B, K, L, C).permute(0, 1, 3, 2)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)

    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    # dts = torch.einsum("b k r l, k d r n -> b k d l", dts, dt_projs_weight)

    B, K, R, L = dts.shape
    dt_projs_weight = dt_projs_weight.squeeze(-1)
    dts_reshaped = dts.reshape(B * K, R, L)
    dt_projs_weight_expanded = dt_projs_weight.unsqueeze(0).expand(B, -1, -1, -1)
    dt_projs_weight_reshaped = dt_projs_weight_expanded.reshape(B * K, D, R)
    dts = torch.bmm(dt_projs_weight_reshaped, dts_reshaped)
    dts = dts.view(B, K, D, L)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class ODPM(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            d_conv=3,
            conv_bias=True,
            dropout=0.0,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            out_channels=96,
            **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        def checkpostfix(tag, value):
            ret = value.endswith(tag)
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        if forward_type.endswith("none"):
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type.endswith("dwconv3"):
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type.endswith("softmax"):
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type.endswith("sigmoid"):
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        d_proj = d_inner if self.disable_z else (d_inner * 2)

        self.in_proj = nn.Conv2d(in_channels=d_model, out_channels=d_proj, kernel_size=1, bias=bias)

        self.act = act_layer()

        self.x_proj = [
            nn.Conv1d(in_channels=d_inner, out_channels=(dt_rank + d_state * 2), kernel_size=1, bias=False)
            for _ in range(8)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            nn.Conv1d(in_channels=dt_rank, out_channels=d_inner, kernel_size=1, bias=True)
            for _ in range(8)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = nn.Parameter(torch.randn((8 * d_inner, d_state)))

        self.Ds = nn.Parameter(torch.ones((8 * d_inner)))

        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        self.out_proj = nn.Conv2d(in_channels=d_inner, out_channels=d_model, kernel_size=1, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.outconv = nn.Conv2d(d_model, out_channels, kernel_size=3, stride=2, padding=1)

    def forward_corev2(self, x: torch.Tensor, channel_first=False, force_fp32=None, SelectiveScan=SelectiveScanCore):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)

        # x_reshaped = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        # x = x.permute(0, 3, 2, 1)
        x = self.in_proj(x)
        x = x.permute(0, 3, 2, 1)

        if not self.disable_z:
            x, z = x.chunk(2, dim=-1)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z

        y = y.permute(0, 3, 2, 1)
        out = self.dropout(self.out_proj(y))
        output = self.outconv(out)

        return output

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.inplanes = 64
        vgg = models.vgg16(pretrained=True)
        self.layer0 = nn.Sequential(*list(vgg.children())[0][0:5]) 
        self.layer1 = nn.Sequential(*list(vgg.children())[0][5:10]) 
        self.layer2 = nn.Sequential(*list(vgg.children())[0][10:17]) 
        self.layer3 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 256, embed_dims=[320], num_heads=[5], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2], sr_ratios=[1], num_stages = 1)
        self.layer4 = PyramidVisionTransformerV2(
        patch_size=2, in_chans = 320, embed_dims=[512], num_heads=[8], mlp_ratios=[4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2], sr_ratios=[1], num_stages = 1)

    def forward(self, x):
        out1 = self.layer0(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)[0]
        out5 = self.layer4(out4)[0]
        
        return out1, out2, out3, out4, out5  
        
    def initialize(self):
        #self.load_state_dict(torch.load('./data/vgg16-397923af.pth'), strict=False)
        pass


class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        self.inplanes = 64
        vgg = models.vgg16(pretrained=True)
        self.layer0 = nn.Sequential(*list(vgg.children())[0][0:5])
        self.layer1 = nn.Sequential(*list(vgg.children())[0][5:10])
        self.layer2 = nn.Sequential(*list(vgg.children())[0][10:17])
        self.layer3 = ODPM(d_model=256, forward_type="v2", out_channels=320)
        self.layer4 = ODPM(d_model=320, forward_type="v2", out_channels=512)

        self.Corr1 = ASCRM(64, 64)
        self.Corr2 = ASCRM(128, 128)
        self.Corr3 = ASCRM(256, 256)
        self.Corr4 = ASCRM(320, 320)
        self.Corr5 = ASCRM(512, 512)

        # self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(640, 320, kernel_size=3, stride=1, padding=1)

    def forward(self, x, s1, s2, s3, s4, s5):
        x0 = self.layer0(x)

        fus_1 = self.Corr1(s1, x0)
        # fus_1 = self.conv1(torch.cat((s1, out1), dim=1))
        x1 = self.layer1(fus_1)

        fus_2 = self.Corr2(s2, x1)
        # fus_2 = self.conv2(torch.cat((s2, out2), dim=1))
        x2 = self.layer2(fus_2)

        fus_3 = self.Corr3(s3, x2)
        # fus_3 = self.conv3(torch.cat((s3, out3), dim=1))
        x3 = self.layer3(fus_3)

        fus_4 = self.Corr4(s4, x3)
        # fus_4 = self.conv4(torch.cat((s4, out4), dim=1))
        x4 = self.layer4(fus_4)

        fus_5 = self.Corr5(s5, x4)
        # x5 = self.out(fus_5)

        return fus_1, fus_2, fus_3, fus_4, fus_5

    def initialize(self):
        # self.load_state_dict(torch.load('./data/vgg16-397923af.pth'), strict=False)
        pass

class PredictBlock(nn.Module):
    """
    Input: 
           Type: Tensor 
           Note: feature maps after AFAM for Deep Supervised
           Channel: 64
           Size: B * 64 * 448 * 448
    Output:
           Type: Tensor list
           Len: 2
           Note: salient maps & edge maps
           Channel: 1
           Size: B * 1 * 448 * 448
    """
    def __init__(self):
        super(PredictBlock, self).__init__()
        self.down1 = nn.Conv2d(64, 32, kernel_size=3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.smaps = nn.Conv2d(32, 1, kernel_size = 3, padding = 1)
        self.edges = nn.Conv2d(32, 1, kernel_size = 3, padding = 1)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.down1(x)))
        smaps = self.smaps(out)
        edges = self.edges(out)
        return  smaps, edges
        
    def initialize(self):
        weight_init(self)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class ASCRM(nn.Module):
    def __init__(self, all_channel=128, all_dim=32):
        super(ASCRM, self).__init__()
        self.conv_e = nn.Conv1d(all_channel, all_channel, kernel_size=1, bias=False)
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.gate_1 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_2 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv2 = DSConv3x3(all_channel, all_channel, stride=1)
        self.conv_fusion = DSConv3x3(all_channel * 2, all_channel, stride=1)
        # self.pred = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)
        # self.pred2 = nn.Conv2d(all_channel, 1, kernel_size=1, bias=True)

    def forward(self, exemplar, query):
        fea_size = query.size()[2:]
        N, C, H, W = query.size()

        k = 7
        s = 4

        exemplar_patches = exemplar.unfold(2, k, s).unfold(3, k, s)  # N x C x num_patches_h x num_patches_w x k x k
        query_patches = query.unfold(2, k, s).unfold(3, k, s)

        num_patches_h = exemplar_patches.size(2)
        num_patches_w = exemplar_patches.size(3)
        num_patches = num_patches_h * num_patches_w

        exemplar_patches = exemplar_patches.contiguous().view(N * num_patches, C, k, k)
        query_patches = query_patches.contiguous().view(N * num_patches, C, k, k)

        all_dim = k * k
        exemplar_flat = exemplar_patches.view(-1, C, all_dim)  # (N*num_patches) x C x (k*k)
        query_flat = query_patches.view(-1, C, all_dim)

        exemplar_t = exemplar_flat  # (N*num_patches) x C x (k*k)

        exemplar_corr = self.conv_e(exemplar_t)  # (N*num_patches) x C x (k*k)

        exemplar_corr = exemplar_corr.transpose(1, 2)  # (N*num_patches) x (k*k) x C

        A = torch.bmm(exemplar_corr, query_flat)  # (N*num_patches) x (k*k) x (k*k)

        A1 = F.softmax(A, dim=1)
        B = F.softmax(A.transpose(1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1)  # (N*num_patches) x C x (k*k)
        exemplar_att = torch.bmm(query_flat, B)   # (N*num_patches) x C x (k*k)

        query_att = query_att.view(N, num_patches_h, num_patches_w, C, k, k)
        exemplar_att = exemplar_att.view(N, num_patches_h, num_patches_w, C, k, k)

        query_att = query_att.permute(0, 3, 1, 4, 2, 5).contiguous()
        query_att = query_att.view(N, C, num_patches_h * k, num_patches_w * k)
        exemplar_att = exemplar_att.permute(0, 3, 1, 4, 2, 5).contiguous()
        exemplar_att = exemplar_att.view(N, C, num_patches_h * k, num_patches_w * k)

        query_att = query_att[:, :, :H, :W]
        exemplar_att = exemplar_att[:, :, :H, :W]

        exemplar_mask = self.gate_1(exemplar_att)
        exemplar_mask = self.gate_s(exemplar_mask)
        exemplar_att = exemplar_att * exemplar_mask
        exemplar_out = self.conv1(exemplar_att + exemplar)

        query_mask = self.gate_2(query_att)
        query_mask = self.gate_s(query_mask)
        query_att = query_att * query_mask
        query_out = self.conv1(query_att + query)

        pred = self.conv_fusion(torch.cat([exemplar_out, query_out], 1))
        return pred




class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # Hybrid Backbone with PVT_b1 and VGG16
        self.bkbone   = VGG()
        self.bkbone1 = VGG1()


        # Fused Blocks for obtaining init feature maps
        self.fuse5 = SGAFF(in_channel_1=512, in_channel_2=1024, out_channel=512)
        self.fuse4 = SGAFF(in_channel_1=320, in_channel_2=512, out_channel=320)
        self.fuse3 = SGAFF(in_channel_1=256, in_channel_2=320, out_channel=256)
        self.fuse2 = SGAFF(in_channel_1=128, in_channel_2=256, out_channel=128)
        self.fuse1 = SGAFF(in_channel_1=64, in_channel_2=128, out_channel=64)


        # Gate_Flod_ASPP Module
        self.gate = nn.Sequential(
            GFASPP(in_channel=512,
                      out_channel=512,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
                   ),
           nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
           nn.BatchNorm2d(1024),
            # nn.BatchNorm2d(512),
            nn.PReLU(),
        )

        #Prediction Module

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1)


        self.prerdict5 = PredictBlock()
        self.prerdict4 = PredictBlock()
        self.prerdict3 = PredictBlock()
        self.prerdict2 = PredictBlock()
        self.prerdict1 = PredictBlock()


        # self.conv_5 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        # self.sem_d5 = SEM(512, 512, 512 // 4)
        # self.ifm_d5 = IFM(512, 512)
        #
        # self.conv_4 = nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1)
        # self.sem_d4 = SEM(320, 320, 320 // 4)
        # self.ifm_d4 = IFM(320, 320)
        #
        # self.conv_3 = nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1)
        # self.sem_d3 = SEM(256, 256, 256 // 4)
        # self.ifm_d3 = IFM(256, 256)
        #
        # self.conv_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # self.sem_d2 = SEM(128, 128, 128 // 4)
        # self.ifm_d2 = IFM(128, 128)
        #
        # self.conv_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.sem_d1 = SEM(64, 64, 64 // 4)
        # self.ifm_d1 = IFM(64, 64)
        #
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.initialize()

    def forward(self, x):
        
        s1, s2, s3, s4, s5 = self.bkbone(x)
        s_1, s_2, s_3, s_4, s_5 = self.bkbone1(x, s1, s2, s3, s4, s5)

        s6 = self.gate(s_5)

        # d5 = self.conv_5(s6)
        # att5 = self.sem_d5(d5)
        # out5 = self.ifm_d5(s_5, self.upsample2(d5), att5)
        #
        # d4 = self.conv_4(out5)
        # att4 = self.sem_d4(d4)
        # out4 = self.ifm_d4(s_4, self.upsample2(d4), att4)
        #
        # d3 = self.conv_3(out4)
        # att3 = self.sem_d3(d3)
        # out3 = self.ifm_d3(s_3, self.upsample2(d3), att3)
        #
        # d2 = self.conv_2(out3)
        # att2 = self.sem_d2(d2)
        # out2 = self.ifm_d2(s_2, self.upsample2(d2), att2)
        #
        # d1 = self.conv_1(out2)
        # att1 = self.sem_d1(d1)
        # out1 = self.ifm_d1(s_1, self.upsample2(d1), att1)


        out5 = self.fuse5(s_5, s6)
        out4 = self.fuse4(s_4, out5)
        out3 = self.fuse3(s_3, out4)
        out2 = self.fuse2(s_2, out3)
        out1 = self.fuse1(s_1, out2)
        
        # if self.training:
        #     smap1, edge1 = self.prerdict1(out1)
        #     smap2, edge2 = self.prerdict2(out2)
        #     smap3, edge3 = self.prerdict3(out3)
        #     smap4, edge4 = self.prerdict4(out4)
        #     smap5, edge5 = self.prerdict5(out5)
        #
        #     smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     smap2 = F.interpolate(smap2, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     smap3 = F.interpolate(smap3, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     smap4 = F.interpolate(smap4, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     smap5 = F.interpolate(smap5, size = x.size()[2:], mode='bilinear',align_corners=True)
        #
        #     edge1 = F.interpolate(edge1, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     edge2 = F.interpolate(edge2, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     edge3 = F.interpolate(edge3, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     edge4 = F.interpolate(edge4, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     edge5 = F.interpolate(edge5, size = x.size()[2:], mode='bilinear',align_corners=True)
        #
        #     return smap1, smap2, smap3, smap4, smap5,   edge1, edge2, edge3, edge4, edge5
        #
        # else:
        #     smap1, edge1 = self.prerdict1(out1)
        #     smap1 = F.interpolate(smap1, size = x.size()[2:], mode='bilinear',align_corners=True)
        #     return torch.sigmoid(smap1)

        smap1, edge1 = self.prerdict1(self.conv1(out1))
        smap2, edge2 = self.prerdict2(self.conv2(out2))
        smap3, edge3 = self.prerdict3(self.conv3(out3))
        smap4, edge4 = self.prerdict4(self.conv4(out4))
        smap5, edge5 = self.prerdict5(self.conv5(out5))

        smap1 = F.interpolate(smap1, size=x.size()[2:], mode='bilinear', align_corners=True)
        smap2 = F.interpolate(smap2, size=x.size()[2:], mode='bilinear', align_corners=True)
        smap3 = F.interpolate(smap3, size=x.size()[2:], mode='bilinear', align_corners=True)
        smap4 = F.interpolate(smap4, size=x.size()[2:], mode='bilinear', align_corners=True)
        smap5 = F.interpolate(smap5, size=x.size()[2:], mode='bilinear', align_corners=True)

        edge1 = F.interpolate(edge1, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge4 = F.interpolate(edge4, size=x.size()[2:], mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=x.size()[2:], mode='bilinear', align_corners=True)

        return smap1, smap2, smap3, smap4, smap5, edge1, edge2, edge3, edge4, edge5

    def initialize(self):
        weight_init(self)

# import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    model.eval()

    # input_tensor = torch.randn(4, 3, 512, 512).to(device)
    input_tensor = torch.randn(4, 3, 448, 448).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    # print(f"Output shape: {output.shape}")

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            print(f"Output {i + 1} shape: {out.shape}")
    else:
        print(f"Output shape: {output.shape}")

    # if output.shape[1] == 1:
    #     output_image = output.squeeze().cpu().numpy()  # 移除批次维度并转换为 numpy 数组
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(output_image, cmap='gray')
    #     plt.title("Output Image")
    #     plt.show()
    # else:
    #     print("Output is not a single-channel image, skipping display.")
