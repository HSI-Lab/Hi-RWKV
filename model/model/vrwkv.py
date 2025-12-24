# Copyright (c) Shanghai AI Lab. All rights reserved.

from typing import Sequence
import math, os

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone

from .utils import DropPath
from einops import rearrange
logger = logging.getLogger(__name__)


from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", sources=["/home/rdg22/Code/MambaHSI/model/model/cuda_new/bi_wkv.cpp", "/home/rdg22/Code/MambaHSI/model/model/cuda_new/bi_wkv_kernel.cu"],
                verbose=True, extra_cuda_cflags=[
                                    "--use_fast_math",
                                    "-O3",
                                    "-Xptxas=-O3",
                                    "-DTmax=262144",
                                    "-allow-unsupported-compiler"
                                ]
                )


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape

    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=[1, 2, 4], init_mode='fancy', 
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel[0] > 0 or isinstance(shift_pixel, (list, tuple)):
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

        
    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        else:
            raise NotImplementedError

    def jit_func(self, x, patch_resolution):
        B, T, C = x.size()
        
        if isinstance(self.shift_pixel, (list, tuple)):
            xk = x * self.spatial_mix_k
            xv = x * self.spatial_mix_v
            xr = x * self.spatial_mix_r
            for shift in self.shift_pixel:
                if shift > 0:
                    xx = self.shift_func(x, shift, self.channel_gamma, patch_resolution)
                    xk += xx * (1 - self.spatial_mix_k)
                    xv += xx * (1 - self.spatial_mix_v)
                    xr += xx * (1 - self.spatial_mix_r)
                else:
                    xk = xv = xr = x
        else:
            TypeError

        # 特征投影
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)
        return sr, k, v
    

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x, patch_resolution)
            sr = torch.sigmoid(x)
            # x = RUN_CUDA(self.spatial_decay / T, self.spatial_first / T, k, v)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=[0], hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel[0] >= 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0
        self.whiten = nn.Linear(n_embd, n_embd, bias=False)
    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
                self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        else:
            raise NotImplementedError
    def regularization_loss(self) -> torch.Tensor:
        if self.whiten is None: 
            return torch.tensor(0., device=next(self.parameters()).device)
        W = self.whiten.weight  # [C, C]
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        reg = torch.norm(W.t() @ W - I, p='fro')  # Frobenius
        return reg
    
    def forward(self, x, patch_resolution=None):

        def _inner_forward(x):
            x = self.whiten(x)
            if isinstance(self.shift_pixel, (list, tuple)):
                xk = x * self.spatial_mix_k
                xr = x * self.spatial_mix_r
                for shift in self.shift_pixel:
                    if shift > 0:
                        xx = self.shift_func(x, shift, self.channel_gamma, patch_resolution)
                        xk += xx * (1 - self.spatial_mix_k)# * scale_weight
                        xr += xx * (1 - self.spatial_mix_r)# * scale_weight
                else:
                    xk = xr = x
            else:
                TypeError
            # import ipdb
            # ipdb.set_trace()
            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd) # nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, [1], init_mode,
                                   key_norm=key_norm)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, [1], hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        b, c, h, w = x.shape
        patch_resolution = (h, w)

        def _inner_forward(x):
            if self.layer_id == 0:
                x = rearrange(x, 'b c h w -> b (h w) c')
                x = self.ln0(x)
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = x + self.att(self.ln1(x), patch_resolution)
            x = x + self.ffn(self.ln2(x), patch_resolution)
            # x = x_sm + x_cm
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class HSI_RWKV(nn.Module):
    def __init__(self, 
        dim = 128,
        num_blocks = [1], 
    ):
        super(HSI_RWKV, self).__init__() 
        self.encoder_level1 = nn.Sequential(*[Block(n_embd=dim, n_layer=num_blocks[0], layer_id=i) for i in range(num_blocks[0])])
           
    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)
        x = rearrange(inp_img, 'b c h w -> b c w h')
        out_enc_level1 = self.encoder_level1(x).permute(0, 1, 3, 2) + out_enc_level1
        return out_enc_level1
