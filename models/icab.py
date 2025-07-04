import torch
from torch import nn as nn
from torch.nn import init as init
from einops import rearrange
import numbers
# import math
# from torch.nn import functional as F
# from torch.nn.modules.batchnorm import _BatchNorm
# from timm.models.layers import DropPath, trunc_normal_, to_2tuple


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
 
class MultiheadChannelAttention(nn.Module):
    # CrossViewChannelFusion
    def __init__(self, dim, input_resolution, num_heads, bias):
        super(MultiheadChannelAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.num_heads = num_heads

    def forward(self, x, input_num):
        b, c, h, w = x.shape
        x_anchor = x[:,(input_num//2)*(c//input_num):(input_num//2+1)*(c//input_num),:,:]
        y = x_anchor.repeat(1, input_num, 1, 1)
        q = self.q(y)  
        k = self.k(x)  
        v = self.v(x)  
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads * input_num)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads * input_num)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads * input_num)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads * input_num, h=h, w=w)
        out = self.project_out(out)
        return out
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += 3 * H * W * self.dim * self.dim
        flops += self.num_heads * (self.dim // self.num_heads) * H * W * (self.dim // self.num_heads)
        flops += self.num_heads * H * W * (self.dim // self.num_heads) * (self.dim // self.num_heads)
        flops += H * W * self.dim * self.dim
        return flops


##########################################################################
class ICAB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4, bias=False, LayerNorm_type='WithBias'):
        super(ICAB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.norm1_x = LayerNorm(dim, LayerNorm_type)
        self.norm1_y = LayerNorm(dim, LayerNorm_type)
        self.attn = MultiheadChannelAttention(dim, input_resolution, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, x, input_num, x_size):
        b, n, c = x.shape
        h, w = x_size
        assert n == h*w
        x = x.permute(0,2,1).view(b, c, h, w)
        fused = x + self.attn(self.norm1_x(x), input_num)  # b, c, h, w
        # mlp
        fused = to_3d(fused)  # b, h*w, c
        # x = to_3d(x) # b, h*w, c
        fused = fused + self.mlp(self.norm2(fused))
        return fused
    
    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # MCA
        flops += self.attn.flops()
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops