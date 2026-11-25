"""FDNet: feature decomposition network for SAR RFI suppression.

Fang et al., "Synthetic-Aperture Radar Radio-Frequency Interference
Suppression Based on Regularized Optimization Feature Decomposition Network",
Remote Sensing 16(14):2540, 2024. Official code:
https://github.com/fangfuping/FDNet (``model.py``, class ``FDUformer``).

The paper describes FDNet as Uformer with a feature decomposition block
embedded at the bottleneck. The official repository holds two networks with
the decomposition block: ``FDNet`` (a convolutional U-Net) and ``FDUformer``
(the Uformer variant). The paper selects the transformer variant, so this
module ports ``FDUformer``, with ``get_arch``'s FDUformer arguments:
``win_size=8, token_projection='linear', token_mlp='leff', modulator=True``,
default depths ``[2]*9`` and heads ``[1, 2, 4, 8, 16, 16, 8, 4, 2]``, and the
paper's initial width ``C = 16``.

Structure (paper Sec. 3.1, Figs. 3-4; official ``FDUformer.forward``):

* Input projection: 3x3 convolution to ``C`` channels and LeakyReLU.
* Encoder: four levels. Each is a stack of LeWin transformer blocks
  followed by a 4x4, stride-2 convolution that halves the size and doubles
  the channels.
* LeWin block: window self-attention in 8x8 windows with a relative position
  bias (global feature extraction), alternating with shifted windows, then
  the LeFF feed-forward (linear, 3x3 depthwise convolution, linear) for local
  features. Decoder blocks also add a learned window "modulator" to the
  tokens before attention.
* Feature decomposition at the bottleneck: two parallel transformer stacks
  read the level-4 features. ``conv`` gives the signal features, ``noise``
  gives the interference features. Only the signal branch goes on to the
  decoder; the interference branch is used only by the cosine
  decomposition loss.
* Decoder: four levels of 2x2, stride-2 transposed convolution, then
  concatenation with the encoder features of the same level, then LeWin
  blocks.
* Output: 3x3 convolution back to the input channels, added to the input
  (residual).

Module and parameter names match the official code, so an official
``FDUformer`` state dict loads into :class:`FDNet` with ``strict=True``
(``tests/test_fdnet.py`` checks this when the official repository is
available).

Adaptations, all mechanical and none architectural:

* Input channels. The official model reads one real channel. This
  repository feeds every baseline the 2-channel (real, imaginary) STFT, so
  ``in_channels`` defaults to 2. Use ``in_channels=1`` for the paper's
  single-channel maps.
* Shapes. The official layers recover ``H = W = sqrt(L)`` from the token
  count, so they accept only square inputs. Here ``H`` and ``W`` are passed
  along explicitly. An input is zero-padded at the bottom and right to a
  multiple of ``2**4 * win_size`` (128). That keeps every level divisible
  into 8x8 windows down to the bottleneck. The output is cropped back to
  the input size.
* Shift rule. The official block turns the shift off when the level's
  resolution is at most the window size, decided at construction from
  ``img_size``. Here the same rule is applied at run time to the padded
  size of each level. For the paper's 512x512 input no level is that small,
  so nothing changes.
* ``timm`` and ``einops`` are not dependencies. ``DropPath`` and the
  rearranges are rewritten with plain torch, with the same semantics.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

PAPER_EMBED_DIM = 16
PAPER_WIN_SIZE = 8
OFFICIAL_DEPTHS = (2, 2, 2, 2, 2, 2, 2, 2, 2)
OFFICIAL_HEADS = (1, 2, 4, 8, 16, 16, 8, 4, 2)


# --------------------------------------------------------------------------- helpers (timm equivalents)


class DropPath(nn.Module):
    """Stochastic depth per sample, as ``timm.models.layers.DropPath``."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    # timm's trunc_normal_(std=.02) truncates at absolute bounds [-2, 2], as torch's does.
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)


def _tokens_to_map(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``(B, H*W, C)`` tokens to a ``(B, C, H, W)`` map, row-major like the official code."""
    B, L, C = x.shape
    if L != H * W:
        raise ValueError(f"token count {L} does not match {H}x{W}")
    return x.transpose(1, 2).contiguous().view(B, C, H, W)


def _map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """``(B, H, W, C)`` to ``(B * nW, win, win, C)``."""
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)


def window_reverse(windows: torch.Tensor, win_size: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`."""
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# --------------------------------------------------------------------------- attention


class LinearProjection(nn.Module):
    """Q, K, V projection (official ``LinearProjection``)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, bias: bool = True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x: torch.Tensor):
        B_, N, C = x.shape
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(x).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return q[0], kv[0], kv[1]


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one window, with a relative position bias."""

    def __init__(self, dim: int, win_size: tuple[int, int], num_heads: int, qkv_bias: bool = True,
                 qk_scale: float | None = None, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        coords_h = torch.arange(win_size[0])
        coords_w = torch.arange(win_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_size[0] - 1
        relative_coords[:, :, 1] += win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * win_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        _trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        n_tok = self.win_size[0] * self.win_size[1]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n_tok, n_tok, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"


class LeFF(nn.Module):
    """Locally-enhanced feed-forward: linear, 3x3 depthwise conv, linear (the paper's local module)."""

    def __init__(self, dim: int = 32, hidden_dim: int = 128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.linear1(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.linear2(x)


# --------------------------------------------------------------------------- transformer block and stage


class LeWinTransformerBlock(nn.Module):
    """Window attention (global module) then LeFF (local module), both pre-norm and residual."""

    def __init__(self, dim: int, num_heads: int, win_size: int = 8, shift_size: int = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale: float | None = None, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, modulator: bool = False):
        super().__init__()
        if not 0 <= shift_size < win_size:
            raise ValueError("shift_size must be in [0, win_size)")
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.modulator = nn.Embedding(win_size * win_size, dim) if modulator else None
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, win_size=(win_size, win_size), num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = LeFF(dim, int(dim * mlp_ratio), act_layer=act_layer)

    def _shift_mask(self, H: int, W: int, shift: int, like: torch.Tensor) -> torch.Tensor:
        """Official SW-MSA mask: -100 between tokens that the cyclic shift brought from different regions."""
        shift_mask = torch.zeros((1, H, W, 1), dtype=like.dtype, device=like.device)
        slices = (slice(0, -self.win_size), slice(-self.win_size, -shift), slice(-shift, None))
        cnt = 0
        for h in slices:
            for w in slices:
                shift_mask[:, h, w, :] = cnt
                cnt += 1
        windows = window_partition(shift_mask, self.win_size).view(-1, self.win_size * self.win_size)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        if H % self.win_size or W % self.win_size:
            raise ValueError(f"{H}x{W} is not divisible into {self.win_size}x{self.win_size} windows")
        # Official rule: no shift once the level is no larger than one window.
        shift = 0 if min(H, W) <= self.win_size else self.shift_size
        attn_mask = self._shift_mask(H, W, shift, x) if shift > 0 else None

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        shifted = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2)) if shift > 0 else x
        windows = window_partition(shifted, self.win_size).view(-1, self.win_size * self.win_size, C)
        if self.modulator is not None:
            windows = windows + self.modulator.weight
        attn_windows = self.attn(windows, mask=attn_mask).view(-1, self.win_size, self.win_size, C)
        shifted = window_reverse(attn_windows, self.win_size, H, W)
        x = torch.roll(shifted, shifts=(shift, shift), dims=(1, 2)) if shift > 0 else shifted
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_heads={self.num_heads}, win_size={self.win_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, modulator={self.modulator is not None}")


class BasicUformerLayer(nn.Module):
    """``depth`` LeWin blocks; odd blocks use shifted windows (official ``shift_flag=True``)."""

    def __init__(self, dim: int, depth: int, num_heads: int, win_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, qk_scale: float | None = None, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: Sequence[float] | float = 0.0, norm_layer=nn.LayerNorm, shift_flag: bool = True,
                 modulator: bool = False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(
                dim=dim, num_heads=num_heads, win_size=win_size,
                shift_size=(win_size // 2 if (shift_flag and i % 2 == 1) else 0),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                norm_layer=norm_layer, modulator=modulator)
            for i in range(depth)])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for blk in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, H, W, use_reentrant=False)
            else:
                x = blk(x, H, W)
        return x


# --------------------------------------------------------------------------- sampling and projections


class Downsample(nn.Module):
    """4x4 stride-2 convolution: half the size, ``out_channel`` channels."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return _map_to_tokens(self.conv(_tokens_to_map(x, H, W)))


class Upsample(nn.Module):
    """2x2 stride-2 transposed convolution: twice the size, ``out_channel`` channels."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return _map_to_tokens(self.deconv(_tokens_to_map(x, H, W)))


class InputProj(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _map_to_tokens(self.proj(x))


class OutputProj(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.proj(_tokens_to_map(x, H, W))


# --------------------------------------------------------------------------- FDNet


class FDNet(nn.Module):
    """FDUformer from the official FDNet code, for ``(B, C, H, W)`` real input of any size.

    ``forward(x)`` returns the restored map, the same shape as ``x``.
    ``forward(x, return_features=True)`` also returns the signal and
    interference features of the decomposition block, both
    ``(B, H/16 * W/16, 16 * embed_dim)`` tokens of the padded input, for
    the cosine decomposition loss.
    """

    def __init__(self, in_channels: int = 2, embed_dim: int = PAPER_EMBED_DIM,
                 depths: Sequence[int] = OFFICIAL_DEPTHS, num_heads: Sequence[int] = OFFICIAL_HEADS,
                 win_size: int = PAPER_WIN_SIZE, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 qk_scale: float | None = None, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, shift_flag: bool = True,
                 modulator: bool = True):
        super().__init__()
        depths, num_heads = list(depths), list(num_heads)
        if len(depths) != 9 or len(num_heads) != 9:
            raise ValueError("FDNet needs 9 depths and 9 head counts: 4 encoder, bottleneck, 4 decoder")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.num_levels = 4
        self.pad_multiple = (2 ** self.num_levels) * win_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:4]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        common = dict(win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate, norm_layer=norm_layer, shift_flag=shift_flag)

        self.input_proj = InputProj(in_channels, embed_dim)
        self.output_proj = OutputProj(2 * embed_dim, in_channels)

        # Encoder: widths C, 2C, 4C, 8C; each level ends with a stride-2 conv.
        self.encoderlayer_0 = BasicUformerLayer(embed_dim, depths[0], num_heads[0],
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])], **common)
        self.dowsample_0 = Downsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(embed_dim * 2, depths[1], num_heads[1],
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])], **common)
        self.dowsample_1 = Downsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(embed_dim * 4, depths[2], num_heads[2],
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])], **common)
        self.dowsample_2 = Downsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(embed_dim * 8, depths[3], num_heads[3],
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])], **common)
        self.dowsample_3 = Downsample(embed_dim * 8, embed_dim * 16)

        # Feature decomposition block: signal branch (official name "conv") and interference branch ("noise").
        self.conv = BasicUformerLayer(embed_dim * 16, depths[4], num_heads[4], drop_path=conv_dpr, **common)
        self.noise = BasicUformerLayer(embed_dim * 16, depths[4], num_heads[4], drop_path=conv_dpr, **common)

        # Decoder: upsample, concatenate the skip, LeWin blocks with the window modulator.
        self.upsample_0 = Upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(embed_dim * 16, depths[5], num_heads[5],
                                                drop_path=dec_dpr[:depths[5]], modulator=modulator, **common)
        self.upsample_1 = Upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(embed_dim * 8, depths[6], num_heads[6],
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                modulator=modulator, **common)
        self.upsample_2 = Upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(embed_dim * 4, depths[7], num_heads[7],
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                modulator=modulator, **common)
        self.upsample_3 = Upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(embed_dim * 2, depths[8], num_heads[8],
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                modulator=modulator, **common)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def padded_size(self, H: int, W: int) -> tuple[int, int]:
        m = self.pad_multiple
        return math.ceil(H / m) * m, math.ceil(W / m) * m

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        H0, W0 = x.shape[-2:]
        H, W = self.padded_size(H0, W0)
        xp = F.pad(x, (0, W - W0, 0, H - H0)) if (H, W) != (H0, W0) else x

        y = self.pos_drop(self.input_proj(xp))
        conv0 = self.encoderlayer_0(y, H, W)
        pool0 = self.dowsample_0(conv0, H, W)
        conv1 = self.encoderlayer_1(pool0, H // 2, W // 2)
        pool1 = self.dowsample_1(conv1, H // 2, W // 2)
        conv2 = self.encoderlayer_2(pool1, H // 4, W // 4)
        pool2 = self.dowsample_2(conv2, H // 4, W // 4)
        conv3 = self.encoderlayer_3(pool2, H // 8, W // 8)
        pool3 = self.dowsample_3(conv3, H // 8, W // 8)

        # Feature decomposition: the interference features are not passed on.
        conv4 = self.conv(pool3, H // 16, W // 16)
        interference = self.noise(pool3, H // 16, W // 16) if return_features else None

        up0 = self.upsample_0(conv4, H // 16, W // 16)
        deconv0 = self.decoderlayer_0(torch.cat([up0, conv3], -1), H // 8, W // 8)
        up1 = self.upsample_1(deconv0, H // 8, W // 8)
        deconv1 = self.decoderlayer_1(torch.cat([up1, conv2], -1), H // 4, W // 4)
        up2 = self.upsample_2(deconv1, H // 4, W // 4)
        deconv2 = self.decoderlayer_2(torch.cat([up2, conv1], -1), H // 2, W // 2)
        up3 = self.upsample_3(deconv2, H // 2, W // 2)
        deconv3 = self.decoderlayer_3(torch.cat([up3, conv0], -1), H, W)

        out = xp + self.output_proj(deconv3, H, W)
        out = out[..., :H0, :W0]
        if return_features:
            return out, conv4, interference
        return out
