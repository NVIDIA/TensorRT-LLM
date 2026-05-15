import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda import amp
from torch.nn import Parameter
from torch.nn.utils import spectral_norm, weight_norm

# ---------------------------------------------------------------------------
# VAE Bottleneck
# ---------------------------------------------------------------------------


class VAEBottleneck(nn.Module):
    """
    Variational Autoencoder (VAE) bottleneck.

    Applies VAE reparameterization trick during encoding.
    """

    def __init__(self) -> None:
        super().__init__()

    def sample(self, mean: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

    def encode(
        self, x: torch.Tensor, return_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encode input through VAE bottleneck.

        Args:
            x: Input tensor with shape [B, C*2, T] where C*2 contains
               concatenated mean and scale parameters
            return_info: Whether to return additional info dict

        Returns:
            Sampled latents (and optionally info dict with KL divergence)
        """
        info = {}

        mean, scale = x.chunk(2, dim=1)
        x, kl = self.sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(
        self, x: torch.Tensor, return_info: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Decode from latents (identity operation for VAE).

        Args:
            x: Latent tensor
            return_info: Whether to return additional info dict

        Returns:
            Latents (and optionally empty info dict)
        """
        info = {}
        if return_info:
            return x, info
        else:
            return x


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version
        based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)

        return x + (1.0 / (beta + 1e-9)) * pow(torch.sin(x * alpha), 2)


# ---------------------------------------------------------------------------
# LayerNorm (fp32-safe)
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. Forces fp32 to avoid numerical issues."""

    def __init__(self, size: int, eps: float = 1e-5, use_bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size)) if use_bias else None
        self.eps = eps

    def forward(self, tensor: Tensor) -> Tensor:
        dtype = tensor.dtype
        with amp.autocast(enabled=True, dtype=torch.float32):
            tensor = F.layer_norm(tensor, self.weight.shape, self.weight, self.bias, self.eps)
        return tensor.to(dtype)


# ---------------------------------------------------------------------------
# ConvNeXt helpers
# ---------------------------------------------------------------------------


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out all parameters of a module (identity-friendly init)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def may_mask(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Apply optional mask tensor to activations."""
    if mask is not None:
        x = x * mask
    return x


# ---------------------------------------------------------------------------
# WN wrappers
# ---------------------------------------------------------------------------


def WNConv1d(*args: Any, **kwargs: Any) -> nn.Conv1d:
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args: Any, **kwargs: Any) -> nn.ConvTranspose1d:
    """Weight-normalized 1D transpose convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# ---------------------------------------------------------------------------
# ConvNeXt block
# ---------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 1D Block adapted from https://github.com/charactr-platform/vocos
    which is adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.
    Supports causal and non-causal mode.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        identity_init (bool): If True, initializes the 1x1 conv in residual paths to zero (identity-friendly).
        use_snake (bool): If True, uses SnakeBeta activation; otherwise, GELU.
        causal (bool): If True, applies causal padding; otherwise, applies symmetric padding for non-causal.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        identity_init: bool = False,
        use_snake: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        if causal:
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((6, 0), 0),
                nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
            )
        else:
            self.dwconv = nn.Sequential(
                nn.ConstantPad1d((3, 3), 0),
                nn.Conv1d(dim, dim, kernel_size=7, groups=dim),
            )

        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)
        self.act = SnakeBeta(intermediate_dim) if use_snake else nn.GELU()

        if identity_init:
            self.pwconv2 = zero_module(nn.Conv1d(intermediate_dim, dim, 1))
        else:
            self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.dwconv(may_mask(x, mask))
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = residual + x
        return may_mask(x, mask)

    def remove_weight_norm(self) -> None:
        """No weight norm is applied in ConvNeXtBlock."""
        pass


# ---------------------------------------------------------------------------
# EnCodec-style conv helpers (SConv1d / SConvTranspose1d)
# ---------------------------------------------------------------------------

CONV_NORMALIZATIONS = frozenset(
    ["none", "weight_norm", "spectral_norm", "time_layer_norm", "layer_norm", "time_group_norm"]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.LayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tuple, mode: str = "zero", value: float = 0.0) -> torch.Tensor:
    """Tiny wrapper around F.pad that handles reflect padding on short inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tuple) -> torch.Tensor:
    """Remove padding from x. Only for 1D."""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Conv1d with optional weight_norm / spectral_norm."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d with optional weight_norm / spectral_norm."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with builtin asymmetric/causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with builtin asymmetric/causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert 0.0 <= self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
