"""Attention building blocks for the video VAE."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction


class _RMSNorm2D(nn.Module):
    r"""Channel-first RMSNorm for 4D tensors with a learnable per-channel gain.
    Args:
        channels: The number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels**0.5
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * (self.scale * self.gamma)


class AttnBlock3D(nn.Module):
    r"""Single-head per-frame spatial self-attention block.
    Frames are folded into the batch dimension and attention is computed over
    H*W per frame -- no cross-frame interaction.
    The block is single-head, matching the reference implementation. An
    ``attention_head_dim`` carried in the VAE block config is intentionally not
    honored: the reference (which trained the checkpoints) is single-head, and a
    real-image encode/decode round-trip on the 16x16x4 checkpoint reconstructs at
    ~29 dB PSNR, confirming the single-head layout is the faithful one.
    Args:
        in_channels: The number of input channels.
        attention: The attention backend, given either as an :class:`AttentionFunction`
            enum (resolved once here via :meth:`~AttentionFunction.to_callable`) or as an
            already-resolved :class:`AttentionCallable`. Defaults to the PyTorch (SDPA)
            backend rather than ``AUTOMATIC``: this block is single-head, so
            ``head_dim == in_channels``, which exceeds the FlashAttention head-dim limit at
            the VAE bottleneck. SDPA's dispatcher falls back to an efficient/math kernel for
            such large head dims, whereas ``AUTOMATIC`` would hand back a raw flash kernel on
            Hopper/Blackwell that the kernel cannot serve.
    """

    def __init__(
        self,
        in_channels: int,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.PYTORCH,
    ) -> None:
        super().__init__()
        self._attention = attention.to_callable() if isinstance(attention, AttentionFunction) else attention
        self.norm = _RMSNorm2D(in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:  # noqa: ARG002
        identity = hidden_states
        b = hidden_states.shape[0]
        x = rearrange(hidden_states, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        qkv = rearrange(self.to_qkv(x), "bt c h w -> bt (h w) c")
        q, k, v = qkv.chunk(3, dim=-1)
        x = self._attention(q, k, v, 1)
        x = rearrange(x, "bt (h w) c -> bt c h w", h=hidden_states.shape[3])
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return identity + x
