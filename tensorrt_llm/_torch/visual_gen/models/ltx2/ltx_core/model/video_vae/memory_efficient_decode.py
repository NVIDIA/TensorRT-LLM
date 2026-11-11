"""Memory-efficient VAE decoder operations.
Reduces peak VRAM usage during video decoding through in-place operations
and workspace buffer reuse.  The main optimizations are:
1. **Workspace buffers** -- Pre-allocated tensors with temporal padding replace
   dynamic padding (``F.pad`` / ``concatenate``) in ``CausalConv3d``.  A
   workspace of shape ``[B, C, T+2, H, W]`` holds the data in positions
   ``[1:-1]`` with replicate padding at ``[0]`` and ``[-1]``.
2. **In-place temporal-chunked Conv3d** *(non-causal only)* -- The convolution
   output is written back into the workspace buffer, avoiding a separate
   output allocation.  Temporal chunking with boundary save/restore ensures
   correct reads despite in-place writes.
3. **In-place normalization and affine transforms** -- PixelNorm, scale/shift,
   and SiLU are applied in-place on workspace views.
4. **Free-before-conv** -- For ``DepthToSpaceUpsample`` blocks the input
   tensor is freed before the convolution runs so that peak VRAM never holds
   input *and* output simultaneously.
Both causal and non-causal modes are supported.  Non-causal mode benefits
from all four optimizations.  Causal mode benefits from optimizations 1, 3,
and 4; in-place conv (2) is skipped because the asymmetric causal padding
layout prevents clean in-place overwrites.
Usage via the ``ModuleOps`` pattern (preferred)::
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae import (
        CHANNELS_LAST_3D_WEIGHTS,
        MEMORY_EFFICIENT_DECODE,
        VAE_DECODER_COMFY_KEYS_FILTER,
    )
    sd_ops = SDOps(
        name=f"sd_ops_chain_{VAE_DECODER_COMFY_KEYS_FILTER.name}+{CHANNELS_LAST_3D_WEIGHTS.name}",
        mapping=(*VAE_DECODER_COMFY_KEYS_FILTER.mapping, *CHANNELS_LAST_3D_WEIGHTS.mapping),
    excluded_prefixes=VAE_DECODER_COMFY_KEYS_FILTER.excluded_prefixes,
    )
    builder = decoder_builder.with_module_ops(
        (*decoder_builder.module_ops, MEMORY_EFFICIENT_DECODE)
    ).with_sd_ops(sd_ops)
Or applied directly to an existing decoder::
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.memory_efficient_decode import (
        enable_memory_efficient_decode,
    )
    enable_memory_efficient_decode(decoder)
Weights must be loaded with :data:`CHANNELS_LAST_3D_WEIGHTS` chained into sd_ops
(or equivalent ``channels_last_3d`` layout) so the registry caches NHWC; the
module op only formats activations and installs the workspace forward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.common.normalization import PixelNorm
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import CausalConv3d, memory_format_of
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import unpatchify
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.sampling import DepthToSpaceUpsample

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import ConvVideoDecoder


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _find_temporal_split_size(num_frames: int) -> int:
    """Find chunk size for in-place temporal convolution.
    The chunk size ensures the last chunk has at least 3 frames
    (the temporal kernel size), avoiding degenerate chunks.
    """
    for s in range(16, 2, -1):
        remainder = num_frames % s
        if remainder == 0 or remainder >= 3:
            return s

    raise ValueError(
        f"Unable to find a valid temporal split size for num_frames={num_frames}. "
        "Expected a split size between 3 and 16 such that the final chunk is "
        "either exact or has at least 3 frames."
    )


def _pad_workspace_temporal(workspace: torch.Tensor) -> None:
    """Apply non-causal replicate padding to temporal boundaries.
    Sets ``workspace[:, :, 0]`` to a copy of ``workspace[:, :, 1]`` and
    ``workspace[:, :, -1]`` to a copy of ``workspace[:, :, -2]``.
    """
    workspace[:, :, 0, :, :].copy_(workspace[:, :, 1, :, :])
    workspace[:, :, -1, :, :].copy_(workspace[:, :, -2, :, :])


# ---------------------------------------------------------------------------
# In-place Conv3d (non-causal only)
# ---------------------------------------------------------------------------


def inplace_conv3d_temporal_chunked(workspace: torch.Tensor, conv: nn.Conv3d) -> None:
    """Run a 3x3x3 Conv3d in-place on a temporally-padded workspace.
    The workspace has shape ``[B, C, T+2, H, W]`` where positions ``[1:-1]``
    hold the real data and positions ``[0]`` and ``[-1]`` are padding slots.
    The convolution must have ``kernel_size=(3,3,3)``, ``stride=(1,1,1)``,
    ``padding=(0,1,1)`` -- no temporal padding, symmetric spatial padding.
    The output (T frames) overwrites positions ``[1:-1]``.  Temporal chunking
    with boundary save/restore ensures each chunk reads unmodified input even
    though earlier chunks already wrote to the same buffer.
    Only valid for **non-causal** mode (symmetric replicate padding).
    Args:
        workspace: Tensor ``[B, max(C_in, C_out), T+2, H, W]``.
            Modified in-place; after the call ``workspace[:, :C_out, 1:-1]``
            holds the convolution result.
        conv: ``nn.Conv3d`` with the constraints above.
    """
    if conv.kernel_size != (3, 3, 3):
        raise ValueError(f"Expected kernel_size=(3,3,3), got {conv.kernel_size}")
    if conv.stride != (1, 1, 1):
        raise ValueError(f"Expected stride=(1,1,1), got {conv.stride}")
    if conv.padding != (0, 1, 1):
        raise ValueError(f"Expected padding=(0,1,1), got {conv.padding}")

    _pad_workspace_temporal(workspace)

    total_frames = workspace.shape[2]
    out_channels = conv.out_channels
    in_channels = conv.in_channels

    if total_frames > 16:
        split_size = _find_temporal_split_size(total_frames)
        num_splits = (total_frames + split_size - 1) // split_size
    else:
        split_size = total_frames - 1
        num_splits = 1

    # 1-frame buffers for saving / restoring boundary frames across chunks.
    x_buf = torch.empty(
        workspace.shape[0],
        workspace.shape[1],
        1,
        workspace.shape[3],
        workspace.shape[4],
        device=workspace.device,
        dtype=workspace.dtype,
        memory_format=memory_format_of(workspace),
    )
    o_buf = torch.empty_like(x_buf)

    # Helper: extract a chunk and make it contiguous.  Workspace views can
    # inherit strides > 2^31 from the full buffer, which makes Conv3d's
    # reflect-padding path (F.pad) crash with "input tensor must fit into
    # 32-bit index math".  A small .clone() per chunk avoids this.
    needs_clone = workspace.untyped_storage().nbytes() > (2**31 - 1) * workspace.element_size()

    def _chunk(t_start: int, t_end: int) -> torch.Tensor:
        s = workspace[:, :in_channels, t_start:t_end]
        return s.clone() if needs_clone else s

    # --- First chunk ---
    if num_splits > 1:
        # Save the boundary now so the loop below can restore it.  Skipped
        # when there is only one chunk: the loop never runs, and the save
        # would be a wasted full HW slice copy.
        x_buf[:, :, 0] = workspace[:, :, split_size - 1].clone()
    workspace[:, :out_channels, 1:split_size] = conv(_chunk(0, split_size + 1))

    # --- Remaining chunks ---
    for i in range(1, num_splits):
        start = i * split_size
        end = min((i + 1) * split_size, total_frames - 1)

        # Save the value at start-1 (now holds previous chunk's output).
        o_buf[:, :, 0] = workspace[:, :, start - 1].clone()
        # Restore the original input value needed by this chunk's conv.
        workspace[:, :, start - 1] = x_buf[:, :, 0]
        # Save the boundary for the *next* chunk before we overwrite it.
        x_buf[:, :, 0] = workspace[:, :, end - 1].clone()

        workspace[:, :out_channels, start:end] = conv(_chunk(start - 1, end + 1))

        # Put back the previous chunk's output at the boundary.
        workspace[:, :, start - 1] = o_buf[:, :, 0]


# ---------------------------------------------------------------------------
# Causal conv helper (free-before-conv)
# ---------------------------------------------------------------------------


def _causal_pad(x: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Build a causal-padded buffer of shape ``[B, C, T+pad_size, H, W]``.
    Copies ``x`` into ``padded[:, :, pad_size:]`` and replicates the first
    real frame into the leading ``pad_size`` slots.  The caller still owns
    ``x`` after this returns.
    """
    padded = torch.empty(
        x.shape[0],
        x.shape[1],
        x.shape[2] + pad_size,
        x.shape[3],
        x.shape[4],
        device=x.device,
        dtype=x.dtype,
        memory_format=memory_format_of(x),
    )
    padded[:, :, pad_size:].copy_(x)
    for i in range(pad_size):
        padded[:, :, i] = padded[:, :, pad_size]
    return padded


def _causal_pad_free_and_conv(x: torch.Tensor, causal_conv: CausalConv3d) -> torch.Tensor:
    """Causal-pad *x*, free it, then run the raw ``nn.Conv3d``.
    This avoids the peak where both the original and padded tensors are
    live simultaneously (as happens inside ``CausalConv3d.forward``).
    Args:
        x: Input ``[B, C_in, T, H, W]``.  **Deleted** inside this function;
            the caller must not use it afterwards.
    Returns:
        Convolution output ``[B, C_out, T, H, W]``.
    """
    padded = _causal_pad(x, causal_conv.time_kernel_size - 1)
    del x
    result = causal_conv.conv(padded)
    del padded
    return result


# ---------------------------------------------------------------------------
# In-place normalization
# ---------------------------------------------------------------------------


def _pixel_norm_inplace(x: torch.Tensor, eps: float = 1e-8) -> None:
    """In-place RMS (pixel) normalization along the channel dimension."""
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    x.div_(rms)


def _norm_inplace(norm: nn.Module, x: torch.Tensor) -> None:
    """Apply *norm* in-place, using an optimised path for ``PixelNorm``."""
    if isinstance(norm, PixelNorm):
        _pixel_norm_inplace(x, eps=norm.eps)
    else:
        # GroupNorm or other -- fall back to allocating a temporary.
        result = norm(x)
        x.copy_(result)
        del result


# ---------------------------------------------------------------------------
# Per-block efficient forwards
# ---------------------------------------------------------------------------


def _resnet_block_forward_inplace(
    resnet: ResnetBlock3D,
    workspace: torch.Tensor,
    causal: bool,
    timestep: torch.Tensor | None,
    generator: torch.Generator | None,
) -> None:
    """Run a ``ResnetBlock3D`` in-place on a workspace buffer.
    The workspace has shape ``[B, C, T+2, H, W]`` with real data in
    ``[1:-1]``.  After this call ``workspace[:, :, 1:-1]`` holds the
    residual-branch output ``F(x)`` (without the skip connection --
    the caller adds it back to the hidden state).
    Only valid when ``in_channels == out_channels`` (true for all
    ``ResnetBlock3D`` instances inside a ``UNetMidBlock3D``).
    """
    if resnet.in_channels != resnet.out_channels:
        raise ValueError(
            "In-place resnet forward requires in_channels == out_channels, "
            f"got {resnet.in_channels} != {resnet.out_channels}"
        )

    interior = workspace[:, :, 1:-1]

    # --- norm1 + [ada scaling] + SiLU + conv1 ---
    _norm_inplace(resnet.norm1, interior)

    if resnet.timestep_conditioning and timestep is not None:
        ada = resnet.scale_shift_table[None, ..., None, None, None].to(
            device=interior.device, dtype=interior.dtype
        ) + timestep.reshape(
            interior.shape[0],
            4,
            -1,
            timestep.shape[-3],
            timestep.shape[-2],
            timestep.shape[-1],
        )
        shift1, scale1, shift2, scale2 = ada.unbind(dim=1)
        interior.mul_(1 + scale1).add_(shift1)

    F.silu(interior, inplace=True)

    if causal:
        result = resnet.conv1(interior, causal=True)
        interior.copy_(result)
        del result
    else:
        inplace_conv3d_temporal_chunked(workspace, resnet.conv1.conv)

    if resnet.inject_noise:
        spatial_shape = interior.shape[-2:]
        scale = resnet.per_channel_scale1.to(device=interior.device, dtype=interior.dtype)
        noise = torch.randn(spatial_shape, device=interior.device, dtype=interior.dtype, generator=generator)
        interior.add_((noise * scale)[None, :, None, ...])

    # --- norm2 + [ada scaling] + SiLU + conv2 ---
    _norm_inplace(resnet.norm2, interior)

    if resnet.timestep_conditioning and timestep is not None:
        interior.mul_(1 + scale2).add_(shift2)  # type: ignore[possibly-undefined]

    F.silu(interior, inplace=True)
    # dropout is always 0.0 during inference -- skip.

    if causal:
        result = resnet.conv2(interior, causal=True)
        interior.copy_(result)
        del result
    else:
        inplace_conv3d_temporal_chunked(workspace, resnet.conv2.conv)

    if resnet.inject_noise:
        spatial_shape = interior.shape[-2:]
        scale = resnet.per_channel_scale2.to(device=interior.device, dtype=interior.dtype)
        noise = torch.randn(spatial_shape, device=interior.device, dtype=interior.dtype, generator=generator)
        interior.add_((noise * scale)[None, :, None, ...])


def _midblock_forward_efficient(
    block: UNetMidBlock3D,
    hidden_states: torch.Tensor,
    causal: bool,
    timestep: torch.Tensor | None,
    generator: torch.Generator | None,
    prefer_channels_last_3d: bool = False,
) -> torch.Tensor:
    """Memory-efficient ``UNetMidBlock3D`` forward.
    Allocates a single workspace buffer that is reused across all
    ``ResnetBlock3D`` iterations.  For each block the workspace is
    populated with the current hidden state, processed in-place, and
    the result is added back (residual connection).
    """
    timestep_embed = None
    if block.timestep_conditioning:
        if timestep is None:
            raise ValueError("'timestep' required when timestep_conditioning=True")
        batch_size = hidden_states.shape[0]
        timestep_embed = block.time_embedder(
            timestep=timestep.flatten(),
            hidden_dtype=hidden_states.dtype,
        )
        timestep_embed = timestep_embed.view(batch_size, timestep_embed.shape[-1], 1, 1, 1)

    workspace = torch.empty(
        hidden_states.shape[0],
        hidden_states.shape[1],
        hidden_states.shape[2] + 2,
        hidden_states.shape[3],
        hidden_states.shape[4],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
        memory_format=memory_format_of(hidden_states, prefer_channels_last_3d),
    )

    for resnet in block.res_blocks:
        workspace[:, :, 1:-1].copy_(hidden_states)
        _resnet_block_forward_inplace(resnet, workspace, causal, timestep_embed, generator)
        hidden_states.add_(workspace[:, :, 1:-1])

    del workspace
    return hidden_states


def _upsample_forward_efficient(
    block: DepthToSpaceUpsample,
    x: torch.Tensor,
    causal: bool,
    prefer_channels_last_3d: bool = False,
) -> torch.Tensor:
    """Memory-efficient ``DepthToSpaceUpsample`` forward.
    For non-causal mode the input is copied into a workspace and the
    convolution runs in-place.  For causal mode the input is manually
    padded and freed before the convolution runs.  Both paths avoid
    the peak where input *and* output coexist.
    """
    if block.residual:
        x_in = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=block.stride[0],
            p2=block.stride[1],
            p3=block.stride[2],
        )
        num_repeat = math.prod(block.stride) // block.out_channels_reduction_factor
        x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
        if block.stride[0] == 2:
            x_in = x_in[:, :, 1:, :, :]

    conv = block.conv.conv  # underlying nn.Conv3d inside CausalConv3d
    in_channels = x.shape[1]
    out_channels = conv.out_channels

    if causal:
        x = _causal_pad_free_and_conv(x, block.conv)
    else:
        mem_fmt = memory_format_of(x, prefer_channels_last_3d)
        workspace = torch.empty(
            x.shape[0],
            max(in_channels, out_channels),
            x.shape[2] + 2,
            x.shape[3],
            x.shape[4],
            device=x.device,
            dtype=x.dtype,
            memory_format=mem_fmt,
        )
        workspace[:, :in_channels, 1:-1].copy_(x)
        del x
        inplace_conv3d_temporal_chunked(workspace, conv)
        x = workspace[:, :out_channels, 1:-1].contiguous(memory_format=mem_fmt)
        del workspace

    x = rearrange(
        x,
        "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
        p1=block.stride[0],
        p2=block.stride[1],
        p3=block.stride[2],
    )
    if block.stride[0] == 2:
        x = x[:, :, 1:, :, :]
    if block.residual:
        x.add_(x_in)
        del x_in
    return x


# ---------------------------------------------------------------------------
# Final norm + conv_out
# ---------------------------------------------------------------------------


def _final_norm_and_conv_out(
    decoder: ConvVideoDecoder,
    sample: torch.Tensor,
    causal: bool,
    scaled_timestep: torch.Tensor | None,
    batch_size: int,
    prefer_channels_last_3d: bool = False,
) -> torch.Tensor:
    """Workspace-based final norm + [ada] + SiLU + conv_out + unpatchify."""
    conv_out_mod: CausalConv3d = decoder.conv_out  # type: ignore[assignment]
    conv_out = conv_out_mod.conv
    feature_channels = sample.shape[1]

    mem_fmt = memory_format_of(sample, prefer_channels_last_3d)
    workspace = torch.empty(
        sample.shape[0],
        max(feature_channels, conv_out.out_channels),
        sample.shape[2] + 2,
        sample.shape[3],
        sample.shape[4],
        device=sample.device,
        dtype=sample.dtype,
        memory_format=mem_fmt,
    )
    workspace[:, :feature_channels, 1:-1].copy_(sample)
    del sample

    interior = workspace[:, :feature_channels, 1:-1]
    _norm_inplace(decoder.conv_norm_out, interior)

    if decoder.timestep_conditioning:
        embedded_timestep = decoder.last_time_embedder(
            timestep=scaled_timestep.flatten(),
            hidden_dtype=interior.dtype,
        )
        embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
        ada_values = decoder.last_scale_shift_table[None, ..., None, None, None].to(
            device=interior.device, dtype=interior.dtype
        ) + embedded_timestep.reshape(
            batch_size,
            2,
            -1,
            embedded_timestep.shape[-3],
            embedded_timestep.shape[-2],
            embedded_timestep.shape[-1],
        )
        shift, scale = ada_values.unbind(dim=1)
        interior.mul_(1 + scale).add_(shift)

    F.silu(interior, inplace=True)

    if causal:
        # Causal: build padded tensor directly from the interior view,
        # then free the workspace before running the conv.
        padded = _causal_pad(interior, conv_out_mod.time_kernel_size - 1)
        del workspace, interior
        result = conv_out(padded)
        del padded
    else:
        inplace_conv3d_temporal_chunked(workspace, conv_out)
        result = workspace[:, : conv_out.out_channels, 1:-1].contiguous(memory_format=mem_fmt)
        del workspace, interior

    return unpatchify(result, patch_size_hw=decoder.patch_size, patch_size_t=1)


# ---------------------------------------------------------------------------
# Top-level efficient decoder forward
# ---------------------------------------------------------------------------


def _memory_efficient_forward(
    decoder: ConvVideoDecoder,
    sample: torch.Tensor,
    timestep: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Full memory-efficient ``ConvVideoDecoder.forward`` replacement.
    Orchestrates the entire decode through workspace-based operations:
    ``UNetMidBlock3D`` and ``DepthToSpaceUpsample`` blocks use efficient
    paths; standalone ``ResnetBlock3D`` blocks fall back to the standard
    forward.  The final norm + ada + SiLU + conv_out is also workspace-based.
    All workspaces are allocated ``channels_last_3d`` so cuDNN's NHWC 3D
    conv kernels run end-to-end. Weights must already be NHWC (via
    :data:`CHANNELS_LAST_3D_WEIGHTS` chained at load); this path only formats the
    input sample.
    """
    causal = decoder.causal
    batch_size = sample.shape[0]
    sample = sample.to(next(decoder.parameters()).dtype)

    # --- Noise injection and de-normalisation (identical to standard path) ---
    if decoder.timestep_conditioning:
        noise = (
            torch.randn(sample.size(), generator=generator, dtype=sample.dtype, device=sample.device)
            * decoder.decode_noise_scale
        )
        sample = noise + (1.0 - decoder.decode_noise_scale) * sample

    sample = decoder.per_channel_statistics.un_normalize(sample)

    if timestep is None and decoder.timestep_conditioning:
        timestep = torch.full((batch_size,), decoder.decode_timestep, device=sample.device, dtype=sample.dtype)

    # --- conv_in (latent tensor is small -- standard path is fine) ---
    sample = decoder.conv_in(sample, causal=causal)

    upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype
    sample = sample.to(upscale_dtype)

    scaled_timestep = None
    if decoder.timestep_conditioning:
        if timestep is None:
            raise ValueError("'timestep' required when timestep_conditioning=True")
        scaled_timestep = timestep * decoder.timestep_scale_multiplier.to(sample)

    # Workspaces are unconditionally NHWC: rearrange + slice + residual-add
    # inside _upsample_forward_efficient produces NCHW-default output, so
    # per-tensor inspection would silently fall back to NCHW for every
    # workspace after the first upsample.

    # --- Up blocks (dispatch to efficient path per block type) ---
    for up_block in decoder.up_blocks:
        if isinstance(up_block, UNetMidBlock3D):
            sample = _midblock_forward_efficient(
                up_block,
                sample,
                causal=causal,
                timestep=scaled_timestep if decoder.timestep_conditioning else None,
                generator=generator,
                prefer_channels_last_3d=True,
            )
        elif isinstance(up_block, DepthToSpaceUpsample):
            sample = _upsample_forward_efficient(up_block, sample, causal=causal, prefer_channels_last_3d=True)
        elif isinstance(up_block, ResnetBlock3D):
            sample = up_block(sample, causal=causal, generator=generator)
        else:
            sample = up_block(sample, causal=causal)

    return _final_norm_and_conv_out(decoder, sample, causal, scaled_timestep, batch_size, prefer_channels_last_3d=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _to_channels_last_3d(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Store 5D conv weights in NHWC so the mem-efficient path never reformats them."""
    value = inputs[key]
    if value.dim() == 5:
        value = value.to(memory_format=torch.channels_last_3d)
    return {key: value}


# Load-time NHWC for 5D weights. Chain after the VAE decoder filter at the builder
# (same pattern as QuantizationPolicy.sd_ops) so the registry keeps this layout
# under a distinct sd_ops name.
CHANNELS_LAST_3D_WEIGHTS = SDOps("channels_last_3d_weights").with_kv_operation(_to_channels_last_3d)


def enable_memory_efficient_decode(decoder: nn.Module) -> nn.Module:
    """Patch a ``ConvVideoDecoder`` to use the memory-efficient forward path.
    The mem-efficient path runs the decoder in ``channels_last_3d`` memory
    format. **Weights** must already be NHWC — chain :data:`CHANNELS_LAST_3D_WEIGHTS`
    into the builder's sd_ops at load so the registry stores that layout. This
    patch only formats the **activation** and swaps in the workspace-based forward.
    The original ``forward`` is saved as ``decoder._original_forward`` so
    that it can be restored later with :func:`disable_memory_efficient_decode`.
    """
    # Import here to avoid circular dependency at module level.
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import ConvVideoDecoder  # noqa: PLC0415

    if not isinstance(decoder, ConvVideoDecoder):
        raise TypeError(f"Expected ConvVideoDecoder, got {type(decoder).__name__}")

    if hasattr(decoder, "_original_forward"):
        return decoder

    original_forward = decoder.forward

    def efficient_forward(
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample.dim() == 5:
            sample = sample.to(memory_format=torch.channels_last_3d)
        return _memory_efficient_forward(decoder, sample, timestep, generator)

    decoder._original_forward = original_forward  # type: ignore[attr-defined]
    decoder.forward = efficient_forward  # type: ignore[assignment]
    return decoder


def disable_memory_efficient_decode(decoder: nn.Module) -> nn.Module:
    """Restore the original ``forward`` method on a patched ``ConvVideoDecoder``."""
    if hasattr(decoder, "_original_forward"):
        decoder.forward = decoder._original_forward  # type: ignore[attr-defined]
        del decoder._original_forward  # type: ignore[attr-defined]
    return decoder


def _is_video_decoder(model: nn.Module) -> bool:
    """Matcher for the ``MEMORY_EFFICIENT_DECODE`` module op."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import ConvVideoDecoder  # noqa: PLC0415

    return isinstance(model, ConvVideoDecoder)


MEMORY_EFFICIENT_DECODE = ModuleOps(
    name="memory_efficient_vae_decode",
    matcher=_is_video_decoder,
    mutator=enable_memory_efficient_decode,
)
