"""Install BLACKWELL_DSL attention + fused blocks onto a DiffusionVideoDecoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.attn import (
    NA_SOFTMAX_BOUND_BUFFER,
    DSLAttention,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.block import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    DSLDiffusionNABlock,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.chain import DSLChannelLinear, DSLDiffusionBlockChain
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.layers import ChannelLinear


def enable_blackwell_dsl(decoder: nn.Module) -> nn.Module:
    """Install CuTe DSL attention + fused deferred stage-5 blocks.
    Runs on the meta model before weights exist: the softmax bound (NaN sentinel) and
    the fused-context weights get empty buffers here, which
    ``video_decoder_sd_ops_for_checkpoint(na_dsl_kernel=True)`` fills at load. The two
    must be paired -- ``ltx_pipelines.utils.blocks.VideoDecoder`` does that from one
    flag; a custom builder that applies this without the matching SDOps fails at first
    forward (meta / NaN / empty fused buffers), not at install.
    GPU capability is *not* gated here so wiring can be unit-tested on any host; the
    kernel launchers still require datacenter Blackwell + cutlass-dsl. Callers that
    build a runnable decoder (e.g. ``VideoDecoder``) should still refuse early via
    ``na_dsl_available()``.
    Every block must already carry its ``stage4_upsample`` -- ``load_state_dict`` checks
    shapes even with ``assign=True``, so the fused buffers can only be registered once
    the hop that decides their shape is known.
    The stage-4 ``upsamples[3].proj`` and each block's ``context_proj`` stay on the
    module after install: the hot path reads only the fused buffers, but the Linears
    remain as load/shape ballast (and for reference / non-fused comparisons).
    """
    try:
        import ltx_kernels.vae  # noqa: F401, PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "DiffVAEMode.BLACKWELL_DSL needs the ltx-kernels package "
            "(uv sync --group kernels); kernel launch additionally needs datacenter "
            "Blackwell + nvidia-cutlass-dsl"
        ) from exc

    dsl_attention = DSLAttention()
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = dsl_attention
            if getattr(module, NA_SOFTMAX_BOUND_BUFFER, None) is None:
                # NaN until SDOps overwrite: an apply-without-fill path must not look "set".
                module.register_buffer(
                    NA_SOFTMAX_BOUND_BUFFER,
                    torch.full((), float("nan"), device=module.k_norm.weight.device, dtype=torch.float32),
                )
        elif isinstance(module, DiffusionNABlock):
            module.__class__ = DSLDiffusionNABlock
            _register_fused_ctx_buffers(module)

    if isinstance(getattr(decoder, "diff_blocks", None), nn.ModuleList):
        decoder.diff_blocks.__class__ = DSLDiffusionBlockChain
    if isinstance(getattr(decoder, "conv_in_x_t", None), ChannelLinear):
        decoder.conv_in_x_t.__class__ = DSLChannelLinear

    return decoder


def _register_fused_ctx_buffers(block: DSLDiffusionNABlock) -> None:
    """Give ``block`` empty ``(P*C, Ctx)`` / ``(P*C,)`` buffers for SDOps to load into.
    The shape is the stage-4 upsample's own Linear: it expands ``Ctx`` to ``P*C`` and the
    pixel-shuffle only permutes that, so composing a *square* ``context_proj`` onto it
    leaves the shape alone. Fusion requires ``context_proj`` in_features == out_features
    (``stage5_channels == stage_channels[-1]``); a non-square checkpoint fails here at
    apply / pack time rather than silently mis-sizing the fused buffers.
    """
    if getattr(block, FUSED_CTX_WEIGHT_BUFFER, None) is not None:
        return
    upsample = block.stage4_upsample
    if upsample is None:
        raise RuntimeError(
            "BLACKWELL_DSL needs each block's stage4_upsample wired before its fused-context "
            "buffers can be sized; apply_diffvae_config does that first"
        )
    weight = upsample.proj.weight
    channels = block.context_proj.out_features
    if block.context_proj.in_features != channels:
        raise RuntimeError(
            f"BLACKWELL_DSL fused context requires square context_proj (C, C); got "
            f"{tuple(block.context_proj.weight.shape)} — stage5_channels must equal "
            f"stage_channels[-1] (context width after stage-4 shuffle)"
        )
    if weight.shape[0] != math.prod(upsample.stride) * channels:
        raise RuntimeError(
            f"stage4_upsample {tuple(weight.shape)} at stride {tuple(upsample.stride)} does not "
            f"shuffle out to this block's {channels} channels, so it cannot fuse into context_proj"
        )
    block.register_buffer(
        FUSED_CTX_WEIGHT_BUFFER,
        torch.empty(weight.shape, device=weight.device, dtype=torch.bfloat16),
    )
    block.register_buffer(
        FUSED_CTX_BIAS_BUFFER,
        torch.empty(weight.shape[:1], device=weight.device, dtype=torch.bfloat16),
    )
