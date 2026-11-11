"""Install a resolved ``DiffVAEConfig`` onto a loaded ``DiffusionVideoDecoder``.
Owns all module mutation and ModuleOps builders. Preset → recipe lives in
``config.py``; ``torch.compile`` mechanics live in ``compiling.py``.
CHUNKED / BLACKWELL_DSL modes wire ``decoder.upsamples[3]`` onto each diff block
for deferred stage-4 (upsample Linear then ``context_proj`` — fused into the CuTe
kernel for BLACKWELL_DSL).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.compiling import CompilationConfig, _config_fingerprint
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import (
    NeighborhoodAttention3D,
    configure_w_chunks,
    natten_available,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.chunked.block import ChunkedDiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.combined.block import CombinedDiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.compiling import (
    compile_diffusion_decoder,
    configure_natten_backend,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.config import (
    DiffVAEBlockKind,
    DiffVAEConfig,
    DiffVAEMode,
    NAttentionKind,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.block import DSLDiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.fallback_na import (
    EagerSdpaAttention,
    TritonNaAttention,
    joint_na_attention,
    triton_na_available,
    warn_no_natten,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder


def _block_cls_for(cfg: DiffVAEConfig) -> type[DiffusionNABlock]:
    if cfg.block is DiffVAEBlockKind.COMBINED:
        return CombinedDiffusionNABlock
    if cfg.block is DiffVAEBlockKind.CHUNKED:
        return ChunkedDiffusionNABlock
    if cfg.block is DiffVAEBlockKind.BLACKWELL_DSL:
        return DSLDiffusionNABlock
    raise ValueError(f"Unknown DiffVAEBlockKind: {cfg.block!r}")


def resolve_attention_for_host(cfg: DiffVAEConfig) -> DiffVAEConfig:
    """Remap NATTEN recipes to Triton/eager when natten is missing.
    Both fallback backends are opaque ``custom_op``s, so Dynamo traces straight over them and
    ``COMBINED`` keeps its compilation. Chunked modes remapped to fallback drop ``torch.compile``
    instead: there it would only cover the attn+mlp region, which is exactly what
    ``CHUNKED_COMPILE``'s higher memory coefficient pays for, so a no-natten host is better served
    by the eager chunked recipe. Auto tiling reads this decision back through
    :func:`~ltx_core.model.video_vae.diffusion_tiling.stage5_mem_coef`.
    """
    if cfg.attention is not NAttentionKind.NATTEN:
        return cfg
    if natten_available():
        return cfg
    kind = NAttentionKind.TRITON if triton_na_available() else NAttentionKind.EAGER_SDPA
    if cfg.block is DiffVAEBlockKind.COMBINED:
        return replace(cfg, attention=kind, natten_backend=None)
    return replace(
        cfg,
        attention=kind,
        natten_backend=None,
        compile_blocks=False,
        compile_det_stages=False,
    )


def _configure_deferred_stage4(decoder: DiffusionVideoDecoder, enabled: bool) -> None:
    """Toggle deferred stage-4 and wire ``stage4_upsample`` on deferred blocks."""
    decoder.deferred_stage4_upsample = enabled
    upsample = decoder.upsamples[3] if enabled else None
    for block in decoder.diff_blocks:
        if isinstance(block, (ChunkedDiffusionNABlock, DSLDiffusionNABlock)):
            block.stage4_upsample = upsample  # type: ignore[assignment]
        elif hasattr(block, "stage4_upsample"):
            # Drop leftover from a prior deferred → COMBINED class swap.
            delattr(block, "stage4_upsample")


def _set_attention_function(decoder: DiffusionVideoDecoder, attn_fn: object) -> None:
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = attn_fn  # type: ignore[assignment]
            module.natten_backend = None


def _install_joint_attention(decoder: DiffusionVideoDecoder, cfg: DiffVAEConfig) -> None:
    """Install the keyframe-aware backend on every NA module, for every mode.
    This is the whole "natten only when installed **and** no keyframes" gate: it lives in
    a slot of its own, so keyframe-less decode keeps whatever ``attention_function`` the
    mode chose (NATTEN when installed) while keyframe decode always routes here -- NATTEN
    cannot express a joint window. It also survives ``configure_natten_backend``-style
    wholesale overwrites of ``attention_function``.
    The CuTe DSL kernel *can* express it, and under ``BLACKWELL_DSL`` it is preferred, with
    ``joint_na_attention()`` behind it for the calls it cannot serve -- a CPU module, a volume
    under its halo panel, or a checkpoint loaded without the DSL SDOps.
    """
    joint = joint_na_attention()
    if cfg.attention is NAttentionKind.BLACKWELL_DSL:
        from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels import DSLJointAttention  # noqa: PLC0415

        joint = DSLJointAttention(joint)
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.joint_attention_function = joint  # type: ignore[assignment]


def _install_attention(decoder: DiffusionVideoDecoder, cfg: DiffVAEConfig) -> None:
    if cfg.attention is NAttentionKind.BLACKWELL_DSL:
        from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.apply_dsl import (  # noqa: PLC0415
            enable_blackwell_dsl,
        )

        enable_blackwell_dsl(decoder)
        return
    if cfg.attention is NAttentionKind.TRITON:
        warn_no_natten(backend="Triton na3d")
        _set_attention_function(decoder, TritonNaAttention())
        return
    if cfg.attention is NAttentionKind.EAGER_SDPA:
        warn_no_natten(backend="eager tiled SDPA na3d")
        _set_attention_function(decoder, EagerSdpaAttention())
        return
    if cfg.natten_backend is not None:
        configure_natten_backend(decoder, cfg.natten_backend)


def apply_diffvae_config(
    decoder: DiffusionVideoDecoder,
    cfg: DiffVAEConfig,
    *,
    compilation_config: CompilationConfig | None = None,
) -> DiffusionVideoDecoder:
    """Install ``cfg`` onto ``decoder`` (only mutator for DiffVAE presets)."""
    cfg = resolve_attention_for_host(cfg)
    block_cls = _block_cls_for(cfg)
    for block in decoder.diff_blocks:
        if isinstance(block, DiffusionNABlock):
            block.__class__ = block_cls

    deferred = cfg.block in (DiffVAEBlockKind.CHUNKED, DiffVAEBlockKind.BLACKWELL_DSL)
    _configure_deferred_stage4(decoder, deferred)
    # W-chunking + rope_num_tiles=1 coupling is for diffusion residual only.
    for block in decoder.diff_blocks:
        configure_w_chunks(block, cfg.w_chunks)

    _install_attention(decoder, cfg)
    _install_joint_attention(decoder, cfg)

    if cfg.compile_blocks:
        compile_diffusion_decoder(
            decoder,
            config=compilation_config,
            compile_det_stages=cfg.compile_det_stages,
        )
    else:
        decoder.mark_dynamic_shapes = False
    return decoder


def apply_diffvae_mode(
    decoder: DiffusionVideoDecoder,
    mode: DiffVAEMode,
    *,
    compilation_config: CompilationConfig | None = None,
) -> DiffusionVideoDecoder:
    """Resolve ``mode`` then :func:`apply_diffvae_config`."""
    return apply_diffvae_config(decoder, mode.resolve(), compilation_config=compilation_config)


def _diffusion_decoder_matcher(model: torch.nn.Module) -> bool:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder  # noqa: PLC0415

    return isinstance(model, DiffusionVideoDecoder)


def build_diffvae_mode_op(
    mode: DiffVAEMode,
    compilation_config: CompilationConfig | None = None,
) -> ModuleOps:
    """Apply ``mode.resolve()`` on the meta model, keyed by non-default compile config."""

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_diffvae_mode(model, mode, compilation_config=compilation_config)  # type: ignore[arg-type]

    return ModuleOps(
        name=f"diffvae_mode_{mode.value}"
        + (
            f":{_config_fingerprint(compilation_config)}"
            if compilation_config is not None and compilation_config != CompilationConfig()
            else ""
        ),
        matcher=_diffusion_decoder_matcher,
        mutator=mutator,
    )


def build_cutlass_fna_diffusion_decoder_op() -> ModuleOps:
    """Backward-compat ModuleOps name; mutator applies :attr:`DiffVAEMode.CHUNKED_EAGER`."""
    inner = build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER)
    return ModuleOps(
        name="cutlass_fna_diffusion_decoder",
        matcher=inner.matcher,
        mutator=inner.mutator,
    )


def build_compile_diffusion_decoder_op(config: CompilationConfig | None = None) -> ModuleOps:
    """Apply combined compilation, retaining the legacy name for default config."""
    inner = build_diffvae_mode_op(DiffVAEMode.COMBINED_COMPILE, compilation_config=config)
    return ModuleOps(
        name="compile_diffusion_decoder"
        + (f":{_config_fingerprint(config)}" if config is not None and config != CompilationConfig() else ""),
        matcher=inner.matcher,
        mutator=inner.mutator,
    )
