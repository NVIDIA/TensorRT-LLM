"""``torch.compile`` mechanics for DiffusionVideoDecoder (no DiffVAEMode / resolve)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.compiling import CompilationConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import DiffusionVideoDecoder


def configure_natten_backend(module_root: nn.Module, backend: str | None) -> None:
    """Pin NATTEN on every ``NeighborhoodAttention3D`` under ``module_root``.
    Installs a :class:`~ltx_core.model.video_vae.transformer.attention.NattenAttention`
    as ``attention_function`` (combined / det / chunked all route through it) and mirrors
    the pin onto ``natten_backend`` for introspection. ``backend=None`` restores NATTEN
    auto-selection.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.attention import (  # noqa: PLC0415
        NattenAttention,
        NeighborhoodAttention3D,
    )

    attn_fn = NattenAttention(backend)
    for module in module_root.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = attn_fn
            module.natten_backend = backend


def configure_cutlass_fna_diffusion_decoder(decoder: DiffusionVideoDecoder) -> None:
    """Eager DiffVAE path: pin all NA modules to ``cutlass-fna`` for lower peak VRAM."""
    configure_natten_backend(decoder, "cutlass-fna")


def compile_diffusion_decoder(
    decoder: DiffusionVideoDecoder,
    *,
    config: CompilationConfig | None = None,
    compile_det_stages: bool = True,
) -> DiffusionVideoDecoder:
    """``torch.compile`` pre-diffusion and/or diffusion-step block methods.
    Diff-step (matches ``feature/diffvae-attn-w-chunking-backup``):
      - Combined: ``forward_combined``
      - Chunked: ``forward_attn_mlp`` only (inject stays eager in-place outside
        the compiled region — keeps peak VRAM near ChunkedEager).
    Det stages optional via ``compile_det_stages``.
    """
    cfg = config or CompilationConfig()

    try:
        import natten  # noqa: PLC0415

        natten.use_kv_parallelism_in_fused_na(False)
    except ImportError:
        pass

    compile_kwargs: dict[str, Any] = {
        "mode": cfg.mode,
        "backend": cfg.backend,
        "fullgraph": cfg.fullgraph,
        "dynamic": cfg.dynamic,
    }

    def _compile(fn: Callable[..., Any]) -> Callable[..., Any]:
        with (
            torch._inductor.config.patch(**cfg.inductor_config),
            torch._dynamo.config.patch(**cfg.dynamo_config),  # type: ignore[attr-defined]
        ):
            return torch.compile(fn, **compile_kwargs)

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.block import ChunkedDiffusionNABlock  # noqa: PLC0415
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.combined.block import CombinedDiffusionNABlock  # noqa: PLC0415
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.dsl_kernels.block import DSLDiffusionNABlock  # noqa: PLC0415

    if compile_det_stages:
        for stage_blocks in decoder.det_stages:
            for block in stage_blocks:
                block.forward = _compile(block.forward)  # type: ignore[method-assign]

    for block in decoder.diff_blocks:
        if isinstance(block, DSLDiffusionNABlock):
            # Fused stage-5 is a CuTe launch; do not torch.compile it.
            continue
        if isinstance(block, ChunkedDiffusionNABlock):
            block.forward_attn_mlp = _compile(block.forward_attn_mlp)  # type: ignore[method-assign]
        elif isinstance(block, CombinedDiffusionNABlock):
            block.forward_combined = _compile(block.forward_combined)  # type: ignore[method-assign]

    decoder.mark_dynamic_shapes = True
    return decoder
