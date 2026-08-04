# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cache-dit enable/refresh helpers for TensorRT-LLM diffusion pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import cache_dit
import torch.nn as nn
from cache_dit import (
    BlockAdapter,
    DBCacheConfig,
    ForwardPattern,
    ParamsModifier,
    TaylorSeerCalibratorConfig,
)

from tensorrt_llm.logger import logger

from ..config import CacheDiTConfig

# Batched CFG: default enable_separate_cfg False (single cond+uncond batch).
_WAN_CFG_DEFAULT = False
_FLUX_CFG_DEFAULT = False
_LTX2_CFG_DEFAULT = False

# Wan 2.2 dual-transformer: stricter caps on the low-noise expert stack (second ParamsModifier).
_WAN22_LOW_NOISE_MAX_WARMUP_STEPS = 2
_WAN22_LOW_NOISE_MAX_CACHED_STEPS = 20


@dataclass
class CacheDiTEnableResult:
    """Return value from a cache-dit enabler."""

    refresh: Callable[[int], None]
    disable_target: Any
    summary_modules: List[nn.Module]


def _resolved_enable_separate_cfg(cfg: CacheDiTConfig, default: bool) -> bool:
    if cfg.enable_separate_cfg is not None:
        return cfg.enable_separate_cfg
    return default


def db_cache_kwargs_from_cache_dit_config(
    user: CacheDiTConfig,
    **overrides: Any,
) -> dict[str, Any]:
    """Keyword arguments for cache_dit.DBCacheConfig built from user config.

    Same keyword set as the reference _build_db_cache_config pattern for DBCacheConfig.
    Callers add overrides (e.g. enable_separate_cfg for Wan).
    """
    kwargs: dict[str, Any] = dict(
        num_inference_steps=None,
        Fn_compute_blocks=user.Fn_compute_blocks,
        Bn_compute_blocks=user.Bn_compute_blocks,
        max_warmup_steps=user.max_warmup_steps,
        max_cached_steps=user.max_cached_steps,
        max_continuous_cached_steps=user.max_continuous_cached_steps,
        residual_diff_threshold=user.residual_diff_threshold,
        force_refresh_step_hint=user.force_refresh_step_hint,
        force_refresh_step_policy=user.force_refresh_step_policy,
    )
    kwargs.update(overrides)
    return kwargs


def split_wan22_inference_steps(pipeline: Any, num_inference_steps: int) -> tuple[int, int]:
    """Split total steps into (high-noise expert steps, low-noise expert steps) for Wan 2.2."""
    boundary_ratio = getattr(pipeline, "boundary_ratio", None)
    if boundary_ratio is not None:
        boundary_timestep = boundary_ratio * pipeline.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None

    device = next(pipeline.transformer.parameters()).device
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    num_high_noise_steps = 0
    for t in timesteps:
        if boundary_timestep is None or t >= boundary_timestep:
            num_high_noise_steps += 1
    num_low_noise_steps = num_inference_steps - num_high_noise_steps
    return num_high_noise_steps, num_low_noise_steps


def _build_db_cache_config(user: CacheDiTConfig, **overrides: Any) -> Any:
    return DBCacheConfig(**db_cache_kwargs_from_cache_dit_config(user, **overrides))


def _maybe_calibrator(cache_dit_cfg: CacheDiTConfig):
    if not cache_dit_cfg.enable_taylorseer:
        return None
    return TaylorSeerCalibratorConfig(taylorseer_order=cache_dit_cfg.taylorseer_order)


def _refresh_ctx(
    module: nn.Module,
    cache_dit_cfg: CacheDiTConfig,
    num_inference_steps: int,
    verbose: bool,
    *,
    extra_reset: Optional[dict] = None,
) -> None:
    if cache_dit_cfg.scm_steps_mask_policy is None:
        cache_dit.refresh_context(module, num_inference_steps=num_inference_steps, verbose=verbose)
        return
    # cache_dit predefined steps_mask only allows total_steps in {4, 6} when total_steps < 8.
    # Pipeline warmup (e.g. 2 denoise steps) would assert inside steps_mask().
    if num_inference_steps < 8 and num_inference_steps not in (4, 6):
        logger.warning(
            f"Cache-DiT: scm_steps_mask_policy={cache_dit_cfg.scm_steps_mask_policy!r} is "
            f"incompatible with num_inference_steps={num_inference_steps} (predefined masks "
            f"require 4, 6, or >= 8). Using plain refresh_context."
        )
        cache_dit.refresh_context(module, num_inference_steps=num_inference_steps, verbose=verbose)
        return
    reset_kw: dict = dict(
        num_inference_steps=num_inference_steps,
        steps_computation_mask=cache_dit.steps_mask(
            mask_policy=cache_dit_cfg.scm_steps_mask_policy,
            total_steps=num_inference_steps,
        ),
        steps_computation_policy=cache_dit_cfg.scm_steps_policy,
    )
    if extra_reset:
        reset_kw.update(extra_reset)
    cache_dit.refresh_context(
        module,
        cache_config=DBCacheConfig().reset(**reset_kw),
        verbose=verbose,
    )


def enable_cache_dit_for_wan(pipeline: Any, cache_dit_cfg: CacheDiTConfig) -> CacheDiTEnableResult:
    """Wan T2V / I2V: single transformer (2.1) or dual (2.2)."""
    calibrator = _maybe_calibrator(cache_dit_cfg)
    separate = _resolved_enable_separate_cfg(cache_dit_cfg, _WAN_CFG_DEFAULT)

    transformer_2 = getattr(pipeline, "transformer_2", None)

    if transformer_2 is None:
        db_cfg = _build_db_cache_config(cache_dit_cfg, enable_separate_cfg=separate)
        logger.info("Cache-DiT: Wan single-transformer mode (Pattern_2 BlockAdapter).")
        adapter = BlockAdapter(
            transformer=pipeline.transformer,
            blocks=[pipeline.transformer.blocks],
            forward_pattern=[ForwardPattern.Pattern_2],
            params_modifiers=[ParamsModifier(cache_config=db_cfg)],
        )
        disable_target = cache_dit.enable_cache(
            adapter,
            cache_config=db_cfg,
            calibrator_config=calibrator,
        )

        def refresh(num_inference_steps: int) -> None:
            _refresh_ctx(
                pipeline.transformer,
                cache_dit_cfg,
                num_inference_steps,
                False,
            )

        return CacheDiTEnableResult(
            refresh=refresh,
            disable_target=disable_target,
            summary_modules=[pipeline.transformer],
        )

    assert DBCacheConfig is not None
    shared = DBCacheConfig(
        **db_cache_kwargs_from_cache_dit_config(cache_dit_cfg, enable_separate_cfg=separate),
    )

    # Wan 2.2: dual BlockAdapter, two ParamsModifier stacks (high-noise / low-noise experts),
    # shared DBCacheConfig for enable_cache; batched CFG → enable_separate_cfg from _WAN_CFG_DEFAULT.
    adapter = BlockAdapter(
        transformer=[pipeline.transformer, transformer_2],
        blocks=[pipeline.transformer.blocks, transformer_2.blocks],
        forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
        params_modifiers=[
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=cache_dit_cfg.max_warmup_steps,
                    max_cached_steps=cache_dit_cfg.max_cached_steps,
                ),
            ),
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=_WAN22_LOW_NOISE_MAX_WARMUP_STEPS,
                    max_cached_steps=_WAN22_LOW_NOISE_MAX_CACHED_STEPS,
                ),
            ),
        ],
    )
    disable_target = cache_dit.enable_cache(
        adapter,
        cache_config=shared,
        calibrator_config=calibrator,
    )

    def refresh_dual(num_inference_steps: int) -> None:
        hi, lo = split_wan22_inference_steps(pipeline, num_inference_steps)
        if cache_dit_cfg.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=hi, verbose=False)
            cache_dit.refresh_context(transformer_2, num_inference_steps=lo, verbose=False)
        else:
            _refresh_ctx(pipeline.transformer, cache_dit_cfg, hi, False)
            _refresh_ctx(transformer_2, cache_dit_cfg, lo, False)

    logger.info("Cache-DiT: Wan 2.2 dual-transformer mode enabled.")
    return CacheDiTEnableResult(
        refresh=refresh_dual,
        disable_target=disable_target,
        summary_modules=[pipeline.transformer, transformer_2],
    )


def enable_cache_dit_for_flux(
    pipeline: Any,
    cache_dit_cfg: CacheDiTConfig,
    *,
    is_flux2: bool,
) -> CacheDiTEnableResult:
    """Cache-DiT BlockAdapter for FLUX.1 / FLUX.2."""
    calibrator = _maybe_calibrator(cache_dit_cfg)
    separate = _resolved_enable_separate_cfg(cache_dit_cfg, _FLUX_CFG_DEFAULT)
    db_cfg = _build_db_cache_config(cache_dit_cfg, enable_separate_cfg=separate)

    if calibrator is not None:
        logger.info(f"TaylorSeer enabled with order={cache_dit_cfg.taylorseer_order}")

    # TaylorSeer: set ParamsModifier(calibrator_config=...) only when calibrator is non-None.
    if calibrator is not None:
        modifier = ParamsModifier(cache_config=db_cfg, calibrator_config=calibrator)
    else:
        modifier = ParamsModifier(cache_config=db_cfg)

    if is_flux2:
        block_lists = [
            pipeline.transformer.transformer_blocks,
            pipeline.transformer.single_transformer_blocks,
        ]
        forward_pattern = [ForwardPattern.Pattern_1, ForwardPattern.Pattern_3]
        tag = "FLUX.2"
    else:
        block_lists = [
            pipeline.transformer.transformer_blocks,
            pipeline.transformer.single_transformer_blocks,
        ]
        forward_pattern = [ForwardPattern.Pattern_1, ForwardPattern.Pattern_1]
        tag = "FLUX.1"

    adapter = BlockAdapter(
        transformer=pipeline.transformer,
        blocks=block_lists,
        forward_pattern=forward_pattern,
        params_modifiers=[modifier],
        check_forward_pattern=True,
    )

    logger.info(
        f"Cache-DiT: {tag} — Fn={db_cfg.Fn_compute_blocks}, Bn={db_cfg.Bn_compute_blocks}, "
        f"W={db_cfg.max_warmup_steps}",
    )

    disable_target = cache_dit.enable_cache(
        adapter,
        cache_config=db_cfg,
        calibrator_config=calibrator,
    )

    def refresh_flux(num_inference_steps: int) -> None:
        _refresh_ctx(
            pipeline.transformer,
            cache_dit_cfg,
            num_inference_steps,
            False,
        )

    return CacheDiTEnableResult(
        refresh=refresh_flux,
        disable_target=disable_target,
        summary_modules=[pipeline.transformer],
    )


def enable_cache_dit_for_ltx2(pipeline: Any, cache_dit_cfg: CacheDiTConfig) -> CacheDiTEnableResult:
    """Native LTX-2: Pattern_0 passes video latents and audio latents.

    Second arg uses cache-dit encoder_hidden_states slot.
    """
    calibrator = _maybe_calibrator(cache_dit_cfg)
    separate = _resolved_enable_separate_cfg(cache_dit_cfg, _LTX2_CFG_DEFAULT)
    db_cfg = _build_db_cache_config(cache_dit_cfg, enable_separate_cfg=separate)

    if calibrator is not None:
        logger.info(f"TaylorSeer enabled with order={cache_dit_cfg.taylorseer_order}")
        modifier = ParamsModifier(cache_config=db_cfg, calibrator_config=calibrator)
    else:
        modifier = ParamsModifier(cache_config=db_cfg)

    transformer = pipeline.transformer
    adapter = BlockAdapter(
        transformer=transformer,
        blocks=[transformer.transformer_blocks],
        forward_pattern=[ForwardPattern.Pattern_0],
        params_modifiers=[modifier],
        check_forward_pattern=False,
    )

    logger.info(
        f"Cache-DiT: LTX2 — Fn={db_cfg.Fn_compute_blocks}, Bn={db_cfg.Bn_compute_blocks}, "
        f"W={db_cfg.max_warmup_steps}, R={cache_dit_cfg.residual_diff_threshold:.3f}",
    )

    disable_target = cache_dit.enable_cache(
        adapter,
        cache_config=db_cfg,
        calibrator_config=calibrator,
    )

    def refresh_ltx2(num_inference_steps: int) -> None:
        _refresh_ctx(
            transformer,
            cache_dit_cfg,
            num_inference_steps,
            False,
        )

    return CacheDiTEnableResult(
        refresh=refresh_ltx2,
        disable_target=disable_target,
        summary_modules=[transformer],
    )


CUSTOM_CACHE_DIT_ENABLERS = {
    "WanPipeline": enable_cache_dit_for_wan,
    "WanImageToVideoPipeline": enable_cache_dit_for_wan,
    "FluxPipeline": lambda p, c: enable_cache_dit_for_flux(p, c, is_flux2=False),
    "Flux2Pipeline": lambda p, c: enable_cache_dit_for_flux(p, c, is_flux2=True),
    "LTX2Pipeline": enable_cache_dit_for_ltx2,
}


def enable_cache_dit_for_pipeline(
    pipeline: Any, cache_dit_cfg: CacheDiTConfig
) -> CacheDiTEnableResult:
    """Dispatch to a registered enabler or raise."""
    name = pipeline.__class__.__name__
    if name not in CUSTOM_CACHE_DIT_ENABLERS:
        raise ValueError(
            f"Cache-DiT: no enabler registered for pipeline class '{name}'. "
            f"Supported: {sorted(CUSTOM_CACHE_DIT_ENABLERS.keys())}."
        )
    return CUSTOM_CACHE_DIT_ENABLERS[name](pipeline, cache_dit_cfg)
