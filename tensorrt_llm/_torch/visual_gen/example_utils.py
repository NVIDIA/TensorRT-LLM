#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared argparse helpers for visual-generation example scripts."""

from tensorrt_llm._torch.visual_gen.config import CacheDiTConfig, TeaCacheConfig


def add_cache_dit_args(parser, residual_threshold_default=None):
    """Add Cache-DiT override arguments to an ArgumentParser.

    Args:
        parser: ArgumentParser (or argument group) to add args to.
        residual_threshold_default: Default for --cache_dit_residual_threshold.
            Pass None (default) to defer to CacheDiTConfig's own default;
            pass a float (e.g. 0.16 for LTX2) to set a model-specific default.
    """
    parser.add_argument(
        "--cache_dit_fn_compute_blocks",
        type=int,
        default=None,
        help="DBCache Fn_compute_blocks (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_bn_compute_blocks",
        type=int,
        default=None,
        help="DBCache Bn_compute_blocks (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_max_warmup_steps",
        type=int,
        default=None,
        help="DBCache max_warmup_steps (default: from CacheDiTConfig).",
    )
    parser.add_argument(
        "--cache_dit_max_cached_steps",
        type=int,
        default=None,
        help="DBCache max_cached_steps (-1 = no cap; default: from CacheDiTConfig).",
    )
    if residual_threshold_default is not None:
        residual_help = (
            f"DBCache residual_diff_threshold "
            f"(model default: {residual_threshold_default}; global CacheDiTConfig default is 0.24)."
        )
    else:
        residual_help = "DBCache residual_diff_threshold (default: from CacheDiTConfig)."
    parser.add_argument(
        "--cache_dit_residual_threshold",
        type=float,
        default=residual_threshold_default,
        help=residual_help,
    )
    parser.add_argument(
        "--cache_dit_enable_taylorseer",
        action="store_true",
        help="Enable TaylorSeer calibrator (default: off).",
    )
    parser.add_argument(
        "--cache_dit_taylorseer_order",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="TaylorSeer order; implies TaylorSeer on if set. Default order from CacheDiTConfig.",
    )
    parser.add_argument(
        "--cache_dit_scm_mask_policy",
        type=str,
        default=None,
        help="SCM steps_mask policy name (e.g. fast, medium, slow, ultra). Omit to disable SCM.",
    )
    parser.add_argument(
        "--cache_dit_scm_steps_policy",
        type=str,
        default=None,
        choices=["dynamic", "static"],
        help="SCM steps_computation_policy (default: dynamic if not overridden).",
    )


def add_cache_args(parser, teacache_thresh_default=0.2, teacache_thresh_help=None):
    """Add the full cache-acceleration group (TeaCache + Cache-DiT, mutually exclusive).

    Use for models that support both backends (e.g. Wan T2V/I2V, FLUX).
    Callers that only support Cache-DiT (e.g. LTX2) should add --enable_cache_dit directly
    and call add_cache_dit_args instead.

    Args:
        teacache_thresh_default: Default for --teacache_thresh. Pass None for models where
            the threshold is auto-detected per variant (e.g. FLUX.1 vs FLUX.2).
        teacache_thresh_help: Custom help string for --teacache_thresh. Defaults to a generic
            description when None.
    """
    # Diffusion cache acceleration (TeaCache vs Cache-DiT; mutually exclusive)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--enable_teacache", action="store_true", help="Enable TeaCache acceleration"
    )
    cache_group.add_argument(
        "--enable_cache_dit",
        action="store_true",
        help=(
            "Enable Cache-DiT per-block acceleration (requires the cache_dit package; "
            "see https://github.com/vipshop/cache-dit). Incompatible with --enable_teacache."
        ),
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=teacache_thresh_default,
        help=teacache_thresh_help
        or "TeaCache similarity threshold (rel_l1_thresh); ignored when using --enable_cache_dit",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Use ret_steps mode for TeaCache. "
        "Using Retention Steps will result in faster generation speed and better generation quality. "
        "Ignored when using --enable_cache_dit.",
    )
    # Cache-DiT overrides (only apply with --enable_cache_dit; omitted fields use CacheDiTConfig defaults)
    add_cache_dit_args(parser)


def build_cache_dit_config(args) -> CacheDiTConfig:
    """Build CacheDiTConfig from CLI args; unset options keep Pydantic defaults."""
    overrides: dict = {}
    if args.cache_dit_fn_compute_blocks is not None:
        overrides["Fn_compute_blocks"] = args.cache_dit_fn_compute_blocks
    if args.cache_dit_bn_compute_blocks is not None:
        overrides["Bn_compute_blocks"] = args.cache_dit_bn_compute_blocks
    if args.cache_dit_max_warmup_steps is not None:
        overrides["max_warmup_steps"] = args.cache_dit_max_warmup_steps
    if args.cache_dit_max_cached_steps is not None:
        overrides["max_cached_steps"] = args.cache_dit_max_cached_steps
    if args.cache_dit_residual_threshold is not None:
        overrides["residual_diff_threshold"] = args.cache_dit_residual_threshold
    if args.cache_dit_enable_taylorseer or args.cache_dit_taylorseer_order is not None:
        overrides["enable_taylorseer"] = True
    if args.cache_dit_taylorseer_order is not None:
        overrides["taylorseer_order"] = args.cache_dit_taylorseer_order
    if args.cache_dit_scm_mask_policy is not None:
        overrides["scm_steps_mask_policy"] = args.cache_dit_scm_mask_policy
    if args.cache_dit_scm_steps_policy is not None:
        overrides["scm_steps_policy"] = args.cache_dit_scm_steps_policy
    return CacheDiTConfig(**overrides)


def build_teacache_config(args) -> TeaCacheConfig:
    """Build TeaCacheConfig from CLI args; unset options keep Pydantic defaults."""
    kwargs: dict = {"use_ret_steps": args.use_ret_steps}
    if args.teacache_thresh is not None:
        kwargs["teacache_thresh"] = args.teacache_thresh
    return TeaCacheConfig(**kwargs)


def build_cache_config(args) -> dict:
    """Build the appropriate cache kwarg dict from CLI args.

    Returns a dict ready to unpack into VisualGenArgs (e.g. **build_cache_config(args)).
    Handles the mutually exclusive TeaCache / Cache-DiT selection used by Wan T2V/I2V.
    """
    if args.enable_cache_dit:
        return {"cache": build_cache_dit_config(args)}
    if args.enable_teacache:
        return {"cache": build_teacache_config(args)}
    return {}


def add_quant_args(parser):
    """Add --linear_type dynamic quantization argument."""
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=["default", "trtllm-fp8-per-tensor", "trtllm-fp8-blockwise", "trtllm-nvfp4"],
        help=(
            "Dynamic quantization mode for linear layers. "
            "Quantizes weights on-the-fly during loading from an unquantized checkpoint."
        ),
    )


def add_attention_backend_args(parser, include_sage=False):
    """Add --attention_backend argument (VANILLA, TRTLLM, FA4).

    Args:
        include_sage: If True, also add --enable_sage_attention. Only models with
            SageAttention wiring (FLUX, Wan T2V/I2V) should pass True.
    """
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM", "FA4"],
        help="Attention backend (VANILLA: PyTorch SDPA, TRTLLM: optimized kernels, "
        "FA4: Flash Attention 4). "
        "Note: TRTLLM falls back to VANILLA for cross-attention.",
    )
    if include_sage:
        parser.add_argument(
            "--enable_sage_attention",
            action="store_true",
            help="Enable SageAttention (per-block quantized Q/K/V). Requires TRTLLM backend.",
        )


def add_optimization_args(parser):
    """Add torch.compile, CUDA graph, autotune, and NVTX profiling arguments."""
    parser.add_argument(
        "--enable_cudagraph", action="store_true", help="Enable CudaGraph acceleration"
    )
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable TorchCompile acceleration"
    )
    parser.add_argument(
        "--enable_fullgraph", action="store_true", help="Enable fullgraph for TorchCompile"
    )
    parser.add_argument(
        "--disable_autotune", action="store_true", help="Disable autotuning during warmup"
    )
    parser.add_argument(
        "--enable_layerwise_nvtx_marker", action="store_true", help="Enable layerwise NVTX markers"
    )
