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


def add_cache_args(
    parser,
    *,
    backends=("teacache", "cache_dit"),
    teacache_thresh_default=0.2,
    teacache_thresh_help=None,
    cache_dit_residual_threshold_default=None,
):
    """Add cache-acceleration args; expose only the backends the model supports.

    Args:
        backends: Subset of ("teacache", "cache_dit") to expose on the CLI. Models that
            support both (Wan T2V/I2V, FLUX) take the default; models that only support
            one (e.g. LTX2: ("cache_dit",)) pass that subset. Multiple backends become a
            mutually-exclusive group; a single backend is added as a plain flag.
        teacache_thresh_default: Default for --teacache_thresh. Pass None for models where
            the threshold is auto-detected per variant (e.g. FLUX.1 vs FLUX.2). Ignored if
            "teacache" is not in `backends`.
        teacache_thresh_help: Custom help string for --teacache_thresh. Defaults to a
            generic description when None. Ignored if "teacache" is not in `backends`.
        cache_dit_residual_threshold_default: Forwarded to add_cache_dit_args as its
            `residual_threshold_default`. Ignored if "cache_dit" is not in `backends`.
    """
    if not backends:
        raise ValueError("add_cache_args: `backends` must contain at least one backend")

    enable_target = parser.add_mutually_exclusive_group() if len(backends) > 1 else parser
    if "teacache" in backends:
        enable_target.add_argument(
            "--enable_teacache", action="store_true", help="Enable TeaCache acceleration"
        )
    if "cache_dit" in backends:
        enable_target.add_argument(
            "--enable_cache_dit",
            action="store_true",
            help=(
                "Enable Cache-DiT per-block acceleration (requires the cache_dit package; "
                "see https://github.com/vipshop/cache-dit)."
                + (" Incompatible with --enable_teacache." if "teacache" in backends else "")
            ),
        )
    if "teacache" in backends:
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
    if "cache_dit" in backends:
        add_cache_dit_args(parser, residual_threshold_default=cache_dit_residual_threshold_default)


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
    Tolerates callers that exposed only a subset of backends via add_cache_args(backends=...);
    flags from an unexposed backend are treated as off.
    """
    if getattr(args, "enable_cache_dit", False):
        return {"cache": build_cache_dit_config(args)}
    if getattr(args, "enable_teacache", False):
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


def add_attention_backend_args(
    parser,
    *,
    backends=("VANILLA", "TRTLLM", "FA4"),
    expose_sage=True,
):
    """Add --attention_backend (and optionally --enable_sage_attention).

    Args:
        backends: Allowed values for --attention_backend. Models that don't support a
            particular backend (e.g. LTX2 doesn't yet support FA4) should pass a narrower
            subset so it isn't surfaced on the CLI. First entry is used as the default.
        expose_sage: When True, also expose --enable_sage_attention. Models that don't
            wire SageAttention through to their attention config (e.g. LTX2) should pass
            False so the flag isn't silently accepted-and-ignored.
    """
    if not backends:
        raise ValueError("add_attention_backend_args: `backends` must be non-empty")
    backend_descriptions = {
        "VANILLA": "VANILLA: PyTorch SDPA",
        "TRTLLM": "TRTLLM: optimized kernels",
        "FA4": "FA4: Flash Attention 4",
    }
    help_choices = ", ".join(backend_descriptions.get(b, b) for b in backends)
    parser.add_argument(
        "--attention_backend",
        type=str,
        default=backends[0],
        choices=list(backends),
        help=(
            f"Attention backend ({help_choices}). "
            "Note: TRTLLM falls back to VANILLA for cross-attention."
        ),
    )
    if expose_sage:
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


def add_parallelism_args(parser, *, expose_cfg_size=False, expose_parallel_vae=False):
    """Add diffusion parallelism args.

    Ulysses + Attention2D (row/col) are always added; every visual-gen example uses them.
    --cfg_size and --disable_parallel_vae are gated by kwargs because not every model
    exposes them (e.g. FLUX has no CFG parallel knob; only Wan T2V/I2V drive parallel VAE).

    Args:
        expose_cfg_size: Add --cfg_size (choices [1, 2]). Wan T2V/I2V and LTX2 use this.
        expose_parallel_vae: Add --disable_parallel_vae. Only Wan T2V/I2V use this today.
    """
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses (head-sharding) parallel size within each CFG group. "
        "Cannot be combined with --attn2d_row_size / --attn2d_col_size (not yet implemented).",
    )
    parser.add_argument(
        "--attn2d_row_size",
        type=int,
        default=1,
        help="Attention2D row mesh size (Q all-gather dimension). "
        "Can be set independently of --attn2d_col_size; asymmetric meshes (e.g. 1x4 or 4x1) are valid. "
        "Total context parallelism degree = attn2d_row_size * attn2d_col_size. "
        "Cannot be combined with --ulysses_size (not yet implemented).",
    )
    parser.add_argument(
        "--attn2d_col_size",
        type=int,
        default=1,
        help="Attention2D column mesh size (K/V all-gather dimension). "
        "Can be set independently of --attn2d_row_size; asymmetric meshes (e.g. 1x4 or 4x1) are valid. "
        "Cannot be combined with --ulysses_size (not yet implemented).",
    )
    if expose_cfg_size:
        parser.add_argument(
            "--cfg_size",
            type=int,
            default=1,
            choices=[1, 2],
            help="CFG parallel size (1 or 2). Set to 2 for CFG Parallelism.",
        )
    if expose_parallel_vae:
        parser.add_argument(
            "--disable_parallel_vae", action="store_true", help="Disable parallel VAE"
        )


def validate_parallelism_args(args) -> None:
    """Raise if --ulysses_size and --attn2d_row/col_size are both >1 (not yet implemented)."""
    attn2d_size = args.attn2d_row_size * args.attn2d_col_size
    if attn2d_size > 1 and args.ulysses_size > 1:
        raise ValueError(
            "Combining --ulysses_size with --attn2d_row_size/--attn2d_col_size is not yet implemented."
        )


def format_parallelism_str(args) -> str:
    """Format the parallelism mode for logging (e.g. 'Ulysses(size=2)', 'Attention2D(...)', 'None')."""
    attn2d_size = args.attn2d_row_size * args.attn2d_col_size
    if args.ulysses_size > 1:
        return f"Ulysses(size={args.ulysses_size})"
    if attn2d_size > 1:
        return (
            f"Attention2D(row={args.attn2d_row_size}, col={args.attn2d_col_size}, "
            f"total={attn2d_size})"
        )
    return "None"
