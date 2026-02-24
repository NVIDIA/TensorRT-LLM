"""Shared utilities for visual generation example scripts."""

import argparse
from typing import Any, Dict, Optional

from tensorrt_llm.llmapi.visual_gen import VisualGenParams

LINEAR_TYPE_CHOICES = ["default", "trtllm-fp8-per-tensor", "trtllm-fp8-blockwise", "trtllm-nvfp4"]

_LINEAR_TYPE_TO_QUANT = {
    "trtllm-fp8-per-tensor": {"quant_algo": "FP8", "dynamic": True},
    "trtllm-fp8-blockwise": {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
    "trtllm-nvfp4": {"quant_algo": "NVFP4", "dynamic": True},
}


def add_common_args(parser: argparse.ArgumentParser, *, prompt_required: bool = True) -> None:
    """Add CLI arguments shared across all visual generation examples.

    Args:
        parser: The argument parser to add arguments to.
        prompt_required: Whether --prompt is required (False for scripts
            that support batch mode via --prompts_file).
    """
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace Hub model ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace Hub revision (branch, tag, or commit SHA)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=prompt_required,
        default=None,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt. Default is model-specific.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.png",
        help="Path to save the output image/video",
    )

    # Generation params (defaults are None; each script supplies a defaults map)
    parser.add_argument("--height", type=int, default=None, help="Output height")
    parser.add_argument("--width", type=int, default=None, help="Output width")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to generate")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale",
    )
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=None,
        help="Second-stage guidance scale for two-stage denoising",
    )
    parser.add_argument(
        "--boundary_ratio",
        type=float,
        default=None,
        help="Custom boundary ratio for two-stage denoising (default: auto-detect)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # TeaCache
    parser.add_argument(
        "--enable_teacache", action="store_true", help="Enable TeaCache acceleration"
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache similarity threshold (rel_l1_thresh)",
    )

    # Quantization
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=LINEAR_TYPE_CHOICES,
        help=(
            "Dynamic quantization mode for linear layers. "
            "Quantizes weights on-the-fly during loading from an unquantized checkpoint."
        ),
    )

    # Attention backend
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM"],
        help="Attention backend (VANILLA: PyTorch SDPA, TRTLLM: optimized kernels)",
    )

    # Parallelism
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size (1 or 2)",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses sequence parallel size within each CFG group",
    )

    # Compilation
    parser.add_argument(
        "--enable_cudagraph", action="store_true", help="Enable CudaGraph acceleration"
    )
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable TorchCompile acceleration"
    )
    parser.add_argument(
        "--torch_compile_models",
        type=str,
        nargs="+",
        default=[],
        help="Components to torch.compile (empty = auto detect)",
    )
    parser.add_argument(
        "--enable_fullgraph", action="store_true", help="Enable fullgraph for TorchCompile"
    )
    parser.add_argument(
        "--disable_autotune", action="store_true", help="Disable autotuning during warmup"
    )

    # Debug / profiling
    parser.add_argument(
        "--enable_layerwise_nvtx_marker", action="store_true", help="Enable layerwise NVTX markers"
    )


def build_diffusion_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the diffusion_config dict from parsed CLI arguments."""
    config: Dict[str, Any] = {
        "attention": {"backend": args.attention_backend},
        "teacache": {
            "enable_teacache": args.enable_teacache,
            "teacache_thresh": args.teacache_thresh,
        },
        "parallel": {
            "dit_cfg_size": args.cfg_size,
            "dit_ulysses_size": args.ulysses_size,
        },
        "compilation": {
            "enable_cuda_graph": args.enable_cudagraph,
            "enable_torch_compile": not args.disable_torch_compile,
            "torch_compile_models": args.torch_compile_models,
            "enable_fullgraph": args.enable_fullgraph,
            "enable_autotune": not args.disable_autotune,
        },
        "pipeline": {
            "enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker,
        },
    }

    if args.revision is not None:
        config["revision"] = args.revision

    quant_config = _LINEAR_TYPE_TO_QUANT.get(args.linear_type)
    if quant_config is not None:
        config["quant_config"] = quant_config

    return config


_GENERATION_PARAM_FIELDS = {
    "height": "height",
    "width": "width",
    "steps": "num_inference_steps",
    "guidance_scale": "guidance_scale",
    "seed": "seed",
    "num_frames": "num_frames",
    "guidance_scale_2": "guidance_scale_2",
    "boundary_ratio": "boundary_ratio",
}


def build_generation_params(
    args: argparse.Namespace,
    defaults: Optional[Dict[str, Any]] = None,
    **overrides,
) -> VisualGenParams:
    """Build VisualGenParams from parsed CLI arguments.

    Resolution order for each field (first non-None wins):
        explicit override  >  CLI arg  >  defaults map

    Args:
        args: Parsed CLI arguments.
        defaults: Model-specific defaults map (arg-name -> value).
        **overrides: Extra keyword arguments forwarded to VisualGenParams
            (e.g. ``input_reference`` for I2V).
    """
    defaults = defaults or {}
    kwargs: Dict[str, Any] = {}
    for arg_name, param_name in _GENERATION_PARAM_FIELDS.items():
        cli_value = getattr(args, arg_name, None)
        kwargs[param_name] = cli_value if cli_value is not None else defaults.get(arg_name)
    kwargs.update(overrides)
    return VisualGenParams(**kwargs)
