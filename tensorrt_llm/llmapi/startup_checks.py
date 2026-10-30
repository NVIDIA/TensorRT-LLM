# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap, deterministic checks that fail before any weights are loaded.

Only certain lower bounds are checked here. Model-, kernel- or feature-specific
compatibility is left to the existing errors raised where the support is
implemented.
"""

import math
import os
from typing import Optional

import torch
from transformers import PretrainedConfig

from tensorrt_llm._errors import _DOCS_SITE, ResourceError, UnsupportedError
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs, _ParallelConfig
from tensorrt_llm.llmapi.utils import get_device_count
from tensorrt_llm.quantization.mode import QuantAlgo

# Lowest architecture marked as supported in the "Hardware Support Matrix" of
# docs/source/features/quantization.md. Formats missing from the matrix are not
# checked, and neither are KV cache formats: attention kernels dequantize the KV
# cache before the MMA, so its support depends on the kernel, not on a lower bound.
_MIN_SM_BY_QUANT_ALGO = {
    QuantAlgo.FP8: 89,
    QuantAlgo.FP8_BLOCK_SCALES: 90,
    QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN: 90,
    QuantAlgo.W4A8_AWQ: 89,
    QuantAlgo.NVFP4: 100,
    QuantAlgo.W4A8_MXFP4_MXFP8: 100,
}
# The opt-in Marlin NVFP4 GEMM runs NVFP4 linear layers on Ada and Hopper.
_MIN_SM_NVFP4_MARLIN = 89


def _visible_devices_env() -> str:
    return ", ".join(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    )


def _nvml_facts() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            driver = pynvml.nvmlSystemGetDriverVersion()
            cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:  # noqa: BLE001 - facts are best effort
        # nvmlInit fails without the driver libraries, e.g. in a container started without GPUs.
        return f"NVML unavailable ({type(e).__name__})"
    if isinstance(driver, bytes):
        driver = driver.decode()
    return (
        f"NVML sees {count} GPU(s), driver {driver} supports CUDA up to "
        f"{cuda // 1000}.{cuda % 1000 // 10}"
    )


def no_visible_gpu_error() -> ResourceError:
    """Build the error for "PyTorch sees no GPU", listing the observed facts and likely causes."""
    cuda = torch.version.cuda or "unknown"
    causes = (
        "Possible causes: (1) the container was started without GPUs (docker --gpus, or "
        "nvidia.com/gpu in Kubernetes); (2) CUDA_VISIBLE_DEVICES hides all GPUs; "
        f"(3) the NVIDIA driver is too old for CUDA {cuda}: upgrade it, or install the "
        "matching cuda-compat package."
    )
    return ResourceError(
        "No GPU is visible to PyTorch (torch.cuda.device_count() == 0). "
        f"{_visible_devices_env()}; {_nvml_facts()}; this build uses CUDA {cuda}.\n{causes}",
        reason="no_gpu",
        hint=causes,
    )


def check_visible_gpus(parallel_config: _ParallelConfig, all_ranks_local: bool) -> None:
    """Fail when this node shows no GPU, or fewer GPUs than its share of the world needs.

    Args:
        parallel_config: The LLM's parallel configuration.
        all_ranks_local: Whether every rank is spawned on this node, as the LLM's own
            MPI pool does. Otherwise (mpirun / Slurm launches) the world is assumed to
            span ``ceil(world_size / gpus_per_node)`` nodes.
    """
    visible = get_device_count()
    if visible == 0:
        raise no_visible_gpu_error()
    world_size = parallel_config.world_size
    if all_ranks_local:
        needed, placement = world_size, "on this node"
    else:
        nodes = math.ceil(world_size / parallel_config.gpus_per_node)
        needed, placement = parallel_config.world_size_per_node, f"over {nodes} node(s)"
    if visible >= needed:
        return
    raise ResourceError(
        f"Only {visible} GPUs are available, but {needed} are required on this node: "
        "tensor_parallel_size x pipeline_parallel_size x context_parallel_size = "
        f"{parallel_config.tp_size} x {parallel_config.pp_size} x "
        f"{parallel_config.cp_size} = {world_size} ranks {placement} "
        f"({_visible_devices_env()}).",
        hint=(
            "Make more GPUs visible (CUDA_VISIBLE_DEVICES, docker --gpus) "
            "or lower the parallel sizes."
        ),
    )


def check_quantization_compute_capability(llm_args: TorchLlmArgs) -> None:
    """Fail when GPU 0 is older than the lowest architecture supporting the quantization format.

    Must run after the checkpoint's quantization config has been merged into ``llm_args``.

    Args:
        llm_args: The resolved LLM arguments.
    """
    sm = get_sm_version()
    if sm <= 0:
        return
    algo = llm_args.quant_config.quant_algo
    min_sm = _MIN_SM_BY_QUANT_ALGO.get(algo)
    if algo == QuantAlgo.NVFP4 and "marlin" in llm_args.nvfp4_gemm_config.allowed_backends:
        min_sm = _MIN_SM_NVFP4_MARLIN
    if min_sm is None or sm >= min_sm:
        return
    message = (
        f"Quantization {algo} requires SM{min_sm} or newer, but GPU 0 "
        f"({torch.cuda.get_device_name(0)}) is SM{sm}."
    )
    if algo == QuantAlgo.NVFP4 and sm >= _MIN_SM_NVFP4_MARLIN:
        message += (
            " On Ada and Hopper GPUs, NVFP4 linear layers can use the opt-in Marlin "
            "backend: add 'marlin' to nvfp4_gemm_config.allowed_backends."
        )
    raise UnsupportedError(
        message,
        hint=(
            "Use a checkpoint quantized for this GPU, or a newer GPU. See the hardware "
            f"support matrix at {_DOCS_SITE}/features/quantization.html."
        ),
    )


def check_architecture_registered(
    llm_args: TorchLlmArgs, pretrained_config: Optional[PretrainedConfig]
) -> None:
    """Fail when the Hugging Face checkpoint's architecture has no registered implementation.

    Args:
        llm_args: The resolved LLM arguments.
        pretrained_config: The checkpoint's Hugging Face config, or None if it was not loaded.
    """
    if llm_args.checkpoint_format not in (None, "HF"):
        return
    if llm_args.mm_encoder_only or llm_args.encode_only:
        return
    architectures = getattr(pretrained_config, "architectures", None) or []
    if not architectures:
        return
    # The static index lists every built-in architecture without importing the model
    # zoo; MODEL_CLASS_MAPPING adds custom models registered in this process.
    from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    if any(arch in MODEL_CLASS_MAPPING or arch in MODEL_ARCH_TO_MODULE for arch in architectures):
        return
    raise UnsupportedError(
        f"Unknown model architecture: {architectures[0]}. TensorRT-LLM has no "
        f"implementation registered for the checkpoint's architectures {architectures}.",
        hint=(
            f"See the supported models at {_DOCS_SITE}/models/supported-models.html. A custom "
            "model must be registered with register_auto_model before the LLM is created."
        ),
    )
