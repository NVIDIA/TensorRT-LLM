# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's BF16 split-K GEMM at a fixed tactic for the TP8 drafter's down and o_proj projections (7 draft
tokens, 1..8 rows).

With the drafter's fused all-reduce + norm (DRAFT_AR_NORM_FUSED) both row-parallel projections defer their all-reduce:
their Linear.forward is a plain GEMM whose output the fused boundary reduces. On those calls the scoped forward runs
FlashInfer's low-M split-K kernel (``flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk``) at the pinned tactic and
returns an ordinary tensor to the same consumer. Every other call (other rows, an all-reduce or fusion request, LoRA,
grad mode, a replaced weight) runs the original complete Linear.forward. The provider is admitted by behavior, not by
version: the module must expose the split-K entry points and accept the pinned tactics at every admitted shape.
Kill switches TRTLLM_M31_DISABLE_DRAFTER_DOWN_SPLITK=1 / TRTLLM_M31_DISABLE_DRAFTER_OPROJ_SPLITK=1 restore the stock
BF16 GEMM of the down / o_proj projection.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_FI_ENTRY_POINTS = (
    "SplitKTactic",
    "run_splitk_dense",
    "validate_tactic",
    "_get_compiled_splitk_kernel",
)
TACTIC = (64, 8, 2, 12)
OPROJ_TACTIC = (64, 8, 1, 12)
ROWS = range(1, 9)
DOWN_SHAPE = (6144, 1536)
OPROJ_SHAPE = (6144, 1024)


def _module_admitted(
    linear, weight, bf16, row_mode, method_type, allreduce_type, shape=DOWN_SHAPE
) -> bool:
    """Metadata-only admission; no device reads or synchronization."""
    mapping = linear.mapping
    method = linear.quant_method
    ar = linear.all_reduce
    return (
        linear.weight is weight
        and weight.is_cuda
        and weight.dtype == bf16
        and tuple(weight.shape) == shape
        and weight.is_contiguous()
        and not weight.requires_grad
        and linear.bias is None
        and linear.tp_mode == row_mode
        and linear.reduce_output
        and not linear.partition_plan.enabled
        and not linear.use_custom_cublas_mm
        and not linear.use_cute_dsl_bf16_gemm
        and mapping.tp_size == 8
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and not mapping.enable_attention_dp
        and list(mapping.tp_group) == list(range(8))
        and type(method) is method_type
        and getattr(method.apply, "__func__", None) is method_type.apply
        and method.supports_nccl_symmetric_memory_window_output
        and type(ar) is allreduce_type
        and getattr(ar.forward, "__func__", None) is allreduce_type.forward
        and not ar._disable_mpi
        and ar.uses_nccl_symmetric_memory_window()
    )


def _wrap_forward(
    linear, provider, tactic, torch, low_m, types, layer, log_taken, shape=DOWN_SHAPE
):
    original = linear.forward
    weight = linear.weight
    bf16, row_mode, method_type, allreduce_type, fusion_none = types
    out_features, in_features = shape
    logged = False

    def forward(input, *, all_reduce_params=None, lora_params=None, layer_idx=None):
        nonlocal logged
        if (
            not low_m.LOW_M_GEMM_ACTIVE
            and not torch.is_grad_enabled()
            and isinstance(input, torch.Tensor)
            and input.dtype == bf16
            and input.device == weight.device
            and not input.requires_grad
            and input.ndim == 2
            and input.shape[0] in ROWS
            and input.shape[1] == in_features
            and input.is_contiguous()
            and lora_params is None
            and all_reduce_params is not None
            and all_reduce_params.enable_allreduce is False
            and all_reduce_params.fusion_op == fusion_none
            and _module_admitted(linear, weight, *types[:4], shape)
        ):
            output = torch.empty((input.shape[0], out_features), dtype=bf16, device=input.device)
            # The original AllReduce.forward is an identity for this explicit
            # no-allreduce call. The existing deferred consumer still runs.
            provider.run_splitk_dense(input, weight.t(), None, output, True, tactic)
            if not logged:
                log_taken(layer, input.shape[0])
                logged = True
            return output
        return original(
            input,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            layer_idx=layer_idx,
        )

    return forward


def _load_provider():
    """FlashInfer's split-K module when it exposes the entry points this route calls, else None (nothing launched)."""
    try:
        module = importlib.import_module("flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk")
    except ImportError:
        return None
    if not all(callable(getattr(module, name, None)) for name in _FI_ENTRY_POINTS):
        return None
    return module


def _tactic_admitted(provider, tactic_args, shape) -> bool:
    """Whether the provider accepts the pinned tactic at every admitted row count of ``shape`` on this device."""
    out_features, in_features = shape
    try:
        tactic = provider.SplitKTactic(*tactic_args)
        for rows in ROWS:
            provider.validate_tactic(tactic, rows, out_features, in_features)
    except (TypeError, ValueError):
        return False
    return True


def _prepare_aot(provider) -> None:
    """Use the existing object export/load path for the pinned FI compiler."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot

    if getattr(provider, "_m31_down_splitk_aot", False):
        return
    sources = (str(Path(provider.__file__).resolve()), str(Path(__file__).resolve()))
    provider._get_compiled_splitk_kernel = cute_aot.cached_compile(
        "drafter_down_splitk.flashinfer_splitk", lambda: sources
    )(provider._get_compiled_splitk_kernel)
    provider._m31_down_splitk_aot = True


def _route(draft_model, enabled, name, select, shape, tactic_args, attr) -> int:
    """Prewarm then install the five scoped forwards, outside CUDA capture.

    The caller latches the target kill switch at import/prebuild. A killed
    process never imports or prewarms FI through this route. Kernel/compile
    errors propagate; they are never retried through stock after a launch.
    """
    if not enabled:
        return 0
    import torch

    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.modules import low_m_gemm
    from tensorrt_llm._torch.modules.linear import (
        Linear,
        TensorParallelMode,
        UnquantizedLinearMethod,
    )
    from tensorrt_llm.functional import AllReduceFusionOp
    from tensorrt_llm.logger import logger

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"{name} must be prepared outside capture")
    layers = list(getattr(getattr(draft_model, "model", None), "layers", None) or ())
    if len(layers) != 5 or low_m_gemm.LOW_M_GEMM_ACTIVE:
        return 0
    linears = [select(layer) for layer in layers]
    if all(getattr(linear, attr, None) is not None for linear in linears):
        return len(linears)
    types = (
        torch.bfloat16,
        TensorParallelMode.ROW,
        UnquantizedLinearMethod,
        AllReduce,
        AllReduceFusionOp.NONE,
    )
    if any(
        type(linear) is not Linear
        or getattr(linear.forward, "__func__", None) is not Linear.forward
        or not _module_admitted(linear, linear.weight, *types[:4], shape)
        or torch.cuda.get_device_capability(linear.weight.device) != (10, 7)
        for linear in linears
    ):
        return 0
    provider = _load_provider()
    if provider is None or not _tactic_admitted(provider, tactic_args, shape):
        logger.info(
            f"M3.1 {name}: FlashInfer's split-K GEMM does not take the pinned tactic; stock forwards kept"
        )
        return 0
    _prepare_aot(provider)
    tactic = provider.SplitKTactic(*tactic_args)
    out_features, in_features = shape
    # Prewarm every admitted layout before installing any forward. These
    # ordinary temporary outputs never enter an all-reduce or serving state.
    with torch.inference_mode():
        for linear in linears:
            for rows in ROWS:
                x = torch.zeros(
                    (rows, in_features), dtype=torch.bfloat16, device=linear.weight.device
                )
                out = torch.empty((rows, out_features), dtype=torch.bfloat16, device=x.device)
                provider.run_splitk_dense(x, linear.weight.t(), None, out, True, tactic)
        torch.cuda.synchronize(linears[0].weight.device)

    def log_taken(layer, rows):
        logger.info(
            f"M3.1 {name} taken layer={layer} rows={rows} provider=flashinfer_splitk "
            f"tactic={','.join(map(str, tactic_args))} output=ordinary"
        )

    for layer, linear in enumerate(linears):
        original = linear.forward
        linear.forward = _wrap_forward(
            linear, provider, tactic, torch, low_m_gemm, types, layer, log_taken, shape
        )
        setattr(linear, attr, (original, linear.weight))
    return len(linears)


def route_down_splitk(draft_model, *, enabled: bool = True) -> int:
    return _route(
        draft_model,
        enabled,
        "DRAFTER_DOWN_SPLITK",
        lambda layer: layer.mlp.down_proj,
        DOWN_SHAPE,
        TACTIC,
        "_m31_down_splitk",
    )


def route_oproj_splitk(draft_model, *, enabled: bool = True) -> int:
    return _route(
        draft_model,
        enabled,
        "DRAFTER_OPROJ_SPLITK",
        lambda layer: layer.self_attn.o_proj,
        OPROJ_SHAPE,
        OPROJ_TACTIC,
        "_m31_oproj_splitk",
    )
