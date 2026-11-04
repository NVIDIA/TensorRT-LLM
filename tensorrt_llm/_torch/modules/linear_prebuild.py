# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the GEMM kernels of a model's Linear layers without a model forward.

The decoder warmup builds them as a side effect of its forwards: DeepGEMM compiles a
swap-AB kernel for every tile layout its heuristic picks for an FP8 block-scale weight
over the autotuner's token buckets, and FlashInfer's MXFP8 GEMM its CUTLASS module. Which
kernels follows from the weight shapes and the token counts the warmup runs, both known
once the model is constructed, so ``warmup_overlap`` builds them while the weights load,
by running the same kernels once on dummy operands with the shapes, dtypes and layouts of
the loaded weights. Nothing is timed: the warmup still runs its sweeps, and finds the
kernels built.

Only builds that run with the GIL released are done next to the loader: DeepGEMM's nvcc
and FlashInfer's ninja run in subprocesses. A CuTe DSL or Triton compile holds the GIL for
most of its time, and the loading thread, which releases and re-acquires it around every
copy, would then wait a full switch interval each time. Those kernels, like attention and
fused-MoE kernels, stay with the warmup.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, List, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger

#: A model that builds FP8 block-scale GEMMs while it loads (Kimi K3 fuses KDA projections
#: and quantizes BF16 MLP projections) returns their ``(out, in)`` from this method.
FP8_BLOCK_SCALE_SHAPES_HOOK = "linear_prebuild_fp8_block_scale_shapes"


@dataclasses.dataclass
class Plan:
    """The GEMMs to build and the token counts to build them for."""

    #: The autotuner warmup's token counts, and per FP8 block-scale GEMM on DeepGEMM
    #: operands that ``fp8_swap_ab_gemm`` sweeps there: (N, K, the sweep's token counts).
    tuning_m: Tuple[int, ...]
    fp8: List[Tuple[int, int, Tuple[int, ...]]]
    flashinfer_mxfp8: bool

    def empty(self) -> bool:
        return not (self.fp8 or self.flashinfer_mxfp8)

    def summary(self) -> str:
        shapes = " ".join(f"{n}x{k}:{len(sweep)}" for n, k, sweep in self.fp8)
        return (
            f"autotuner token counts {self.tuning_m}, {len(self.fp8)} DeepGEMM shapes "
            f"[N x K: token counts] [{shapes}], FlashInfer MXFP8 {self.flashinfer_mxfp8}"
        )


def _tuning_token_counts(llm_args: Any, mapping: Any) -> Tuple[int, ...]:
    """Tokens of the autotuner warmup's forwards, as ``DecoderWarmup._run_autotuner_warmup``
    computes them: one batch of generation requests on a Helix CP generation worker;
    otherwise max_num_tokens capped by the sequence budget (the KV cache may allow fewer,
    then some kernels are built that the warmup would not need) and one generation request.
    """
    max_num_tokens = llm_args.max_num_tokens or 8192
    max_batch_size = llm_args.max_batch_size or 2048
    spec = getattr(llm_args, "speculative_config", None)
    draft = 0
    if spec is not None:
        draft = (
            getattr(spec, "max_total_draft_tokens", None)
            or getattr(spec, "max_draft_len", None)
            or 0
        )
    per_request = 1 + draft
    if mapping.has_cp_helix():
        return (min(max_batch_size, max_num_tokens // per_request) * per_request,)
    tokens = max_num_tokens
    if llm_args.max_seq_len:
        tokens = min(tokens, max_batch_size * (llm_args.max_seq_len - 1))
    return tokens, per_request


def _uses_flashinfer_mxfp8(model: nn.Module, llm_args: Any, methods: List[Any]) -> bool:
    """Whether the warmup tunes FlashInfer's MXFP8 GEMM (``_run_autotuner_warmup``)."""
    from .linear import mxfp8_gemm_backend_override

    if not any(method.use_cutlass for method in methods):
        return False
    override = mxfp8_gemm_backend_override()
    if override is not None:
        return override in ("flashinfer", "auto")
    if getattr(llm_args, "cuda_graph_config", None) is None:
        return False
    return any(
        getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
        for module in model.modules()
    )


def collect(model: nn.Module, llm_args: Any, mapping: Any) -> Plan:
    """The GEMMs of the constructed (possibly meta) ``model`` that the warmup builds."""
    from tensorrt_llm._utils import is_sm_100f

    from .linear import FP8BlockScalesLinearMethod, MXFP8LinearMethod

    sm100f = is_sm_100f()
    fp8 = set()
    mxfp8_methods = []
    for module in model.modules():
        # On the text model, which a multimodal model wraps.
        hook = getattr(module, FP8_BLOCK_SCALE_SHAPES_HOOK, None)
        if callable(hook) and sm100f:
            try:
                fp8.update(tuple(shape) for shape in hook())
            except Exception as e:  # noqa: BLE001 - the warmup builds those kernels
                logger.warning(
                    f"Linear prebuild: {FP8_BLOCK_SCALE_SHAPES_HOOK} failed. {type(e).__name__}: {e}"
                )
        method = getattr(module, "quant_method", None)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            continue
        if isinstance(method, FP8BlockScalesLinearMethod):
            # FP8BlockScalesLinearMethod.apply: DeepGEMM operands unless opted out.
            if sm100f and not (
                getattr(module, "use_cute_dsl_blockscaling_mm", False)
                or getattr(module, "disable_deep_gemm", False)
            ):
                fp8.add(tuple(weight.shape))
        elif isinstance(method, MXFP8LinearMethod):
            mxfp8_methods.append(method)
    tuning_m = _tuning_token_counts(llm_args, mapping)
    # On this thread: the low-M range comes from CuTe DSL host code, which the
    # background threads must not run.
    sweeps = [(n, k, deep_gemm_sweep((n, k), tuning_m)) for n, k in fp8]
    return Plan(
        tuning_m=tuning_m,
        fp8=sorted((s for s in sweeps if s[2]), key=lambda s: -s[0] * s[1]),
        flashinfer_mxfp8=bool(mxfp8_methods)
        and _uses_flashinfer_mxfp8(model, llm_args, mxfp8_methods),
    )


def _dummy_fp8_weight(n: int, k: int, device: torch.device):
    """A zero FP8 weight and its scale in the layout ``transform_weights`` gives it."""
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.ones((math.ceil(n / 128), math.ceil(k / 128)), dtype=torch.float32, device=device)
    scale = transform_sf_into_required_layout(scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False)
    return weight, scale


def _low_m_max(n: int, k: int) -> int:
    """Rows up to which the forward runs the low-M GEMM instead of ``fp8_swap_ab_gemm``
    (``Fp8BlockScaleLowMGemmRange``, set by ``transform_weights`` except on SM107)."""
    from tensorrt_llm._utils import get_sm_version

    from .linear import Fp8BlockScaleLowMGemmRange

    if get_sm_version() == 107:
        return 0
    try:
        return Fp8BlockScaleLowMGemmRange.for_weight(n, k).max_m
    except Exception as e:  # noqa: BLE001 - sweep every count: more kernels, none missing
        logger.warning(f"Linear prebuild: no low-M range for {n}x{k}. {type(e).__name__}: {e}")
        return 0


def deep_gemm_sweep(shape: Tuple[int, int], tuning_m: Tuple[int, ...]) -> Tuple[int, ...]:
    """Token counts of ``fp8_swap_ab_gemm``'s sweep for ``shape`` in the autotuner warmup:
    of each forward whose rows the low-M GEMM does not serve, its buckets and its count
    (``DynamicTensorSpec``); none when the low-M GEMM serves them all."""
    from ..utils import deep_gemm_jit_warmup_buckets

    low_m_max = _low_m_max(*shape)
    sweep = set()
    for m in tuning_m:
        if m > low_m_max:
            sweep.add(m)
            sweep.update(deep_gemm_jit_warmup_buckets(m))
    return tuple(sorted(sweep))


def deep_gemm_footprint(item: Tuple[int, int, Tuple[int, ...]]) -> int:
    """Bytes of ``build_deep_gemm``'s dummy operands at its largest token count: the FP8
    weight and its scale, the BF16 activation, its FP8 copy and scale, and the output."""
    n, k, sweep = item
    m, scale_cols = max(sweep), math.ceil(k / 512)
    return n * k + 4 * n * scale_cols + 3 * m * k + 4 * m * scale_cols + 2 * m * n


def build_deep_gemm(item: Tuple[int, int, Tuple[int, ...]], device: torch.device) -> int:
    """Run ``fp8_swap_ab_gemm``'s runner at every token count of a plan's sweep
    ``(N, K, token counts)``. Returns the GEMM launches."""
    from ..custom_ops.torch_custom_ops import fp8SwapABGemmRunner

    n, k, sweep = item
    weight, scale = _dummy_fp8_weight(n, k, device)
    x = torch.zeros((max(sweep), k), dtype=torch.bfloat16, device=device)
    # The runner the op builds (fp8_block_scale_ue8m0_gemm: BF16 output, packed UE8M0
    # scale) with the CUDA quantizer; the GEMM kernel does not depend on the quantizer.
    runner = fp8SwapABGemmRunner(torch.bfloat16, True, quant_tactic=0)
    for m in sweep:
        runner.forward([x[:m], weight, scale])
    return len(sweep)


def build_flashinfer_mxfp8() -> None:
    """Build and load FlashInfer's CUTLASS MXFP8 GEMM module (``mm_mxfp8``)."""
    from flashinfer.gemm.gemm_base import get_gemm_sm100_module_cutlass_mxfp8

    get_gemm_sm100_module_cutlass_mxfp8()
