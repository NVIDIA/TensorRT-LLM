# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GEMM kernels of a model's Linear layers, built without a model forward.

The decoder warmup builds them as a side effect of its forwards: DeepGEMM compiles a
swap-AB kernel for every tile layout its heuristic picks for an FP8 block-scale weight
over the autotuner's token buckets, and FlashInfer's MXFP8 GEMM its CUTLASS module. Which
kernels follows from the weight shapes and the token counts the warmup runs, both known
once the model is constructed. They are built by running the same kernels once on dummy
operands with the shapes, dtypes and layouts of the loaded weights. Nothing is timed: the
warmup still runs its sweeps, and finds the kernels built.

Some models create FP8 GEMMs only while their weights load; ``_load_time_fp8_shapes`` lists
them from the model's own rules for those transforms.

Only builds that run with the GIL released are done next to the loader: DeepGEMM's nvcc
and FlashInfer's ninja run in subprocesses. A CuTe DSL or Triton compile holds the GIL for
most of its time, and the loading thread, which releases and re-acquires it around every
copy, would then wait a full switch interval each time. Those kernels stay with the warmup.
``TLLM_OVERLAP_PREWARM_LINEAR=0`` leaves all of them to the warmup.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import threading
from types import ModuleType
from typing import Any, List, Optional, Set, Tuple

import torch
from torch import nn

from tensorrt_llm.logger import logger

from .group import BuildGroup

_ENV = "TLLM_OVERLAP_PREWARM_LINEAR"
# nvcc runs with the GIL released, so DeepGEMM compiles in parallel threads.
_DEEP_GEMM_THREADS_ENV = "TLLM_OVERLAP_PREWARM_DEEP_GEMM_THREADS"
# Free device memory left to the loader next to the dummy operands.
_LOADER_RESERVE_BYTES = 8 << 30
_KIMI_LINEAR_MODULE = "tensorrt_llm._torch.models.modeling_kimi_linear"
# Kimi K3's weight-loading helpers: in this module where it exists, else in modeling_kimi_linear.
_KIMI_K3_LOADER_MODULE = "tensorrt_llm._torch.models.modeling_kimi_k3_loader"


@dataclasses.dataclass
class Plan:
    """The GEMMs to build and the token counts to build them for."""

    #: The autotuner warmup's token counts, and per FP8 block-scale GEMM on DeepGEMM
    #: operands that ``fp8_swap_ab_gemm`` sweeps there: (N, K, the sweep's token counts).
    tuning_m: Tuple[int, ...]
    fp8: List[Tuple[int, int, Tuple[int, ...]]]
    flashinfer_mxfp8: bool

    def summary(self) -> str:
        shapes = " ".join(f"{n}x{k}:{len(sweep)}" for n, k, sweep in self.fp8)
        return (
            f"autotuner token counts {self.tuning_m}, {len(self.fp8)} DeepGEMM shapes "
            f"[N x K: token counts] [{shapes}], FlashInfer MXFP8 {self.flashinfer_mxfp8}"
        )


def _tuning_token_counts(
    llm_args: Any, mapping: Any, max_seq_len: Optional[int]
) -> Tuple[int, ...]:
    """Tokens of the autotuner warmup's forwards, as ``DecoderWarmup._run_autotuner_warmup``
    computes them: one batch of generation requests on a Helix CP generation worker;
    otherwise max_num_tokens capped by the sequence budget and, without pipeline parallelism
    or guided decoding, one generation request. Known only after the weights load, and so
    not applied here: the KV cache's cap on the context tokens, a max_seq_len the engine
    infers from the model, and a context-only KV cache, which drops the generation request.
    The kernels planned beyond them are built for nothing.
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
    if max_seq_len:
        tokens = min(tokens, max_batch_size * (max_seq_len - 1))
    if mapping.has_pp() or getattr(llm_args, "guided_decoding_backend", None) is not None:
        return (tokens,)
    return tokens, per_request


def _compiles_all_batches(model: nn.Module, llm_args: Any) -> bool:
    """Whether torch.compile covers every batch rather than prefill only, as
    ``apply_torch_compile`` decides."""
    config = getattr(llm_args, "torch_compile_config", None)
    if config is None:
        return False
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
    from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

    prefill_only = (
        isinstance(model, DecoderModelForCausalLM)
        and llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
        and (config.compile_only_piecewise_graphs or not model.use_fx_for_pcg_fallback)
    )
    return not prefill_only


def _uses_flashinfer_mxfp8(model: nn.Module, llm_args: Any, methods: List[Any]) -> bool:
    """Whether the warmup tunes FlashInfer's MXFP8 GEMM (``_run_autotuner_warmup``)."""
    from tensorrt_llm._torch.modules.linear import mxfp8_gemm_backend_override

    if not any(method.use_cutlass for method in methods):
        return False
    override = mxfp8_gemm_backend_override()
    # With every batch compiled, MXFP8 GEMMs dispatch natively unless FlashInfer is forced.
    compile_all = _compiles_all_batches(model, llm_args)
    if override is not None:
        return override == "flashinfer" or (override == "auto" and not compile_all)
    if compile_all or getattr(llm_args, "cuda_graph_config", None) is None:
        return False
    return any(
        getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
        for module in model.modules()
    )


def _kimi_linear_fp8_shapes(
    kimi: ModuleType, loader: ModuleType, model: nn.Module
) -> List[Tuple[int, int]]:
    """``(out, in)`` of the FP8 block-scale GEMMs on DeepGEMM operands that
    ``KimiLinearForCausalLM._finalize_weight_load`` creates: the attention projections it
    rewraps, the KDA ``qkvg_proj`` that ``finalize_decode_weights`` fuses, and the BF16
    shared/latent MLP projections it quantizes. ``loader`` is the module with the model's
    weight-loading helpers."""
    if kimi._rubin_mxfp8_is_available():
        return []
    shapes = []
    for layer in model.model.layers:
        if not loader._has_weights(layer):
            continue
        is_kda = getattr(layer, "is_kda", False)
        attention = layer.linear_attn if is_kda else layer.self_attn.mixer
        shapes += [
            tuple(linear.weight.shape)
            for linear in attention.children()
            if isinstance(linear, kimi.TrtllmLinear) and linear.has_fp8_block_scales
        ]
        if is_kda and (fused := attention._fp8_qkvg_modules()) is not None:
            shapes.append((sum(module.out_features for module in fused), fused[0].in_features))
    if model._fp8_weight_read_moe_mlp:
        shapes += [
            tuple(getattr(parent, attr).weight.shape)
            for parent, attr, linear_types in loader._moe_mlp_fp8_weight_read_targets(
                model.model, include_fused_gate_up=model.model_config.mapping.enable_attention_dp
            )
            if isinstance(getattr(parent, attr, None), linear_types)
        ]
    return shapes


def _load_time_fp8_shapes(model: nn.Module) -> Set[Tuple[int, int]]:
    """FP8 block-scale GEMMs on DeepGEMM operands that a model creates while its weights
    load, which the walk over the constructed model cannot see. Looked up on every
    submodule: a multimodal model wraps its text model."""
    shapes = set()
    # Models are imported on demand: an unimported model module has no instance here.
    kimi = sys.modules.get(_KIMI_LINEAR_MODULE)
    if kimi is None:
        return shapes
    loader = sys.modules.get(_KIMI_K3_LOADER_MODULE, kimi)
    for module in model.modules():
        if isinstance(module, kimi.KimiLinearForCausalLM):
            try:
                shapes.update(_kimi_linear_fp8_shapes(kimi, loader, module))
            except Exception as e:  # noqa: BLE001 - the warmup builds those kernels
                logger.warning(
                    f"Overlapped Linear GEMM builds: no load-time shapes of "
                    f"{type(module).__name__}. {type(e).__name__}: {e}"
                )
    return shapes


def collect(model: nn.Module, llm_args: Any, mapping: Any) -> Plan:
    """The GEMMs of the constructed (possibly meta) ``model`` that the warmup builds."""
    from tensorrt_llm._torch.modules.linear import FP8BlockScalesLinearMethod, MXFP8LinearMethod
    from tensorrt_llm._utils import is_sm_100f

    if not getattr(llm_args, "enable_autotuner", True):
        # The autotuner warmup then runs no tuning forward.
        return Plan(tuning_m=(), fp8=[], flashinfer_mxfp8=False)

    sm100f = is_sm_100f()
    fp8 = _load_time_fp8_shapes(model) if sm100f else set()
    mxfp8_methods = []
    for module in model.modules():
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
    tuning_m = _tuning_token_counts(llm_args, mapping, getattr(llm_args, "max_seq_len", None))
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
    from tensorrt_llm._torch.modules.linear import Fp8BlockScaleLowMGemmRange
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() == 107:
        return 0
    try:
        return Fp8BlockScaleLowMGemmRange.for_weight(n, k).max_m
    except Exception as e:  # noqa: BLE001 - sweep every count: more kernels, none missing
        logger.warning(
            f"Overlapped Linear GEMM builds: no low-M range for {n}x{k}. {type(e).__name__}: {e}"
        )
        return 0


def deep_gemm_sweep(shape: Tuple[int, int], tuning_m: Tuple[int, ...]) -> Tuple[int, ...]:
    """Token counts of ``fp8_swap_ab_gemm``'s sweep for ``shape`` in the autotuner warmup:
    of each forward whose rows the low-M GEMM does not serve, its buckets and its count
    (``DynamicTensorSpec``); none when the low-M GEMM serves them all."""
    from tensorrt_llm._torch.utils import deep_gemm_jit_warmup_buckets

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


def _deep_gemm_threads(plan: Plan) -> int:
    """Threads for the DeepGEMM builds: the requested count, as far as their dummy operands
    fit next to the loader in free device memory (0: none fit)."""
    threads = max(1, int(os.environ.get(_DEEP_GEMM_THREADS_ENV, "4")))
    largest = max((deep_gemm_footprint(item) for item in plan.fp8), default=0)
    if largest == 0:
        return threads
    free_bytes, _ = torch.cuda.mem_get_info()
    return max(0, min(threads, (free_bytes - _LOADER_RESERVE_BYTES) // largest))


def build_deep_gemm(
    item: Tuple[int, int, Tuple[int, ...]],
    device: torch.device,
    stop: threading.Event,
    free_samples: List[int],
) -> int:
    """Run ``fp8_swap_ab_gemm``'s runner at every token count of a plan's sweep
    ``(N, K, token counts)``, stopping between counts once ``stop`` is set. Skips the sweep
    when its dummy operands would leave less than the loader's reserve of free memory,
    which the loader's own allocations shrink as it goes. Returns the GEMM launches."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import fp8SwapABGemmRunner

    n, k, sweep = item
    free_bytes, _ = torch.cuda.mem_get_info()
    free_samples.append(free_bytes)
    if free_bytes - deep_gemm_footprint(item) < _LOADER_RESERVE_BYTES:
        logger.info(
            f"[warmup] overlap: DeepGEMM {n}x{k} left to the warmup, "
            f"{free_bytes / (1 << 30):.1f} GiB free"
        )
        return 0
    weight, scale = _dummy_fp8_weight(n, k, device)
    x = torch.zeros((max(sweep), k), dtype=torch.bfloat16, device=device)
    # The runner the op builds (fp8_block_scale_ue8m0_gemm: BF16 output, packed UE8M0
    # scale) with the CUDA quantizer; the GEMM kernel does not depend on the quantizer.
    runner = fp8SwapABGemmRunner(torch.bfloat16, True, quant_tactic=0)
    launches = 0
    for m in sweep:
        if stop.is_set():
            break
        runner.forward([x[:m], weight, scale])
        launches += 1
    return launches


def build_flashinfer_mxfp8(_item: Any = None, _stop: Optional[threading.Event] = None) -> int:
    """Build and load FlashInfer's CUTLASS MXFP8 GEMM module (``mm_mxfp8``)."""
    from flashinfer.gemm.gemm_base import get_gemm_sm100_module_cutlass_mxfp8

    get_gemm_sm100_module_cutlass_mxfp8()
    return 1


def build_groups(
    model: nn.Module, model_config: Any, llm_args: Any, mapping: Any
) -> List[BuildGroup]:
    """The builds of the GEMM kernels of ``model``'s Linear layers."""
    if os.environ.get(_ENV, "1") == "0":
        return []
    plan = collect(model, llm_args, mapping)
    device = torch.device("cuda", torch.cuda.current_device())
    deep_gemm_threads = _deep_gemm_threads(plan)
    if plan.fp8 and deep_gemm_threads == 0:
        logger.info("[warmup] overlap: no free memory for DeepGEMM's operands")
    logger.info(f"[warmup] overlap plan: {plan.summary()}")
    # Free device memory before each DeepGEMM sweep, for the group's final log.
    free_samples: List[int] = []
    return [
        BuildGroup(
            "deep_gemm",
            plan.fp8 if deep_gemm_threads else [],
            lambda item, stop: build_deep_gemm(item, device, stop, free_samples),
            deep_gemm_threads,
            summary=lambda: f"{min(free_samples) / (1 << 30):.1f} GiB free at least"
            if free_samples
            else "no sweep started",
        ),
        BuildGroup(
            "flashinfer_mxfp8",
            ["module"] if plan.flashinfer_mxfp8 else [],
            build_flashinfer_mxfp8,
            1,
        ),
    ]
