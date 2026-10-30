# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``prefill_cutedsl_experts``: the NVFP4 layers' large expert calls on the Rubin CuTe DSL GEMMs (sm107).

Installed on Rubin (sm107) at tp 4 only (``serves``); every other GPU and tp size keeps
the TRTLLM-gen expert calls. Every NVFP4 MoE layer whose leaf is
the native TRTLLM-gen one on this rank's contiguous expert block
(``ring_buddy.eligible``) gets a wrapper around its expert forward: an
unfinalized call (``do_finalize=False``) of ``MIN_ROWS..MAX_ROWS`` NVFP4 rows --
the prefill and mixed steps of at least 4096 tokens -- runs
``ops/moe/prefill_cutedsl_experts`` on the leaf's own tensors and returns the
same ``(gemm2_output, weights, expanded_idx_to_permuted_idx)`` triple the
TRTLLM-gen call returns, which the model's per-expert post-norm finalize then
consumes. Every other call (decode steps, smaller prefill chunks, the BF16 layers
0 and 61) keeps the expert forward it had.

The routing is the routing method's own ``apply`` (memoized on the logits, so the
finalize reads the same weights). Rounding-different from the TRTLLM-gen call;
expert selection is unchanged.

Install after the weights are final and before any CUDA graph capture: the
kernels are compiled here (``PrefillCuteDslExperts.prepare``), so no capture
compiles them. Agreed over the TP group (``agreed_install``).
"""

from __future__ import annotations

import types
from typing import List, Optional, Tuple

import torch

from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm.logger import logger

from ... import dispatch as _tp_agreement
from ...ops.moe import prefill_cutedsl_experts as _op
from ...ops.moe.ring_buddy_assign import EXPERTS, TOP_K
from ..replicated_experts import ring_buddy as _ring_buddy

#: The SM version and tp size this path is installed at: Rubin (sm107) tep4.
SERVED_SM = 107
SERVED_TP_SIZE = 4

#: Rows of the expert calls this path serves.
MIN_ROWS = 4096
MAX_ROWS = 8192


def serves(compute_capability: Optional[Tuple[int, int]], tp_size: int) -> bool:
    """Whether a model on ``compute_capability`` at ``tp_size`` installs this path."""
    return _tp_agreement.sm_of(compute_capability) == SERVED_SM and tp_size == SERVED_TP_SIZE


def _refusal(moe, mapping) -> Optional[str]:
    reason = _ring_buddy.eligible(moe, mapping)
    if reason is not None:
        return reason
    if _ring_buddy.expert_format(moe) != "nvfp4":
        return "the experts are not the native NVFP4 TRTLLM-gen leaf"
    backend = _ring_buddy._backend(moe)
    if not _ring_buddy._is_plain_swiglu(backend):
        return "the experts are not plain SwiGLU"
    if getattr(backend, "fc2_input_scale", None) is None:
        return "the leaf has no FC2 input scale"
    return None


def _wrap(moe, op: "_op.PrefillCuteDslExperts") -> None:
    experts = moe.experts
    prior = experts.forward
    routing = experts.routing_method

    def forward(self, x, router_logits, *args, do_finalize=True, **kwargs):
        if (
            do_finalize
            or args
            or not isinstance(x, Fp4QuantizedTensor)
            or x.is_sf_swizzled
            or not MIN_ROWS <= x.fp4_tensor.shape[0] <= MAX_ROWS
            or torch.compiler.is_compiling()
        ):
            return prior(x, router_logits, *args, do_finalize=do_finalize, **kwargs)
        selected, weights = routing.apply(router_logits)
        gemm2_output, expanded_idx_to_permuted_idx = op(
            x.fp4_tensor, x.scaling_factor, selected, weights
        )
        return gemm2_output, weights, expanded_idx_to_permuted_idx

    experts.forward = types.MethodType(forward, experts)


def install(text_model, mapping) -> List[int]:
    """Wrap every eligible NVFP4 layer's expert forward; returns the layer indices.

    Collective over the TP group (the outcome is agreed). Raises ValueError on every
    rank alike when some rank refuses (not sm107, no eligible layer) or cannot
    compile the kernels.
    """

    def build() -> List[int]:
        if torch.cuda.get_device_capability() != (10, 7):
            raise ValueError(
                "[musespark_open] prefill_cutedsl_experts is written for sm107 (Rubin) only"
            )
        installed, skipped, ops = [], {}, []
        for index, layer in enumerate(text_model.layers):
            moe = layer.feed_forward
            reason = _refusal(moe, mapping)
            if reason is not None:
                skipped[index] = reason
                continue
            backend = _ring_buddy._backend(moe)
            op = _op.PrefillCuteDslExperts(backend, EXPERTS, TOP_K, backend.slot_start)
            ops.append((moe, op))
            installed.append(index)
        if not installed:
            raise ValueError(
                "[musespark_open] prefill_cutedsl_experts found no eligible NVFP4 layer"
            )
        try:
            # One compile serves every layer (the kernels' shapes are runtime arguments).
            ops[0][1].prepare(MAX_ROWS)
            torch.cuda.synchronize()
            # Hand the compile call's transient buffers back before the KV cache is sized.
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - the compile's failure must reach the agreement
            raise RuntimeError(
                f"[musespark_open] prefill_cutedsl_experts could not compile: {error!r}"
            ) from error
        for moe, op in ops:
            _wrap(moe, op)
            moe._prefill_cutedsl_experts = op
        logger.info(
            f"[musespark_open] prefill_cutedsl_experts: {len(installed)} NVFP4 MoE layers run "
            f"their {MIN_ROWS}..{MAX_ROWS}-row expert calls on the Rubin CuTe DSL GEMMs "
            f"(tile {_op.TILE}); built-in expert call kept for layers {sorted(skipped)}: "
            f"{sorted(set(skipped.values()))}"
        )
        return installed

    return _tp_agreement.agreed_install(mapping, build, (ValueError, RuntimeError))
