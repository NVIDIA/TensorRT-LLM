# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_TAIL: the MoE layers' top-k reduce folded into the shared-expert add, and where the layer boundary runs
FlashInfer's fused add + RMSNorm, into that norm too (``megamoe_topk_reduce_tail``).

On request the MegaMoE local backend (``TrtllmCutedslMegaMoeLocalNvfp4Impl``, lever TAIL) launches without its top-k
reduce and returns the pre-reduced plane with the top-k weights (``MegaMoEDeferredTopkReduce``). Two forms consume it,
both bitwise the stock path:

* form (a), ``megamoe_topk_reduce_shared_add``: ``shared.add_(routed)``. The layer then runs its boundary norm as
  stock.
* form (b), ``megamoe_topk_reduce_add_rmsnorm``: form (a) and then ``flashinfer_fused_add_rmsnorm`` with the next
  layer's input norm, in one launch. A layer takes it only where ``RMSNorm.forward`` runs FlashInfer's fused add +
  RMSNorm (``norm_decline_reason``) and ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` finds the FlashInfer
  kernel form (b) copies; never on a layer Eagle3 captures. Where that norm is a DEP_NORM_MXFP8 site
  (``norm_mxfp8``), form (b) also writes the next qkv_proj's MXFP8 hand-off.

A launch the backend does not defer (a bucket without the deferred kernel, a chunked forward) returns the reduced rows,
and the layer runs the stock add and norm. ``_moe_forward`` is ``MiniMaxM3MoE.forward``'s attention-DP branch with the
shared expert; a change there must be made here too.

A deferred launch is another kernel class, and once a CUDA graph captured a launch bucket every launch at it must be
the captured class. So whether a launch defers may not change from step to step, and the install declines where it
would (``config_decline_reason``).

``TRTLLM_M31_DISABLE_DEP_TAIL=1`` installs nothing; ``TRTLLM_M31_DISABLE_DEP_TAIL_NORM=1`` keeps every layer on form
(a). The backend's ``TRTLLM_M31_DISABLE_MEGAMOE_LOCAL_TAIL=1`` declines every request. Read at import.
"""

from __future__ import annotations

import functools
import os
import types
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer, MiniMaxM3MoE
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import EventType
from tensorrt_llm.logger import logger

from ..ops.moe.megamoe_local.local_impl import MegaMoEDeferredTopkReduce
from ..ops.moe.megamoe_local.local_plan import SHARED_SERIAL_DISABLE_ENV
from . import norm_mxfp8 as _norm_mxfp8
from . import se_side as _se_side
from .m31_switches import m31_disabled

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig


def _is_deferred_topk_reduce(routed) -> bool:
    """Whether ``routed`` is the routed output of a MegaMoE launch that left its top-k reduce to the caller."""
    return isinstance(routed, MegaMoEDeferredTopkReduce)


def _on_pinned_mxfp8(compute_shared, bucket: int, kernel: str):
    """``compute_shared`` with its MXFP8 linears on the pinned GEMM (``kernel``: ``cute_dsl`` or ``cutlass``) through
    the linear layer's MXFP8 dispatch scopes; reports the pin once per launch bucket. Without those scopes on this
    runtime, ``compute_shared`` itself."""
    cute_dsl = kernel == "cute_dsl"
    scope = getattr(
        _linear, "mxfp8_cute_dsl_dispatch" if cute_dsl else "mxfp8_native_dispatch", None
    )
    if scope is None:
        return compute_shared

    def run():
        logger.info_once(
            f"[M3.1 DEP] shared corun pin {'cute_dsl' if cute_dsl else 'cutlass_tuned'} bucket={bucket}",
            key=f"m31_shared_corun_pin_{bucket}",
        )
        with scope():
            return compute_shared()

    return run


def _routed_beside_shared_expert(moe, hidden_states, max_rows, compute_experts, compute_shared):
    """The routed and shared outputs of a step whose shared expert runs serially, with the router and the route-quant
    seat on the aux stream beside the shared expert. The routed launch starts after both and takes the staged seat;
    reports the fork once per step width."""
    backend = moe.experts.backend
    logger.info_once(
        f"[M3.1 DEP] route beside the serial shared expert at {max_rows} rows",
        key=f"m31_route_beside_shared_{max_rows}",
    )

    def route_and_stage():
        router_logits = moe.gate(hidden_states)
        backend.prestage_route_quant(hidden_states, router_logits, max_rows)
        return router_logits

    shared_output, router_logits = maybe_execute_in_parallel(
        compute_shared,
        route_and_stage,
        moe.event_dict[EventType.Main],
        moe.event_dict[EventType.MoeShared],
        moe.aux_stream,
        disable_on_compile=True,
    )
    return compute_experts(router_logits), shared_output


DEP_TAIL = not m31_disabled("DEP_TAIL")
DEP_TAIL_NORM = not m31_disabled("DEP_TAIL_NORM")

# The (outcome, detail) pairs already logged: one line per distinct reason a forward takes a form or the stock tail.
_LOGGED: set = set()


def _log_once(outcome: str, detail: str) -> None:
    if (outcome, detail) not in _LOGGED:
        _LOGGED.add((outcome, detail))
        logger.info(f"[M3.1 DEP] DEP_TAIL {outcome}: {detail}")


def config_decline_reason(model_config: ModelConfig) -> Optional[str]:
    """Why no layer of a model built with ``model_config`` takes the tail, or None.

    Under torch.compile the compiled forwards could not defer while the eager ones did. With CUDA graphs on and an
    MoE cap below ``max_num_tokens``, a step can run in chunks, whose launches do not defer, at a bucket a graph
    captured deferring.
    """
    if model_config.extra_attrs.get("torch_compile_enabled", False):
        return "torch.compile is on"
    moe_cap, max_tokens = model_config.moe_max_num_tokens, model_config.max_num_tokens
    if model_config.use_cuda_graph and moe_cap is not None and moe_cap < max_tokens:
        return (
            f"moe_max_num_tokens {moe_cap} is below max_num_tokens {max_tokens} with CUDA graphs on, "
            "so a step can run in chunks"
        )
    return None


def moe_decline_reason(moe: Optional[torch.nn.Module]) -> Optional[str]:
    """Why the MoE block ``moe`` keeps its stock forward, or None if its routed output may arrive deferred."""
    if type(moe) is not MiniMaxM3MoE:
        return f"MoE block {type(moe).__name__}"
    if moe.shared_experts is None:
        return "no shared expert"
    if moe.allreduce is not None or not moe.routed_output_is_global:
        return "not attention DP on a fused-communication MoE"
    backend = getattr(moe.experts, "backend", None)
    if not hasattr(backend, "request_deferred_topk_reduce"):
        return f"backend {type(backend).__name__} reduces its own top-k"
    return backend.tail_static_decline_reason()


@functools.cache
def _matches_flashinfer(device: torch.device, hidden_size: int) -> bool:
    from ..kernels.megamoe.megamoe_topk_reduce_tail import (
        megamoe_topk_reduce_add_rmsnorm_matches_flashinfer,
    )

    return megamoe_topk_reduce_add_rmsnorm_matches_flashinfer(device, hidden_size)


def norm_decline_reason(layer: torch.nn.Module) -> Optional[str]:
    """Why ``layer``'s boundary norm stays outside the tail, or None if it is ``flashinfer_fused_add_rmsnorm`` on BF16
    rows and form (b) copies that kernel (``RMSNorm.forward``'s FlashInfer branch)."""
    if not DEP_TAIL_NORM:
        return "TRTLLM_M31_DISABLE_DEP_TAIL_NORM=1"
    return boundary_norm_decline_reason(layer)


def boundary_norm_decline_reason(layer: torch.nn.Module) -> Optional[str]:
    """Why ``layer``'s boundary norm is not ``flashinfer_fused_add_rmsnorm`` on BF16 rows in the kernel form
    ``megamoe_topk_reduce_tail`` copies, or None if it is."""
    norm = getattr(layer, "next_layer_layernorm", None)
    if norm is None:
        return "no next-layer norm"
    if layer.post_feed_forward_fusion:
        return "the boundary norm is fused with an AllReduce"
    return flashinfer_norm_decline_reason(norm)


def flashinfer_norm_decline_reason(norm: torch.nn.Module) -> Optional[str]:
    """Why ``norm``'s add + RMSNorm is not ``flashinfer_fused_add_rmsnorm`` on BF16 rows in the kernel form
    ``megamoe_topk_reduce_tail`` copies, or None if it is."""
    if type(norm) is not RMSNorm:
        return f"norm {type(norm).__name__}"
    if norm.is_nvfp4 and norm.nvfp4_scale is not None:
        return "the norm is fused with an NVFP4 quantize"
    if norm.return_hp_output or norm.use_cuda_tile or norm.use_gemma:
        return "the norm is not FlashInfer's fused add + RMSNorm"
    if not IS_FLASHINFER_AVAILABLE:
        return "FlashInfer is not available"
    weight = norm.weight
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or weight.data_ptr() % 16
    ):
        return "the norm weight is not a contiguous 16-byte-aligned BF16 CUDA tensor"
    if not _matches_flashinfer(weight.device, weight.numel()):
        return "FlashInfer's fused add + RMSNorm is not the kernel form (b) copies"
    return None


def _routed_and_shared(
    moe: MiniMaxM3MoE, hidden_states: torch.Tensor, attn_metadata, lora_params
) -> Tuple[object, torch.Tensor]:
    """``MiniMaxM3MoE.forward``'s routed and shared branches, with the routed launch's top-k reduce requested: the
    shared expert on DEP_SE_SIDE's 28-SM green side stream beside the routed launch where ``se_side.placement`` gives
    it (after the routed launch where it gives ``SERIAL``), else after the routed launch where the backend's
    ``shared_expert_serial`` holds for the step (before it, with the router and the route-quant seat beside it on the
    aux stream, where its ``route_beside_shared`` holds too), else beside it on the aux stream, on the pinned MXFP8 GEMM
    (the backend's ``shared_corun_pin_kernel``) where its ``shared_corun_pin`` holds for the step."""
    all_rank_num_tokens = attn_metadata.all_rank_num_tokens
    backend = moe.experts.backend
    if getattr(moe, "_dep_tail_defer", True):
        if hidden_states.dtype != torch.bfloat16:
            reason = "the activation is not BF16"
        else:
            reason = backend.request_deferred_topk_reduce(
                all_rank_num_tokens, hidden_states.shape[0]
            )
        if reason is not None:
            backend.cancel_deferred_topk_reduce()
            _log_once("stock", reason)

    def _compute_experts(router_logits):
        return moe.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=False,
            lora_params=lora_params,
        )

    def _compute_routed_output():
        return _compute_experts(moe.gate(hidden_states))

    def _compute_shared_output():
        return moe.shared_experts(hidden_states, lora_params=lora_params)

    max_rows = max(all_rank_num_tokens) if all_rank_num_tokens else None
    side = _se_side.placement(moe, backend, max_rows, hidden_states.shape[0])
    if side is _se_side.SERIAL:
        routed_output = _compute_routed_output()
        return routed_output, _compute_shared_output()
    if side is not None:

        def _compute_shared_on_side():
            with _se_side.side_calls():
                return _compute_shared_output()

        return maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_on_side,
            moe.event_dict[EventType.Main],
            moe.event_dict[EventType.MoeShared],
            side,
            disable_on_compile=True,
        )
    # The built-in MoE block's serial / co-run settings (its constructor reads them off the backend; False where the
    # block or the backend has none).
    if getattr(moe, "shared_expert_serial", False) and backend.shared_expert_serial(max_rows):
        if getattr(moe, "route_beside_shared", False) and backend.route_beside_shared(max_rows):
            return _routed_beside_shared_expert(
                moe, hidden_states, max_rows, _compute_experts, _compute_shared_output
            )
        routed_output = _compute_routed_output()
        return routed_output, _compute_shared_output()
    compute_shared = _compute_shared_output
    if getattr(moe, "shared_corun_pinnable", False) and backend.shared_corun_pin(max_rows):
        compute_shared = _on_pinned_mxfp8(
            _compute_shared_output,
            backend.megamoe_launch_bucket(max_rows),
            backend.shared_corun_pin_kernel(),
        )
    return maybe_execute_in_parallel(
        _compute_routed_output,
        compute_shared,
        moe.event_dict[EventType.Main],
        moe.event_dict[EventType.MoeShared],
        moe.aux_stream,
        disable_on_compile=True,
    )


def _tail_rows(shared: torch.Tensor) -> torch.Tensor:
    if shared.is_contiguous() and shared.data_ptr() % 16 == 0:
        return shared
    return shared.contiguous()


def _add(routed, shared: torch.Tensor) -> torch.Tensor:
    if _is_deferred_topk_reduce(routed):
        _log_once("form (a)", "top-k reduce + shared add")
        return routed.add_into(_tail_rows(shared))
    return shared.add_(routed)


def _moe_forward(
    self: MiniMaxM3MoE,
    hidden_states: torch.Tensor,
    attn_metadata,
    final_all_reduce_params=None,
    lora_params: Optional[dict] = None,
) -> torch.Tensor:
    """``MiniMaxM3MoE.forward`` under attention DP with a shared expert, the add on form (a) when the launch
    deferred its top-k reduce."""
    routed, shared = _routed_and_shared(self, hidden_states, attn_metadata, lora_params)
    return _add(routed, shared)


def _forward_moe_norm(
    self: MiniMaxM3DecoderLayer,
    hidden_states: torch.Tensor,
    attn_metadata,
    residual: torch.Tensor,
    spec_metadata=None,
    lora_params: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``MiniMaxM3DecoderLayer.forward_MoE``, the boundary norm on form (b) when the launch deferred its top-k reduce
    and the rows fit form (b); a layer Eagle3 captures this step runs the built-in ``forward_MoE``."""
    if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
        return MiniMaxM3DecoderLayer.forward_MoE(
            self,
            hidden_states,
            attn_metadata,
            residual,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
        )
    hidden_states, residual = self._apply_pre_feed_forward_norm(hidden_states, residual)
    routed, shared = _routed_and_shared(
        self.block_sparse_moe, hidden_states, attn_metadata, lora_params
    )
    norm = self.next_layer_layernorm
    if (
        _is_deferred_topk_reduce(routed)
        and norm.nvfp4_scale is None
        and residual.dtype == torch.bfloat16
        and residual.shape == shared.shape
        and residual.is_contiguous()
        and residual.data_ptr() % 16 == 0
    ):
        _log_once("form (b)", "top-k reduce + shared add + fused add + RMSNorm")
        rows = _tail_rows(shared)
        handoff = _norm_mxfp8.site_handoff(norm, rows, residual)
        if handoff is None:
            return routed.add_rmsnorm_into(rows, residual, norm.weight, norm.variance_epsilon)
        out = routed.add_rmsnorm_into(
            rows, residual, norm.weight, norm.variance_epsilon, q=handoff[0], sf=handoff[1]
        )
        _norm_mxfp8.publish(rows, handoff, "DEP_TAIL form (b)")
        return out
    return self._apply_next_layer_layernorm(_add(routed, shared), residual)


def install(layers: Iterable[torch.nn.Module], model_config: ModelConfig) -> Tuple[int, int]:
    """Install the tail on each decoder layer whose MoE block ``moe_decline_reason`` admits: form (b) where
    ``norm_decline_reason`` finds none, else form (a). Returns how many layers carry each form, (a) then (b), including
    any an earlier call installed; (0, 0) with TRTLLM_M31_DISABLE_DEP_TAIL=1 or where ``config_decline_reason`` finds
    one for the model's ``model_config``. Logs each distinct decline once."""
    if not DEP_TAIL:
        return 0, 0
    reason = config_decline_reason(model_config)
    if reason is not None:
        _log_once("not installed", reason)
        return 0, 0
    form_a = form_b = 0
    for layer in layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        reason = moe_decline_reason(moe)
        if reason is None and not _replaceable(moe, "forward", _moe_forward):
            reason = "the MoE block's forward is already replaced"
        if reason is None and not _replaceable(layer, "forward_MoE", _forward_moe_norm):
            reason = "the layer's forward_MoE is already replaced"
        if reason is not None:
            _log_once("not installed", reason)
            continue
        moe.forward = types.MethodType(_moe_forward, moe)
        moe._dep_tail_defer = True
        _set_shared_plan(moe)
        reason = (
            None
            if _replaced(layer, "forward_MoE", _forward_moe_norm)
            else norm_decline_reason(layer)
        )
        if reason is None:
            layer.forward_MoE = types.MethodType(_forward_moe_norm, layer)
            form_b += 1
        else:
            _log_once("form (a) only", reason)
            form_a += 1
    return form_a, form_b


def serves(moe: torch.nn.Module) -> bool:
    """Whether ``moe``'s forward is this module's (the placements in ``_routed_and_shared`` apply to it)."""
    return _replaced(moe, "forward", _moe_forward)


def _replaced(module: torch.nn.Module, name: str, func) -> bool:
    return getattr(module.__dict__.get(name), "__func__", None) is func


def _replaceable(module: torch.nn.Module, name: str, func) -> bool:
    return name not in module.__dict__ or _replaced(module, name, func)


def shared_plan_decline_reason(moe: Optional[torch.nn.Module]) -> Optional[str]:
    """Why the MoE block ``moe`` keeps the built-in placement of its shared expert, or None if this module's forward
    may place it by its backend's answers (``shared_expert_serial``, ``route_beside_shared``, ``shared_corun_pin``)."""
    if type(moe) is not MiniMaxM3MoE:
        return f"MoE block {type(moe).__name__}"
    if moe.shared_experts is None:
        return "no shared expert"
    if moe.allreduce is not None or not moe.routed_output_is_global:
        return "not attention DP on a fused-communication MoE"
    backend = getattr(moe.experts, "backend", None)
    if not hasattr(backend, "shared_expert_serial") or not hasattr(
        backend, "request_deferred_topk_reduce"
    ):
        return f"backend {type(backend).__name__} has no shared-expert placement"
    return None


def _set_shared_plan(moe: MiniMaxM3MoE) -> None:
    """The MoE block's per-layer placement switches; each step's answer is its backend's.

    The shared expert runs after the routed launch where the backend's ``shared_expert_serial`` holds for the step
    (``TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL=1`` keeps it beside the launch at every step), with the router and the
    route-quant seat beside it on the aux stream where its ``route_beside_shared`` holds too; beside the launch, its
    MXFP8 linears take the pinned GEMM where its ``shared_corun_pin`` holds.
    """
    backend = moe.experts.backend
    has_plan = hasattr(backend, "shared_expert_serial")
    moe.shared_expert_serial = has_plan and os.environ.get(SHARED_SERIAL_DISABLE_ENV, "0") != "1"
    moe.shared_corun_pinnable = has_plan
    moe.route_beside_shared = hasattr(backend, "prestage_route_quant")


def install_shared_plan(layers: Iterable[torch.nn.Module]) -> int:
    """Give every MoE block the tail did not take this module's forward without the top-k deferral, so its shared
    expert is placed by its backend's answers (``_set_shared_plan``); returns how many blocks run this module's forward
    afterwards, the tail's included. Logs each distinct decline once."""
    count = 0
    for layer in layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is None:
            continue
        if serves(moe):
            count += 1
            continue
        reason = shared_plan_decline_reason(moe)
        if reason is None and not _replaceable(moe, "forward", _moe_forward):
            reason = "the MoE block's forward is already replaced"
        if reason is not None:
            _log_once("shared placement not installed", reason)
            continue
        moe.forward = types.MethodType(_moe_forward, moe)
        moe._dep_tail_defer = False
        _set_shared_plan(moe)
        count += 1
    return count
