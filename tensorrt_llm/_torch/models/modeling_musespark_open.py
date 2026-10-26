# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MuseSparkOpen text decoder for the PyTorch backend.

Text-to-text only. The checkpoint's architecture string covers a multimodal
model, but perception towers are never constructed here and their checkpoint
keys are skipped without reading their payloads.

Shape summary for the released checkpoint: 62 MoE decoder layers, hidden 8192,
128 query heads / 16 KV heads of dim 64, 256 experts with top-8 routing, expert
state 4096 wide (``compression_factor=2``) and expert FFN intermediate 4096.
Layers ``l`` with ``(num_layers - l - 1) % 4 == 0`` are NoPE + full causal; the
rest are interleaved-RoPE with a 2048-key sliding window.

Environment switches (each keeps its unfused/recomputing arm reachable so any
lever can be A/B'd on one image; see the ``_*_enabled`` helpers for rationale):

- ``MUSESPARK_OPEN_FUSED_RESIDUAL`` (default on): fused residual join.
- ``MUSESPARK_OPEN_ROUTING_MEMO`` (default on): reuse of the first routing pass.
- ``MUSESPARK_OPEN_FROZEN_GATE_CACHE`` (default on): cached gate coefficients.
- ``MUSESPARK_OPEN_FUSED_FINALIZE_AR`` (default on): fused expert finalize +
  TP4 all-reduce.
- ``MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL`` (default off): extend finalize fusion
  to TP8/EP8 across two NVLink-connected four-GPU GB300 nodes.
- ``MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED`` (default off): serve batches of
  129..512 tokens with the fused finalize + all-reduce, as consecutive calls of
  at most 128 tokens.
- ``MUSESPARK_OPEN_SHARDED_PROJECTIONS`` (default off, experimental): column-shard
  pool/unpool projections for batches of at most eight tokens, at TP8/EP8 over
  two four-GPU GB300 nodes or TP4/EP4 on one.
- ``MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT`` (default on): fused pre-expert
  RMSNorm + NVFP4 quantization.
- ``MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT`` (default off): BF16 TP4/TP8 QKV
  weight layout for single-row decode with separate native normalization.
- ``MUSESPARK_OPEN_FUSED_PREATTN`` (default OFF, opt-in): fused pre-attention.
- ``MUSESPARK_OPEN_ALLREDUCE`` (default NCCL): all-reduce strategy name.
- ``MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL`` (default off): shard replicated
  prefill work across single-node BF16 TP4/EP4 ranks.
- ``MUSESPARK_OPEN_BF16X3_ROUTER`` (default off): BF16x3 projection for large routers.
- ``MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES`` (default off): fuse large-batch routing;
  ``MUSESPARK_OPEN_FULL_ROUTING_TORCH_ORDER`` preserves Torch's large-batch order.
- ``MUSESPARK_OPEN_LARGE_FINALIZE_4WARPS`` (default off): use four warps for large
  expert finalize kernels.
- ``MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE`` (default off): finalize batches of 160
  tokens or more by visiting only each token's local expert slots.
- ``MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP`` (default off): overlap the router with the
  pool projection for batches of 129..512 tokens too.
- ``MUSESPARK_OPEN_LARGE_NORM_GATE`` (default off): fuse H64 attention norm/gate with
  the reference reduction order above 128 tokens.
- ``MUSESPARK_OPEN_OVERLAP_SP_PROJECTION`` (default off): overlap two output-projection
  chunks with reduce-scatter in eager sequence-parallel prefills of >=8192 tokens.
- ``MUSESPARK_OPEN_PAD_SP_PROJECTION`` (default off): pad narrower projection inputs
  instead of outputs in eager sequence-parallel prefills; overlap takes precedence.

Default-on switches read as disabled for the values ``0``/``false``/``no``/
``off``.
"""

import contextlib
import copy
import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch.configs.musespark_open import SLIDING_ATTENTION, MuseSparkOpenConfig
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...models.modeling_utils import QuantConfig
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..attention.qk_norm_attention import QKNormRoPEAttention
from ..distributed import AllReduce, AllReduceParams, allgather, reducescatter
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_ops.rmsnorm_fp4_quant import rmsnorm_fp4_quant_available
from ..modules.linear import Linear, TensorParallelMode
from ..modules.logits_processor import LogitsProcessor
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe
from ..moe.fused_moe.moe_op_backend import TRTLLMOpBackend
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel

if TYPE_CHECKING:
    from .musespark_open_finalize_ar import FinalizeIpcWorkspace
    from .musespark_open_projection_gather import ProjectionGatherWorkspace

# Perception subtrees in the checkpoint. Text requests never build these, and
# their tensors must not be materialized (they are hundreds of GiB).
PERCEPTION_KEY_PREFIXES = ("vision_model.", "modalities.", "audio_model.")


# Per-layer, per-op NVTX ranges. The decode step of this model is dominated by
# small elementwise kernels spread across the router, the MoE finalize and the
# two all-reduces, and a flat kernel timeline cannot say which layer or which
# op a launch belongs to. Gated on the same environment variables as
# ``nvtx_range_debug`` (``TLLM_LLMAPI_ENABLE_NVTX`` / ``TLLM_NVTX_DEBUG``), but
# resolved once per module at construction instead of per call: when profiling
# is off, ``forward`` enters one shared no-op context and does no environment
# lookup, string formatting or allocation per step.
#
# The ranges are host-side, so they bracket the Python call, not a CUDA-graph
# replay: with graphs on they annotate the capture pass and not the steady-state
# iterations. Attribute from an eager (graph-disabled) profile, and remember
# that eager and graphed iteration times are not comparable.
NVTX_DOMAIN = "MuseSparkOpen"
_NVTX_OFF = contextlib.nullcontext()


def nvtx_layer_ranges_enabled() -> bool:
    return (
        os.getenv("TLLM_LLMAPI_ENABLE_NVTX", "0") == "1" or os.getenv("TLLM_NVTX_DEBUG", "0") == "1"
    )


class LayerNvtx:
    """Prebuilt ``<module>.L<idx>.<op>`` NVTX ranges for one layer."""

    def __init__(self, module: str, layer_idx: int, ops: Tuple[str, ...]):
        self.enabled = nvtx_layer_ranges_enabled()
        self.names = {op: f"{module}.L{layer_idx:03d}.{op}" for op in ops} if self.enabled else {}

    def __call__(self, op: str):
        if not self.enabled:
            return _NVTX_OFF
        return nvtx_range(self.names[op], domain=NVTX_DOMAIN)


def normalize_exclude_modules(patterns: List[str]) -> List[str]:
    """Rewrite the checkpoint's bare-substring exclusions as fnmatch globs.

    ``hf_quant_config.json`` lists ``["self_attn", "lm_head", "layers.0.",
    "layers.61."]``. TRT-LLM matches an exclusion against the module name and
    its ancestors with ``fnmatch``, so a bare substring like ``layers.0.``
    matches nothing and the BF16 layers would be built as NVFP4. Anchor each
    substring with ``*`` unless the producer already wrote a glob or regex.
    """
    normalized = []
    for pattern in patterns:
        if pattern.startswith("re:") or "*" in pattern or "?" in pattern:
            normalized.append(pattern)
        elif pattern == "lm_head":
            normalized.append(pattern)
        else:
            normalized.append(
                f"*{pattern.strip('.')}.*" if pattern.endswith(".") else f"*{pattern}*"
            )
    return normalized


def post_expert_norm_rows(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weight-first RMSNorm of each selected expert's output, before combining.

    Returns ``(rows, valid)`` with ``rows`` shaped ``[num_tokens, top_k,
    hidden_size]`` in FP32 and ``valid`` marking the slots whose expert is
    local to this rank.

    Separate from the combine below because this is the boundary the source's
    fused finalize kernel destroys: it unpermutes, normalizes, weights and
    reduces in one pass, so the per-expert row exists only inside its
    registers. Naming it here is what lets a parity harness compare the individual
    post-norm rows rather than only their sum -- against which a norm-after-sum
    bug and a norm-before-sum implementation look the same.

    ``expanded_idx_to_permuted_idx`` holds ``-1`` for slots whose expert is not
    local to this rank (or is padding). Those rows of ``gemm2_output`` are
    never written by the kernel and may hold NaN, so they are *selected out*
    rather than multiplied by zero.
    """
    index = expanded_idx_to_permuted_idx.reshape(num_tokens, top_k)
    valid = index >= 0
    safe_index = torch.where(valid, index, torch.zeros_like(index))
    gathered = gemm2_output.index_select(0, safe_index.reshape(-1).int())
    gathered = gathered[:, :hidden_size].reshape(num_tokens, top_k, hidden_size)
    gathered = torch.where(
        valid.unsqueeze(-1),
        gathered.to(torch.float32),
        torch.zeros_like(gathered, dtype=torch.float32),
    )

    scaled = gathered * (norm_weight.to(torch.float32) + 1.0)
    variance = scaled.pow(2).mean(-1, keepdim=True)
    normed = scaled * torch.rsqrt(variance + eps)
    # Invalid slots stay exactly zero: their gathered rows were replaced by
    # zeros, and rsqrt(0 + eps) would otherwise scale them to a finite value.
    return torch.where(valid.unsqueeze(-1), normed, torch.zeros_like(normed)), valid


@triton.jit
def _post_expert_norm_combine_kernel(
    out_ptr,  # [num_tokens, HIDDEN] fp32
    gemm2_ptr,  # [num_permuted, >= HIDDEN], any float dtype
    index_ptr,  # [num_tokens * TOP_K] integer, -1 for a non-local slot
    weights_ptr,  # [num_tokens * TOP_K] any float dtype
    norm_weight_ptr,  # [HIDDEN] any float dtype
    gemm2_row_stride,
    gemm2_num_rows,  # rows the MoE backend actually allocated and wrote
    eps,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per token: gather, weight-first RMSNorm, weight, sum.

    The whole ``[TOP_K, HIDDEN]`` working set for one token stays inside the SM,
    so the FP32 intermediates the unfused chain materializes in HBM (``[T, 8,
    H]``, a dozen times over) never leave it. Only the expert rows are read and
    only the FP32 combined token is written.

    Both reductions are taken once over the whole tile rather than once per
    slot. The slot-at-a-time formulation was written first and measured against
    this one on device time at the checkpoint's width: it is 1.5x slower at
    T=256 and 1.4x at T=32, because it pays a cross-warp reduction barrier per
    slot instead of one for all eight.
    """
    token = tl.program_id(0)
    hoff = tl.arange(0, BLOCK_H)
    koff = tl.arange(0, BLOCK_K)
    in_row = hoff < HIDDEN
    in_slot = koff < TOP_K

    # Weight-first: the stored gamma is an offset from 1.0.
    gain = tl.load(norm_weight_ptr + hoff, mask=in_row, other=0.0).to(tl.float32) + 1.0
    index = tl.load(index_ptr + token * TOP_K + koff, mask=in_slot, other=-1)
    # A non-local (or padding) slot is *selected out*, not multiplied by zero:
    # the MoE kernel never wrote that row and it may hold NaN, and ``0 * NaN``
    # is NaN. Folding validity into the load mask means those bytes are never
    # read at all, so no NaN can enter the accumulator.
    # Bounded on BOTH sides. `index >= 0` rejects the non-local/padding slots
    # the backend marks with -1. `index < gemm2_num_rows` is the other half, and
    # it is not hypothetical bookkeeping: this index is
    # `expanded_idx_to_permuted_idx`, whose value depends on how the WHOLE batch
    # routed, so an index that addressed past the permuted buffer would be a
    # batch-composition-dependent read of memory this kernel does not own --
    # which is exactly the class of failure batch-invariance checks exist to
    # detect, and it
    # would look like a numerics divergence rather than a fault. Triton masks
    # only what it is told to mask, so without this the upper end is unchecked.
    # Costs one comparison per slot and cannot change a correct result: every
    # valid permuted index is inside the buffer by construction.
    local = in_slot & (index >= 0) & (index < gemm2_num_rows)
    safe = tl.where(local, index, 0).to(tl.int64)

    rows = tl.load(
        gemm2_ptr + safe[:, None] * gemm2_row_stride + hoff[None, :],
        mask=local[:, None] & in_row[None, :],
        other=0.0,
    ).to(tl.float32)
    scaled = rows * gain[None, :]
    variance = tl.sum(scaled * scaled, axis=1) / HIDDEN
    normed = scaled * tl.rsqrt(variance + eps)[:, None]
    weights = tl.load(weights_ptr + token * TOP_K + koff, mask=in_slot, other=0.0).to(tl.float32)
    # Second guard on the same fact, and free: an invalid slot contributes exact
    # zeros even if ``eps`` were ever zero, where ``rsqrt(0)`` would otherwise
    # turn the zeroed row into a NaN.
    acc = tl.sum(tl.where(local[:, None], normed * weights[:, None], 0.0), axis=0)
    tl.store(out_ptr + token * HIDDEN + hoff, acc, mask=in_row)


# Threads per program. Measured, not reasoned: at the checkpoint's width eight
# warps is the fastest or within 8% of the fastest at T=1, 32 and 256, and the
# sweep over 4/8/16/32 spans 2.4x at T=256, so the choice is worth making from
# device time rather than from a register-pressure argument. Four and 32 warps
# are both worse at some shape. See the bench report cited above.
_FINALIZE_NUM_WARPS = 8


def _note_unfused_fallback(site: str, **conditions: bool) -> None:
    """Say once when a fused path is skipped for a reason that is not routine.

    Both fused paths in this file fall back to the reference composition rather
    than asserting, which is the right behaviour: the composition computes the
    same function, so nothing is wrong with the output. What is wrong is that
    the fusion can be completely inactive with nothing anywhere to say so. The
    accuracy canaries still pass, because the answer is the same, and the perf
    run then reports no gain -- which reads as "this fusion does not help"
    rather than "this fusion never ran". That is a false negative, and an
    expensive one, because the conclusion gets drawn from a tray boot and the
    lever gets priced on it.

    The zero-token DP pad is deliberately NOT reported. It is routine, it fires
    constantly, and a guard that goes off on the normal path is one nobody
    reads. Only conditions that should never hold for this model are worth a
    line, which is why the caller passes them by name.
    """
    failed = [name for name, ok in conditions.items() if not ok]
    if failed:
        logger.warning_once(
            f"MuseSparkOpen {site}: using the unfused reference composition "
            f"because these do not hold: {', '.join(failed)}. The result is "
            f"correct, but the fusion is inactive, so a performance measurement "
            f"taken in this state is not measuring it.",
            key=f"musespark_open_unfused_{site}",
        )


def _fused_post_expert_norm_combine(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    num_warps: int = _FINALIZE_NUM_WARPS,
) -> torch.Tensor:
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=gemm2_output.device)
    _post_expert_norm_combine_kernel[(num_tokens,)](
        out,
        gemm2_output,
        expanded_idx_to_permuted_idx.reshape(-1),
        routing_weights.reshape(-1),
        norm_weight,
        gemm2_output.stride(0),
        gemm2_output.shape[0],
        eps,
        TOP_K=top_k,
        HIDDEN=hidden_size,
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_H=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
    )
    return out


@triton.jit
def _post_expert_norm_combine_sparse_kernel(
    out_ptr,  # [num_tokens, HIDDEN] fp32
    gemm2_ptr,  # [num_permuted, >= HIDDEN], any float dtype
    index_ptr,  # [num_tokens * TOP_K] integer, -1 for a non-local slot
    weights_ptr,  # [num_tokens * TOP_K] any float dtype
    norm_weight_ptr,  # [HIDDEN] any float dtype
    gemm2_row_stride,
    gemm2_num_rows,
    eps,
    TOP_K: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per token that visits only the slots local to this rank.

    The same function as ``_post_expert_norm_combine_kernel``: gather,
    weight-first RMSNorm, weight and sum, and a slot that is invalid or past the
    end of ``gemm2`` is never read. The difference is the working set. That
    kernel holds the whole ``[TOP_K, HIDDEN]`` tile for every slot, local or
    not; this one branches on each slot's scalar index and keeps one
    ``[HIDDEN]`` row plus the accumulator, so a token with two local experts out
    of eight (the expectation at EP4) loads and normalizes two rows. Large
    batches are bandwidth-bound on exactly those rows.

    ROUNDING-DIFFERENT from the tile kernel: the slots are summed in slot order
    rather than in the tile reduction's order, and each variance is reduced over
    a 1-D row rather than one row of a 2-D tile.
    """
    token = tl.program_id(0)
    hoff = tl.arange(0, BLOCK_H)
    in_row = hoff < HIDDEN
    gain = tl.load(norm_weight_ptr + hoff, mask=in_row, other=0.0).to(tl.float32) + 1.0
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for slot in tl.static_range(TOP_K):
        index = tl.load(index_ptr + token * TOP_K + slot)
        if (index >= 0) & (index < gemm2_num_rows):
            row = tl.load(
                gemm2_ptr + index.to(tl.int64) * gemm2_row_stride + hoff,
                mask=in_row,
                other=0.0,
            ).to(tl.float32)
            scaled = row * gain
            variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
            normed = scaled * tl.rsqrt(variance + eps)
            weight = tl.load(weights_ptr + token * TOP_K + slot).to(tl.float32)
            acc += normed * weight
    tl.store(out_ptr + token * HIDDEN + hoff, acc, mask=in_row)


_SPARSE_FINALIZE_NUM_WARPS = 4

#: The smallest batch that ``MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE`` finalizes
#: with the sparse-slot kernel. The tile kernel runs one program per token, so
#: up to one program per SM (152 on GB300) it finishes in a single wave and is
#: the faster kernel; past that it needs a second wave, and the sparse kernel's
#: smaller working set makes it the faster one on EP4 inputs.
_SPARSE_FINALIZE_MIN_TOKENS = 160


def _sparse_post_expert_norm_combine(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> torch.Tensor:
    out = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=gemm2_output.device)
    _post_expert_norm_combine_sparse_kernel[(num_tokens,)](
        out,
        gemm2_output,
        expanded_idx_to_permuted_idx.reshape(-1),
        routing_weights.reshape(-1),
        norm_weight,
        gemm2_output.stride(0),
        gemm2_output.shape[0],
        eps,
        TOP_K=top_k,
        HIDDEN=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        num_warps=_SPARSE_FINALIZE_NUM_WARPS,
    )
    return out


def post_expert_norm_combine(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    num_warps: int = _FINALIZE_NUM_WARPS,
    sparse_slots: bool = False,
) -> torch.Tensor:
    """Weight-first RMSNorm each selected expert output, then combine, in FP32.

    This is the step the fused kernel's own finalize cannot do: RMSNorm is
    nonlinear, so normalizing after the top-k sum is a different function.

    On CUDA this is one Triton kernel; everywhere else it is the reference
    composition below, which is also what ``post_expert_norm_rows`` still spells
    out op by op. The two are not bit-identical -- the fused reduction sums the
    squares and the top-k in a different order -- so this is a
    ROUNDING-DIFFERENT change, declared as such before it was measured. The
    difference is ~1e-6 relative on an FP32 value that is cast to BF16 two
    statements later, i.e. two orders of magnitude below the grain of the dtype
    it lands in.

    ``sparse_slots`` selects ``_post_expert_norm_combine_sparse_kernel``, which
    reads only the local slots of each token instead of the whole top-k tile.
    ``num_warps`` sets the tile kernel's width only; the sparse kernel always
    runs ``_SPARSE_FINALIZE_NUM_WARPS`` warps.
    """
    # ``num_tokens == 0`` happens on the zero-token DP pad and would launch an
    # empty grid; a row-major ``gemm2_output`` is what the kernel's single row
    # stride assumes. Both fall back rather than assert, because both are
    # exactly what the reference composition below already handles.
    on_cuda = gemm2_output.is_cuda
    row_major = gemm2_output.stride(-1) == 1
    if num_tokens > 0:
        # Not for num_tokens == 0: that is the routine pad, and reporting it
        # would bury the two conditions that actually mean something.
        _note_unfused_fallback(
            "post-expert-norm combine",
            gemm2_output_on_cuda=on_cuda,
            gemm2_output_row_major=row_major,
        )
    if on_cuda and num_tokens > 0 and row_major and sparse_slots:
        return _sparse_post_expert_norm_combine(
            gemm2_output,
            expanded_idx_to_permuted_idx,
            routing_weights,
            norm_weight,
            eps,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
        )
    if on_cuda and num_tokens > 0 and row_major:
        return _fused_post_expert_norm_combine(
            gemm2_output,
            expanded_idx_to_permuted_idx,
            routing_weights,
            norm_weight,
            eps,
            num_tokens=num_tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            num_warps=num_warps,
        )
    normed, _ = post_expert_norm_rows(
        gemm2_output,
        expanded_idx_to_permuted_idx,
        norm_weight,
        eps,
        num_tokens=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
    )
    weights = routing_weights.reshape(num_tokens, top_k, 1).to(torch.float32)
    return (normed * weights).sum(dim=1)


def _version_or_none(tensor: torch.Tensor) -> Optional[int]:
    """``tensor._version``, or ``None`` when the tensor cannot carry one.

    An inference tensor has no version counter, and merely touching
    ``_version`` on one raises ``RuntimeError: Inference tensors do not track
    version counter``. The serving path runs the whole forward under
    ``torch.inference_mode()``, so this is the ordinary case there rather than
    an edge one: reading the counter unguarded killed all four ranks at engine
    warmup, in the routing memo below, while every unit test passed because
    pytest does not run under inference mode.

    Returning ``None`` degrades the memo's guard to object identity for that
    call, which is the guard that actually carries it. The memo is written by
    the backend's routing call and consumed and cleared by the model's, both
    inside one ``MuseSparkOpenMoE.forward``, and nothing between them writes to
    ``router_logits``. Were anything to, the pre-memo code would have been
    wrong too: the kernel would have been driven by one routing and the
    finalize weighted by another.
    """
    if tensor.is_inference():
        return None
    return tensor._version


@triton.jit
def _gated_residual_after_rmsnorm_kernel(
    out_ptr,  # [T, HIDDEN] fp32, the new residual stream
    residual_ptr,  # [T, HIDDEN] fp32, the old one
    branch_ptr,  # [T, HIDDEN] branch output, activation dtype
    gain_ptr,  # [HIDDEN] (1 + w) when HAS_GAIN, else unread
    alpha_ptr,  # [HIDDEN] fp32
    beta_ptr,  # [HIDDEN] fp32
    eps,
    next_out_ptr,
    next_weight_ptr,
    next_eps,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_GAIN: tl.constexpr,
    FUSE_NEXT_NORM: tl.constexpr,
    NORM_FP32_RESIDUAL: tl.constexpr,
):
    """``alpha*residual + beta*rmsnorm(branch)`` for one token, in one kernel.

    Both of a decoder layer's residual joins take this shape. ``HAS_GAIN``
    selects between them: the FFN side scales by ``1 + w`` before normalizing
    (``MuseSparkOpenRMSNormWeightFirst``), the attention side does not
    (``RMSNorm(has_weights=False)``).

    Every rounding the torch chain performs is reproduced, so the only
    arithmetic difference is the order of the variance reduction: the gain
    product is rounded back to the activation dtype before the variance, which
    ``MuseSparkOpenRMSNormWeightFirst`` does deliberately to match the reference's
    fused path, and the norm result is rounded again before the gated add reads
    it. Dropping either round would make this MORE accurate and therefore not a
    fusion of this chain.

    ``NORM_FP32_RESIDUAL`` selects the ordering of the fused NEXT prenorm
    (``FUSE_NEXT_NORM``) only: unset, the updated FP32 residual is rounded to
    the activation dtype before the next variance (the historical ordering);
    set, the variance and the normalization read the FP32 residual directly and
    the single round to the activation dtype happens on the output, after the
    ``(1 + w)`` gain. The residual stream written to ``out_ptr`` is identical
    either way.
    """
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < HIDDEN
    base = token * HIDDEN + offs

    branch = tl.load(branch_ptr + base, mask=mask, other=0.0)
    scaled = branch.to(tl.float32)
    if HAS_GAIN:
        gain = tl.load(gain_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scaled = (scaled * gain).to(branch.dtype).to(tl.float32)
    variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
    normed = (scaled * tl.rsqrt(variance + eps)).to(branch.dtype).to(tl.float32)

    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    alpha = tl.load(alpha_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    updated = alpha * residual + beta * normed
    tl.store(out_ptr + base, updated, mask=mask)
    if FUSE_NEXT_NORM:
        if NORM_FP32_RESIDUAL:
            # Reference ordering: the next variance and normalization read the
            # FP32 residual; the store below takes the single activation-dtype
            # round, after the gain.
            next_input = updated
        else:
            # Preserve the explicit activation cast before the consuming RMSNorm.
            next_input = updated.to(branch.dtype).to(tl.float32)
        next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
        next_weight = tl.load(next_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        next_output = next_input * tl.rsqrt(next_variance + next_eps)
        next_output = next_output * (next_weight + 1.0)
        tl.store(next_out_ptr + base, next_output, mask=mask)


# Threads per program for the residual join, measured the same way the finalize's
# was: device time at the checkpoint's 8192-wide residual stream, T in
# {1, 32, 256}. Sixteen is the fastest or within noise of it at every shape.
_RESIDUAL_NUM_WARPS = 16


def _fused_residual_enabled() -> bool:
    """Whether each decoder layer's residual join runs as one Triton kernel.

    On unless MUSESPARK_OPEN_FUSED_RESIDUAL names a falsy value, the convention
    every accepted lever in this file uses. ``_fused_preattention_enabled`` is
    the deliberate exception: it opts in, because its canary is red.

    The unfused chain stays reachable for the same reason it does there: it is
    this fusion's control leg. The two arms round identically at every step
    except the RMSNorm reduction order -- the kernel reproduces the gain product
    and the norm result's rounds to the activation dtype deliberately -- so a
    difference between them is attributable to that reduction and nothing else.

    Having the switch at all is what lets the A/B run on ONE image. Without it
    the only way to time the layer without this fusion is a second build, and
    two images differ by more than the lever under test.
    """
    return os.environ.get("MUSESPARK_OPEN_FUSED_RESIDUAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _sm107_fused_kernels_enabled() -> bool:
    """Whether the fused kernels this file measured on SM103 (GB300) also run on SM107 (VR).

    Off unless MUSESPARK_OPEN_SM107_FUSED_KERNELS names a truthy value. The
    kernels behind ``fused_kernels_supported`` run on SM107 but are not certified
    there, and two of them are on by default on SM103 (the cluster residual
    boundary and the TP4 fused finalize all-reduce): enabling them on SM107
    changes this model's default arithmetic on that GPU, so it is an opt-in.
    """
    return os.environ.get("MUSESPARK_OPEN_SM107_FUSED_KERNELS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def fused_kernels_supported(capability: Tuple[int, int]) -> bool:
    """Whether the SM103-measured fused kernels of this model run on ``capability``.

    SM103: yes. SM107: only with ``_sm107_fused_kernels_enabled``. Any other GPU:
    no. Read once where each kernel is set up (model construction, its workspace
    or availability check), never per step.
    """
    capability = tuple(capability)
    if capability == (10, 3):
        return True
    if capability == (10, 7):
        return _sm107_fused_kernels_enabled()
    return False


def _preqkv_norm_fp32_residual_enabled() -> bool:
    """Whether the pre-QKV prenorm reads the FP32 residual (cast after norm).

    Off unless MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL names a truthy value. This
    is an opt-in, unlike the accepted fusion levers, because it selects between
    two well-defined numerics orderings rather than fusing an unchanged one:
    set, the pre-QKV RMSNorm in later layers computes its variance on the FP32
    residual stream and rounds to the activation dtype once, on its output
    (the reference training recipe's ordering); unset, the residual is rounded
    to the activation dtype before the norm reads it (the historical serving
    ordering, applied uniformly to both prenorm boundaries). The pre-FFN
    prenorm is cast-first in both recipes and is not affected.

    The default stays on the historical side until the served numerics are
    confirmed to switch; flipping it is a bf16-grain change to every later
    layer's attention input. Both arms exist in one build so the A/B is a
    one-image environment flip, the same reason every other lever here keeps
    its control leg. Resolved at module construction, before warmup and any
    graph capture.
    """
    return os.environ.get("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _rmsnorm_fp32_in(x: torch.Tensor, norm: "RMSNorm") -> torch.Tensor:
    """Gemma RMSNorm over the FP32 residual, one round to the weight dtype.

    The unfused arm of the reference pre-QKV ordering. ``RMSNorm`` itself
    cannot serve it: the flashinfer branch only accepts fp16/bf16 inputs, and
    the torch branch returns the input dtype (fp32 here). The variance, the
    normalization, and the ``(1 + w)`` gain all evaluate in fp32 -- the same
    single-round chain as the fused kernels' NORM_FP32_RESIDUAL arm, so the
    arms differ only in reduction order.
    """
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + norm.variance_epsilon)
    return (normed * (norm.weight.to(torch.float32) + 1.0)).to(norm.weight.dtype)


def _routing_memo_enabled() -> bool:
    """Whether the second routing pass reuses the first one's result.

    On unless MUSESPARK_OPEN_ROUTING_MEMO names a falsy value, the convention this
    file already uses for ``MUSESPARK_OPEN_FUSED_RESIDUAL``.

    The recomputing path stays reachable because it is this change's control
    leg, and this one needs a control more than the fusions do, not less: the
    memo is declared BIT-EXACT -- it returns the tensors the second pass would
    have recomputed from the same logits -- and the only way to demonstrate
    bit-exactness is to run both arms and compare tokens. That comparison is
    valid only within a single process, because this build is not
    greedy-deterministic across server boots, so two images cannot answer it and
    a switch on one image is the only thing that can.
    """
    return os.environ.get("MUSESPARK_OPEN_ROUTING_MEMO", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _fused_finalize_ar_enabled() -> bool:
    """Whether the expert finalize may fuse with the TP4 all-reduce.

    On unless MUSESPARK_OPEN_FUSED_FINALIZE_AR names a falsy value, the convention
    every accepted lever in this file uses. The unfused finalize + AllReduce
    chain stays reachable as this fusion's control leg: the fused kernel is the
    most novel machinery in the file (IPC, Lamport signaling, epochs), and a
    one-image A/B is the only practical way to attribute a regression to it.
    Resolved at model construction, before warmup and any graph capture.
    """
    return os.environ.get("MUSESPARK_OPEN_FUSED_FINALIZE_AR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


#: The largest batch ``MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED`` serves with the
#: fused finalize + all-reduce: decode steps of up to 64 requests at eight rows
#: each. Larger batches (prefill chunks) keep the unfused finalize and NCCL.
_FINALIZE_AR_CHUNKED_MAX_TOKENS = 512


def _fused_finalize_ar_chunked_enabled() -> bool:
    """Whether batches of 129..512 tokens take the fused finalize + all-reduce in 128-token calls.

    Off unless MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED names a truthy value.
    The fused kernel serves at most 128 tokens per call; this runs it once per
    128-token chunk instead of falling back to the unfused finalize and the
    NCCL all-reduce. Each token's result is bitwise what the fused kernel
    returns for it at up to 128 tokens; against the unfused fallback it
    differs as the fused path does there (per-rank BF16 partials summed in
    FP32). Resolved at module construction, before warmup and any graph
    capture.
    """
    return os.environ.get("MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _large_finalize_sparse_enabled() -> bool:
    """Whether the unfused finalize of a large batch takes the sparse-slot kernel.

    Off unless MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE names a truthy value. It
    acts on ``MuseSparkOpenMoE.finalize`` calls of at least
    ``_SPARSE_FINALIZE_MIN_TOKENS`` tokens. Batches that the fused finalize +
    all-reduce takes (up to 128 tokens, or up to
    ``_FINALIZE_AR_CHUNKED_MAX_TOKENS`` with
    MUSESPARK_OPEN_FUSED_FINALIZE_AR_CHUNKED) never reach that finalize.
    Resolved at module construction, before warmup and any graph capture.
    """
    return os.environ.get("MUSESPARK_OPEN_LARGE_FINALIZE_SPARSE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


#: The largest batch whose router and pool projection run on two streams:
#: 128 tokens by default, decode steps of up to 64 requests at eight rows each
#: with ``MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP``.
_PREPARE_OVERLAP_MAX_TOKENS = 128
_LARGE_PREPARE_OVERLAP_MAX_TOKENS = 512


def _large_prepare_overlap_enabled() -> bool:
    """Whether batches of 129..512 tokens overlap the router with the pool projection.

    Off unless MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP names a truthy value. The
    overlap only places the same kernels on two streams, so its outputs are
    bitwise the serial ones. Resolved at module construction, before warmup
    and any graph capture.
    """
    return os.environ.get("MUSESPARK_OPEN_LARGE_PREPARE_OVERLAP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _fused_pre_expert_quant_enabled() -> bool:
    """Whether the pre-expert RMSNorm may fuse with NVFP4 quantization.

    On unless MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT names a falsy value, the
    convention every accepted lever in this file uses. The unfused
    pre_expert_norm path stays reachable as this fusion's control leg: the
    fused kernel quantizes the FP32 norm result without the intermediate BF16
    cast, a deliberate (slightly more precise) numerics change that a one-image
    A/B must be able to isolate. Resolved at module construction, before warmup
    and any graph capture.
    """
    return os.environ.get("MUSESPARK_OPEN_FUSED_PRE_EXPERT_QUANT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


@triton.jit
def _routing_weights_kernel(
    scores_ptr,
    indices_ptr,
    selected_ptr,
    weights_ptr,
    score_stride,
    eps,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK_K)
    mask = slots < TOP_K
    indices = tl.load(indices_ptr + row * TOP_K + slots, mask=mask, other=0)
    scores = tl.load(scores_ptr + row * score_stride + indices, mask=mask, other=0.0)
    denominator = tl.sum(scores, axis=0) + eps
    weights = tl.div_rn(scores, denominator)
    tl.store(selected_ptr + row * TOP_K + slots, indices.to(tl.int32), mask=mask)
    tl.store(weights_ptr + row * TOP_K + slots, weights, mask=mask)


def _routing_weights(
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather and normalize unbiased scores without changing top-k selection."""
    if (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.is_contiguous()
        and topk_idx.is_cuda
        and topk_idx.dtype == torch.int64
        and topk_idx.ndim == 2
        and topk_idx.is_contiguous()
        and topk_idx.shape[0] == scores.shape[0]
        and scores.shape[0] > 0
        and topk_idx.shape[1] == 8
        and output_dtype == torch.float32
    ):
        selected = torch.empty_like(topk_idx, dtype=torch.int32)
        weights = torch.empty_like(topk_idx, dtype=output_dtype)
        _routing_weights_kernel[(scores.shape[0],)](
            scores,
            topk_idx,
            selected,
            weights,
            scores.stride(0),
            eps,
            TOP_K=8,
            BLOCK_K=8,
            num_warps=1,
        )
        return selected, weights
    weights = scores.gather(1, topk_idx)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    return topk_idx.to(torch.int32), weights.to(output_dtype)


@triton.jit
def _router_projection_kernel(
    hidden_states,
    weight,
    output,
    scale,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, HIDDEN)
    x = tl.load(hidden_states + token * HIDDEN + offsets).to(tl.float32)
    w = tl.load(
        weight + expert[:, None] * HIDDEN + offsets[None, :],
        expert[:, None] < EXPERTS,
        other=0.0,
    )
    logits = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(output + token * EXPERTS + expert, logits, expert < EXPERTS)


@triton.jit
def _router_tensorcore_kernel(
    hidden_states,
    weight,
    partials,
    TOKENS: tl.constexpr,
):
    # Split the 8192-wide dot product into sixteen 512-element chunks.
    # TF32x3 retains FP32-weight precision while reusing each weight tile
    # across sixteen tokens. The second kernel sums the FP32 partials.
    tokens = tl.program_id(0) * 16 + tl.arange(0, 16)
    experts = tl.program_id(1) * 32 + tl.arange(0, 32)
    split = tl.program_id(2)
    acc = tl.zeros((16, 32), tl.float32)
    for block in range(4):
        hidden = split * 512 + block * 128 + tl.arange(0, 128)
        x = tl.load(
            hidden_states + tokens[:, None] * 8192 + hidden[None, :],
            tokens[:, None] < TOKENS,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight + experts[None, :] * 8192 + hidden[:, None])
        acc = tl.dot(x, w, acc, input_precision="tf32x3")
    tl.store(
        partials + split * TOKENS * 256 + tokens[:, None] * 256 + experts[None, :],
        acc,
        tokens[:, None] < TOKENS,
    )


@triton.jit
def _router_tensorcore_reduce_kernel(partials, output, scale, SIZE: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    splits = tl.arange(0, 16)
    values = tl.load(
        partials + splits[:, None] * SIZE + offsets[None, :],
        offsets[None, :] < SIZE,
        other=0.0,
    )
    tl.store(output + offsets, tl.sum(values, axis=0) * scale, offsets < SIZE)


@triton.jit
def _full_routing_kernel(
    logits,
    bias,
    selected,
    weights,
    eps,
    EXPERTS: tl.constexpr,
    TORCH_ORDER: tl.constexpr = False,
):
    # The (255 - expert) tie-break key below underflows past 256 experts; the
    # call site pins EXPERTS to 256, and this keeps the invariant explicit if
    # the kernel is ever generalized.
    tl.static_assert(EXPERTS <= 256)
    row = tl.program_id(0)
    expert = tl.arange(0, EXPERTS)
    scores = tl.sigmoid(tl.load(logits + row * EXPERTS + expert))
    ranked = scores + tl.load(bias + expert)
    # Treat signed zeros as equal before adding the expert-ID tie breaker.
    ranked = tl.where(ranked == 0.0, 0.0, ranked)
    bits = ranked.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | (255 - expert).to(tl.uint64)
    sorted_keys = tl.sort(keys, descending=True)
    ids = 255 - (sorted_keys & 0xFFFFFFFF).to(tl.int32)
    if TORCH_ORDER:
        # The installed CUDA topk(sorted=False) gathers values above the kth
        # threshold in expert-ID order, then fills threshold ties in ID order.
        # Preserve that accumulation order for the experimental large-T path.
        ranked_picked = tl.gather(ranked, ids, axis=0)
        threshold = tl.sum(tl.where(expert == 7, ranked_picked, 0.0), axis=0)
        gather_order = tl.where(
            expert < 8, ids + tl.where(ranked_picked > threshold, 0, 256), 512 + ids
        )
        ids = tl.sort(gather_order, descending=False) % 256
    if TORCH_ORDER:
        slots = tl.arange(0, 8)
        selected_ids = tl.gather(ids, slots, axis=0)
        picked = tl.gather(scores, selected_ids, axis=0)
        normalized = tl.div_rn(picked, tl.sum(picked, axis=0) + eps)
        tl.store(selected + row * 8 + slots, selected_ids)
        tl.store(weights + row * 8 + slots, normalized)
    else:
        picked = tl.gather(scores, ids, axis=0)
        picked = tl.where(expert < 8, picked, 0.0)
        normalized = tl.div_rn(picked, tl.sum(picked, axis=0) + eps)
        tl.store(selected + row * 8 + expert, ids, expert < 8)
        tl.store(weights + row * 8 + expert, normalized, expert < 8)


@triton.jit
def _attention_norm_gate_kernel(
    out,
    attention,
    gate,
    attention_stride,
    gate_token_stride,
    gate_head_stride,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(attention + token * attention_stride + head * HEAD_DIM + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    g = tl.load(gate + token * gate_token_stride + head * gate_head_stride + offsets).to(tl.float32)
    # The unfused BF16 sigmoid materializes before the multiply.
    sigmoid = tl.sigmoid(g).to(tl.bfloat16)
    result = normed.to(tl.float32) * sigmoid.to(tl.float32)
    tl.store(out + row * HEAD_DIM + offsets, result)


class MuseSparkOpenRoutingMethod(MiniMaxM2MoeRoutingMethod):
    """sigmoid -> add expert bias (selection only) -> top-k -> renormalize.

    Same algorithm as MiniMax-M2, with the source's renormalization epsilon
    (``router_score_division_eps``) instead of the MiniMax constant. Routing is
    forced onto the Python path so the kernel consumes exactly the top-k ids and
    weights the post-expert-norm finalize uses; letting the kernel route again
    would give it a different epsilon.
    """

    def __init__(
        self, *, top_k: int, num_experts: int, callable_e_score_correction_bias, renorm_eps: float
    ):
        super().__init__(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=callable_e_score_correction_bias,
        )
        self.renorm_eps = renorm_eps
        # One-slot memo for the duplicated routing pass; see ``apply``.
        # (logits tensor, its version or None, topk_idx, top_k_weights) or None.
        self._routing_memo = None
        # Read ONCE here rather than per call. A forward that consults the
        # environment is a forward whose behaviour can change between CUDA-graph
        # capture and replay, and the captured graph is what actually runs.
        self._memo_enabled = _routing_memo_enabled()
        # Experimental high-concurrency path: selection keeps the fused kernel's
        # lowest-ID tie policy and sorted accumulation order at larger batches.
        self._full_routing_all_batches = (
            os.getenv("MUSESPARK_OPEN_FULL_ROUTING_ALL_BATCHES", "0") == "1"
        )
        self._full_routing_torch_order = (
            os.getenv("MUSESPARK_OPEN_FULL_ROUTING_TORCH_ORDER", "0") == "1"
        )

    @property
    def requires_separated_routing(self) -> bool:
        return True

    def clear_routing_memo(self) -> None:
        """Drop the memo so it does not pin device buffers past the layer."""
        self._routing_memo = None

    def apply(self, router_logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        """Route, memoized on the identity of the logits tensor.

        ``MuseSparkOpenMoE.forward`` routes the same logits twice per layer: once
        inside the MoE backend (``moe/fused_moe/moe_scheduler.py:401``, reached
        because ``requires_separated_routing`` is True) and once again in the
        model, because the backend's ``expert_weights`` output is left
        uninitialized when ``topk_weights`` are supplied to it -- it is allocated
        with ``at::detail::empty_cuda`` and then orphaned
        (``cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:234,274,500``), so reading
        it back would be reading uninitialized device memory. The second pass is
        a sigmoid, a top-k, a gather, a sum, a divide and a cast over tiny
        ``[T, num_experts]`` and ``[T, top_k]`` tensors: about seven
        launch-latency-bound kernels per layer. The profile shows the duplication
        directly -- ``_scatter_gather_elementwise_kernel`` appears 124 times per
        decode iteration across 62 layers, exactly twice each.

        Keying on the tensor OBJECT (``is``, while holding a reference to it)
        rather than on ``id()`` is deliberate: an id can be reused once a tensor
        is freed, and holding the reference makes that impossible. The version
        counter, where one exists, additionally catches an in-place write to the
        same object between the two calls.

        Three source facts make the memo safe, each checked rather than assumed:
        the backend-side call at ``:543`` always runs before the model-side one
        at ``:567``; on the single-chunk, unpadded path the tensor reaches the
        scheduler unmodified; and the second call today recomputes bit-for-bit
        what the first produced. The three places that replace the tensor object
        -- the zero-token DP pad ``torch.cat``, the multi-chunk ``split`` and the
        synthetic perfect-router logits -- all miss the memo and recompute, which
        is exactly today's behaviour.
        """
        version = _version_or_none(router_logits)
        memo = self._routing_memo if self._memo_enabled else None
        if memo is not None and memo[0] is router_logits and memo[1] == version:
            return memo[2], memo[3]
        bias = self.e_score_correction_bias
        if (
            router_logits.is_cuda
            and router_logits.dtype == torch.float32
            and router_logits.ndim == 2
            and router_logits.is_contiguous()
            and router_logits.shape[0] > 0
            and (router_logits.shape[0] <= 128 or self._full_routing_all_batches)
            and router_logits.shape[1] == 256
            and self.top_k == 8
            and self.output_dtype == torch.float32
            and bias is not None
            and bias.is_cuda
            and bias.dtype == torch.float32
            and bias.shape == (256,)
            and bias.is_contiguous()
        ):
            # Selection is sorted and breaks exact ties by lowest expert ID.
            # Unlike torch.topk(sorted=False), this fixes expert accumulation order.
            selected = torch.empty(
                (router_logits.shape[0], 8), device=router_logits.device, dtype=torch.int32
            )
            weights = torch.empty_like(selected, dtype=torch.float32)
            _full_routing_kernel[(router_logits.shape[0],)](
                router_logits,
                bias,
                selected,
                weights,
                self.renorm_eps,
                EXPERTS=256,
                TORCH_ORDER=self._full_routing_torch_order and router_logits.shape[0] > 128,
                num_warps=1 if router_logits.shape[0] >= 512 else 4,
            )
            self._routing_memo = (router_logits, version, selected, weights)
            return selected, weights
        scores, scores_with_bias = self.get_scores(router_logits, self.e_score_correction_bias)
        _, topk_idx = torch.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=False)
        # Weights come from the *unbiased* scores; the bias only steers which
        # experts are picked.
        selected, weights = _routing_weights(scores, topk_idx, self.renorm_eps, self.output_dtype)
        self._routing_memo = (router_logits, version, selected, weights)
        return selected, weights


def _frozen_gate_cache_enabled() -> bool:
    """Whether the frozen-parameter derived constants are cached at weight load.

    On unless MUSESPARK_OPEN_FROZEN_GATE_CACHE names a falsy value, the same
    convention as this file's other switches.

    Falsy restores the pre-cache behaviour exactly: every call recomputes
    ``1 + w`` for the weight-first norm and the sigmoid/sigmoid/multiply/clamp/
    sqrt chain for the gated residual's coefficients. That path is not dead
    code kept for sentiment -- it is the control leg for a change declared
    BIT-EXACT, and bit-exactness can only be shown by running both arms in ONE
    process and comparing tokens, because this build is not greedy-deterministic
    across server boots. Two images cannot answer it; a switch on one image can.
    """
    return os.environ.get("MUSESPARK_OPEN_FROZEN_GATE_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class MuseSparkOpenRMSNormWeightFirst(nn.Module):
    """``rmsnorm((1 + w) * x, eps)`` with no output gain.

    The source calls this the "input scale" norm form; it is selected by
    ``use_ffn_learnable_scales`` for ``post_expert_norm`` and ``post_ffn_norm``
    only. It is not the same function as ``RMSNorm(use_gemma=True)``, which
    applies ``(1 + w)`` *after* normalizing.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: Optional[torch.dtype]):
        super().__init__()
        # Zero-centered gamma: the stored value is an offset from 1.0.
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps
        # ``weight + 1`` over a frozen parameter, cached by
        # cache_derived_state() after the weights land. Non-persistent: it
        # is derived from ``weight`` and must not enter a checkpoint.
        self.register_buffer("_weight_plus_one", None, persistent=False)
        # Read once, not per call: a forward that consults the environment can
        # behave differently under CUDA-graph capture than under replay.
        self._cache_enabled = _frozen_gate_cache_enabled()

    def cache_derived_state(self) -> None:
        """Framework post-load stage; see MuseSparkOpenGatedResidual for why."""
        if self.weight.is_meta:
            return
        self._weight_plus_one = self.weight + 1

    def post_load_weights(self) -> None:
        """Delegate to the cache stage; see MuseSparkOpenGatedResidual for why."""
        self.cache_derived_state()

    def gain(self) -> torch.Tensor:
        """``1 + w``, cached where it has been refreshed."""
        if self._weight_plus_one is None or not self._cache_enabled:
            return self.weight + 1
        return self._weight_plus_one

    def _apply(self, *args, **kwargs):
        # nn.Module funnels .to()/.cuda()/.float()/.half() through _apply, and it
        # would carry the cached buffer along with the parameter. That is wrong
        # for a dtype move: casting ``w + 1`` is not the same as computing
        # ``cast(w) + 1``, so the cache would quietly stop matching the
        # expression it stands for. Drop it instead; the forward below recomputes
        # per call until cache_derived_state() runs again, which is exactly
        # the pre-cache behaviour.
        out = super()._apply(*args, **kwargs)
        self._weight_plus_one = None
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Round the scaled input to the activation dtype before the variance,
        # matching the reference's fused path.
        scaled = self.gain() * hidden_states
        scaled_fp32 = scaled.to(torch.float32)
        variance = scaled_fp32.pow(2).mean(-1, keepdim=True)
        return (scaled_fp32 * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)


class MuseSparkOpenGatedResidual(nn.Module):
    """Norm-preserving gated residual add, evaluated in FP32.

    ``h' = alpha * h + beta * y`` with ``beta = sigmoid(g / 0.3)`` and
    ``alpha = sqrt(max(sigmoid(-g / 0.3) * (1 + beta), 1e-3))``. Both branches
    are gated (this is not ``h + g * y``), which keeps ``alpha^2 + beta^2 == 1``
    away from the clamp.
    """

    TEMPERATURE = 0.3
    EPS = 1e-3

    def __init__(self, hidden_size: int, dtype: Optional[torch.dtype]):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype), requires_grad=False)
        # alpha/beta depend only on the frozen gate, so they are constant for
        # the life of a loaded engine; cache_derived_state() fills these in
        # once the weights are in. Non-persistent: derived, never checkpointed.
        self.register_buffer("_alpha", None, persistent=False)
        self.register_buffer("_beta", None, persistent=False)
        # Same switch, same reason, as MuseSparkOpenRMSNormWeightFirst: falsy
        # restores the per-call recompute this cache replaced, so the bit-exact
        # claim has a control leg inside one process.
        self._cache_enabled = _frozen_gate_cache_enabled()
        # Read once here, not per call, which is how MuseSparkOpenAttention takes
        # its own switch (``self._fuse_qk_norm_rope_gate``). Per call it would
        # be an environment lookup twice per layer per step, and -- worse -- a
        # value that could change under a captured CUDA graph, where the arm
        # recorded at capture is the one that replays no matter what the
        # variable says afterwards. Fixing it at construction makes the engine's
        # choice a property of the engine.
        self._fused_enabled = _fused_residual_enabled()
        # Same construction-time read, for the same graph-capture reason: the
        # ordering of the fused next prenorm must not change between capture
        # and replay.
        self._preqkv_norm_fp32 = _preqkv_norm_fp32_residual_enabled()

    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = self.gate.to(torch.float32) / self.TEMPERATURE
        beta = torch.sigmoid(scaled)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), min=self.EPS))
        return alpha, beta

    def cache_derived_state(self) -> None:
        """Fill the frozen-gate cache. This is the loader's post-load stage.

        The name is the framework contract, not ours: ``ModelLoader``'s
        ``_walk_cache_state`` visits ``model.modules()`` and calls
        ``cache_derived_state()`` on every module that defines it
        (``pyexecutor/model_loader.py``). That walk is how a GPU-memory-service
        READ-ONLY worker gets here at all -- it never calls the MuseSparkOpen
        weight mapper, it zero-copy imports weights another rank committed, runs
        ``setup_aliases()``, materializes, then this stage, and explicitly does
        NOT run ``transform_weights`` (``memory/gpu_memory_backend.py`` says in
        as many words that derived Python state has to live in this stage to
        reach RO readers). Keyed only off the mapper, the cache stayed ``None``
        on every RO rank and the lever silently degraded to the per-call
        recomputation it exists to remove -- no error, just the old kernel count
        back on some ranks and not others.

        Returns early on a meta parameter: both walks can reach a module before
        its weights are real, and ``_compute_coefficients`` on a meta tensor
        would cache meta tensors that the forward then reads as if they were
        values. ``MuseSparkOpenRMSNormWeightFirst`` guards the same way.

        Idempotent, as the stage requires: it recomputes from the live parameter
        every time.
        """
        if self.gate.is_meta:
            return
        self._alpha, self._beta = self._compute_coefficients()

    def post_load_weights(self) -> None:
        """Delegate to the cache stage so the *normal* load path reaches it.

        There are two post-load walks and they look up different names.
        ``_walk_cache_state`` (used by the RO-reader and staged-receiver paths)
        looks up ``cache_derived_state``; ``_walk_full_post_load`` -- the branch
        the ordinary load takes -- looks up ``post_load_weights`` and skips any
        module that lacks it. A leaf module implementing only the cache stage is
        therefore invisible to the common path, which is where this cache
        actually has to be warm.

        Implementing both is the idiom the repo already uses for a leaf module
        in this position (``modeling_llama_min_latency.py``), so this adds no
        new convention. Nothing here depends on another module's weights -- the
        cache is a function of this module's own frozen parameter -- so the
        walk's arbitrary module order is not a hazard.
        """
        self.cache_derived_state()

    def _apply(self, *args, **kwargs):
        # See MuseSparkOpenRMSNormWeightFirst._apply. This case is the sharper of
        # the two: alpha and beta are FP32 by construction whatever dtype the
        # module is in, because _compute_coefficients upcasts the gate. A
        # .to(bfloat16) after the cache was filled would carry them down to
        # bf16, and coefficients() would then return something the uncached path
        # never produces. Drop the cache on any move.
        out = super()._apply(*args, **kwargs)
        self._alpha = None
        self._beta = None
        return out

    def coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._alpha is None or not self._cache_enabled:
            return self._compute_coefficients()
        return self._alpha, self._beta

    def forward(self, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.coefficients()
        return alpha * residual + beta * branch.to(torch.float32)

    def _add_after_rmsnorm_and_norm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        next_norm: RMSNorm,
        gain: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse a residual join with the next Gemma prenorm.

        Retain the branch-normalization and residual-to-BF16 rounding points.
        The next variance reduction may differ from FlashInfer's reduction order.
        The residual-to-BF16 round feeding the next prenorm is itself switched
        by ``_preqkv_norm_fp32`` (see ``_preqkv_norm_fp32_residual_enabled``):
        set, all three arms norm the FP32 residual and round once on the output.
        """
        if not (
            self._fused_enabled
            and residual.is_cuda
            and residual.dtype == torch.float32
            and branch.dtype == torch.bfloat16
            and residual.ndim == 2
            and residual.shape == branch.shape
            and residual.shape[0] > 0
            and residual.is_contiguous()
            and branch.is_contiguous()
            and next_norm.use_gemma
            and next_norm.weight.dtype == branch.dtype
            and next_norm.weight.is_contiguous()
            and (gain is None or (gain.dtype == branch.dtype and gain.is_contiguous()))
        ):
            updated = self.add_after_rmsnorm(residual, branch, eps, gain)
            if self._preqkv_norm_fp32:
                return updated, _rmsnorm_fp32_in(updated, next_norm)
            return updated, next_norm(updated.to(branch.dtype))

        alpha, beta = self.coefficients()
        if (
            not torch.compiler.is_compiling()
            and residual.shape[1] == 8192
            and residual.shape[0] <= 32
        ):
            from .musespark_open_boundary import cluster_boundary, cluster_boundary_supported

            if cluster_boundary_supported(residual, branch, gain, alpha, beta, next_norm.weight):
                return cluster_boundary(
                    residual,
                    branch,
                    gain,
                    alpha,
                    beta,
                    next_norm.weight,
                    eps,
                    next_norm.variance_epsilon,
                    norm_fp32_residual=self._preqkv_norm_fp32,
                )
        updated = torch.empty_like(residual)
        next_input = torch.empty_like(branch)
        hidden_size = residual.shape[-1]
        _gated_residual_after_rmsnorm_kernel[(residual.shape[0],)](
            updated,
            residual,
            branch,
            gain if gain is not None else alpha,
            alpha,
            beta,
            eps,
            next_input,
            next_norm.weight,
            next_norm.variance_epsilon,
            HIDDEN=hidden_size,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            HAS_GAIN=gain is not None,
            FUSE_NEXT_NORM=True,
            NORM_FP32_RESIDUAL=self._preqkv_norm_fp32,
            num_warps=_RESIDUAL_NUM_WARPS,
        )
        return updated, next_input

    def add_after_rmsnorm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``self(residual, rmsnorm(branch))``, as one kernel on CUDA.

        A decoder layer joins its two branches this way, so the pair runs 124
        times per forward over the 8192-wide FP32 residual stream: seventeen
        elementwise launches per layer, and the residual makes a round trip
        through HBM between the norm and the add. ``gain`` selects which norm --
        ``1 + w`` for the weight-first FFN one, ``None`` for the weightless
        attention one.

        Everywhere but CUDA -- and anywhere MUSESPARK_OPEN_FUSED_RESIDUAL is set
        falsy, which is how the A/B gets both arms out of one image -- this is
        the composition it replaces, spelled out, so the two can be compared
        directly. ROUNDING-DIFFERENT and declared so
        before it was measured: only the variance reduction order changes, at
        ~9e-08 relative, because the kernel reproduces both of the chain's
        intermediate rounds to the activation dtype.
        """
        alpha, beta = self.coefficients()
        num_tokens = residual.shape[0]
        hidden_size = residual.shape[-1]
        # A zero-token batch would launch an empty grid; a non-contiguous input
        # breaks the kernel's single flat index. Both fall back rather than
        # assert, because the composition below already handles them.
        #
        # The dtype clause is the third of the same kind. The kernel rounds the
        # gain product to ``branch.dtype`` because that is what the composition
        # does when the gain and the activation share a dtype, which is the only
        # way this model builds them: ``gain`` is ``weight + 1`` over a parameter
        # created in the module dtype. If they ever differed, torch would promote
        # ``gain * branch`` to the wider type and skip that round while the kernel
        # still took it, and the two would silently compute different functions --
        # measured at 5.1e-04 relative for an FP32 gain against an FP16 branch.
        # Falling back keeps the fused and unfused paths the same function by
        # construction, at the cost of one dtype comparison per call.
        on_cuda = residual.is_cuda
        residual_contiguous = residual.is_contiguous()
        branch_contiguous = branch.is_contiguous()
        gain_dtype_ok = gain is None or gain.dtype == branch.dtype
        if num_tokens > 0:
            # Not for num_tokens == 0: that is the routine pad, and reporting it
            # would bury the conditions that actually mean something.
            #
            # The switch is reported alongside the involuntary conditions rather
            # than short-circuited quietly above them, because a run that was
            # left with MUSESPARK_OPEN_FUSED_RESIDUAL=0 set is exactly the false
            # negative _note_unfused_fallback exists to catch: correct output,
            # no gain, and the lever priced at zero for the wrong reason.
            _note_unfused_fallback(
                "gated-residual join",
                fused_residual_enabled=self._fused_enabled,
                residual_on_cuda=on_cuda,
                residual_contiguous=residual_contiguous,
                branch_contiguous=branch_contiguous,
                gain_dtype_matches_branch=gain_dtype_ok,
            )
        if not (
            self._fused_enabled
            and on_cuda
            and num_tokens > 0
            and residual_contiguous
            and branch_contiguous
            and gain_dtype_ok
        ):
            scaled = (gain * branch if gain is not None else branch).to(torch.float32)
            normed = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(
                branch.dtype
            )
            return alpha * residual + beta * normed.to(torch.float32)

        out = torch.empty_like(residual, dtype=torch.float32)
        _gated_residual_after_rmsnorm_kernel[(num_tokens,)](
            out,
            residual,
            branch,
            # Unread when HAS_GAIN is False, but Triton still needs a tensor
            # rather than None for a pointer argument.
            gain if gain is not None else alpha,
            alpha,
            beta,
            eps,
            out,
            alpha,
            eps,
            HIDDEN=hidden_size,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            HAS_GAIN=gain is not None,
            FUSE_NEXT_NORM=False,
            NORM_FP32_RESIDUAL=False,
            num_warps=_RESIDUAL_NUM_WARPS,
        )
        return out


class MuseSparkOpenLogitsProcessor(LogitsProcessor):
    """LM head epilogue: metap multiplier then a tanh soft cap, both FP32."""

    def __init__(self, logit_multiplier: float, soft_cap: float):
        super().__init__()
        self.logit_multiplier = logit_multiplier
        self.soft_cap = soft_cap

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        attn_metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        logits = super().forward(hidden_states, lm_head, attn_metadata, return_context_logits)
        logits = logits.float() * self.logit_multiplier
        return self.soft_cap * torch.tanh(logits / self.soft_cap)


def _attention_allreduce_strategy() -> AllReduceStrategy:
    """Strategy for the attention output projection's all-reduce.

    NCCL unless MUSESPARK_OPEN_ALLREDUCE names another member of the enum. An
    unknown name keeps the pin rather than failing engine construction: the
    variable exists for attribution legs, and a typo in one must not decide
    correctness.
    """
    name = os.environ.get("MUSESPARK_OPEN_ALLREDUCE", "NCCL").strip().upper() or "NCCL"
    try:
        return AllReduceStrategy[name]
    except KeyError:
        logger.warning(
            f"MUSESPARK_OPEN_ALLREDUCE={name!r} is not an all-reduce strategy; "
            f"keeping NCCL on the attention output projection"
        )
        return AllReduceStrategy.NCCL


def _fused_preattention_enabled() -> bool:
    """Whether attention preprocessing runs as one Triton kernel.

    OFF unless MUSESPARK_OPEN_FUSED_PREATTN names a truthy value. This is the one
    switch in this file that opts in rather than out, and the asymmetry is the
    point. Two independent multi-turn canaries on this fusion's own image fail
    the structural check that a streamed answer and an unstreamed one agree:
    at the same serial turn the streamed text is a 105-character prefix of the
    106-character unstreamed one, with a 0.3225 logprob margin at the first
    divergent token, and the two samples agree to the byte. No build without
    this fusion reproduces it. Until that truncation is traced to either the
    kernel's last-token store or the streaming path, a caller who asks for
    nothing must not get the fused chain.

    The path stays reachable because the question is still open and answering it
    needs the fusion on: it is the treatment leg against the unfused control,
    and the two arms round identically at every step except the RMSNorm
    reduction order and the output gate's sigmoid, so a difference between them
    is attributable to exactly those two.

    An unrecognized value reads as off. The on-by-default switches spell this
    the other way round because there a typo costs throughput; here it would
    hand back the defect.
    """
    return os.environ.get("MUSESPARK_OPEN_FUSED_PREATTN", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class MuseSparkOpenAttention(QKNormRoPEAttention):
    """Attention with weightless per-head Q/K norms, a query pre-scale, an
    output norm+gate fused into the QKV projection, and a per-layer scope.

    ``QKNormRoPEAttention(is_qk_norm=False)`` gives exactly the weightless
    per-head RMSNorm the source uses (``RMSNorm(has_weights=False)`` over
    ``head_dim``); the query pre-scale and the output norm are layered on top.
    """

    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig], layer_idx: int):
        config = model_config.pretrained_config
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.uses_rope = config.use_rope_per_layer[layer_idx]
        self.attention_window_size = config.sliding_window if self.is_sliding else None
        self.scale_query_by = config.scale_query_by

        if self.uses_rope:
            rope_params = RopeParams(
                dim=config.head_dim,
                theta=config.rope_theta,
                scale_type=RotaryScalingType.none,
                scale=1.0,
                max_positions=config.max_position_embeddings,
            )
            # GPT-J / interleaved rotation over adjacent channel pairs.
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=rope_params,
                is_neox=False,
            )
        else:
            pos_embd_params = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            skip_rope=not self.uses_rope,
            # The C++ fused qk-norm+RoPE op is not taken: the Triton
            # preprocessing kernel selected below covers the same work plus the
            # gate deinterleave and the query pre-scale, in one pass.
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            # The kernel keeps its plain 1/sqrt(head_dim); the source's extra
            # query multiplier is applied explicitly in apply_qk_norm so its
            # rounding stays observable.
            q_scaling=1.0,
            attn_output_gate=True,
            is_qk_norm=False,
        )
        # One Triton kernel for everything between the QKV projection and the
        # attention call: gate deinterleave, the two weightless per-head
        # RMSNorms, the query pre-scale, and the interleaved rotation. It
        # replaces the chunk-and-reshape copies, two norms, one elementwise
        # multiply and the in-place RoPE launch that the generic path spends
        # per layer, and it is what makes `apply_qk_norm` below unreachable on
        # the fused arm. `apply_output_gate` picks up the fused sigmoid-mul
        # from the same switch.
        self._fuse_qk_norm_rope_gate = _fused_preattention_enabled()
        self._qkv_weight_layout = os.getenv("MUSESPARK_OPEN_LL_QKV_WEIGHT_LAYOUT", "0") == "1"
        self.register_buffer("_qkv_layout_inverse", None, persistent=False)
        self._large_norm_gate_enabled = os.getenv("MUSESPARK_OPEN_LARGE_NORM_GATE", "0") == "1"
        self._sp_projection_stream: Optional[torch.cuda.Stream] = None
        self.fused_prep_q_scale = self.scale_query_by
        # Weightless post-attention norm applied after o_proj, at the
        # post-norm epsilon.
        self.o_head_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        # An explicit strategy for the row-parallel output projection's
        # all-reduce, for the same reason the MoE one above is pinned but a
        # different failure. `o_proj` takes its strategy from the model config,
        # which defaults to AUTO. At TP4 on one node a decode step reduces a
        # tiny message, and AUTO sends tiny messages down the NCCL
        # symmetric-memory path, which reuses a shared buffer across launches
        # without synchronising: ranks overwrite each other's payload and the
        # layer output comes back non-finite. Downstream that is the token-0
        # collapse -- argmax over an all-NaN row is index 0 -- which is why it
        # arrives on all four ranks at once, at the same engine iteration for
        # every request in flight, and only when the batch (and so the message)
        # is small.
        #
        # Measured on the concurrency gate, one variable, same node, same
        # session: AUTO failed with non-finite decode steps on nearly every
        # prompt, NCCL passed. Candidate identified by an independent
        # investigation; the symmetric-memory corruption itself is
        # filed as a follow-up rather than fixed here.
        #
        # MUSESPARK_OPEN_ALLREDUCE names a different strategy for one run --
        # `AUTO` restores the racy default, which is what a control leg needs
        # in order to reproduce the fault at all.
        if self.o_proj.all_reduce is not None:
            self.o_proj.all_reduce = AllReduce(
                mapping=self.o_proj.mapping,
                strategy=_attention_allreduce_strategy(),
                dtype=self.o_proj.dtype,
            )

    def prepare_qkv_weight_layout(self):
        if getattr(self, "_qkv_weight_layout", False):
            from .musespark_open_qkv_layout import prepare_weight_layout

            self._qkv_layout_inverse = prepare_weight_layout(self)

    def preprocess_qkv(self, qkv, position_ids):
        inverse = getattr(self, "_qkv_layout_inverse", None)
        if inverse is not None:
            if (
                not torch.compiler.is_compiling()
                and qkv.is_cuda
                and qkv.dtype == torch.bfloat16
                and qkv.shape == (1, 2 * (self.q_size + self.kv_size))
                and qkv.stride(-1) == 1
                and self.support_fused_qkv
            ):
                from .musespark_open_qkv_layout import preprocess_one_row

                return preprocess_one_row(self, qkv, position_ids)
            # Larger batches/prefill retain native arithmetic and dispatch.
            # Weights stay permuted, so reconstruct the original projection.
            qkv = qkv.index_select(-1, inverse)
        return super().preprocess_qkv(qkv, position_ids)

    def apply_qk_norm(self, q, k):
        q, k = super().apply_qk_norm(q, k)
        q = q * self.scale_query_by
        return q, k

    def apply_output_gate(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Weightless per-head RMSNorm on the attention result, then the gate.

        Order matters: the source normalizes each head of the attention output
        before multiplying by ``sigmoid(o_gate)``, and only then projects.
        """
        if (
            attention_output.is_cuda
            and attention_output.dtype == torch.bfloat16
            and attention_output.ndim == 2
            and attention_output.stride(-1) == 1
            and (
                0 < attention_output.shape[0] <= 128
                or (
                    getattr(self, "_large_norm_gate_enabled", False)
                    and attention_output.shape[0] > 128
                    and not torch.compiler.is_compiling()
                    and attention_output.stride(0) % 8 == 0
                    and attention_output.data_ptr() % 16 == 0
                    and gate.ndim in (2, 3)
                    and gate.data_ptr() % 16 == 0
                    and gate.stride(0) % 8 == 0
                    and (gate.ndim != 3 or gate.stride(1) % 8 == 0)
                )
            )
            and self.head_dim == 64
            and attention_output.shape[1] % self.head_dim == 0
            and gate.is_cuda
            and gate.dtype == torch.bfloat16
            and gate.ndim in (2, 3)
            and gate.stride(-1) == 1
            and gate.shape[0] == attention_output.shape[0]
            and gate.numel() == attention_output.numel()
            and (gate.ndim == 2 or gate.shape[-1] == self.head_dim)
        ):
            if attention_output.shape[0] > 128:
                # Bitwise the unfused norm and gate below, at any token count.
                from ..cute_dsl_kernels.blackwell.rmsnorm_gate import rmsnorm_gate

                return rmsnorm_gate(attention_output, gate, self.o_head_norm.variance_epsilon)
            output = torch.empty_like(attention_output, memory_format=torch.contiguous_format)
            heads = attention_output.shape[1] // self.head_dim
            _attention_norm_gate_kernel[(attention_output.shape[0] * heads,)](
                output,
                attention_output,
                gate,
                attention_output.stride(0),
                gate.stride(0),
                gate.stride(1) if gate.ndim == 3 else self.head_dim,
                self.o_head_norm.variance_epsilon,
                HEADS=heads,
                HEAD_DIM=self.head_dim,
                num_warps=4,
            )
            return output
        normed = self.o_head_norm(attention_output.reshape(-1, self.head_dim)).reshape(
            attention_output.shape
        )
        return super().apply_output_gate(normed, gate)

    def _pad_sp_output_projection(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Pad the narrower input, preserving the ordinary projection and reduction."""
        padded = _pad_sequence_parallel_tokens(attention_output, self.mapping.tp_size)
        partial = self.o_proj(padded, all_reduce_params=AllReduceParams(enable_allreduce=False))
        return reducescatter(partial, self.mapping, dim=0)

    def _overlap_sp_output_projection(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Project two chunks while reducing the preceding chunk to its token owners."""
        main = torch.cuda.current_stream(attention_output.device)
        if self._sp_projection_stream is None:
            self._sp_projection_stream = torch.cuda.Stream(device=attention_output.device)
        stream = self._sp_projection_stream
        padded = _pad_sequence_parallel_tokens(attention_output, self.mapping.tp_size)
        shard = padded.shape[0] // self.mapping.tp_size
        split = (shard + 1) // 2
        results = []
        partials = []
        for begin, end in ((0, split), (split, shard)):
            # Each collective must see owner-major rows. Splitting the original
            # tensor in half would send rows to the wrong rank.
            packed = torch.cat(
                [
                    padded[rank * shard + begin : rank * shard + end]
                    for rank in range(self.mapping.tp_size)
                ],
                dim=0,
            )
            partial = self.o_proj(packed, all_reduce_params=AllReduceParams(enable_allreduce=False))
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                reduced = reducescatter(partial, self.mapping, dim=0)
            # Keep main-stream allocations alive through the final join. Their
            # next reuse on main is then ordered after the collective, without
            # record_stream delaying allocator reuse until CPU event polling.
            partials.append(partial)
            results.append(reduced)
        main.wait_stream(stream)
        output = torch.cat(results, dim=0)
        # Results were allocated on stream but consumed by cat on main. Order
        # their next reuse on their allocating stream after that consumption.
        stream.wait_stream(main)
        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        overlap_sp_output_projection: bool = False,
        pad_sp_output_projection: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if overlap_sp_output_projection:
            kwargs = dict(kwargs, output_projection=self._overlap_sp_output_projection)
        elif pad_sp_output_projection:
            kwargs = dict(kwargs, output_projection=self._pad_sp_output_projection)
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


def _pad_sequence_parallel_tokens(tensor: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Pad the token dimension for equal-size TP shards, preserving real rows."""
    padding = -tensor.shape[0] % tp_size
    if padding == 0:
        return tensor
    return torch.cat((tensor, tensor.new_zeros((padding, *tensor.shape[1:]))), dim=0)


class MuseSparkOpenMoE(nn.Module):
    """Router + pooled experts + per-expert post-norm combine + unpooler.

    Order (source ``feed_forward``): route on the 8192-wide pre-FFN activation,
    pool it to 4096, pre-expert norm, run top-8 experts, apply the weight-first
    post-expert norm to *each* expert output before the weighted sum, all-reduce
    the partial sums across expert-parallel ranks, then unpool back to 8192.

    The post-expert norm is why the fused kernel runs with ``do_finalize=False``:
    RMSNorm is nonlinear, so normalizing the already-combined result is a
    different model.
    """

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.expert_hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.metap_multiplier = config.metap_multiplier
        self.mapping = model_config.mapping
        self._large_finalize_4warps = os.getenv("MUSESPARK_OPEN_LARGE_FINALIZE_4WARPS", "0") == "1"
        self._large_finalize_sparse = _large_finalize_sparse_enabled()
        self._finalize_ar_max_tokens = (
            _FINALIZE_AR_CHUNKED_MAX_TOKENS if _fused_finalize_ar_chunked_enabled() else 128
        )
        self._prepare_overlap_max_tokens = (
            _LARGE_PREPARE_OVERLAP_MAX_TOKENS
            if _large_prepare_overlap_enabled()
            else _PREPARE_OVERLAP_MAX_TOKENS
        )
        self._bf16x3_router_enabled = (
            os.getenv("MUSESPARK_OPEN_BF16X3_ROUTER", "0") == "1"
            and torch.cuda.is_available()
            and fused_kernels_supported(torch.cuda.get_device_capability())
        )
        # EP reduces normalized local experts; pure TP must instead reduce
        # each expert's FC2 partials before its nonlinear normalization.
        pure_ep = self.mapping.moe_tp_size == 1 and (
            self.mapping.moe_ep_size == self.mapping.tp_size
        )
        pure_tp = (
            self.mapping.moe_ep_size == 1
            and (self.mapping.moe_tp_size == self.mapping.tp_size)
            and not self.mapping.enable_attention_dp
        )
        if not (pure_ep or pure_tp):
            raise ValueError(
                "MuseSparkOpen requires MoE EP == attention TP with MoE TP == 1, "
                "or pure MoE TP without attention DP, got "
                f"moe_tp_size={self.mapping.moe_tp_size}, "
                f"moe_ep_size={self.mapping.moe_ep_size}, "
                f"tp_size={self.mapping.tp_size}, "
                f"enable_attention_dp={self.mapping.enable_attention_dp}"
            )

        # Router: unquantized FP32 linear, replicated on every rank.
        self.router = Linear(
            self.hidden_size, self.num_experts, bias=False, dtype=torch.float32, quant_config=None
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )

        # Pooler / unpooler stay replicated and unquantized: NVFP4 PTQ keeps
        # them in original precision, and replication avoids an allgather.
        self.pooler = Linear(
            self.hidden_size,
            self.expert_hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
        )
        self.unpooler = Linear(
            self.expert_hidden_size,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
        )

        self.pre_expert_norm = RMSNorm(
            hidden_size=self.expert_hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self._prepare_stream = aux_stream
        self.post_expert_norm_eps = config.rms_post_norm_eps
        self.register_buffer("_pre_expert_quant_gain", None, persistent=False)
        # Resolved once, before warmup and any graph capture.
        self._pre_expert_quant_fused = _fused_pre_expert_quant_enabled()
        # Weight-first post-expert norm, applied per selected expert output.
        self.post_expert_norm_weight = nn.Parameter(
            torch.zeros(self.expert_hidden_size, dtype=config.torch_dtype), requires_grad=False
        )

        aux_stream_dict = (
            {AuxStreamType.MoeChunkingOverlap: aux_stream} if aux_stream is not None else None
        )
        # The checkpoint may quant-exclude whole layers ("model.layers.N." in
        # exclude_modules), leaving their experts BF16. Backend resolution
        # only sees the global quant config, so without help an excluded layer
        # is resolved to an NVFP4 leaf and then fails that leaf's format check
        # once the post-init exclusion pass rewrites its quant_config (the
        # leaves carry one format each; there is no construct-then-requant
        # path any more). Hand resolution the format the layer actually has,
        # so it lands on the TRTLLM-Gen BF16 leaf, which honors
        # ``do_finalize=False`` -- the per-expert norm needs the unfinalized
        # triple on every layer, and the generic BF16 fallback
        # (CutlassFusedMoE) does not provide it. The MEGAMOE families have no
        # BF16 member at all, so those requests also drop to TRTLLM here.
        layer_model_config = model_config
        layer_quant_config = None
        global_quant_config = model_config.quant_config
        if (
            global_quant_config is not None
            and global_quant_config.exclude_modules
            and global_quant_config.is_module_excluded_from_quantization(
                f"model.layers.{layer_idx}"
            )
        ):
            layer_quant_config = QuantConfig(
                kv_cache_quant_algo=global_quant_config.kv_cache_quant_algo
            )
            if str(getattr(model_config, "moe_backend", "")).upper() in (
                "MEGAMOE_CUTEDSL",
                "MEGAMOE_DEEPGEMM",
            ):
                layer_model_config = copy.copy(model_config)
                layer_model_config._frozen = False
                layer_model_config.moe_backend = "TRTLLM"
                layer_model_config._frozen = True
        self.experts = create_moe(
            routing_method=MuseSparkOpenRoutingMethod(
                top_k=self.top_k,
                num_experts=self.num_experts,
                callable_e_score_correction_bias=lambda: self.expert_bias,
                renorm_eps=config.router_score_division_eps,
            ),
            num_experts=self.num_experts,
            hidden_size=self.expert_hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            # The per-expert post-norm has to run before the combine, so the
            # partial sums are reduced by this module, not inside the backend.
            reduce_results=False,
            model_config=layer_model_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            override_quant_config=layer_quant_config,
        )
        # Under attention DP the MoE layer acquires a cross-rank combine
        # between the expert GEMM and this module, and no combine can carry an
        # unfinalized expert output. Hand the scheduler this module's finalize
        # so it runs on the dispatched rows instead, before the combine. That
        # is the same function: ``w * RMSNorm(gain * y)`` is computed per
        # (token, expert) row and non-local slots contribute exact zeros, so
        # summing the per-rank finalized partials equals finalizing the whole
        # set.
        #
        # Registered ONLY when the layer has a communication strategy. The
        # scheduler runs a registered ``unfinalized_combine_fn`` on every
        # ``do_finalize=False`` forward, comm or not, and then returns the
        # dense tensor; without a comm the triple must come back instead so
        # ``forward`` can finalize it here, where the small-batch fused
        # finalize + all-reduce path lives. ``comm`` is decided in the
        # ConfigurableMoE constructor; the only later reassignment swaps one
        # strategy for another, never between None and a strategy.
        if getattr(self.experts, "comm", None) is not None:
            self.experts.unfinalized_combine_fn = self.finalize
        # An explicit strategy, not the AUTO default. AUTO routes through
        # `torch.ops.trtllm.tunable_allreduce`, whose autotuner profiles each
        # tactic inside a `torch.cuda.graph` capture; at TP4 every rank hung in
        # `torch.cuda.synchronize()` inside that capture during engine warmup,
        # because the ranks do not enter it as one collective. Pinning the
        # strategy skips the tunable path. Worth revisiting after bring-up:
        # this is a throughput question, not a correctness one.
        self.all_reduce = AllReduce(mapping=self.mapping, strategy=AllReduceStrategy.NCCL)

        self.nvtx = LayerNvtx(
            "moe",
            layer_idx,
            ("router", "pool", "experts", "finalize", "all_reduce", "unpool"),
        )
        self._pool_gather_workspace: "ProjectionGatherWorkspace | None" = None
        self._unpool_gather_workspace: "ProjectionGatherWorkspace | None" = None

    def _replicated_projection(
        self,
        module: Linear,
        hidden_states: torch.Tensor,
        workspace: "ProjectionGatherWorkspace | None",
    ) -> torch.Tensor:
        """Shard small BF16 ``[tokens, hidden]`` projections, then gather columns.

        The workspace is shared across layers in CUDA execution order. Larger
        batches and compiled execution retain the original replicated Linear.
        """
        if (
            workspace is None
            or torch.compiler.is_compiling()
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 2
            or not 0 < hidden_states.shape[0] <= 8
            or not hidden_states.is_contiguous()
        ):
            return module(hidden_states)
        from ..modules.low_m_gemm import apply_low_m_gemm

        columns = module.weight.shape[0] // self.mapping.tp_size
        begin = self.mapping.tp_rank * columns
        weight = module.weight[begin : begin + columns]
        projected = apply_low_m_gemm(module, hidden_states, weight, None, force_active=True)
        if projected is None:
            projected = torch.nn.functional.linear(hidden_states, weight)
        return workspace.gather(projected)

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits, scaled by the metap multiplier before routing."""
        weight = getattr(self.router, "weight", None)
        if (
            getattr(self, "_bf16x3_router_enabled", False)
            and weight is not None
            and not torch.compiler.is_compiling()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.shape[0] > 128
            and hidden_states.shape[1] == 8192
            and hidden_states.is_contiguous()
            and weight.device == hidden_states.device
            and weight.dtype == torch.float32
            and weight.shape == (256, 8192)
            and weight.is_contiguous()
        ):
            from ..cute_dsl_kernels.blackwell.bf16x3_router_gemm import bf16x3_router_gemm

            # Split long accumulation chains even when occupancy needs no split.
            # Weights remain FP32 and are read afresh on every graph replay.
            return bf16x3_router_gemm(hidden_states, weight, min_split_k=4) * self.metap_multiplier
        tensorcore = hidden_states.ndim == 2 and 16 <= hidden_states.shape[0] <= 128
        if (
            weight is not None
            and not torch.compiler.is_compiling()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and 0 < hidden_states.shape[0] <= 128
            and hidden_states.shape[1] == 8192
            and hidden_states.is_contiguous()
            and weight.device == hidden_states.device
            and weight.dtype == torch.float32
            and weight.shape == (256, 8192)
            and weight.is_contiguous()
        ):
            # Retain FP32 weights and accumulation. Only the reduction order
            # differs from the unfused GEMM; input casting and scaling no
            # longer require separate launches.
            tokens = hidden_states.shape[0]
            output = torch.empty((tokens, 256), device=hidden_states.device, dtype=torch.float32)
            if tensorcore:
                partials = torch.empty(
                    (16, tokens, 256), device=hidden_states.device, dtype=torch.float32
                )
                _router_tensorcore_kernel[(triton.cdiv(tokens, 16), 8, 16)](
                    hidden_states,
                    weight,
                    partials,
                    TOKENS=tokens,
                    num_warps=4,
                )
                _router_tensorcore_reduce_kernel[(triton.cdiv(tokens * 256, 128),)](
                    partials,
                    output,
                    self.metap_multiplier,
                    SIZE=tokens * 256,
                    num_warps=4,
                )
                return output
            rows = 1 if tokens <= 2 else 2
            warps = 16 if tokens <= 2 else 4
            if (
                getattr(self, "_prepare_stream", None) is not None
                and getattr(getattr(self, "experts", None), "comm", None) is None
            ):
                # Minimize router+routing and pool+quant join time. The
                # isolated projection winner can contend with the pool GEMM.
                if tokens in (3, 5):
                    rows, warps = 1, 16
                elif 4 <= tokens <= 11:
                    rows, warps = 2, 16
                elif tokens in (13, 15):
                    rows, warps = 2, 8
                elif tokens == 14:
                    rows, warps = 4, 8
            _router_projection_kernel[(tokens, 256 // rows)](
                hidden_states,
                weight,
                output,
                self.metap_multiplier,
                EXPERTS=256,
                HIDDEN=8192,
                ROWS=rows,
                num_warps=warps,
                enable_fp_fusion=False,
            )
            return output
        return self.router(hidden_states.to(torch.float32)) * self.metap_multiplier

    def prepare_expert_input(self, pooled: torch.Tensor) -> torch.Tensor | Fp4QuantizedTensor:
        """Fold the pre-expert norm into static NVFP4 quantization.

        Keep communication and unsupported backends on the original path.
        The fused kernel quantizes the FP32 norm result without an intermediate BF16 cast.
        """
        backend = getattr(self.experts, "backend", self.experts)
        if (
            not getattr(self, "_pre_expert_quant_fused", True)
            or not pooled.is_cuda
            or pooled.dtype != torch.bfloat16
            or torch.compiler.is_compiling()
            or getattr(self.experts, "comm", None) is not None
            or not isinstance(getattr(backend, "op_backend", None), TRTLLMOpBackend)
            or not getattr(backend, "has_nvfp4", False)
            or getattr(backend, "fc31_act_scale", None) is not None
            or pooled.shape[0] == 0
            or pooled.shape[0] > getattr(self.experts, "moe_max_num_tokens", pooled.shape[0])
            or not rmsnorm_fp4_quant_available()
        ):
            return self.pre_expert_norm(pooled)
        scale = getattr(backend, "fc31_input_scale", None)
        if scale is None or scale.numel() != 1:
            return self.pre_expert_norm(pooled)
        if self._pre_expert_quant_gain is None:
            # Serving weights are loaded before the first warmup forward.
            self._pre_expert_quant_gain = (self.pre_expert_norm.weight.float() + 1).to(pooled.dtype)
        from flashinfer.norm import rmsnorm_fp4quant

        fp4, sf = rmsnorm_fp4quant(
            pooled,
            self._pre_expert_quant_gain,
            eps=self.pre_expert_norm.variance_epsilon,
            global_scale=scale.reshape(1),
            block_size=16,
            scale_format="e4m3",
            is_sf_swizzled_layout=False,
        )
        return Fp4QuantizedTensor(fp4.view(torch.uint8), sf.view(torch.uint8), is_sf_swizzled=False)

    def prepare_moe_inputs(self, hidden_states: torch.Tensor):
        """Overlap small-batch pooling with router preparation."""
        overlap = (
            getattr(self, "_prepare_stream", None) is not None
            and hidden_states.is_cuda
            and 0
            < hidden_states.shape[0]
            <= getattr(self, "_prepare_overlap_max_tokens", _PREPARE_OVERLAP_MAX_TOKENS)
            and not torch.compiler.is_compiling()
            and getattr(self.experts, "comm", None) is None
        )
        main_stream = torch.cuda.current_stream() if overlap else None
        if overlap:
            self._prepare_stream.wait_stream(main_stream)
        with self.nvtx("router"):
            router_logits = self.router_logits(hidden_states)
            if overlap:
                # The backend consumes the same memoized tensors. Replacement
                # logits (padding/chunking) still miss the memo and recompute.
                self.experts.routing_method.apply(router_logits)
        stream_context = (
            torch.cuda.stream(self._prepare_stream) if overlap else contextlib.nullcontext()
        )
        with stream_context, self.nvtx("pool"):
            pooled = self.prepare_expert_input(
                self._replicated_projection(self.pooler, hidden_states, self._pool_gather_workspace)
            )
        if overlap:
            main_stream.wait_stream(self._prepare_stream)
            if not torch.cuda.is_current_stream_capturing():
                hidden_states.record_stream(self._prepare_stream)
                tensors = (
                    (pooled.fp4_tensor, pooled.scaling_factor)
                    if isinstance(pooled, Fp4QuantizedTensor)
                    else (pooled,)
                )
                for tensor in tensors:
                    tensor.record_stream(main_stream)
        return router_logits, pooled

    def finalize(
        self,
        gemm2_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        routing_weights: torch.Tensor,
        num_tokens: int,
        token_selected_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ``token_selected_slots`` is part of the ``unfinalized_combine_fn``
        # contract for per-expert transforms; the post-expert norm is the same
        # for every expert, so this finalize does not consult it.
        del token_selected_slots
        if self.mapping.moe_tp_size > 1:
            # Backend row permutations are rank-local. Reduce in canonical
            # (token, selected-expert) order, never in GEMM scratch order.
            index = expanded_idx_to_permuted_idx.reshape(-1)
            valid = (index >= 0) & (index < gemm2_output.shape[0])
            safe_index = torch.where(valid, index, torch.zeros_like(index))
            rows = gemm2_output.index_select(0, safe_index.int())[:, : self.expert_hidden_size]
            rows = torch.where(valid[:, None], rows.float(), 0.0)
            # FP32 preserves small residuals between opposing FC2 shards
            # before RMSNorm can amplify them. This is the sole MoE reduce
            # for pure TP; the resulting experts are complete on every rank.
            gemm2_output = self.all_reduce(rows.contiguous())
            expanded_idx_to_permuted_idx = torch.arange(
                num_tokens * self.top_k, dtype=torch.int32, device=gemm2_output.device
            )
        return post_expert_norm_combine(
            gemm2_output,
            expanded_idx_to_permuted_idx,
            routing_weights,
            self.post_expert_norm_weight,
            self.post_expert_norm_eps,
            num_tokens=num_tokens,
            top_k=self.top_k,
            hidden_size=self.expert_hidden_size,
            num_warps=(
                4
                if getattr(self, "_large_finalize_4warps", False)
                and num_tokens >= 8192
                and self.expert_hidden_size == 4096
                and self.top_k == 8
                else _FINALIZE_NUM_WARPS
            ),
            sparse_slots=(
                getattr(self, "_large_finalize_sparse", False)
                and num_tokens >= _SPARSE_FINALIZE_MIN_TOKENS
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        sp_num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0] if sp_num_tokens is None else sp_num_tokens
        finalize_ar_used = False
        routing_method = self.experts.routing_method
        if sp_num_tokens is None:
            router_logits, pooled = self.prepare_moe_inputs(hidden_states)
        else:
            if getattr(self.experts, "comm", None) is not None:
                raise ValueError("Prefill sequence parallelism requires MoE without external comm")
            with self.nvtx("router"):
                router_logits = self.router_logits(hidden_states)
            with self.nvtx("pool"):
                pooled = self.pooler(hidden_states)
                router_logits, pooled = allgather([router_logits, pooled], self.mapping, dim=0)
                router_logits = router_logits[:num_tokens]
                # Gather raw BF16 pooled activations before the existing norm +
                # NVFP4 fusion. Normalizing before gather would add a BF16 round.
                pooled = self.prepare_expert_input(pooled[:num_tokens])

        with self.nvtx("experts"):
            outputs = self.experts(
                pooled,
                router_logits,
                do_finalize=False,
                output_dtype=hidden_states.dtype,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens
                if attn_metadata is not None
                else None,
                use_dp_padding=False,
            )
        # A Tensor is only trustworthy here if THIS layer actually had a
        # communication strategy. ``comm is not None`` is what makes
        # ``ExternalCommMoEScheduler`` run ``self.finalize`` before its combine,
        # and it is the only reason a finalized tensor may come back from a
        # ``do_finalize=False`` call.
        #
        # Testing ``isinstance(outputs, torch.Tensor)`` alone is NOT equivalent,
        # and the difference is silent: a backend that ignores
        # ``do_finalize=False`` and finalizes internally also returns a Tensor,
        # with no comm anywhere. Accepting that would skip this model's
        # per-expert RMSNorm and substitute the backend's plain weighted sum --
        # a different model that still produces fluent text. That is precisely
        # what the refusal in the ``else`` branch exists to catch, and the
        # unguarded ``isinstance`` check let it through.
        scheduler_finalized = isinstance(outputs, torch.Tensor) and (
            getattr(self.experts, "comm", None) is not None
        )
        if scheduler_finalized:
            # Attention-DP wide-EP: the layer had a communication strategy, so
            # ``ExternalCommMoEScheduler`` ran ``self.finalize`` on the
            # dispatched rows and then reduced across the EP ranks. What comes
            # back is this rank's own tokens, per-expert normed, weighted and
            # summed over every expert on every rank. There is nothing left to
            # finalize, and the routing memo could not have supplied the
            # weights anyway: it holds this rank's tokens, while the finalize
            # had to run over the dispatched ones.
            routing_method.clear_routing_memo()
            # The combine returns one row per LOCAL token, after the scheduler
            # strips the DP padding. Checked because the rest of this function
            # is shape-agnostic: a combine that handed back the dispatched row
            # count instead would flow through the cast, the unpooler and the
            # residual add before anything noticed.
            if outputs.shape[0] != num_tokens:
                raise RuntimeError(
                    f"The MoE communication combine returned {outputs.shape[0]} rows "
                    f"for {num_tokens} local tokens. The finalize ran before the "
                    "combine, so what comes back must already be per local token."
                )
            combined = outputs
        else:
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
                raise RuntimeError(
                    "MuseSparkOpen MoE expects the unfinalized (gemm2_output, "
                    "expert_weights, expanded_idx_to_permuted_idx) triple from the "
                    f"selected backend, got {type(outputs).__name__} "
                    f"{getattr(outputs, 'shape', '')}. The chosen MoE backend "
                    "finalizes internally, which would apply the per-expert norm "
                    "after the combine."
                )
            gemm2_output, _kernel_weights, expanded_idx_to_permuted_idx = outputs

            # The kernel's expert_weights output buffer is left uninitialized when
            # the routing weights are supplied to it, so recover them here from the
            # same logits. The routing method memoizes on the logits tensor, so this
            # returns the very tensors the backend routed with rather than
            # recomputing an identical top-k assignment; if anything replaced the
            # tensor on the way down (DP padding, chunking) the memo misses and this
            # recomputes, which is what it always did.
            with self.nvtx("router"):
                _, token_final_scales = routing_method.apply(router_logits)
            # The memo would otherwise keep this layer's routing buffers alive for
            # the rest of the forward, once per layer.
            routing_method.clear_routing_memo()
            with self.nvtx("finalize"):
                workspace = getattr(self, "_finalize_ar_workspace", None)
                # Batches above one fused call's 128 tokens need a workspace
                # that chunks them; a wrapper without ``finalize_chunked``
                # leaves them on the unfused path.
                finalize = None
                if workspace is not None and not torch.compiler.is_compiling():
                    if 1 <= num_tokens <= 128:
                        finalize = workspace.finalize
                    elif num_tokens <= getattr(self, "_finalize_ar_max_tokens", 128):
                        finalize = getattr(workspace, "finalize_chunked", None)
                if finalize is not None:
                    combined = finalize(
                        gemm2_output,
                        expanded_idx_to_permuted_idx,
                        token_final_scales,
                        self.post_expert_norm_weight,
                        self.post_expert_norm_eps,
                        num_tokens,
                    )
                    finalize_ar_used = True
                else:
                    combined = self.finalize(
                        gemm2_output, expanded_idx_to_permuted_idx, token_final_scales, num_tokens
                    )
        # For EP, cast down BEFORE the reduce, not after. `combined` is this rank's
        # partial sum -- the finalize zeroes the slots whose expert lives on
        # another rank -- and the sum is cast to the activation dtype for the
        # unpooler either way, so the cast's position changes only the width the
        # partials cross the wire and get summed in, and it halves the bytes.
        # The capture puts this reduce at 2.672 ms per decode iteration in fp32
        # against 2.055 ms for the bf16 reduce the attention side runs at the
        # same cadence.
        #
        # ROUNDING-DIFFERENT, declared before it was measured: with EP the
        # partials are rounded once before being added instead of once after.
        # Most slots on a rank are already exactly zero, so the rounding falls
        # on the few nonzero partials that the trailing cast would have rounded
        # anyway -- but it is a real numerics change and the parity gates
        # decide it, not this argument. For pure MoE TP, finalize has already
        # reduced each expert before normalization; only the cast remains.
        combined = combined.to(hidden_states.dtype)
        # ``has_tp()`` is only ``tp_size > 1``; it knows nothing about attention
        # DP. Under attention DP the ranks of the TP group hold DIFFERENT
        # tokens, and the MoE's own cross-rank combine has already reduced the
        # expert partials, so an all-reduce here would both double-count the
        # partials and mix unrelated tokens. Same gate as
        # ``modeling_deepseekv3.py:1280``.
        # Pure MoE TP also skips this reduction: finalize returned complete
        # expert results on every rank, so reducing again would multiply them.
        if sp_num_tokens is not None:
            with self.nvtx("all_reduce"):
                combined = reducescatter(
                    _pad_sequence_parallel_tokens(combined, self.mapping.tp_size),
                    self.mapping,
                    dim=0,
                )
        elif (
            not finalize_ar_used
            and not self.mapping.enable_attention_dp
            and self.mapping.has_tp()
            and self.mapping.moe_tp_size == 1
        ):
            with self.nvtx("all_reduce"):
                combined = self.all_reduce(combined)
        with self.nvtx("unpool"):
            return self._replicated_projection(
                self.unpooler, combined, self._unpool_gather_workspace
            )


class MuseSparkOpenDecoderLayer(DecoderLayer):
    #: The sequence-parallel arm's attention-input all-gather, ``(attn_input, sp_num_tokens) ->
    #: rows``; None (the default) runs the NCCL all-gather. A model-specific path may set it
    #: on a layer, together with a QKV projection that consumes what it returns.
    _sp_attn_input_gather = None

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = MuseSparkOpenAttention(model_config, layer_idx)
        # Projection joins this stream before the next branch/layer. Reusing
        # the model stream avoids a separate allocator pool for every layer.
        self.self_attn._sp_projection_stream = aux_stream
        self._overlap_sp_projection_enabled = (
            os.getenv("MUSESPARK_OPEN_OVERLAP_SP_PROJECTION", "0") == "1"
        )
        self._pad_sp_projection_enabled = os.getenv("MUSESPARK_OPEN_PAD_SP_PROJECTION", "0") == "1"
        self.feed_forward = MuseSparkOpenMoE(model_config, layer_idx, aux_stream)

        self.pre_qkv_proj_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        # post_norm_affine=false -> no weight on the post-attention norm.
        self.post_attn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_post_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        self.pre_ffn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.post_ffn_norm = MuseSparkOpenRMSNormWeightFirst(
            hidden_size=config.hidden_size, eps=config.rms_post_norm_eps, dtype=config.torch_dtype
        )

        self.add_attn = MuseSparkOpenGatedResidual(config.hidden_size, config.torch_dtype)
        self.add_ffn = MuseSparkOpenGatedResidual(config.hidden_size, config.torch_dtype)
        self.dtype = config.torch_dtype

        self.nvtx = LayerNvtx(
            "layer",
            layer_idx,
            (
                "pre_qkv_proj_norm",
                "attn",
                # One range per residual join, not two: the norm and the gated
                # add are a single fused kernel here, so there is no boundary
                # between them to annotate.
                "norm_add_attn",
                "moe",
                "norm_add_ffn",
            ),
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        precomputed_attn_input: Optional[torch.Tensor] = None,
        next_layer_norm: Optional[RMSNorm] = None,
        sp_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states is the FP32 residual stream; each branch reads a BF16
        # cast of it and writes back through the gated add.
        with self.nvtx("pre_qkv_proj_norm"):
            # This arm is layer 0's (later layers receive precomputed_attn_input
            # from the previous layer's fused boundary). The ordering switch is
            # read off the gated residual -- the single construction-time
            # resolution point -- rather than re-consulting the environment.
            # Layer 0's residual is bf16-representable, so both orderings
            # produce identical values there; honoring the switch keeps the
            # arms the same function by construction.
            if precomputed_attn_input is not None:
                attn_input = precomputed_attn_input
            elif self.add_attn._preqkv_norm_fp32:
                attn_input = _rmsnorm_fp32_in(hidden_states, self.pre_qkv_proj_norm)
            else:
                attn_input = self.pre_qkv_proj_norm(hidden_states.to(self.dtype))
        # The attention projection and its TP reduction stay in this range.
        with self.nvtx("attn"):
            overlap_projection = (
                sp_num_tokens is not None
                and sp_num_tokens >= 8192
                and self._overlap_sp_projection_enabled
                and not torch.cuda.is_current_stream_capturing()
            )
            pad_projection = (
                sp_num_tokens is not None
                and self._pad_sp_projection_enabled
                and not torch.cuda.is_current_stream_capturing()
            )
            if sp_num_tokens is not None:
                gather = self._sp_attn_input_gather
                if gather is None:
                    attn_input = allgather(attn_input, self.mapping, dim=0)[:sp_num_tokens]
                else:
                    attn_input = gather(attn_input, sp_num_tokens)
                kwargs = dict(kwargs, all_reduce_params=AllReduceParams(enable_allreduce=False))
            attn_out = self.self_attn(
                position_ids=position_ids,
                hidden_states=attn_input,
                attn_metadata=attn_metadata,
                overlap_sp_output_projection=overlap_projection,
                pad_sp_output_projection=pad_projection,
                **kwargs,
            )
            if sp_num_tokens is not None and not (overlap_projection or pad_projection):
                attn_out = reducescatter(
                    _pad_sequence_parallel_tokens(attn_out, self.mapping.tp_size),
                    self.mapping,
                    dim=0,
                )
        # post_attn_norm + add_attn, and post_ffn_norm + add_ffn, each as one
        # kernel: seventeen elementwise launches per layer became two, and the
        # 8192-wide FP32 residual no longer round-trips through HBM between the
        # norm and the add. The range names say norm_add rather than the
        # prep branch's separate post_attn_norm and add_attn, because there is
        # no longer a boundary between them to annotate.
        with self.nvtx("norm_add_attn"):
            hidden_states, ffn_input = self.add_attn._add_after_rmsnorm_and_norm(
                hidden_states, attn_out, self.post_attn_norm.variance_epsilon, self.pre_ffn_norm
            )
        with self.nvtx("moe"):
            if sp_num_tokens is None:
                ffn_out = self.feed_forward(ffn_input, attn_metadata)
            else:
                ffn_out = self.feed_forward(ffn_input, attn_metadata, sp_num_tokens=sp_num_tokens)
        with self.nvtx("norm_add_ffn"):
            next_attn_input = None
            if next_layer_norm is not None:
                hidden_states, next_attn_input = self.add_ffn._add_after_rmsnorm_and_norm(
                    hidden_states,
                    ffn_out,
                    self.post_ffn_norm.variance_epsilon,
                    next_layer_norm,
                    self.post_ffn_norm.gain(),
                )
            else:
                hidden_states = self.add_ffn.add_after_rmsnorm(
                    hidden_states,
                    ffn_out,
                    self.post_ffn_norm.variance_epsilon,
                    self.post_ffn_norm.gain(),
                )
        # DFlash captures the FP32 residual, independently of prenorm carry.
        spec_metadata = kwargs.get("spec_metadata")
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)
        if next_attn_input is not None:
            return hidden_states, next_attn_input
        return hidden_states


def _build_finalize_ar_workspace(
    mapping: Mapping, *, use_mnnvl: bool = False
) -> "FinalizeIpcWorkspace | None":
    """Construct the finalize+all-reduce workspace, or None to stay unfused.

    The constructor probes the TP group's local or fabric topology and raises
    when peers are inaccessible or the group's configuration disagrees. That is
    a property of the deployment, so it must degrade
    to the unfused finalize + AllReduce path instead of failing model init.
    """
    if not _fused_finalize_ar_enabled():
        return None

    from .musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace

    if not HAS_CUTEDSL:
        return None
    try:
        if use_mnnvl:
            return FinalizeIpcWorkspace(mapping, use_mnnvl=True)
        return FinalizeIpcWorkspace(mapping)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            f"MuseSparkOpen finalize+all-reduce fusion unavailable ({exc}); "
            "using the unfused finalize and AllReduce path."
        )
        return None


class MuseSparkOpenModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        # Weightless RMSNorm on the token embeddings, then the m_emb scale.
        self.embed_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        self.m_emb = config.m_emb
        self.layers = nn.ModuleList(
            [
                MuseSparkOpenDecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._finalize_ar_workspace = None
        mapping = model_config.mapping
        self._sp_mapping = mapping
        self._prefill_sp_enabled = os.getenv("MUSESPARK_OPEN_PREFILL_SEQUENCE_PARALLEL", "0") == "1"
        if self._prefill_sp_enabled:
            if not (
                mapping.tp_size == 4
                and mapping.moe_ep_size == 4
                and mapping.moe_tp_size == 1
                and mapping.cp_size == 1
                and mapping.pp_size == 1
                and not mapping.enable_attention_dp
                and len({rank // mapping.gpus_per_node for rank in mapping.tp_group}) == 1
                and config.torch_dtype == torch.bfloat16
                and all(
                    getattr(layer.feed_forward.experts, "comm", None) is None
                    for layer in self.layers
                )
            ):
                raise ValueError("Prefill sequence parallelism requires single-node BF16 TP4/EP4")
            logger.info("MuseSparkOpen prefill sequence parallelism enabled for >=4096 tokens")
        local_finalize_tp4 = (
            mapping.tp_size == 4
            and len({rank // mapping.gpus_per_node for rank in mapping.tp_group}) == 1
        )
        fabric_finalize_tp8 = (
            os.getenv("MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL", "0") == "1"
            and mapping.tp_size == 8
            and mapping.gpus_per_node == 4
            and mapping.cp_size == 1
            and mapping.pp_size == 1
            and len({rank // mapping.gpus_per_node for rank in mapping.tp_group}) == 2
        )
        if (
            (local_finalize_tp4 or fabric_finalize_tp8)
            and mapping.moe_ep_size == mapping.tp_size
            and not mapping.enable_attention_dp
            and torch.cuda.is_available()
            and config.torch_dtype == torch.bfloat16
            and fused_kernels_supported(torch.cuda.get_device_capability())
            and all(
                layer.feed_forward.expert_hidden_size == 4096 and layer.feed_forward.top_k == 8
                for layer in self.layers
            )
        ):
            self._finalize_ar_workspace = _build_finalize_ar_workspace(
                mapping, use_mnnvl=fabric_finalize_tp8
            )
            if self._finalize_ar_workspace is not None:
                if fabric_finalize_tp8:
                    logger.info("MuseSparkOpen TP8 fabric finalize enabled")
                for layer in self.layers:
                    layer.feed_forward._finalize_ar_workspace = self._finalize_ar_workspace
        if os.getenv("MUSESPARK_OPEN_SHARDED_PROJECTIONS", "0") == "1":
            tp_nodes = len({rank // mapping.gpus_per_node for rank in mapping.tp_group})
            if mapping.tp_size == 8:
                sharded_topology = mapping.moe_ep_size == 8
            elif mapping.tp_size == 4:
                sharded_topology = mapping.moe_ep_size == 4 and tp_nodes == 1
            else:
                sharded_topology = False
            if not (
                sharded_topology
                and mapping.gpus_per_node == 4
                and mapping.cp_size == mapping.pp_size == 1
                and not mapping.enable_attention_dp
                and config.torch_dtype == torch.bfloat16
                and fused_kernels_supported(torch.cuda.get_device_capability())
                and all(
                    layer.feed_forward.expert_hidden_size == 4096
                    and getattr(layer.feed_forward.experts, "comm", None) is None
                    for layer in self.layers
                )
            ):
                raise ValueError(
                    "Sharded projections require SM103 (or SM107 with "
                    "MUSESPARK_OPEN_SM107_FUSED_KERNELS=1) BF16 TP8/EP8 or single-node TP4/EP4, "
                    "four GPUs per node, CP1/PP1, expert width 4096, and no attention DP or "
                    "external MoE communication"
                )
            from .musespark_open_projection_gather import ProjectionGatherWorkspace

            # Pool runs on the preparation stream and unpool on the main stream.
            # Keep their epochs independent; successive layers already join the
            # preparation stream before consuming its result.
            pool_workspace = ProjectionGatherWorkspace(mapping)
            unpool_workspace = ProjectionGatherWorkspace(mapping)
            for layer in self.layers:
                layer.feed_forward._pool_gather_workspace = pool_workspace
                layer.feed_forward._unpool_gather_workspace = unpool_workspace
            logger.info(
                f"MuseSparkOpen experimental TP{mapping.tp_size} sharded projections enabled for M<=8"
            )
        # Final norm keeps the raw stored weight (gain center 0).
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.dtype = config.torch_dtype

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embeds = self.embed_norm(inputs_embeds.to(self.dtype))
        if self.m_emb != 1.0:
            embeds = embeds * self.m_emb
        sp_num_tokens = None
        if (
            self._prefill_sp_enabled
            and not torch.compiler.is_compiling()
            and attn_metadata.num_contexts > 0
            and embeds.shape[0] >= 4096
            and kwargs.get("spec_metadata") is None
            and kwargs.get("lora_params") is None
            and kwargs.get("all_reduce_params") is None
        ):
            sp_num_tokens = embeds.shape[0]
            mapping = self._sp_mapping
            embeds = _pad_sequence_parallel_tokens(embeds, mapping.tp_size)
            shard_size = embeds.shape[0] // mapping.tp_size
            embeds = embeds.narrow(0, mapping.tp_rank * shard_size, shard_size)
        # FP32 residual stream, entered once.
        hidden_states = embeds.to(torch.float32)

        precomputed_attn_input = None
        for index, decoder_layer in enumerate(self.layers):
            next_norm = (
                self.layers[index + 1].pre_qkv_proj_norm if index + 1 < len(self.layers) else None
            )
            result = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                precomputed_attn_input=precomputed_attn_input,
                next_layer_norm=next_norm,
                sp_num_tokens=sp_num_tokens,
                **kwargs,
            )
            if next_norm is None:
                hidden_states = result
            else:
                hidden_states, precomputed_attn_input = result
        output = self.norm(hidden_states.to(self.dtype))
        if sp_num_tokens is not None:
            output = allgather(output, self._sp_mapping, dim=0)[:sp_num_tokens]
        return output


def assert_supported_spec_config(spec_config: Any) -> None:
    """Admit only the speculative modes this decoder can serve.

    External drafters (NGram, user-provided) run outside the model and need
    nothing from it. The suffix automaton is one-engine and
    weight-free, so ``SpecDecOneEngineForCausalLM`` runs its worker inside
    this forward. DFlash is one-engine but its drafter weights come from the
    separate ``speculative_model`` checkpoint, not this one; the decoder
    layers feed it via ``spec_metadata.maybe_capture_hidden_states``. The
    remaining one-engine modes (MTP, Eagle3, PARD, DSpark) need draft weights
    the target checkpoint does not carry.
    """
    if spec_config is None:
        return
    mode = spec_config.spec_dec_mode
    assert (not mode.use_one_engine()) or mode.is_sa() or mode.is_dflash(), (
        "MuseSparkOpen supports speculative decoding only with external "
        "drafters (NGram, user-provided), the suffix automaton, or DFlash; "
        f"{mode.name} needs draft weights the checkpoint does not carry"
    )


# No @register_auto_model here on purpose. The checkpoint's only architecture
# string is "MuseSparkOpenOmniForCausalLM", and it now names
# MuseSparkOpenOmniForConditionalGeneration in modeling_musespark_open_omni, which
# wraps this class. Registration is first-wins and the omni module imports this
# one, so a registration left here would win the slot and the perception tower
# would never be built. This class keeps no architecture string of its own and
# is reached by import: from the omni wrapper, and from its tests.
class MuseSparkOpenForCausalLM(
    SpecDecOneEngineForCausalLM[MuseSparkOpenModel, MuseSparkOpenConfig]
):
    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig]):
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.exclude_modules:
            quant_config.exclude_modules = normalize_exclude_modules(quant_config.exclude_modules)
        config = model_config.pretrained_config
        assert model_config.mapping.pp_size == 1, (
            "MuseSparkOpen does not support pipeline parallelism"
        )
        assert_supported_spec_config(getattr(model_config, "spec_config", None))
        super().__init__(
            MuseSparkOpenModel(model_config),
            model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )
        self.logits_processor = MuseSparkOpenLogitsProcessor(
            logit_multiplier=config.metap_multiplier, soft_cap=config.output_soft_cap_temp
        )

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """V2 sizes the 2048-key local pool and the full-context pool
        independently instead of splitting one budget statically."""
        return "V2"

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """The V1 manager is not a supported route for this model.

        Routes that cannot keep V2 (disaggregated serving outside NIXL with
        the Python transceiver) are refused at configuration time rather
        than demoted to V1.
        """
        return True

    def load_weights(
        self, weights: Dict[str, Any], weight_mapper: Optional[Any] = None, **kwargs
    ) -> Dict[str, int]:
        """Load the checkpoint and return the key-accounting summary.

        The generic module walk cannot express this checkpoint's stacked
        experts, per-half expert scales, fused Q/output-gate projection, or its
        out-of-range layer keys, so loading is fully owned here.
        """
        from .checkpoints.hf.musespark_open_weight_mapper import load_musespark_open_weights

        return load_musespark_open_weights(self, weights)

    def infer_max_seq_len(self) -> int:
        return self.config.max_position_embeddings


def head_interleaved_q_gate(
    q_weight: torch.Tensor, gate_weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Interleave the query and output-gate projections head by head.

    ``Attention.preprocess_qkv`` reads the first ``2 * q_size`` columns as
    ``[tokens, num_heads, 2 * head_dim]`` and splits the last dim in half, so
    each head's query rows must be immediately followed by that head's gate
    rows. Column-parallel sharding then keeps each rank's heads (and their
    gates) contiguous.
    """
    hidden = q_weight.shape[-1]
    q_heads = q_weight.reshape(num_heads, head_dim, hidden)
    gate_heads = gate_weight.reshape(num_heads, head_dim, hidden)
    return torch.stack([q_heads, gate_heads], dim=1).reshape(num_heads * 2 * head_dim, hidden)
