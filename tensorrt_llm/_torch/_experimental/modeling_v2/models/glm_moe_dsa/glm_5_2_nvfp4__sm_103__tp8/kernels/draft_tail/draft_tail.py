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
"""draft_tail: greedy draft-token selection over a vocab-sharded draft LM head.

Replaces, for one MTP draft step, the native sequence

    logits = lm_head(hidden)                     # [R, V/tp] BF16, vocab shard
    tokens = greedy_sample_draft_with_tp_gather(logits)

with ``trtllm::glm_moe_dsa_draft_tail`` (GEMV, rank-local maximum and a
compact cross-rank exchange in one launch) or, in ``logits`` mode, the native
GEMM followed by ``trtllm::glm_moe_dsa_draft_tail_from_logits``. Both return, on
every rank, the token id of the largest BF16 logit over all vocab shards,
lowest global id on ties, NaN highest: the selection rule of the native
gather. The two modes differ in whose BF16 logits they select from:

* ``logits`` mode selects from the native GEMM's own BF16 logits, so its draft
  tokens are the native ones by construction.
* ``fused`` mode computes the logits itself (FP32 accumulation, then BF16
  rounding). Where the two largest logits are within one BF16 rounding step
  of each other (a BF16 near-tie), the token depends on the exact FP32 sum,
  i.e. on the accumulation order. The native-order fused variants (11-14,
  ``NATIVE_ORDER_VARIANTS``; one of them is the default) use the order of the
  native cuBLAS LM-head GEMM selected for this shape on SM103 (K walked in
  16-value tensor-core steps, one FP32 accumulator), so their BF16 logits,
  and draft tokens, equal native bit for bit;
  ``test_draft_tail_draft_tail.py::test_native_order_logits_bitwise_equal_native``
  checks this against ``F.linear`` and must pass again whenever cuBLAS or
  the GPU changes. The other fused variants (0-10) accumulate in other
  orders: decision-level variants that can pick a different draft token at
  BF16 near-ties (draft acceptance can change; target verification and the
  emitted tokens stay native).

The exchange buffer is an MNNVL multicast allocation owned by this package
(one per TP group); rank-local scratch holds the partial keys, the CTA arrival
counter and the exchange epoch. Both are created collectively, outside CUDA
graph capture, by ``DraftTailRunner.prepare``; all ranks agree on the outcome
before any rank uses the candidate path.
"""

import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from ...dispatch import log_capability

FEATURE = "draft_tail"
MAX_ROWS = 8
MAX_RANKS = 8
HIDDEN_SIZE = 6144
NUM_SLOTS = 3
SCRATCH_WORDS = MAX_ROWS + 2
MAX_DIRECT_CTAS = 512
# Last-CTA region (3 slots x 8 ranks x 8 rows), then the direct region
# (3 slots x 8 rows x 8 ranks x 512 CTAs); see draftTail.h.
EXCHANGE_WORDS = (
    NUM_SLOTS * MAX_RANKS * MAX_ROWS + NUM_SLOTS * MAX_ROWS * MAX_RANKS * MAX_DIRECT_CTAS
)
EXCHANGE_NONE, EXCHANGE_MULTICAST, EXCHANGE_UNICAST = 0, 1, 2
EXCHANGE_MULTICAST_DIRECT, EXCHANGE_UNICAST_DIRECT = 3, 4
_EXCHANGE_IDS = {
    "multicast": EXCHANGE_MULTICAST,
    "unicast": EXCHANGE_UNICAST,
    "multicast-direct": EXCHANGE_MULTICAST_DIRECT,
    "unicast-direct": EXCHANGE_UNICAST_DIRECT,
}
NUM_VARIANTS = 15
FIRST_MMA_VARIANT = 7
# Fused variants that accumulate in the native LM-head GEMM's order (they
# differ only in how the weights reach the lanes; see draftTail.cu).
NATIVE_ORDER_VARIANTS = (11, 12, 13, 14)
# Bounded wait for a peer's key, in GPU clock cycles (2**38 is about 2.3 min
# at 1.9 GHz): far above any legitimate skew between ranks of one graph
# replay, after which the kernel records the missing peer and traps.
DEFAULT_TIMEOUT_CYCLES = 1 << 38


@dataclass(frozen=True)
class DraftTailOptions:
    """Launch options of the draft tail; this target run the defaults (``served``).

    ``mode``: ``"fused"`` (default; draft_tail's own GEMV) or ``"logits"`` (native
    GEMM + draft_tail). Both give native draft tokens with the default
    ``variant`` 13 (native accumulation order, see the module docstring);
    fused variants outside ``NATIVE_ORDER_VARIANTS`` are decision-level
    variants at BF16 near-ties.

    One fused kernel shape serves every row count 1..8: a row's packed key
    (BF16-rounded logit and id) is then independent of the other rows of the
    call, so CUDA-graph padding or batch composition cannot change a live
    row's draft token. Kernel families round differently at BF16 near-ties,
    which is why the row count never selects the family. ``variant`` 13 is
    the tensor-core (``mma.sync``) kernel in native accumulation order with
    32-byte weight loads, whose cost is nearly flat in R; 11, 12 and 14 load
    the same values in narrower pieces (same results); 7-10 are
    tensor-core kernels in a permuted K order and 0-6 the CUDA-core family
    (cost grows with R), used only when set explicitly. ``mode="logits"`` (native GEMM + draft_tail)
    likewise serves every row count.
    """

    mode: str = "fused"  # "fused" (own GEMV) or "logits" (native GEMM)
    exchange: str = "multicast"  # see _EXCHANGE_IDS
    variant: int = 13
    ctas_per_sm: int = 3  # CUDA-core variants, when num_ctas is 0
    num_ctas: int = 0  # overrides the default grid when non-zero
    pdl: bool = True
    timeout_cycles: int = DEFAULT_TIMEOUT_CYCLES
    # Per-mille of each warp's rows loaded with an L2 evict_last hint (fused
    # mode, L2-hinted variants); 0 keeps the whole stream evict_first.
    keep_permille: int = 0

    def shape(self, sm_count: int):
        """(variant, num_ctas) of the fused kernel, for every row count."""
        if self.num_ctas:
            return self.variant, self.num_ctas
        if self.variant >= FIRST_MMA_VARIANT:
            return self.variant, 0
        return self.variant, self.ctas_per_sm * sm_count

    @staticmethod
    def served() -> "DraftTailOptions":
        """The defaults, with PDL as the native kernels have it (``TRTLLM_ENABLE_PDL``, on
        unless set to 0)."""
        return DraftTailOptions(pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") != "0")

    def validate(self) -> None:
        if self.mode not in ("fused", "logits"):
            raise ValueError(f"draft_tail: unknown mode {self.mode!r}")
        if self.exchange not in _EXCHANGE_IDS:
            raise ValueError(f"draft_tail: unknown exchange {self.exchange!r}")
        if not 0 <= self.variant < NUM_VARIANTS:
            raise ValueError(f"draft_tail: unknown variant {self.variant}")


class DraftTailRunner:
    """Scratch, exchange workspace and launch for one TP group on this rank."""

    def __init__(
        self, tp_rank: int, tp_size: int, mapping=None, options: Optional[DraftTailOptions] = None
    ):
        if not 1 <= tp_size <= MAX_RANKS:
            raise ValueError(f"draft_tail: tp_size {tp_size} unsupported")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.mapping = mapping
        self.options = options or DraftTailOptions()
        self.options.validate()
        self.scratch: Optional[torch.Tensor] = None
        # 16-byte-aligned rows for step inputs whose layout the kernels
        # cannot read directly (see ``conform``).
        self._staging: Optional[torch.Tensor] = None
        self.staged_steps = 0
        self._mcast = None
        self._uc_buffer: Optional[torch.Tensor] = None
        self._mc_ptr = 0
        self._uc_ptr = 0
        self._peer_ptrs: Optional[torch.Tensor] = None
        self._state = "new"  # new -> ready | failed
        self.calls = 0
        self.sm_count = 0

    @property
    def ready(self) -> bool:
        return self._state == "ready"

    @property
    def exchange_mode(self) -> int:
        if self.tp_size == 1:
            return EXCHANGE_NONE
        return _EXCHANGE_IDS[self.options.exchange]

    def prepare(self) -> bool:
        """Allocate scratch and the exchange workspace once, on every rank.

        Collective for tp_size > 1: every rank of the TP group must call it at
        the same point. Never runs under CUDA graph capture; a runner that is
        not ready there keeps the native path for that capture on all ranks.
        """
        if self._state != "new":
            return self.ready
        if torch.cuda.is_current_stream_capturing():
            return False
        local_error: Optional[Exception] = None
        try:
            self.scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device="cuda")
            self._staging = torch.zeros(MAX_ROWS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
            if not _kernel_layout_ok(self._staging):
                raise RuntimeError("draft_tail: staging rows not 16-byte aligned")
            self.sm_count = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
            if self.tp_size > 1:
                self._allocate_exchange()
            torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            # Converged across ranks below: a rank that raised here must not
            # leave its peers waiting in a collective it never joins.
            local_error = error
        ok = self._agree(local_error is None)
        self._state = "ready" if ok else "failed"
        log_capability(
            FEATURE,
            f"draft_tail[tp{self.tp_size}]",
            ok,
            "" if ok else f"workspace setup failed: {local_error!r}",
        )
        return ok

    def _allocate_exchange(self) -> None:
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
        )

        if self.mapping is None:
            raise ValueError("draft_tail: exchange needs the TP mapping")
        self._comm = _get_mnnvl_workspace_comm(self.mapping)
        use_fabric = (
            self.mapping.is_multi_node() or os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
        )
        self._mcast = _make_mnnvl_mcast_buffer(
            self._comm, EXCHANGE_WORDS * 8, self.mapping, use_fabric
        )
        self._uc_buffer = self._mcast.get_uc_buffer(self.tp_rank, [EXCHANGE_WORDS], torch.int64, 0)
        self._uc_buffer.zero_()
        self._uc_ptr = self._uc_buffer.data_ptr()
        self._mc_ptr = self._mcast.get_mc_buffer([EXCHANGE_WORDS], torch.int64, 0).data_ptr()
        peers = [
            self._mcast.get_uc_buffer(q, [EXCHANGE_WORDS], torch.int64, 0).data_ptr()
            for q in range(self.tp_size)
        ]
        self._peer_ptrs = torch.tensor(peers, dtype=torch.int64, device="cuda")

    def _agree(self, local_ok: bool) -> bool:
        if self.tp_size == 1:
            return local_ok
        comm = getattr(self, "_comm", None)
        if comm is None:
            from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm

            comm = _get_mnnvl_workspace_comm(self.mapping)
            self._comm = comm
        from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

        # The collective also acts as the barrier that orders every rank's
        # zeroed exchange buffer before any rank's first publish.
        return _mnnvl_workspace_all_succeeded(comm, local_ok)

    def _common_args(self, rows: int, vocab_offset: int, keys_out):
        tokens = torch.empty(rows, dtype=torch.int32, device="cuda")
        return tokens, (
            self.scratch,
            keys_out,
            vocab_offset,
            self.tp_rank,
            self.tp_size,
            self.exchange_mode,
            self._mc_ptr,
            self._uc_ptr,
            self._peer_ptrs,
        )

    def run_fused(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        vocab_offset: int,
        keys_out: Optional[torch.Tensor] = None,
        variant: Optional[int] = None,
        num_ctas: Optional[int] = None,
        keep_permille: Optional[int] = None,
        logits_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``logits_out`` (tests only, tensor-core variants): receives the
        kernel's BF16 logits [R, V_local]."""
        opts = self.options
        tokens, common = self._common_args(hidden.shape[0], vocab_offset, keys_out)
        scratch, keys, off, rank, size, mode, mc, uc, peers = common
        torch.ops.trtllm.glm_moe_dsa_draft_tail(
            hidden,
            weight,
            tokens,
            scratch,
            keys,
            off,
            rank,
            size,
            mode,
            mc,
            uc,
            peers,
            opts.variant if variant is None else variant,
            opts.num_ctas if num_ctas is None else num_ctas,
            opts.keep_permille if keep_permille is None else keep_permille,
            opts.pdl,
            opts.timeout_cycles,
            logits_out,
        )
        self.calls += 1
        return tokens

    def run_from_logits(
        self, logits: torch.Tensor, vocab_offset: int, keys_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens, common = self._common_args(logits.shape[0], vocab_offset, keys_out)
        scratch, keys, off, rank, size, mode, mc, uc, peers = common
        torch.ops.trtllm.glm_moe_dsa_draft_tail_from_logits(
            logits,
            tokens,
            scratch,
            keys,
            off,
            rank,
            size,
            mode,
            mc,
            uc,
            peers,
            self.options.pdl,
            self.options.timeout_cycles,
        )
        self.calls += 1
        return tokens

    def select(
        self, hidden: torch.Tensor, lm_head, keys_out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Draft tokens for ``hidden`` [R, H], 1 <= R <= 8, with the
        vocab-sharded head. The same kernel serves every R (see
        ``DraftTailOptions``); ``keys_out`` optionally receives the winning keys."""
        rows = hidden.shape[0]
        if not 1 <= rows <= MAX_ROWS:
            raise ValueError(f"draft_tail: {rows} rows outside [1, {MAX_ROWS}]")
        weight = lm_head.weight
        vocab_offset = self.tp_rank * weight.shape[0]
        if self.options.mode == "fused":
            variant, num_ctas = self.options.shape(self.sm_count)
            return self.run_fused(
                hidden, weight, vocab_offset, keys_out=keys_out, variant=variant, num_ctas=num_ctas
            )
        logits = torch.nn.functional.linear(hidden, weight)
        return self.run_from_logits(logits, vocab_offset, keys_out=keys_out)

    def conform(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` itself when the kernels can read its layout, otherwise
        a copy in this runner's 16-byte-aligned staging rows.

        Strides and alignment come from this rank's allocator, so they must
        never decide between draft_tail and the native path (the two use different
        collectives); a non-conforming layout costs one device copy instead.
        The staging rows are allocated by ``prepare`` (outside capture), so a
        captured copy keeps a stable address across graph replays.
        """
        if _kernel_layout_ok(hidden):
            return hidden
        staged = self._staging[: hidden.shape[0]]
        staged.copy_(hidden)
        self.staged_steps += 1
        return staged

    def step(self, hidden: torch.Tensor, lm_head) -> Optional[torch.Tensor]:
        """Draft tokens of one draft step, or None for the native path.

        None is returned only for reasons that every TP rank shares (see
        ``_shared_step_reason``), so all ranks run the same exchange.
        """
        reason = _shared_step_reason(hidden)
        if reason is not None:
            log_capability(
                FEATURE, f"draft_tail[step: {reason}]", False, "identical on every TP rank"
            )
            return None
        if hidden.device != lm_head.weight.device:
            raise RuntimeError(
                f"draft_tail: draft rows on {hidden.device}, LM head on {lm_head.weight.device}"
            )
        return self.select(self.conform(hidden), lm_head)


def _kernel_layout_ok(hidden: torch.Tensor) -> bool:
    """Unit last stride, 16-byte row stride and base: the kernels' loads."""
    return hidden.stride(1) == 1 and hidden.stride(0) % 8 == 0 and hidden.data_ptr() % 16 == 0


def _shared_step_reason(hidden: torch.Tensor) -> Optional[str]:
    """Why a draft step cannot use draft_tail, judged only from properties that are
    identical on every rank of the TP group, or None.

    The draft LM-head input is ``hidden_states[gather_ids]``: the same
    requests (row count), model dtype and hidden width on every rank. Layout
    (strides, alignment) is rank-local and handled by ``conform`` instead.
    """
    if hidden.dim() != 2:
        return f"rows of rank {hidden.dim()}"
    if not 1 <= hidden.shape[0] <= MAX_ROWS:
        return f"{hidden.shape[0]} rows outside [1, {MAX_ROWS}]"
    if hidden.dtype != torch.bfloat16:
        return f"dtype {hidden.dtype}"
    if hidden.shape[1] != HIDDEN_SIZE:
        return f"hidden width {hidden.shape[1]}"
    return None


_RUNNERS: Dict[tuple, DraftTailRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def get_runner(mapping, options: Optional[DraftTailOptions] = None) -> DraftTailRunner:
    """One runner per (TP group, options) in this process."""
    options = options or DraftTailOptions.served()
    tp_size = getattr(mapping, "tp_size", 1) if mapping is not None else 1
    tp_rank = getattr(mapping, "tp_rank", 0) if mapping is not None else 0
    key = (getattr(mapping, "pp_rank", 0), tp_rank, tp_size, options)
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            runner = DraftTailRunner(tp_rank, tp_size, mapping, options)
            _RUNNERS[key] = runner
    return runner


def _head_is_plain_vocab_shard(lm_head, mapping) -> Optional[str]:
    """Reason the head is not a plain BF16 vocab shard, or None."""
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    if not isinstance(lm_head, LMHead):
        return "draft head is not an LMHead"
    if getattr(lm_head, "has_any_quant", False):
        return "quantized LM head"
    weight = getattr(lm_head, "weight", None)
    if (
        weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or not weight.is_contiguous()
        or weight.shape[1] != HIDDEN_SIZE
    ):
        return "LM head weight is not a contiguous BF16 [V, 6144] tensor"
    tp_size = getattr(mapping, "tp_size", 1)
    if tp_size > 1 and lm_head.tp_mode != TensorParallelMode.COLUMN:
        return "LM head is not vocab (column) parallel"
    if weight.shape[0] * tp_size >= 2**32:
        return "vocabulary too large for 32-bit ids"
    return None


def _max_draft_rows(spec_metadata) -> Optional[int]:
    """Most rows the draft LM head can see per step in this configuration.

    The draft loop gathers one row per request, padded requests included, so
    the engine's sequence-slot pool (``num_seq_slots``) bounds it. That pool
    is fixed per engine and kept by the CUDA-graph metadata copies, whose
    ``max_num_requests`` shrinks to the graph batch size; without a recorded
    pool, ``max_num_requests`` of the eager metadata (the maximum batch size)
    is used, and a graph copy gives no usable bound.
    """
    slots = getattr(spec_metadata, "num_seq_slots", 0) or 0
    if slots > 0:
        return slots
    if getattr(spec_metadata, "is_cuda_graph", False):
        return None
    return getattr(spec_metadata, "max_num_requests", 0) or None


def select_draft_tail(worker, draft_model, spec_metadata) -> Optional[Callable]:
    """Decide, before any device work, whether draft_tail serves this draft loop.

    Returns ``tail(request_hidden_states) -> Optional[int32 tokens]`` or None.
    The decision depends only on configuration and on batch-level flags that
    are identical on every TP rank, so all ranks take the same path. draft_tail
    engages only when no step of the configuration can have more than 8 rows,
    so the row count never moves a step between draft_tail and the native rounding.
    ``tail`` (``DraftTailRunner.step``) returns None (native path for that
    step) only for reasons every TP rank shares (row count, dtype, hidden
    width); a rank-local layout (strides, alignment) is staged into aligned
    rows instead, so no rank can leave draft_tail's exchange while others enter it.
    """
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTPHead

    mapping = getattr(worker, "mapping", None)
    tp_size = getattr(mapping, "tp_size", 1) if mapping is not None else 1
    reason = None
    if not getattr(worker, "is_mtp_eagle", False):
        reason = "not MTP-Eagle"
    elif getattr(worker, "guided_decoder", None) is not None:
        reason = "guided decoding reads draft logits"
    elif getattr(worker, "_d2t", None) is not None:
        reason = "draft-to-target vocab map"
    elif spec_metadata.wants_advanced_draft_sampling:
        reason = "advanced (rejection) draft sampling"
    elif tp_size > MAX_RANKS:
        reason = f"tp_size {tp_size} > {MAX_RANKS}"
    elif (
        mapping is not None
        and tp_size > 1
        and (
            getattr(mapping, "enable_attention_dp", False)
            or getattr(mapping, "enable_lm_head_tp_in_adp", False)
        )
    ):
        reason = "attention DP"
    elif type(draft_model.mtp_layers[0].shared_head) is not DeepseekV3MTPHead:
        reason = "MTP shared head is not DeepseekV3MTPHead"
    else:
        reason = _head_is_plain_vocab_shard(draft_model.lm_head, mapping)
    if reason is None and tp_size > 1:
        draft_vocab = (
            getattr(spec_metadata, "draft_vocab_size", 0)
            or getattr(spec_metadata, "vocab_size", 0)
            or 0
        )
        if not draft_vocab or draft_model.lm_head.weight.shape[0] >= draft_vocab:
            reason = "draft logits are not vocab-sharded"
    if reason is None:
        max_rows = _max_draft_rows(spec_metadata)
        if max_rows is None or max_rows > MAX_ROWS:
            reason = f"draft steps can have {max_rows} rows (> {MAX_ROWS} or unknown)"
    # Logged once per rank for eager and once for CUDA-graph forwards.
    log_key = (
        "draft_tail[graph]"
        if getattr(spec_metadata, "is_cuda_graph", False)
        else "draft_tail[eager]"
    )
    if reason is not None:
        log_capability(FEATURE, log_key, False, reason)
        return None
    runner = get_runner(mapping)
    if not runner.prepare():
        return None
    lm_head = draft_model.lm_head

    def tail(hidden: torch.Tensor) -> Optional[torch.Tensor]:
        return runner.step(hidden, lm_head)

    decision = (
        "native tokens"
        if runner.options.mode == "logits" or runner.options.variant in NATIVE_ORDER_VARIANTS
        else "near-tie variant"
    )
    log_capability(
        FEATURE,
        log_key,
        True,
        f"mode={runner.options.mode} ({decision}) variant="
        f"{runner.options.variant} exchange="
        f"{runner.options.exchange} tp={tp_size}",
    )
    return tail
