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
"""GQA decode FMHA libraries: trtllm-gen plus dedicated generation kernels, for the layers built with a GqaDecodeSetup.

``gqa_decode`` is trtllm-gen for such a layer that also consumes the FP8 query / K / V the attention module already
wrote (``AttentionForwardArgs.kv_append``) and runs a prologue the module deferred to it (``kv_append.fold``). The
libraries registered ahead of it take the generation calls their ``_is_supported`` declares, with an FP8 KV cache:

* ``gqa_kv_replica_fused`` / ``gqa_kv_replica`` (gqa_kv_replica.py): TP replicates this rank's one KV head over a group
  of ranks, which split the pages of each request and merge their partial results; partials from the fused kernel or
  from the segmented trtllm-gen decode;
* ``gqa_split_kv`` (gqa_split_kv_decode.py): one fused split-KV launch, which also runs a deferred prologue;
* ``gqa_segmented`` (gqa_segmented_decode.py): segmented split-KV on top of trtllm-gen.

They apply the output gate the module offers (``AttentionForwardArgs.output_gate``) and fall back to the trtllm-gen
decode kernel for a call whose runtime layout their kernel does not take.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantMode

from . import gqa_kv_replica_fused, gqa_split_kv_decode
from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha, _get_generation_workspace_layout
from .gqa_kv_replica import EXCHANGE_CHANNELS, kv_replica_decode, kv_replica_fused_decode
from .gqa_kv_replica_fused import kv_replica_fused_state, kv_replica_fused_supported
from .gqa_segmented_decode import segmented_decode, segmented_decode_supported
from .gqa_split_kv_decode import (
    split_kv_decode,
    split_kv_decode_buffers,
    split_kv_decode_fold,
    split_kv_decode_fold_supported,
    split_kv_decode_supported,
)
from .interface import FmhaPhase
from .phased import FmhaParams
from .utils import get_attention_chunk_size, get_bmm1_scale

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

# Segments of a segmented call.
_SEGMENTS = 16
# Requests a segmented call takes; above them the trtllm-gen kernel's own split-KV is bandwidth-bound.
_SEGMENTED_MAX_BATCH = 2
# Counter slots of a segmented call per request: segments x folded heads.
_COUNTER_SLOTS_PER_REQUEST = 128
# Smallest sizes of the KV-replica exchange: per-pair payload bytes, rows of one all-to-all merge.
_MIN_EXCHANGE_BYTES = 1 << 20
_MIN_MERGE_ROWS = 2048


@dataclass(frozen=True)
class GqaDecodeSetup:
    """What the attention module tells the GQA decode library about its layer."""

    # TP ranks holding this rank's KV head (consecutive TP ranks of ``mapping``); 1: none.
    kv_replicas: int = 1
    # Model mapping; its TP group carries the KV-replica split exchange.
    mapping: Optional[Mapping] = None
    # Largest q_len of a generation call (1 + draft tokens).
    tokens_per_gen_step: int = 1


@dataclass
class FoldInputs:
    """Raw inputs of the qk-norm / RoPE / gate / KV-append prologue of a folded call (see fold_inputs)."""

    # Raw QKV projection [num_tokens, (2 * Hq + 2 * Hkv) * D] bf16 ((q, gate) per head, K, V).
    qkv: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    cos_sin: torch.Tensor
    positions: torch.Tensor  # [T] or [3, T] (interleaved mrope), contiguous
    eps: float
    rotary_dim: int
    mrope_section: Optional[tuple]


@dataclass
class KvAppendPlan:
    """Paged FP8 destination of a generation-only call whose FP8 query / K / V the attention module writes.

    The module fills q_fp8 / scales and hands the plan over through ``AttentionForwardArgs.kv_append``.
    """

    layer_idx: int
    num_tokens: int
    num_requests: int
    q_len: int
    kv_pool: torch.Tensor  # FP8 [blocks, num_kv_heads, tokens_per_block, head_dim]
    block_tables: torch.Tensor  # int32 [num_requests, 2 (K, V), max_pages]
    seq_lens: torch.Tensor  # int32, KV length including the new tokens
    kv_scale_orig_quant: torch.Tensor
    kv_scale_quant_orig: torch.Tensor
    host_bmm1_scale: float
    q_fp8: Optional[torch.Tensor] = None  # [num_tokens, num_heads, head_dim]
    # float32 [8]: bmm1, bmm1 * log2e at [0:2]; bmm2 at [4] (both views 16-byte aligned)
    scales: Optional[torch.Tensor] = None
    # Set when the module deferred its prologue to the library (Fmha.supports_deferred_prologue); run_generation
    # then runs the folded kernel or the prologue, which fills q_fp8, scales and the contiguous gate.
    fold: Optional[FoldInputs] = None
    gate: Optional[torch.Tensor] = None


def _kv_replica_exchange_sizes(
    max_num_requests: int, tokens_per_gen_step: int, num_heads: int, head_dim: int
) -> Tuple[int, int]:
    """(per-pair payload bytes, all-to-all merge rows) of a layer's KV-replica exchange.

    A generation call moves at most max_num_requests x max(tokens_per_gen_step, 2) x num_heads rows (single-token
    calls send two partials per row) of D / 2 words plus one LSE word each; both sizes are rounded up to powers of
    two.
    """
    rows = max_num_requests * max(tokens_per_gen_step, 2) * num_heads
    merge_rows = max(_MIN_MERGE_ROWS, 1 << (rows - 1).bit_length())
    kib = math.ceil((rows * head_dim // 2 + merge_rows) * 4 / 1024)
    return max(_MIN_EXCHANGE_BYTES, (1 << (kib - 1).bit_length()) * 1024), merge_rows


class GqaDecodeFmha(FlashInferTrtllmGenFmha):
    """trtllm-gen for a layer built with a ``GqaDecodeSetup``; base of the dedicated GQA decode libraries."""

    supports_deferred_prologue = True

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        self._counter_buffer: Optional[torch.Tensor] = None

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            attn.gqa_decode is not None
            and not attn.is_mla_enable
            and attn.sparse_params is None
            and super()._is_available(attn)
        )

    @staticmethod
    def _has_fp8_kv_cache(attn: "TrtllmAttention") -> bool:
        quant_mode = QuantMode(attn.quant_mode)
        return quant_mode.has_fp8_kv_cache() and not quant_mode.has_fp4_kv_cache()

    def _decode_cell(
        self,
        q: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        phase: Optional[FmhaPhase],
        generation_only: bool,
    ) -> Optional[Tuple[int, int]]:
        """(requests, q_len) of a generation phase a dedicated kernel may take, at the upper bound of its selection
        cell (FP8 query, one beam, no cross attention or sinks), else None."""
        from .manager import generation_selection_cell

        if (
            phase == FmhaPhase.CONTEXT
            or metadata.num_generations <= 0
            or (generation_only and metadata.num_contexts != 0)
            or metadata.beam_width != 1
            or metadata.is_cross
            or forward_args.attention_sinks is not None
            or forward_args.output is None
            or not self._use_fp8_context_fmha(
                forward_args.output, forward_args.attention_input_type
            )
        ):
            return None
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)
        return generation_selection_cell(q, metadata, forward_args)

    def run_generation(self, params: FmhaParams) -> None:
        plan = params.fwd.kv_append
        if plan is not None and plan.fold is not None:
            self._run_fold_prologue(params, plan)
        super().run_generation(params)

    def _run_generation_preprocess(
        self,
        params: FmhaParams,
        attention_input: torch.Tensor,
        batch_beam: int,
        bmm1_scale_static: float,
        attention_chunk_size: int,
        fp8_context_fmha: bool,
    ) -> tuple:
        plan = params.fwd.kv_append
        if plan is not None:
            preprocessed = self._appended_preprocess(
                params, plan, batch_beam, attention_chunk_size, fp8_context_fmha
            )
            if preprocessed is not None:
                return preprocessed
        return super()._run_generation_preprocess(
            params,
            attention_input,
            batch_beam,
            bmm1_scale_static,
            attention_chunk_size,
            fp8_context_fmha,
        )

    def _run_generation_decode(
        self,
        params: FmhaParams,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        kv_scale_pool: Optional[torch.Tensor],
        max_kv_len: int,
        window_left: int,
        q_len_per_req: Optional[int],
        max_q_len: Optional[int],
        cu_seqlens: Optional[torch.Tensor],
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> None:
        if (
            q_len_per_req is not None
            and kv_scale_pool is None
            and window_left < 0
            and q.dim() == 3
            and q.is_contiguous()
            and q.dtype == torch.float8_e4m3fn
            and kv_pool.dim() == 4
            and kv_pool.is_contiguous()
            and self._run_dedicated(
                params, q, kv_pool, block_tables, max_kv_len, bmm1_scale, bmm2_scale, fmha_workspace
            )
        ):
            return
        super()._run_generation_decode(
            params,
            q,
            kv_pool,
            block_tables,
            kv_scale_pool,
            max_kv_len,
            window_left,
            q_len_per_req,
            max_q_len,
            cu_seqlens,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
        )

    def _run_dedicated(
        self,
        params: FmhaParams,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> bool:
        """Run this library's kernel on a uniform-q_len generation call with an FP8 query; False: trtllm-gen."""
        return False

    @staticmethod
    def _gate(params: FmhaParams) -> Optional[torch.Tensor]:
        """The offered output gate of a generation-only call (it covers every token of the forward), else None."""
        fwd = params.fwd
        plan = fwd.kv_append
        gate = plan.gate if plan is not None and plan.gate is not None else fwd.output_gate
        if gate is None or params.meta.num_contexts != 0 or gate.shape[0] < params.num_tokens:
            return None
        return gate[: params.num_tokens]

    @staticmethod
    def _counter_slots(metadata: "TrtllmAttentionMetadata") -> int:
        """Size of the counter buffer (_ensure_counter_buffer)."""
        return max(64, metadata.max_num_requests) * _COUNTER_SLOTS_PER_REQUEST

    def _ensure_counter_buffer(
        self, metadata: "TrtllmAttentionMetadata", device: torch.device
    ) -> None:
        """Counters of the segmented trtllm-gen calls: max(b * segments * heads, SMs) of them, more than the regular
        buffer holds. Zeroed once; the kernel resets them at the end of every launch."""
        slots = self._counter_slots(metadata)
        buf = self._counter_buffer
        if buf is None or buf.device != device or buf.numel() < slots:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The GQA decode counter buffer must be allocated before CUDA graph capture."
                )
            self._counter_buffer = torch.zeros(slots, dtype=torch.int32, device=device)

    @staticmethod
    def _run_fold_prologue(params: FmhaParams, plan: KvAppendPlan) -> None:
        """Run the prologue kernel the module skipped for a folded call that misses the folded kernel.

        Sets the packed QKV of this call as params.qkv_input and the prologue outputs on the plan.
        """
        from ...kernels.fused_qk_norm_rope_gate import fused_qkv_rope_gate_fp8_append

        attn = params.attn
        f = plan.fold
        qkv_out, gate_out, plan.q_fp8, plan.scales = fused_qkv_rope_gate_fp8_append(
            f.qkv,
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            f.rotary_dim,
            f.mrope_section,
            plan,
        )
        params.qkv_input = qkv_out[: params.num_tokens]
        if params.fwd.output_gate is not None:
            plan.gate = gate_out

    def _appended_preprocess(
        self,
        params: FmhaParams,
        plan: KvAppendPlan,
        batch_beam: int,
        attention_chunk_size: int,
        fp8_context_fmha: bool,
    ) -> Optional[tuple]:
        """trtllm_gen_generation_preprocess outputs for a call whose FP8 query / K / V the module wrote.

        None when the call does not match the plan; the C++ path then rewrites the same bytes from the bf16 QKV.
        """
        attn = params.attn
        fwd = params.fwd
        if not (
            fp8_context_fmha
            and plan.q_fp8 is not None
            and plan.scales is not None
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == batch_beam
            and plan.q_len == params.input_seq_length
            and params.seq_offset == 0
            and not params.is_cross
            and params.spec_decoding_generation_lengths is None
            and fwd.out_scale is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.qkv_input is not None
            and params.qkv_input.dtype == torch.bfloat16
        ):
            return None
        layout = _get_generation_workspace_layout(
            params.qkv_input.dtype,
            batch_beam,
            params.num_tokens,
            attn.num_heads,
            attn.head_dim,
            attn.num_kv_heads,
            attn.rope_params.dim,
        )
        offset = int(layout["trtllm_gen_workspace_offset"])
        size = int(layout["trtllm_gen_workspace_size"])
        workspace = params.workspace.view(-1).view(torch.uint8)
        if size <= 0 or offset + size > workspace.numel():
            return None
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            attention_chunk_size,
        )
        return (
            plan.q_fp8,
            plan.kv_pool,
            plan.block_tables,
            None,
            plan.scales.narrow(0, 0, 2),
            plan.scales.narrow(0, 4, 1),
            workspace.narrow(0, offset, size),
            None,
            params.input_seq_length,
            params.max_past_kv_length,
            window_left,
            False,
        )

    def kv_append_plan(
        self, metadata: "TrtllmAttentionMetadata", num_tokens: int
    ) -> Optional[KvAppendPlan]:
        """Destination for the module-side FP8 q / K / V write of a generation-only step, or None (stock path).

        Mirrors the conditions under which the C++ generation preprocess (applyBiasRopeUpdateKVCacheV2) writes FP8
        K / V and an FP8 query with a uniform q_len; run_generation re-checks the call before skipping its
        preprocess. None while Dynamo traces the forward: the plan reads the KV cache manager through nanobind,
        which Dynamo cannot trace.
        """
        attn = self.attn
        if num_tokens <= 0 or torch.compiler.is_compiling() or attn.head_dim % 16 != 0:
            return None
        quant_mode = QuantMode(attn.quant_mode)
        if not quant_mode.has_fp8_kv_cache() or quant_mode.has_fp4_kv_cache():
            return None
        if (
            metadata.kv_cache_block_offsets is None
            or metadata.host_kv_cache_pool_pointers is None
            or metadata.host_kv_cache_pool_mapping is None
            or metadata.kv_cache_manager is None
        ):
            return None
        num_seqs = metadata.num_generations
        if (
            metadata.num_contexts != 0
            or num_seqs <= 0
            or metadata.beam_width != 1
            or metadata.is_cross
            or num_tokens % num_seqs != 0
        ):
            return None
        if metadata.is_spec_decoding_enabled and attn.predicted_tokens_per_seq > 1:
            return None
        seq_lens = metadata.kv_lens_cuda_runtime
        if seq_lens is None or seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
            return None
        kv_scale_orig_quant = attn.kv_scale_orig_quant
        kv_scale_quant_orig = attn.kv_scale_quant_orig
        if not (
            isinstance(kv_scale_orig_quant, torch.Tensor)
            and isinstance(kv_scale_quant_orig, torch.Tensor)
            and kv_scale_orig_quant.dtype == torch.float32
            and kv_scale_quant_orig.dtype == torch.float32
        ):
            return None
        tokens_per_block = metadata.tokens_per_block
        if tokens_per_block & (tokens_per_block - 1):
            return None
        layer_idx = attn.get_local_layer_idx(metadata)
        kv_pool, block_tables, kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            metadata.host_kv_cache_pool_pointers,  # host_kv_cache_pool_pointers
            metadata.host_kv_cache_pool_mapping,  # host_kv_cache_pool_mapping
            metadata.kv_cache_block_offsets,  # kv_cache_block_offsets
            layer_idx,  # layer_idx
            attn.num_kv_heads,  # num_kv_heads
            tokens_per_block,  # tokens_per_block
            attn.head_dim,  # head_dim
            self.kv_factor,  # kv_factor
            self._get_total_num_blocks(metadata),  # total_num_blocks
            attn.quant_mode,  # kv_cache_quant_mode
            0,  # batch_start
            num_seqs,  # batch_size
            torch.bfloat16,  # dtype
        )
        if (
            kv_scale_pool is not None
            or kv_pool.dtype != torch.float8_e4m3fn
            or block_tables.dtype != torch.int32
        ):
            return None
        return KvAppendPlan(
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            num_requests=num_seqs,
            q_len=num_tokens // num_seqs,
            kv_pool=kv_pool,
            block_tables=block_tables,
            seq_lens=seq_lens,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            host_bmm1_scale=get_bmm1_scale(attn),
        )

    def fold_inputs(
        self,
        plan: KvAppendPlan,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        rotary_dim: int,
        mrope_section: Optional[tuple],
    ) -> Optional[FoldInputs]:
        """Fold inputs when the folded split-KV decode kernel can take a call with KV-append ``plan``, else None.

        The module then defers its prologue: gqa_split_kv runs the folded kernel, any other GQA decode library the
        prologue kernel.
        """
        attn = self.attn
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(qkv.device)
        if plan.num_tokens != qkv.shape[0] or plan.num_tokens != plan.num_requests * plan.q_len:
            return None
        if mrope_section is not None and positions.numel() != 3 * qkv.shape[0]:
            return None
        if not split_kv_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            plan.q_len,
            plan.num_requests,
            -1,
            False,
            plan.kv_pool.shape[2],
            torch.float8_e4m3fn,
            torch.bfloat16,
            plan.kv_pool.dtype,
            self._multi_processor_count,
        ) or not split_kv_decode_fold_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            rotary_dim,
            plan.q_len,
            qkv,
            positions,
            cos_sin,
        ):
            return None
        if not (
            q_weight.dtype == torch.bfloat16
            and k_weight.dtype == torch.bfloat16
            and q_weight.is_contiguous()
            and k_weight.is_contiguous()
            and q_weight.numel() == attn.head_dim
            and k_weight.numel() == attn.head_dim
        ):
            return None
        return FoldInputs(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin=cos_sin,
            positions=positions,
            eps=float(eps),
            rotary_dim=rotary_dim,
            mrope_section=tuple(mrope_section) if mrope_section is not None else None,
        )


class _GqaKvReplicaFmha(GqaDecodeFmha):
    """KV-replica split of the generation-only calls of two or more requests (gqa_kv_replica.py)."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            super()._is_available(attn)
            and attn.gqa_decode.kv_replicas > 1
            and attn.num_kv_heads == 1
            and cls._has_fp8_kv_cache(attn)
        )

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        setup = attn.gqa_decode
        self._kv_replicas = setup.kv_replicas
        self._mapping = setup.mapping
        self._tokens_per_gen_step = setup.tokens_per_gen_step
        self._exchange = None
        self._exchange_created = False

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        cell = self._decode_cell(q, metadata, forward_args, phase, generation_only=True)
        return (
            cell is not None
            and metadata.num_generations >= 2
            and self._takes(*cell, metadata)
            and super()._is_supported(q, k, v, metadata, forward_args, phase=phase)
        )

    def _takes(self, batch: int, q_len: int, metadata: "TrtllmAttentionMetadata") -> bool:
        return True

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        self._ensure_counter_buffer(metadata, q.device)
        if not self._exchange_created:
            # Collective over the TP group: every member selects this library for the same call.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "The KV-replica split exchange must be created before CUDA graph capture."
                )
            from tensorrt_llm._torch.distributed.ll_exchange import get_tp_group_exchange

            attn = self.attn
            max_bytes, max_merge_rows = _kv_replica_exchange_sizes(
                metadata.max_num_requests, self._tokens_per_gen_step, attn.num_heads, attn.head_dim
            )
            self._exchange = get_tp_group_exchange(
                self._mapping, self._kv_replicas, max_bytes, max_merge_rows, EXCHANGE_CHANNELS
            )
            self._exchange_created = True

    def _run_dedicated(
        self,
        params: FmhaParams,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> bool:
        attn = params.attn
        exchange = self._exchange
        # The q all-gather and the partials of one call must fit the exchange's per-pair payload.
        if (
            exchange is None
            or params.num_tokens * attn.num_heads * (attn.head_dim * 2 + 4) > exchange.max_words * 4
        ):
            return False
        gate = self._gate(params)
        self._run_kv_replica(
            params,
            exchange,
            q,
            kv_pool,
            block_tables,
            max_kv_len,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            gate,
        )
        if gate is not None:
            params.fwd.output_gate_applied = True
        return True


class GqaKvReplicaFusedFmha(_GqaKvReplicaFmha):
    """KV-replica split with the fused partial attention kernel (gqa_kv_replica_fused.py)."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            IS_CUTLASS_DSL_AVAILABLE
            and attn.head_dim == gqa_kv_replica_fused.HEAD_DIM
            and super()._is_available(attn)
        )

    def _takes(self, batch: int, q_len: int, metadata: "TrtllmAttentionMetadata") -> bool:
        return (
            metadata.tokens_per_block == gqa_kv_replica_fused.PAGE
            and q_len <= gqa_kv_replica_fused.MAX_Q_LEN
            and kv_replica_fused_supported(
                q_len, self._kv_replicas, self.attn.num_heads, batch, self._multi_processor_count
            )
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        kv_replica_fused_state(q.device, self._multi_processor_count, metadata.max_num_requests)

    def _run_kv_replica(
        self,
        params,
        exchange,
        q,
        kv_pool,
        block_tables,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace,
        gate,
    ) -> None:
        rows, hq, d = q.shape
        kv_replica_fused_decode(
            exchange,
            q,
            kv_pool,
            block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            bmm1_scale,
            bmm2_scale,
            params.output.view(rows, hq, d),
            kv_replica_fused_state(
                q.device, self._multi_processor_count, params.meta.max_num_requests
            ),
            self._enable_pdl,
            gate=gate,
        )


class GqaKvReplicaFmha(_GqaKvReplicaFmha):
    """KV-replica split with partials from the segmented trtllm-gen decode."""

    @staticmethod
    def _num_segments(num_requests: int, q_len: int) -> int:
        """Local segments per member: two when the batch alone gives the KV-replica group too few CTAs."""
        return 2 if (num_requests <= 2 or q_len == 1) else 1

    def _takes(self, batch: int, q_len: int, metadata: "TrtllmAttentionMetadata") -> bool:
        # One counter per (segment request, folded head) of the segment attention: b * segments x q_len x group
        # heads, at the largest b * segments of a call in this cell (the segments drop from two to one above two
        # requests).
        requests = min(batch, metadata.max_num_requests)
        segment_requests = max(
            b * self._num_segments(b, q_len) for b in (requests, min(requests, 2))
        )
        group_heads = self._kv_replicas * self.attn.num_heads
        return segment_requests * q_len * group_heads <= self._counter_slots(metadata)

    def _run_kv_replica(
        self,
        params,
        exchange,
        q,
        kv_pool,
        block_tables,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace,
        gate,
    ) -> None:
        rows, hq, d = q.shape
        num_segments = self._num_segments(params.num_requests, params.input_seq_length)
        kv_replica_decode(
            exchange,
            q,
            kv_pool,
            block_tables,
            params.sequence_lengths,
            max_kv_len,
            params.input_seq_length,
            num_segments,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            self._counter_buffer,
            params.output.view(rows, hq, d),
            self._enable_pdl,
            gate=gate,
        )


class GqaSplitKvFmha(GqaDecodeFmha):
    """Fused split-KV decode of generation-only calls of up to four requests (gqa_split_kv_decode.py); also runs a
    deferred prologue in the same launch."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return (
            IS_CUTLASS_DSL_AVAILABLE
            and attn.num_kv_heads == 1
            and attn.head_dim == gqa_split_kv_decode.HEAD_DIM
            and cls._has_fp8_kv_cache(attn)
            and super()._is_available(attn)
        )

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        cell = self._decode_cell(q, metadata, forward_args, phase, generation_only=True)
        if cell is None:
            return False
        batch, q_len = cell
        attn = self.attn
        return split_kv_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            q_len,
            batch,
            -1,
            False,
            metadata.tokens_per_block,
            torch.float8_e4m3fn,
            forward_args.output.dtype,
            torch.float8_e4m3fn,
            self._multi_processor_count,
        ) and super()._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        split_kv_decode_buffers(q.device, self._multi_processor_count)

    def run_generation(self, params: FmhaParams) -> None:
        plan = params.fwd.kv_append
        if plan is not None and plan.fold is not None:
            fp8_context_fmha = self._use_fp8_context_fmha(
                params.fwd.output, params.fwd.attention_input_type
            )
            if self._run_folded(params, plan, fp8_context_fmha):
                return
        super().run_generation(params)

    def _run_dedicated(
        self,
        params: FmhaParams,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> bool:
        attn = params.attn
        if not split_kv_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.input_seq_length,
            params.num_requests,
            -1,
            False,
            params.tokens_per_block,
            q.dtype,
            params.output.dtype,
            kv_pool.dtype,
            self._multi_processor_count,
        ):
            return False
        gate = self._gate(params)
        ws, cnt = split_kv_decode_buffers(q.device, self._multi_processor_count)
        split_kv_decode(
            q,
            kv_pool,
            block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            bmm1_scale,
            bmm2_scale,
            params.output,
            ws,
            cnt,
            self._multi_processor_count,
            self._enable_pdl,
            gate=gate,
        )
        if gate is not None:
            params.fwd.output_gate_applied = True
        return True

    def _run_folded(self, params: FmhaParams, plan: KvAppendPlan, fp8_context_fmha: bool) -> bool:
        """Run the folded split-KV decode when the call matches the plan; False otherwise."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        f = plan.fold
        if not (
            fwd.output_gate is not None
            and fp8_context_fmha
            and meta.num_contexts == 0
            and meta.beam_width == 1
            and not params.is_cross
            and params.seq_offset == 0
            and params.spec_decoding_generation_lengths is None
            and plan.layer_idx == attn.local_layer_idx
            and plan.num_tokens == params.num_tokens
            and plan.num_requests == params.num_requests
            and plan.q_len == params.input_seq_length
            and fwd.out_scale is None
            and fwd.attention_sinks is None
            and fwd.kv_scale_orig_quant is plan.kv_scale_orig_quant
            and fwd.kv_scale_quant_orig is plan.kv_scale_quant_orig
            and params.sequence_lengths is not None
            and params.sequence_lengths.data_ptr() == plan.seq_lens.data_ptr()
            and params.output is not None
            and params.output.dtype == torch.bfloat16
            and not QuantMode(attn.quant_mode).has_fp4_kv_cache()
            and self._multi_processor_count is not None
        ):
            return False
        window_left = self._compute_window_left(
            params.cyclic_attention_window_size,
            params.max_past_kv_length,
            get_attention_chunk_size(attn),
        )
        if window_left >= 0 or not split_kv_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.input_seq_length,
            params.num_requests,
            window_left,
            False,
            params.tokens_per_block,
            torch.float8_e4m3fn,
            params.output.dtype,
            plan.kv_pool.dtype,
            self._multi_processor_count,
        ):
            return False
        ws, cnt = split_kv_decode_buffers(params.output.device, self._multi_processor_count)
        split_kv_decode_fold(
            f.qkv[: params.num_tokens],
            f.q_weight,
            f.k_weight,
            f.cos_sin,
            f.positions,
            f.eps,
            f.rotary_dim,
            f.mrope_section,
            plan.kv_pool,
            plan.block_tables,
            params.sequence_lengths,
            params.input_seq_length,
            plan.kv_scale_orig_quant,
            plan.kv_scale_quant_orig,
            plan.host_bmm1_scale,
            params.output,
            ws,
            cnt,
            self._multi_processor_count,
            self._enable_pdl,
        )
        fwd.output_gate_applied = True
        return True


class GqaSegmentedFmha(GqaDecodeFmha):
    """Segmented split-KV on top of trtllm-gen for generation phases of up to two requests
    (gqa_segmented_decode.py)."""

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        return cls._has_fp8_kv_cache(attn) and super()._is_available(attn)

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        cell = self._decode_cell(q, metadata, forward_args, phase, generation_only=False)
        if cell is None or metadata.num_generations > _SEGMENTED_MAX_BATCH:
            return False
        batch, q_len = cell
        attn = self.attn
        return segmented_decode_supported(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            q_len,
            batch,
            _SEGMENTS,
            -1,
            False,
            metadata.tokens_per_block,
            torch.float8_e4m3fn,
            forward_args.output.dtype,
        ) and super()._is_supported(q, k, v, metadata, forward_args, phase=phase)

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        super().prepare_workspace(q, k, v, metadata, forward_args, workspace)
        self._ensure_counter_buffer(metadata, q.device)

    def _run_dedicated(
        self,
        params: FmhaParams,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        max_kv_len: int,
        bmm1_scale,
        bmm2_scale,
        fmha_workspace: torch.Tensor,
    ) -> bool:
        attn = params.attn
        gate = self._gate(params)
        segmented_decode(
            q,
            kv_pool,
            block_tables,
            params.sequence_lengths,
            max_kv_len,
            params.input_seq_length,
            attn.num_kv_heads,
            _SEGMENTS,
            bmm1_scale,
            bmm2_scale,
            fmha_workspace,
            self._counter_buffer,
            params.output,
            self._enable_pdl,
            gate=gate,
        )
        if gate is not None:
            params.fwd.output_gate_applied = True
        return True
