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
"""DSpark speculative-decoding worker of the Kimi K3 family.

``K3DSparkWorker`` drives the family's ``K3DSparkDrafter`` (``drafter.py``) through the engine's
one-engine spec interfaces: it subclasses ``SpecWorkerBase`` and consumes the engine's
``DFlashSpecMetadata``. The drafter's context KV lives in the target KV cache manager's appended
draft layers when the engine unifies them (``draft_layer_ids``, reached through
``get_draft_kv_cache_view``): the worker decodes their pages from ``kv_cache_block_offsets`` in
its own forward, and a generation request that arrives without a prefill here (disaggregated
serving) starts its context at its first position.

Target and greedy draft tokens come from ``K3VocabParallelHead`` (MNNVL pair exchange,
one-launch Markov chains, temperature-only sampling) wherever the target LM head is a plain
vocab shard. The deployment's ``K3DSparkPath`` selects the rest:

* ``spec_tail_fusion``: fused strict acceptance, fused selection + acceptance, the parallel
  block-table decode, the context latents on a side stream and
  ``_prepare_gen_drafter_inputs_fused`` in place of ``_prepare_paged_latent_gen_inputs``; the
  drafter embedding there runs as the pushed MNNVL lookup where it applies.
* ``tail_v2`` (on ``spec_tail_fusion``): the KV rewind in the next-new-tokens kernel, the
  vocab-sharded embedding and the context latents' fused norm + rope.

Vocab-parallel token selection for a vocab-sharded LM head.

A column-parallel LM head leaves every TP rank with a contiguous vocab shard of
the logits. Gathering the full logits to pick one token per row moves
``rows * vocab`` values; this module instead reduces each rank's shard to one
``(value, global index)`` pair per row, exchanges only those pairs, and takes
the winner. The result equals ``torch.argmax`` over the gathered logits bit for
bit: ``NaN`` beats every number, and equal maxima resolve to the lowest global
index (rank shards are contiguous and ordered, so the lowest rank's lowest
column wins).

Three consumers:

* :func:`vocab_parallel_argmax` -- greedy target sampling.
* :func:`vocab_parallel_sample` -- temperature-only sampling through the
  Gumbel-max trick: ``argmax(logits / T + G)`` with ``G`` standard Gumbel noise
  is an exact draw from ``softmax(logits / T)``. The noise of every vocab entry
  comes from a counter-based Philox stream keyed by the row's seed, so all
  ranks agree on the winner. The distribution matches the gathered sampler; the
  individual draws do not (different random stream).
* :func:`vocab_parallel_markov_tokens` -- the DSpark vanilla-Markov greedy
  block chain on shard logits: step ``i`` adds ``markov_w2 @ markov_w1[prev_i]``
  to block position ``i`` and takes the global argmax, which becomes
  ``prev_{i+1}`` and draft token ``i``.

The pair exchange is an all-reduce (sum) of a zero-padded ``[rows, 2 * tp]``
float32 buffer in which each rank fills only its own two columns (``x + 0`` is
exact), over the MNNVL all-reduce when available and NCCL otherwise. Every
launch has a shape fixed by ``(rows, shard width)`` and nothing synchronizes
with the host, so all paths are CUDA-graph capturable.

Non-greedy target rows are sampled vocab-parallel only when the deployment's
``advanced_sampling_mode`` is ``no_topk_no_topp``. The draft Markov chain of up
to 16 requests and 16 block positions over the MNNVL (or single-rank) exchange
runs as one launch: the CuTe DSL kernel
(:meth:`K3VocabParallelHead.markov_tokens_v3`) on SM100-family devices where it
applies, else :meth:`K3VocabParallelHead.markov_tokens_chain`; any other chain
takes :meth:`K3VocabParallelHead.markov_tokens`.

An MNNVL all-reduce with its own Lamport workspace.

The MNNVL all-reduces of one mapping (``layers.TpAllReduce``) share a single
workspace (buffers plus the flags that sequence them), so two of them must not be in
flight at once -- they would interleave in the same Lamport buffers. A reduction that
runs on a side stream concurrently with the main stream's all-reduces needs its own
workspace; this class allocates one the same way
``get_or_scale_allreduce_mnnvl_workspace`` does and calls the same fused kernel, so its
results are those of the shared-workspace all-reduce for the same inputs.
"""

import math
import weakref
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F
import triton
from torch import nn

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.distributed.ops import (
    get_or_scale_allreduce_mnnvl_workspace,
    use_mnnvl_fabric_handle,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager
from tensorrt_llm._torch.speculative import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.accept_stats import maybe_create_recorder
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    SpecWorkerBase,
)
from tensorrt_llm._utils import mpi_comm, prefer_pinned
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, AllReduceStrategy
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..kernels.k3r_spec.dspark_markov_fused import fused_markov_supported
from ..kernels.k3r_spec.spec_tail_kernels import (
    BLOCK_TABLE_CHUNK,
    TAIL_V2_MAX_GENS,
    DraftBlockPlacement,
    vocab_shard_embedding_applies,
)
from ..kernels.k3r_spec.vocab_parallel_kernels import (
    _CHAIN_MAX_REQS,
    _CHAIN_MAX_STEPS,
    _LOCAL_BLOCK_V,
    _MARKOV_BLOCK_V,
    _PairLayout,
)
from ..ops.comm.sm107_allreduce import TpAllReduce, mnnvl_serves, mnnvl_workspace_bytes
from ..ops.spec.k3_dflash_ctx_positions import k3_dflash_ctx_positions
from ..ops.spec.k3_dflash_gen_prepare import k3_dflash_gen_prepare
from ..ops.spec.k3_dflash_noise_embedding import k3_dflash_noise_embedding
from ..ops.spec.k3_dflash_paged_latent_prepare import k3_dflash_paged_latent_prepare
from ..ops.spec.k3_draft_block_table import k3_draft_block_table
from ..ops.spec.k3_fused_markov_chain_logits import k3_fused_markov_chain_logits
from ..ops.spec.k3_fused_select_accept import k3_fused_select_accept
from ..ops.spec.k3_fused_strict_accept import k3_fused_strict_accept
from ..ops.spec.k3_markov_add_local_argmax import k3_markov_add_local_argmax
from ..ops.spec.k3_markov_chain import k3_markov_chain
from ..ops.spec.k3_markov_reduce_embed import k3_markov_reduce_embed
from ..ops.spec.k3_next_new_tokens import k3_next_new_tokens
from ..ops.spec.k3_store_context_latents import k3_store_context_latents
from ..ops.spec.k3_vocab_local_select import k3_vocab_local_select
from ..ops.spec.k3_vocab_reduce_pairs import k3_vocab_reduce_pairs
from ..ops.spec.k3_vocab_shard_embedding import k3_vocab_shard_embedding
from .drafter import K3DSparkPath, _MLABlockFixup
from .primitives import HelixDraftShard, draft_embed_site, to_draft_fp8

_NUM_LAMPORT_BUFFERS = 3


class PrivateMNNVLAllReduce:
    """The MNNVL all-reduce over a private workspace (for a side stream).

    Construction is collective over the mapping's TP group.
    """

    def __init__(self, mapping: Mapping, dtype: torch.dtype, max_tokens: int, hidden: int):
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        self.mapping = mapping
        self.dtype = dtype
        elem = torch.tensor([], dtype=dtype).element_size()
        need = mnnvl_workspace_bytes(max_tokens, hidden, mapping.tp_size, dtype)
        self.buffer_size_bytes = math.ceil(max(need, 1) / (8 * 1024 * 1024)) * (8 * 1024 * 1024)
        self.max_tokens = max_tokens
        self.hidden = hidden
        self._elem = elem
        workspace_bytes = _NUM_LAMPORT_BUFFERS * self.buffer_size_bytes
        comm = mpi_comm().Split(
            int(mapping.pp_rank * mapping.cp_size + mapping.cp_rank), mapping.tp_rank
        )
        self._handle = McastGPUBuffer(
            workspace_bytes,
            mapping.tp_size,
            mapping.tp_rank,
            mapping.local_rank,
            use_mnnvl_fabric_handle(mapping),
            comm.py2f(),
        )
        buffer = self._handle.get_uc_buffer(
            mapping.tp_rank, (workspace_bytes // torch.float32.itemsize,), torch.float32, 0
        )
        buffer.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._comm = comm
        self._buffer_base = buffer.view(dtype).view(_NUM_LAMPORT_BUFFERS, -1)
        # [cur idx, dirty idx, bytes per buffer, dirty num stages, numBytesToClear[4], access count]
        self._buffer_flags = torch.tensor(
            [0, 2, self.buffer_size_bytes, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=torch.device("cuda", mapping.local_rank),
        )

    def fits(self, num_tokens: int, hidden: int) -> bool:
        need = mnnvl_workspace_bytes(num_tokens, hidden, self.mapping.tp_size, self.dtype)
        return need <= self.buffer_size_bytes

    def rms_norm(self, partial: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
        """``RMSNorm(all_reduce(partial))``, as the MNNVL all-reduce with ``RMS_NORM``."""
        assert partial.dtype == self.dtype and self.fits(
            partial.numel() // partial.shape[-1], partial.shape[-1]
        )
        outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
            partial,
            gamma,
            None,
            eps,
            self._buffer_base,
            self._buffer_flags,
            True,
            None,
            int(AllReduceFusionOp.RMS_NORM),
        )
        return outputs[0]


# Programs the local reductions aim to launch (rows x column splits).
_TARGET_PROGRAMS = 256


class _PairExchange:
    """Exchanges ``(value, global index)`` float32 pairs per row across TP.

    Over MNNVL each rank owns ``splits`` consecutive pair slots of a zero-padded
    ``[rows, width]`` buffer that an all-reduce sums; over NCCL each rank
    publishes one pair per row and the pairs are all-gathered.
    """

    def __init__(self, mapping: Mapping):
        self.mapping = mapping
        self.tp_size = mapping.tp_size
        self.tp_rank = mapping.tp_rank
        self._mnnvl = None
        if self.tp_size > 1 and mnnvl_serves(mapping, torch.float32, AllReduceStrategy.AUTO):
            try:
                self._mnnvl = TpAllReduce(mapping, torch.float32)
            except (RuntimeError, ValueError) as e:
                logger.info(f"Vocab-parallel pair exchange: MNNVL unavailable ({e}); using NCCL.")
        self.backend = "mnnvl" if self._mnnvl is not None else "nccl"

    def chain_buffers(self, device):
        """``(multicast view, local view, set floats)`` of the one-launch Markov
        chain's exchange: three sets of ``[steps, requests, ranks, 2]`` float32
        pairs, -0.0 where nothing was written. Allocated collectively on first
        use (every rank reaches it at the same point of the warmup), outside
        graph capture. A single rank uses one local buffer for both views."""
        if getattr(self, "_chain", None) is not None:
            return self._chain
        assert not torch.cuda.is_current_stream_capturing()
        set_floats = _CHAIN_MAX_STEPS * _CHAIN_MAX_REQS * max(self.tp_size, 1) * 2
        numel = 3 * set_floats
        if self.tp_size == 1:
            buf = torch.full((numel,), -0.0, dtype=torch.float32, device=device)
            self._chain = (buf, buf, set_floats)
            return self._chain
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert self.backend == "mnnvl"
        # The communicator of the exchange's MNNVL workspace (allocated with it).
        comm = get_or_scale_allreduce_mnnvl_workspace(self.mapping, torch.float32)["comm"]
        handle = McastGPUBuffer(
            numel * 4,
            self.tp_size,
            self.tp_rank,
            self.mapping.local_rank,
            use_mnnvl_fabric_handle(self.mapping),
            comm.py2f(),
        )
        local = handle.get_uc_buffer(self.tp_rank, [numel], torch.float32, 0)
        mcast = handle.get_mc_buffer([numel], torch.float32, 0)
        local.fill_(-0.0)
        torch.cuda.synchronize()
        comm.Barrier()
        self._chain_handle = handle
        self._chain = (mcast, local, set_floats)
        return self._chain

    def markov_v3_buffers(self, device, entries: int):
        """``(local address, multicast address, state)`` of the CuTe DSL Markov
        chain's key exchange for ``entries`` (ranks x CTAs per rank): its
        rotating sets, zero-filled, and this rank's int32 launch state. Allocated
        collectively on first use, outside graph capture. A single rank uses one
        local buffer for both addresses."""
        cached = getattr(self, "_markov_v3", None)
        if cached is not None and cached[0] == entries:
            return cached[1:4]
        assert cached is None, "the Markov exchange is sized once per process"
        assert not torch.cuda.is_current_stream_capturing()
        from ..kernels.k3r_spec.dspark_markov_greedy import NUM_SETS, STATE_WORDS, set_bytes

        words = NUM_SETS * set_bytes(entries) // 4
        state = torch.zeros((STATE_WORDS,), dtype=torch.int32, device=device)
        if self.tp_size == 1:
            buf = torch.zeros((words,), dtype=torch.int32, device=device)
            self._markov_v3 = (entries, buf.data_ptr(), buf.data_ptr(), state, buf)
            return self._markov_v3[1:4]
        from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

        assert self.backend == "mnnvl"
        # The communicator of the exchange's MNNVL workspace (allocated with it).
        comm = get_or_scale_allreduce_mnnvl_workspace(self.mapping, torch.float32)["comm"]
        handle = McastGPUBuffer(
            words * 4,
            self.tp_size,
            self.tp_rank,
            self.mapping.local_rank,
            use_mnnvl_fabric_handle(self.mapping),
            comm.py2f(),
        )
        local = handle.get_uc_buffer(self.tp_rank, [words], torch.int32, 0)
        mcast = handle.get_mc_buffer([words], torch.int32, 0)
        local.zero_()
        torch.cuda.synchronize()
        comm.Barrier()
        self._markov_v3 = (entries, local.data_ptr(), mcast.data_ptr(), state, handle)
        return self._markov_v3[1:4]

    def layout(self, rows: int, n_cols: int, device) -> _PairLayout:
        # Split each row over enough programs to fill the GPU when rows are few;
        # the per-split pairs travel as extra slots of the same exchange.
        max_splits = triton.cdiv(n_cols, _LOCAL_BLOCK_V)
        splits = 1
        if self.backend == "mnnvl" and self.tp_size > 1:
            splits = max(1, min(max_splits, triton.cdiv(_TARGET_PROGRAMS, max(rows, 1))))
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _LOCAL_BLOCK_V) * _LOCAL_BLOCK_V
        splits = max(1, triton.cdiv(n_cols, chunk))
        if self.tp_size == 1 or self.backend == "nccl":
            buf = torch.empty((rows, 2 * splits), dtype=torch.float32, device=device)
            return _PairLayout(buf, 2 * splits, 0, 0, 0, 0, splits, chunk)
        width = triton.next_power_of_2(2 * self.tp_size * splits)
        buf = torch.empty((rows, width), dtype=torch.float32, device=device)
        lo = 2 * self.tp_rank * splits
        return _PairLayout(
            buf, width, self.tp_rank * splits, lo, lo + 2 * splits, width, splits, chunk
        )

    def exchange(self, layout: _PairLayout) -> tuple[torch.Tensor, int, int, int]:
        """Every rank's pairs: (tensor, row stride, slot stride, number of slots)."""
        buf = layout.buffer
        rows = buf.shape[0]
        if self.tp_size == 1:
            return buf, layout.stride_out, 2, layout.splits
        if self.backend == "mnnvl":
            out = self._mnnvl(buf, all_reduce_params=AllReduceParams())
            return out, layout.stride_out, 2, self.tp_size * layout.splits
        from tensorrt_llm._torch.distributed.ops import allgather

        assert layout.splits == 1
        out = allgather(buf, self.mapping, dim=0)
        return out, 2, 2 * rows, self.tp_size


class K3VocabParallelHead:
    """Vocab-parallel token selection over one column-parallel LM head.

    Args:
        lm_head: an ``LMHead`` (or ``Linear``) in COLUMN tensor-parallel mode
            whose weight holds this rank's contiguous vocab shard.
        vocab_size: the full vocabulary size.
    """

    def __init__(self, lm_head, vocab_size: int):
        self.lm_head = lm_head
        self.mapping = lm_head.mapping
        self.vocab_size = int(vocab_size)
        self.shard_width = int(lm_head.out_features)
        self.col_base = self.mapping.tp_rank * self.shard_width
        self.exchange = _PairExchange(self.mapping)
        self._markov_w2_shard = None
        self._markov_w2_src_ptr = None
        logger.info(
            f"Vocab-parallel LM-head token selection: tp={self.mapping.tp_size}, "
            f"shard={self.shard_width}, vocab={self.vocab_size}, exchange={self.exchange.backend}"
        )

    @staticmethod
    def supported(lm_head, vocab_size: int) -> bool:
        """Whether ``lm_head`` is a plain vocab-sharded head this class handles."""
        mapping = getattr(lm_head, "mapping", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return False
        # The engine's LM head carries its parallel mode as an enum; only its name is read here.
        if getattr(getattr(lm_head, "tp_mode", None), "name", None) != "COLUMN":
            return False
        if not getattr(lm_head, "gather_output", False):
            return False
        if getattr(lm_head, "padding_size", 0) != 0 or getattr(lm_head, "has_any_quant", False):
            return False
        # The sampler draws Philox noise for 4-column groups of the global vocab.
        shard = int(lm_head.out_features)
        return shard % 4 == 0 and shard * mapping.tp_size == int(vocab_size) and vocab_size < 2**24

    def local_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's ``[rows, shard]`` logits in the head's output dtype."""
        return self.lm_head.apply_linear(hidden_states, self.lm_head.bias)

    def shard_logits(self, hidden_states: torch.Tensor) -> "K3VocabShardedLogits":
        """This rank's logits wrapped for a spec worker (see ``K3VocabShardedLogits``)."""
        return K3VocabShardedLogits(self, self.local_logits(hidden_states))

    def gather_full_logits(self, local: torch.Tensor) -> torch.Tensor:
        """Full float32 logits, identical to the gathered head + ``.float()``."""
        from tensorrt_llm._torch.distributed import allgather

        return allgather(local, self.mapping, sizes=self.lm_head.gather_output_sizes).float()

    def _launch_local(self, local: torch.Tensor, sampling: Optional[tuple]) -> _PairLayout:
        rows, n_cols = local.shape
        layout = self.exchange.layout(rows, n_cols, local.device)
        k3_vocab_local_select(local, self.col_base, layout, sampling)
        return layout

    def _exchange_and_reduce(self, layout: _PairLayout, out: torch.Tensor):
        pairs, stride_row, stride_slot, n_slots = self.exchange.exchange(layout)
        return k3_vocab_reduce_pairs(pairs, stride_row, stride_slot, n_slots, out)

    def selection_pairs(self, local: torch.Tensor, spec_metadata) -> tuple:
        """Every rank's ``(value, global index)`` pairs of the target selection,
        not yet reduced: ``(pairs, row stride, slot stride, slots)``.

        The same local kernel and exchange as :meth:`K3VocabShardedLogits.select_tokens`
        (argmax for an all-greedy batch, else the Gumbel-max draw), for a caller
        that reduces the pairs inside a fused consumer kernel.
        """
        assert local.dim() == 2 and local.stride(-1) == 1
        if spec_metadata.is_all_greedy_sample:
            layout = self._launch_local(local, None)
        else:
            from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
                GREEDY_TEMPERATURE_THRESHOLD,
            )

            rows = local.shape[0]
            layout = self._launch_local(
                local,
                (
                    spec_metadata.temperatures[:rows],
                    spec_metadata.seeds[:rows],
                    spec_metadata.offsets[:rows],
                    float(GREEDY_TEMPERATURE_THRESHOLD),
                ),
            )
        return self.exchange.exchange(layout)

    def argmax(self, local: torch.Tensor, out_dtype=torch.int64) -> torch.Tensor:
        """Global argmax over the vocab of shard logits ``[rows, shard]``."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(local, None)
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, out)

    def sample(
        self,
        local: torch.Tensor,
        temperatures: torch.Tensor,
        seeds: torch.Tensor,
        offsets: torch.Tensor,
        greedy_threshold: float,
        out_dtype=torch.int64,
    ) -> torch.Tensor:
        """Temperature-only sampling (Gumbel-max); rows at or below the greedy
        temperature threshold take the plain argmax."""
        assert local.dim() == 2 and local.stride(-1) == 1
        layout = self._launch_local(local, (temperatures, seeds, offsets, float(greedy_threshold)))
        out = torch.empty((local.shape[0],), dtype=out_dtype, device=local.device)
        return self._exchange_and_reduce(layout, out)

    def markov_tokens_chain(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens` with the whole chain in one launch.

        Every step resolves prev, gathers its ``markov_w1`` row, computes this
        rank's ``markov_w2`` bias columns, adds the step's shard logits and
        publishes the rank's (value, global index) pair; the per-rank pairs
        travel through a multicast buffer that every program polls (see
        ``_markov_chain_kernel``) instead of an all-reduce launch. The bias is
        an FP32 dot of the BF16 row and weights rounded to BF16, like the
        ``F.linear`` of :meth:`markov_tokens`, but its summation order differs,
        so a draft token can differ where two vocab entries tie to within that
        rounding. Needs ``num_reqs <= 16`` and ``block <= 16`` (the caller falls
        back otherwise) and the MNNVL (or single-rank) exchange.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        assert num_reqs <= _CHAIN_MAX_REQS and block_size <= _CHAIN_MAX_STEPS
        w2_shard = self.markov_w2_shard(markov_w2)
        assert (
            w2_shard.stride(-1) == 1 and markov_w1.is_contiguous() and local_block.stride(-1) == 1
        )
        programs = triton.cdiv(shard, _MARKOV_BLOCK_V)
        # Every program polls for the others' pairs: all of them must be resident.
        assert programs <= torch.cuda.get_device_properties(device).multi_processor_count
        mcast, local, set_floats = self.exchange.chain_buffers(device)
        state = getattr(self, "_chain_state", None)
        if state is None or state[0] != (programs, str(device)):
            partial = torch.empty(
                (_CHAIN_MAX_STEPS * programs * _CHAIN_MAX_REQS * 2,),
                dtype=torch.float32,
                device=device,
            )
            tickets = torch.zeros((_CHAIN_MAX_STEPS,), dtype=torch.int32, device=device)
            launch = torch.zeros((1,), dtype=torch.int32, device=device)
            self._chain_state = ((programs, str(device)), partial, tickets, launch)
        _, partial, tickets, launch = self._chain_state
        tp = max(self.exchange.tp_size, 1)
        return k3_markov_chain(
            local_block,
            first_prev_tokens,
            markov_w1,
            w2_shard,
            tokens,
            partial,
            tickets,
            launch,
            mcast,
            local,
            set_floats=set_floats,
            vocab=self.vocab_size,
            col_base=self.col_base,
            tp=tp,
            tp_rank=self.mapping.tp_rank if tp > 1 else 0,
        )

    def markov_v3_supported(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> bool:
        """Whether :meth:`markov_tokens_v3` serves these operands: BF16 logits
        and Markov weights (rank a multiple of 16, at most 256), at most 16
        requests and 16 block positions, an SM100-family device with a CTA per
        256 shard rows resident at once, and the MNNVL (or single-rank)
        exchange."""
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

        if not IS_CUTLASS_DSL_AVAILABLE or markov_w1 is None or markov_w2 is None:
            return False
        if self.exchange.tp_size > 1 and self.exchange.backend != "mnnvl":
            return False
        num_reqs, block_size, shard = local_block.shape
        rank_dim = markov_w1.shape[1]
        device = local_block.device
        if not (1 <= num_reqs <= 16 and 1 <= block_size <= 16) or shard != self.shard_width:
            return False
        if (
            local_block.dtype != torch.bfloat16
            or markov_w1.dtype != torch.bfloat16
            or markov_w2.dtype != torch.bfloat16
            or local_block.stride(-1) != 1
            or not markov_w1.is_contiguous()
            or not markov_w2.is_contiguous()
            or markov_w2.shape[1] != rank_dim
            or rank_dim % 16
            or rank_dim > 256
            or first_prev_tokens.dtype not in (torch.int32, torch.int64)
            or first_prev_tokens.stride(0) != 1
            or markov_w2.data_ptr() % 16
        ):
            return False
        from ..kernels.k3r_spec.dspark_markov_greedy import ctas_for, smem_bytes

        props = torch.cuda.get_device_properties(device)
        return (
            props.major == 10
            and ctas_for(shard) <= props.multi_processor_count
            and smem_bytes(rank_dim) <= props.shared_memory_per_block_optin
        )

    def markov_tokens_v3(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
    ) -> torch.Tensor:
        """:meth:`markov_tokens_chain` as one CuTe DSL launch without the
        intra-GPU reduction.

        Each CTA keeps its 256 ``markov_w2`` shard rows in shared memory, adds
        the BF16-rounded bias of every request to the step's shard logits in
        FP32 (the arithmetic of the other chains) and stores its own
        (value, global index) key into every rank's exchange; every CTA polls
        all keys of the step (``dspark_markov_greedy``). Same tokens as
        :meth:`markov_tokens_chain` wherever both compute the same BF16 bias.
        The caller checks :meth:`markov_v3_supported`.
        """
        from ..kernels.k3r_spec.dspark_markov_greedy import ctas_for
        from ..ops.spec.k3_dspark_markov_greedy import k3_dspark_markov_greedy

        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        tp = max(self.exchange.tp_size, 1)
        box_uc, box_mc, state = self.exchange.markov_v3_buffers(device, tp * ctas_for(shard))
        k3_dspark_markov_greedy(
            local_block,
            first_prev_tokens,
            markov_w1,
            self.markov_w2_shard(markov_w2),
            tokens,
            vocab=self.vocab_size,
            tp=tp,
            rank=self.mapping.tp_rank if tp > 1 else 0,
            state=state,
            box_uc=box_uc,
            box_mc=box_mc,
            multicast=tp > 1,
        )
        return tokens

    def markov_w2_shard(self, markov_w2: torch.Tensor) -> torch.Tensor:
        """This rank's rows of ``markov_w2`` (cached; a contiguous row slice)."""
        if self._markov_w2_src_ptr != markov_w2.data_ptr():
            self._markov_w2_shard = markov_w2[self.col_base : self.col_base + self.shard_width]
            self._markov_w2_src_ptr = markov_w2.data_ptr()
        return self._markov_w2_shard

    def markov_tokens(
        self,
        local_block: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        markov_w1: Optional[torch.Tensor],
        markov_w2: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Greedy DSpark block draft tokens from shard block logits.

        Args:
            local_block: ``[B, K, shard]`` shard logits of the block positions.
            first_prev_tokens: ``[B]`` anchor token ids (step 0's prev).
            markov_w1, markov_w2: ``[vocab, rank]`` Markov head weights, or
                ``None`` for a drafter without a Markov head (plain argmax).
        Returns:
            ``[B, K]`` int32 draft tokens, bit-identical to the per-position
            argmax of ``dspark_markov_chain_logits`` on the gathered logits.
        """
        num_reqs, block_size, shard = local_block.shape
        device = local_block.device
        if markov_w1 is None:
            flat = local_block.reshape(num_reqs * block_size, shard)
            return self.argmax(flat, out_dtype=torch.int32).view(num_reqs, block_size)
        tokens = torch.empty((num_reqs, block_size), dtype=torch.int32, device=device)
        if num_reqs == 0 or block_size == 0:
            return tokens
        rank = markov_w1.shape[1]
        w2_shard = self.markov_w2_shard(markov_w2)
        emb = torch.empty((num_reqs, rank), dtype=markov_w1.dtype, device=device)
        pairs, stride_row, stride_slot, n_slots = first_prev_tokens, 0, 0, 1
        for step in range(block_size + 1):
            k3_markov_reduce_embed(
                pairs,
                stride_row,
                stride_slot,
                n_slots,
                first_prev_tokens,
                markov_w1,
                emb,
                tokens,
                self.vocab_size,
                step,
            )
            if step == block_size:
                break
            # Same F.linear as the reference chain, on this rank's w2 rows.
            bias = F.linear(emb, w2_shard)
            step_logits = local_block[:, step]
            layout = self.exchange.layout(num_reqs, shard, device)
            k3_markov_add_local_argmax(step_logits, bias, self.col_base, layout)
            pairs, stride_row, stride_slot, n_slots = self.exchange.exchange(layout)
        return tokens


class K3VocabShardedLogits:
    """This rank's vocab shard of the target logits, handed to a spec worker in
    place of the gathered float32 logits.

    The worker either selects tokens from it vocab-parallel
    (:meth:`select_tokens`) or, for a batch the vocab-parallel samplers do not
    cover, materializes the gathered logits with :meth:`gather`.
    """

    __slots__ = ("head", "local")

    def __init__(self, head: K3VocabParallelHead, local: torch.Tensor):
        self.head = head
        self.local = local

    @property
    def device(self) -> torch.device:
        return self.local.device

    def gather(self) -> torch.Tensor:
        """The gathered ``[rows, vocab]`` float32 logits."""
        return self.head.gather_full_logits(self.local)

    def select_pairs(self, spec_metadata) -> tuple:
        """The unreduced exchanged pairs of :meth:`select_tokens` (see
        :meth:`K3VocabParallelHead.selection_pairs`)."""
        return self.head.selection_pairs(self.local, spec_metadata)

    def select_tokens(self, spec_metadata) -> torch.Tensor:
        """One target token per row: the argmax for an all-greedy batch, else a
        temperature-only draw (the caller guarantees the sampling mode)."""
        if spec_metadata.is_all_greedy_sample:
            return self.head.argmax(self.local)
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        rows = self.local.shape[0]
        return self.head.sample(
            self.local,
            spec_metadata.temperatures[:rows],
            spec_metadata.seeds[:rows],
            spec_metadata.offsets[:rows],
            GREEDY_TEMPERATURE_THRESHOLD,
        )


@lru_cache(maxsize=1)
def get_dflash_paged_append() -> Callable[..., None]:
    """Load flashinfer's paged K/V append: a plain scatter into an HND page pool."""
    if not IS_FLASHINFER_AVAILABLE:
        raise RuntimeError(
            "DFlash paged context cache requires flashinfer, which is not installed."
        )
    import flashinfer

    return flashinfer.page.append_paged_kv_cache


# Backends that keep the draft's private context K/V in a paged HND pool.
# They share the pool layout, page table and paged append op
_PAGED_ATTENTION_BACKENDS = ("TRTLLM", "FA4")

# Eager iterations with context requests in which a unified draft cache's host
# block counts are checked against the decoded table (_check_host_block_counts).
_HOST_BLOCK_COUNT_CHECKS = 8

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig


def dflash_context_dtype(draft_model: nn.Module) -> torch.dtype:
    """Use projection activations for the context cache, not packed weight storage."""
    projection = getattr(draft_model, "fc", None)
    if projection is None:
        return torch.bfloat16
    return getattr(projection, "dtype", projection.weight.dtype)


def last_accepted_hidden(projected_rows: torch.Tensor, num_accepted: torch.Tensor) -> torch.Tensor:
    """Select the target row that predicted each newly committed anchor.

    Rows are [requests, verification tokens, hidden], after the target projection
    and hidden normalization: hidden_norm(fc(target_hidden)). Accepted counts
    include the target's bonus token, so its predicting row is at count minus one.
    """
    indices = (num_accepted - 1).clamp_min(0).long()
    return projected_rows.gather(
        1, indices[:, None, None].expand(-1, 1, projected_rows.shape[-1])
    ).squeeze(1)


def compute_dflash_ctx_buffer_bytes(
    max_batch_size: int,
    max_ctx_len: int,
    block_size: int,
    num_attn_layers: int,
    num_kv_heads_per_rank: int,
    head_dim: int,
    dtype_bytes: int = 2,
    attention_backend: str = "VANILLA",
    page_size: int = 32,
    kv_factor: int = 2,
    paged: Optional[bool] = None,
) -> int:
    """Per-GPU bytes ``_lazy_init_ctx_buffers`` will allocate for the DFlash
    pooled-context K/V buffers.

    Mirrors the allocation exactly: ``max_batch_size + 1`` slots (one scratch
    slot for padding/warmup dummies), capacity ``max_ctx_len + block_size``
    per slot, ``kv_factor`` halves for every drafter attention layer (K and V,
    so 2; the MLA drafter keeps one latent and no V, so 1). The dense layout
    holds ``kv_factor`` contiguous ``[slots, L, capacity, Hkv, D]`` tensors;
    the paged layout holds one ``[L, pages, kv_factor, Hkv, page, D]`` tensor
    whose per-slot capacity is rounded up to whole pages.

    ``paged`` selects the layout. At runtime the caller passes the resolved
    ``_ctx_paged`` flag (the backend's kernels are paged, or the drafter opts
    into pages itself). When ``None`` -- the config-time estimate, which has
    no drafter to ask -- it falls back to whether ``attention_backend`` is a
    paged backend.
    """
    num_slots = max_batch_size + 1
    capacity = max_ctx_len + block_size
    if paged is None:
        paged = attention_backend in _PAGED_ATTENTION_BACKENDS
    if paged:
        pages_per_slot = (capacity + page_size - 1) // page_size
        capacity = pages_per_slot * page_size
    per_slot_elems = kv_factor * num_attn_layers * capacity * num_kv_heads_per_rank * head_dim
    return num_slots * per_slot_elems * dtype_bytes


def dflash_draft_slot_ids(
    num_gens: int,
    block_size: int,
    num_draft_tokens: int,
    shift_label: bool,
    device="cuda",
) -> torch.Tensor:
    """Flat indices into the [num_gens * block_size] drafter block outputs
    whose hidden states produce the K draft-token logits per request.

    Plain DFlash (K2.7 convention, shift_label off): the hidden state at
    mask slot j (j = 1..K) predicts draft token j; slot 0 (which holds the
    anchor/bonus token) is unused.

    DSpark shift_label convention (DeepSpec: labels for a block anchored at
    position p are input_ids[p+1 .. p+block_size], so the hidden state at
    block slot j predicts the token at position p+1+j): slots 0..K-1 are
    used, and slot 0 — the anchor token slot — predicts the first draft
    token.
    """
    request_bases = torch.arange(num_gens, dtype=torch.long, device=device) * block_size
    offsets = torch.arange(num_draft_tokens, dtype=torch.long, device=device)
    first_slot = 0 if shift_label else 1
    return (request_bases.unsqueeze(1) + first_slot + offsets.unsqueeze(0)).flatten()


def dflash_noise_block_embedding(
    embed_tokens: nn.Module,
    bonus: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    trained_mask_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The drafter's input block per request, [len(bonus), block_size, hidden]: slot 0 the bonus token's embedding,
    every other slot the mask token's.

    The mask row is the drafter's own trained one when it has one (``trained_mask_embedding``); otherwise the shared
    lookup's. Both rows go through ``embed_tokens.forward`` (NOT ``.weight[...]``) so TP-sharded vocabs mask out
    ranks that don't own the token id and all-reduce, the mask row included, so every rank makes the same call.

    ``trained_mask_embedding`` ([hidden]) is used as given: the drafter casts it to its own dtype and device at load.
    """
    num_gens = bonus.shape[0]
    mask_tok = torch.full((1,), int(mask_token_id), dtype=torch.long, device=bonus.device)
    combined_embed = embed_tokens(torch.cat([bonus, mask_tok], dim=0))
    embed_bonus = combined_embed[:num_gens]
    embed_mask = combined_embed[num_gens]
    if trained_mask_embedding is not None:
        embed_mask = trained_mask_embedding
    noise_embed_2d = embed_mask.expand(num_gens, block_size, -1).clone()
    noise_embed_2d[:, 0, :] = embed_bonus
    return noise_embed_2d


def dflash_position_ceiling(max_ctx: int, block_size: int, max_draft_len: int) -> int:
    """Positions a drafter that indexes absolute positions must be able to encode.

    The forward reads two position sets past the accumulated context, both
    starting from a ctx_len clamped to ``max_ctx``:

      * block decode  ``ctx_len + num_accepted + j``, ``j in [0, block_size)``
      * context KV    ``ctx_len + [0, max_draft_len]``

    so the largest index is ``max_ctx + max(block_size, max_draft_len + 1) - 1``
    and a table needs one entry more than that. Derived from the runtime
    ``max_ctx`` rather than from model_config: py_executor_creator raises the
    engine's max_seq_len past the configured value and never writes it back, so
    any config-derived reconstruction drifts from what ctx_len is clamped to.
    """
    return int(max_ctx) + max(int(block_size), int(max_draft_len) + 1)


def dflash_allocated_ctx_limit(
    block_counts: torch.Tensor, page_size: int, block_size: int
) -> torch.Tensor:
    """Context length a request may advertise, given the pages it holds.

    Leaves ``block_size`` positions of room: the MLA path writes the block's own
    latents at ``ctx_len .. ctx_len + block_size``, so stopping at exactly the
    allocation puts the first of them on an unallocated page, whose table entry
    ``_refresh_ctx_block_tables`` clamped to 0 -- another request's block. A
    silent cross-request write, not a fault.

    Per-request only on the ``python`` V2 backend, which propagates
    ``BAD_PAGE_INDEX``. The default ``cpp`` one maps it to 0
    (kvCacheManagerV2Utils.cu), every row counts full, and this degenerates to
    the block-table-width bound upstream already applied. Do not read a
    per-request guarantee into it there.
    """
    return (block_counts * page_size - block_size).clamp(min=0)


@dataclass
class K3VocabParallelSelection:
    """The tokens a DFlash worker selects from this rank's vocab shard of the
    target LM head instead of from the gathered logits.

    It exists when the target LM head is a plain vocab-sharded, gathered TP
    head (:meth:`K3VocabParallelHead.supported`) and the deployment has
    neither occurrence penalties nor a guided decoder, which both edit the full
    target logits. :attr:`head` is the private ``K3VocabParallelHead`` (MNNVL
    pair exchange): the sampling decision is
    made per batch (:meth:`K3DFlashWorker._vocab_parallel_target_selection_applies`)
    and the greedy draft tokens come from
    :meth:`K3DFlashWorker._draft_vocab_parallel_head`.
    """

    head: K3VocabParallelHead


class K3DFlashWorker(SpecWorkerBase):
    """
    Worker for DFlash speculative decoding.

    DFlash uses the draft model with mask tokens to predict multiple
    draft tokens in parallel. The DFlash draft model uses cross-attention
    to target model hidden states captured from specific layers.

    The target features are projected through fc + hidden_norm and fed
    to the draft model as K/V context in cross-attention. The context
    accumulates across steps - each step adds newly accepted tokens'
    projected hidden states to a per-request buffer, giving the draft
    model the full history of target features.

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
        *,
        path: K3DSparkPath = K3DSparkPath(),
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping
        # The deployment's DSpark path (the target's constant, the drafter's too).
        self._path = path
        # The type of the logits processor the target's base class built; set by the model.
        # Only that processor makes the target logits ``lm_head(hidden_states)``.
        self.plain_logits_processor_type: Optional[type] = None
        self._resolved_mask_token_id = None
        self._resolved_block_size = None
        self._compute_block_size = None

        # Pre-allocated per-slot context K/V buffers, built lazily on the
        # first forward. Fixed-size so slot-indexed reads/writes are CUDA
        # graph compatible.
        self._ctx_buf_inited = False
        self._ctx_len = None
        # Host shadows of _ctx_len and of each request's prompt progress.
        self._ctx_len_host = None
        self._req_ctx_pos = {}
        # True when the drafter's context lives in the target manager's draft
        # layers (the engine unified them); see _lazy_init_ctx_buffers.
        self._ctx_unified_cache = False
        # The manager layer ids of the drafter's context, and the K-buffer
        # offset that manager's per-layer page tables carry.
        self._ctx_layer_ids = None
        self._ctx_block_offset = 0
        self._ctx_kv_factor = 2
        # Iterations left in which the host block counts are checked against
        # the device-decoded table (see _check_host_block_counts).
        self._draft_table_checks_left = 0
        # False once a check found those counts disagreeing with the table.
        self._ctx_unified_host_counts = True
        # Snapshot for rolling back in-place _ctx_len updates when a forward
        # fails (or after warmup). See _ensure_spec_dec_state_restored.
        self._saved_ctx_len = None
        self._saved_ctx_len_host = None
        self._saved_req_ctx_pos = None
        self._ctx_len_restore_pending = False
        # Deferred kv_lens_cuda rewind state (see _prepare_kv_for_draft_forward,
        # _apply_kv_rewind_after_draft, _ensure_spec_dec_state_restored).
        self._kv_rewind_pending = False
        self._kv_rewind_amount = None
        self._kv_rewind_nc = None
        self._kv_rewind_bs = None
        # path.tail_v2: the rewind rides in the next-new-tokens kernel
        # (see _fused_select_accept, _fused_kv_rewind_operands).
        self._fused_kv_rewind_armed = False
        self._batch_to_slot_buffer = None
        self._max_ctx = 0
        self._ctx_k_buf = None  # [max_batch+1, L, max_ctx+block, nkv, hd]
        self._ctx_v_buf = None
        # Set by _lazy_init_ctx_buffers; every context write switches on the
        # cache layout rather than on which kernel reads it.
        self._ctx_paged = False
        # [L, pages, K/V, nkv, page, hd] when privately allocated, or the draft
        # KV cache manager's per-layer pool views when bound to it.
        self._ctx_kv_buf = None
        # [L] int64 base addresses of the manager's per-layer pool views.
        self._ctx_kv_ptrs = None
        # Whether a generation step's drafter inputs can be prepared by
        # _prepare_paged_latent_gen_inputs; decided with the pool binding.
        self._paged_latent_prepare = False
        # The manager _ctx_kv_buf is borrowed from, so a swap can be detected.
        self._ctx_kv_manager = None
        # Static per-slot page table for the private arena. When bound to the
        # manager it is replaced by _ctx_block_tables, refreshed per iteration.
        self._ctx_page_table = None
        self._ctx_block_tables = None  # [max_batch+1, max_blocks_per_seq] int32
        self._ctx_block_indptr = None
        # Pages the manager handed this request, per batch row. Saturates at
        # the table width on the default backend -- dflash_allocated_ctx_limit.
        self._ctx_block_counts = None
        # Host copy of this iteration's per-row counts, read from the manager's
        # caches (unified draft cache); None when only the device holds them.
        self._ctx_block_counts_host: list[int] | None = None
        self._ctx_block_partial = None  # parallel decode's per-chunk block counts
        self._ctx_pool_idx = 0
        self._ctx_block_divisor = 1  # encoded offset -> raw pool block index
        self._ctx_kv_indptr = None
        self._ctx_kv_last_page_len = None
        self._ctx_page_size = 32
        self._ctx_pages_per_slot = 0
        self._ctx_paged_append = None
        # True only when the drafter reads the draft KV cache manager's pool
        # AND that manager joins the target's reuse protocol, which is what
        # makes a reused prefix's drafter K/V still addressable. Set in
        # _lazy_init_ctx_buffers; read by _store_prefill_context.
        self._ctx_reuse_addressable = False
        self._dflash_attention_backend = spec_config.attention_backend
        # Set by bind_target_lm_head.
        self._vocab_parallel: Optional[K3VocabParallelSelection] = None
        # The private K3VocabParallelHead per LM head (keyed by id), built on
        # first use and shared by the target and draft selections of one head.
        self._vocab_parallel_heads = {}
        # Set by set_target_model and set_draft_model.
        self._target_model_ref: Optional[weakref.ref] = None
        self._draft_model_ref: Optional[weakref.ref] = None
        # The drafter's helix mapping when its context pages are CP-sharded.
        self._helix_mapping = None
        # The side-stream context latents' state (see _launch_ctx_side).
        self._ctx_side = None
        self._ctx_side_ar = None
        self._ctx_side_stream = None

        # Slot management (Python, updated in prepare() and eager mode)
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._dummy_slot = None  # for cudagraph padding or warmup dummy requests
        self._graph_dummy_id_floor = 1 << 62

        # Opt-in acceptance-statistics recorder (None unless
        # TLLM_DFLASH_ACCEPT_STATS_DIR is set; see accept_stats.py). Only
        # consulted behind `is not None` checks — zero overhead when off.
        self._accept_stats = maybe_create_recorder(
            spec_config.max_draft_len, getattr(mapping, "rank", 0) or 0
        )
        if self._accept_stats is not None:
            logger.info(
                f"DFlash: acceptance-statistics recording enabled -> {self._accept_stats.path}"
            )

        # type(self), not a literal: K3DSparkWorker does not override
        # __init__, so a hardcoded name reports the base class and the log
        # cannot evidence which worker a run actually used.
        logger.info(
            f"{type(self).__name__} initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    @property
    def _draft_tokens_per_req(self) -> int:
        """Target-forward tokens per gen request: K drafts + 1 bonus.

        Previously this was 2K (K+1 accepted + K-1 mask fillers), but the
        fillers contribute nothing beyond the K+1 prefix the acceptance
        check and context store consume, so carrying them through every
        target layer is wasted work that dominates step time at large batch.
        """
        return self.max_draft_len + 1

    def _validate_draft_attention_backend(self, draft_model) -> None:
        draft_model_dflash_attention_backend = getattr(
            draft_model, "dflash_attention_backend", None
        )
        if draft_model_dflash_attention_backend is None:
            raise ValueError("DFlash draft model is missing dflash_attention_backend.")
        if self._dflash_attention_backend == "AUTO":
            # The draft model resolved AUTO against its own family and this
            # build; adopt that, because _lazy_init_ctx_buffers keys paging and
            # the trtllm-gen shape checks off the concrete value.
            self._dflash_attention_backend = draft_model_dflash_attention_backend
            return
        if draft_model_dflash_attention_backend != self._dflash_attention_backend:
            raise ValueError(
                "DFlash worker and draft model attention backends must match; "
                f"worker={self._dflash_attention_backend}, "
                f"model={draft_model_dflash_attention_backend}."
            )

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)
        self._validate_draft_attention_backend(draft_model)
        # Non-owning: the draft model is a sibling module of this worker.
        self._draft_model_ref = weakref.ref(draft_model)
        self._helix_mapping = getattr(draft_model, "_helix_mapping", None)
        # The DFlash 2 selector indexes full-vocab codebooks by draft-token
        # id, so a d2t-remapped draft vocab would score the wrong tokens.
        if self._d2t is not None and getattr(draft_model, "is_dflash2", False):
            raise NotImplementedError(
                "DFlash 2 candidate selection requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported)."
            )
        lilicorr = getattr(draft_model, "lilicorr", None)
        if lilicorr is not None:
            if self._d2t is not None:
                raise NotImplementedError("LiLiCorr requires a shared draft/target vocabulary")
            if self.max_draft_len >= lilicorr.block_size:
                raise ValueError(
                    f"LiLiCorr checkpoint was trained with block_size={lilicorr.block_size}. "
                    f"Set max_draft_len to at most {lilicorr.block_size - 1}; "
                    f"got {self.max_draft_len}."
                )
            if not self.spec_config.target_layer_ids:
                raise ValueError(
                    "LiLiCorr requires nonempty target_layer_ids to capture target hidden states."
                )

    def set_target_model(self, target_model: nn.Module) -> None:
        # Non-owning: the target model owns this worker.
        self._target_model_ref = weakref.ref(target_model)

    def bind_target_lm_head(self, lm_head: nn.Module, logits_processor: nn.Module) -> None:
        """Decide the :class:`K3VocabParallelSelection` over the target LM head.

        The shard logits are ``lm_head(hidden_states)``, which are the target's
        logits only when its logits processor is the plain
        processor the target's base class built (``plain_logits_processor_type``,
        recorded by the model); a custom one (a final norm or head transform
        before the LM head) keeps the gathered logits. Binding the same head
        again keeps its selection.

        The head is the private ``K3VocabParallelHead`` (MNNVL pair
        exchange), under the eligibility gates (no guided decoder, no
        occurrence penalties, the plain logits processor) plus its own
        :meth:`K3VocabParallelHead.supported`.
        """
        if (
            self.guided_decoder is not None
            or self.spec_config.enable_penalty
            or self.plain_logits_processor_type is None
            or type(logits_processor) is not self.plain_logits_processor_type
        ):
            self._vocab_parallel = None
            return
        head = self._vocab_parallel_head(lm_head)
        if head is None:
            self._vocab_parallel = None
            return
        if self._vocab_parallel is None or self._vocab_parallel.head is not head:
            # The head decides temperature-only sampling per batch
            # (_vocab_parallel_target_selection_applies), not here.
            self._vocab_parallel = K3VocabParallelSelection(head=head)

    def bind_draft_lm_head(self, draft_model) -> None:
        """Post-load hook for the draft side of the selection.

        The greedy draft tokens are chosen per step by
        :meth:`_draft_vocab_parallel_head`, so nothing is bound here.
        """
        del draft_model

    def setup_aliases(self) -> None:
        """Post-load hook: decide the vocab-parallel selection, the target side
        and then the draft side.

        The model loader runs it after loading: the target model's logits
        processor is final (models install a custom one after the base
        constructor), and the drafter has taken over the target modules it
        shares (``load_weights_from_target_model``).
        """
        target_model = None if self._target_model_ref is None else self._target_model_ref()
        if target_model is not None:
            self.bind_target_lm_head(target_model.lm_head, target_model.logits_processor)
        if self._draft_model_ref is not None:
            self.bind_draft_lm_head(self._draft_model_ref())

    def post_load_weights(self) -> None:
        self.setup_aliases()

    def set_guided_decoder(self, guided_decoder) -> bool:
        # The guided decoder masks the full target logits.
        self._vocab_parallel = None
        return super().set_guided_decoder(guided_decoder)

    @property
    def target_vocab_parallel_head(self) -> Optional[K3VocabParallelHead]:
        return None if self._vocab_parallel is None else self._vocab_parallel.head

    def target_head_rows(self, hidden_states: torch.Tensor, attn_metadata, spec_metadata):
        """On the ``tail_v2`` path a generation-only batch gathers every token in order (gather
        ids 0..num_tokens-1), so the head reads ``hidden_states`` itself instead of an index copy
        of it. The head only reads these rows, so the worker's forward still sees them unchanged."""
        gather_ids = spec_metadata.gather_ids
        if (
            self._path.tail_v2
            and attn_metadata.num_contexts == 0
            and gather_ids is not None
            and gather_ids.shape[0] == hidden_states.shape[0]
        ):
            return hidden_states
        return hidden_states[spec_metadata.gather_ids]

    def target_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        logits_processor,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """This rank's vocab shard (``K3VocabShardedLogits``) once the target
        head is bound for vocab-parallel selection, the gathered logits
        otherwise."""
        head = self.target_vocab_parallel_head
        if head is not None:
            return head.shard_logits(hidden_states)
        return super().target_logits(
            hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
        )

    def _helix_shard(self) -> Optional[HelixDraftShard]:
        """This rank's share of the context pages when the drafter is helix-sharded."""
        helix = self._helix_mapping
        if helix is None:
            return None
        return HelixDraftShard(helix.cp_size, helix.cp_rank, self._ctx_page_size)

    def _vocab_parallel_head(self, lm_head) -> Optional[K3VocabParallelHead]:
        """The cached private ``K3VocabParallelHead`` of ``lm_head``, or None when
        the head is not a plain vocab shard."""
        if lm_head is None:
            return None
        key = id(lm_head)
        if key not in self._vocab_parallel_heads:
            vocab = int(getattr(lm_head, "num_embeddings", 0) or 0)
            supported = vocab > 0 and K3VocabParallelHead.supported(lm_head, vocab)
            self._vocab_parallel_heads[key] = (
                K3VocabParallelHead(lm_head, vocab) if supported else None
            )
        return self._vocab_parallel_heads[key]

    def _vocab_parallel_target_selection_applies(self, spec_metadata) -> bool:
        """Whether this batch's target tokens can come from shard logits.

        Greedy batches always; temperature-only batches only under the
        ``no_topk_no_topp`` sampling mode and without rejection sampling, which
        needs the full target distribution.
        """
        if spec_metadata.is_all_greedy_sample:
            return True
        if self._vocab_parallel is None:
            return False
        if spec_metadata.use_rejection_sampling:
            return False
        return spec_metadata.advanced_sampling_mode == AdvancedSamplingMode.NO_TOPK_NO_TOPP

    def _draft_vocab_parallel_head(self, draft_model, spec_metadata):
        """The private head greedy draft tokens are selected vocab-parallel with,
        or None.

        A drafter with a LiLiCorr or DFlash 2 path selector keeps the
        gathered path, which runs those selectors.
        """
        if (
            spec_metadata.wants_advanced_draft_sampling
            or self.guided_decoder is not None
            or self._d2t is not None
            or getattr(draft_model, "lilicorr", None) is not None
            or getattr(draft_model, "has_candidate_selector", False)
        ):
            return None
        return self._vocab_parallel_head(getattr(draft_model, "lm_head", None))

    def _vocab_parallel_markov_weights(self, draft_model):
        """``(markov_w1, markov_w2)`` of the intra-block Markov head the draft
        tokens chain through, or ``(None, None)`` for the plain per-position
        argmax. Plain DFlash has no Markov head."""
        del draft_model
        return None, None

    def _k3_vocab_parallel_draft_tokens(
        self,
        draft_head: K3VocabParallelHead,
        draft_model,
        hidden_states_out: torch.Tensor,
        gen_gather_ids: torch.Tensor,
        num_gens: int,
        num_draft_tokens: int,
        inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy draft tokens straight from this rank's vocab shard of the
        block logits through the private head, bit-identical to refining the
        gathered logits and taking the per-position argmax.

        Returns ``([num_gens, K]`` int32 draft tokens, the ``[num_gens * K,
        hidden]`` block hidden states they were computed from).
        """
        K = num_draft_tokens
        markov_w1, markov_w2 = self._vocab_parallel_markov_weights(draft_model)
        block_hidden = hidden_states_out[gen_gather_ids]
        local_block = draft_head.local_logits(block_hidden)
        markov = draft_head.markov_tokens
        if (
            markov_w1 is not None
            and num_gens <= TAIL_V2_MAX_GENS
            and (draft_head.exchange.backend == "mnnvl" or draft_head.exchange.tp_size == 1)
            # The one-launch chain holds at most 16 requests and 16 block positions.
            and num_gens <= 16
            and K <= 16
        ):
            markov = draft_head.markov_tokens_chain
        local_block = local_block.view(num_gens, K, -1)
        if draft_head.markov_v3_supported(
            local_block, inputs["first_prev_tokens"], markov_w1, markov_w2
        ):
            markov = draft_head.markov_tokens_v3
        gen_draft_tokens = markov(
            local_block,
            inputs["first_prev_tokens"],
            markov_w1,
            markov_w2,
        )
        return gen_draft_tokens, block_hidden

    def _check_ctx_arena_fits(self, capacity, num_slots, L, nkv, hd, dtype, kv_factor=2):
        """Fail with the arithmetic before allocating the drafter context arena.

        The arena is dense in max_seq_len and lands *after* the KV cache manager
        has already carved its pool, so free_gpu_memory_fraction does not bound
        it. Sized off an unpinned max_seq_len it reaches hundreds of GiB, and the
        bare torch.zeros that follows reports only the failed allocation, naming
        neither the buffer nor the knob that controls it.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()
        # Reuse the allocation-sizing helper so this runtime check and the
        # config-time estimate cannot drift. num_slots and capacity are already
        # resolved here, so feed them directly (max_batch_size = num_slots - 1,
        # block_size = 0, dtype_bytes = itemsize); kv_factor and the resolved
        # paged flag carry the drafter-specific layout.
        arena = compute_dflash_ctx_buffer_bytes(
            num_slots - 1,
            capacity,
            0,
            L,
            nkv,
            hd,
            dtype_bytes=itemsize,
            attention_backend=self._dflash_attention_backend,
            page_size=self._ctx_page_size,
            kv_factor=kv_factor,
            paged=self._ctx_paged,
        )

        # mem_get_info() excludes the caching allocator's reserved-but-unused
        # segments, which torch.zeros() below can serve from.
        free = torch.cuda.mem_get_info()[0] + (
            torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        )
        gib = 1 << 30
        logger.info(
            f"DFlash: drafter ctx arena needs {arena / gib:.2f} GiB "
            f"(slots={num_slots}, layers={L}, capacity={capacity}, "
            f"kv_heads={nkv}, head_dim={hd}), {free / gib:.2f} GiB free"
        )
        if arena >= free:
            # Warn, not raise: this estimate has under-reported real
            # headroom, and torch.zeros below is the authority.
            logger.warning(
                f"DFlash drafter context arena needs {arena / gib:.2f} GiB but only "
                f"{free / gib:.2f} GiB is free. It is dense in sequence length: "
                f"slots={num_slots} x layers={L} x capacity={capacity} x "
                f"kv_heads={nkv} x head_dim={hd} x {itemsize}B x {kv_factor} halves. "
                f"capacity comes from max_seq_len (+{self._resolved_block_size} "
                f"block slack), so pass an explicit max_seq_len matching the "
                f"sequences you actually serve instead of letting it fall back to "
                f"the drafter's max_position_embeddings; lowering max_batch_size "
                f"scales it down linearly too."
            )

    def _managed_ctx_pool(
        self,
        draft_kv_cache_manager: BaseResourceManager | None,
        L: int,
        nkv: int,
        hd: int,
        dtype: torch.dtype,
        kv_factor: int = 2,
    ) -> list[torch.Tensor] | None:
        """Per-layer views of the draft KV cache manager's pool, or None.

        The drafter's layers are the manager's ``draft_layer_ids`` when it
        appends them to the target's (the unified draft cache), else its first
        ``L`` layers. ``get_buffers(..., "HND")`` hands back the per-layer shape
        the private arena uses; ``dtype`` is the pool's (FP8 for a drafter with
        ``context_kv_fp8``).

        A list, not one stacked tensor: only the two consumers that index it by
        layer need it, and the two managers disagree on what a stack would mean.
        V1 slices one page-major allocation, so the layers are strided views of
        a shared storage; V2 wraps each layer's pool base address in its own
        tensor. Everything downstream indexes ``[layer_idx]`` first anyway.

        Note the page count is not the drafter's to interpret: a V2 view spans
        the whole interleaved pool, not this layer's slice, so only shape[1:] is
        checked here; the index space is settled in _init_ctx_block_offsets.
        A unified cache cannot fall back to the private arena (the manager
        allocates, reuses and transfers the drafter's pages), so a mismatch
        there raises.
        """
        if draft_kv_cache_manager is None:
            # No separate draft KV cache. Attention DP alone does NOT disable
            # it: _util.py skips that bail for an external drafter, which is
            # what DSpark is, so this drafter does get a manager under DP.
            return None
        draft_layer_ids = tuple(getattr(draft_kv_cache_manager, "draft_layer_ids", ()))
        layer_ids = draft_layer_ids if draft_layer_ids else tuple(range(L))
        if len(layer_ids) != L:
            reason = (
                f"the KV cache manager exposes {len(layer_ids)} draft layers but the "
                f"drafter has {L}"
            )
            if self._ctx_unified_cache:
                raise ValueError(f"DFlash: {reason}.")
            logger.warning(f"DFlash: {reason}; using the private ctx arena.")
            return None
        self._ctx_layer_ids = layer_ids
        layers = [
            draft_kv_cache_manager.get_buffers(layer_id, kv_layout="HND") for layer_id in layer_ids
        ]
        base = layers[0]
        # kv_factor 1 is the MLA drafter's SELFKONLY pool: one latent, no V.
        expected = (kv_factor, nkv, base.size(-2), hd)
        for i, layer in enumerate(layers):
            if tuple(layer.shape[1:]) != expected or layer.dtype != dtype:
                reason = (
                    f"draft KV pool layer {i} is {tuple(layer.shape)} / {layer.dtype}, "
                    f"not {('pages',) + expected} / {dtype}"
                )
                if self._ctx_unified_cache:
                    raise ValueError(f"DFlash: the unified draft KV pool does not match: {reason}.")
                # Fall back rather than fail: the private arena is what every
                # DFlash model shipped with, so an unrecognized pool costs the
                # old memory footprint, not the run.
                logger.warning(f"DFlash: {reason}; using the private ctx arena.")
                return None
        logger.info(
            f"DFlash: binding the drafter ctx cache to the draft KV cache manager pool: "
            f"{L} x {tuple(base.shape)} {base.dtype}, page_size={base.size(-2)}, "
            f"layers={list(layer_ids)}, unified={self._ctx_unified_cache}"
        )
        return layers

    def _init_ctx_block_offsets(self, draft_kv_cache_manager, pool, L, nkv, hd, page_size) -> bool:
        """Validate the pool layout and resolve its offset decode constants.

        ``kv_cache_block_offsets`` entries are ``pool_block_index *
        num_pool_layers * kv_factor`` plus the K/V field index. What that has to
        be divided by to index a ``get_buffers`` view depends on the manager --
        see the stride check below; attention_backend/utils.py hardcodes the V1
        divisor and is not a general contract. A V2 manager with per-layer page
        tables also offsets every entry by the layer's K buffer within the
        physical slot (``_ctx_block_offset``).
        """
        mgr = draft_kv_cache_manager
        pool_mapping = mgr.kv_cache_pool_mapping
        first_layer_id = self._ctx_layer_ids[0] if self._ctx_layer_ids is not None else 0
        layer_offset = mgr.layer_offsets[first_layer_id]
        self._ctx_pool_idx = int(pool_mapping[layer_offset, 0])
        per_layer_tables = bool(getattr(mgr, "_use_per_layer_page_tables", False))
        num_pool_layers = int((pool_mapping[:, 0] == self._ctx_pool_idx).sum())
        self._ctx_block_offset = 0
        if isinstance(mgr, KVCacheManagerV2):
            converter = mgr.impl.get_page_index_converter(layer_offset, Role.KEY)
            layer_group_id = mgr.layer_to_pool_mapping_dict[layer_offset]
            unsupported = None
            if mgr.enable_swa_scratch_reuse:
                unsupported = "DFlash managed context does not support SWA scratch pages"
            for layer_id in self._ctx_layer_ids:
                local_id = mgr.layer_offsets[layer_id]
                if (
                    mgr.layer_to_pool_mapping_dict[local_id] != layer_group_id
                    or mgr.get_layer_page_index_scale(layer_id) != converter.scale
                ):
                    unsupported = (
                        "DFlash's shared context block table requires draft layers "
                        "with the same lifecycle and page-index scale"
                    )
                    break
            if unsupported is not None:
                # A unified manager was selected for this layout, so the drafter
                # cannot move to private storage without breaking allocation,
                # reuse and transfer. A separate manager keeps the private arena.
                if self._ctx_unified_cache:
                    raise NotImplementedError(unsupported)
                logger.warning(f"{unsupported}; using the private ctx arena.")
                return False
            if per_layer_tables:
                # Attention tables address the physical pool origin, whereas
                # get_buffers() starts at this layer's own K buffer.
                self._ctx_block_offset = int(converter.layer_offset)
        elif num_pool_layers != L:
            logger.warning(
                f"DFlash: draft KV pool {self._ctx_pool_idx} spans {num_pool_layers} layers but "
                f"the drafter has {L}; using the private ctx arena."
            )
            return False
        manager_kv_factor = (
            mgr.get_layer_kv_factor(first_layer_id)
            if hasattr(mgr, "get_layer_kv_factor")
            else mgr.kv_factor
        )
        if manager_kv_factor != self._ctx_kv_factor:
            if self._ctx_unified_cache:
                raise ValueError(
                    f"DFlash: the unified draft layers have KV factor {manager_kv_factor}, "
                    f"the drafter {self._ctx_kv_factor}."
                )
            logger.warning(
                "DFlash: draft KV layer factor does not match the drafter; "
                "using the private ctx arena."
            )
            return False
        # V1 slices a page-major pool, so stride(0) spans every layer; V2 wraps
        # the layer base in a dense view over interleaved pages, so stride(0) is
        # one layer slot. Guessing wrong is silent: acceptance quietly drops.
        layer_block_elems = manager_kv_factor * nkv * page_size * hd
        stride0 = pool[0].stride(0)
        if stride0 == layer_block_elems:
            self._ctx_block_divisor = manager_kv_factor
        elif stride0 == num_pool_layers * layer_block_elems:
            self._ctx_block_divisor = num_pool_layers * manager_kv_factor
        else:
            message = (
                f"DFlash: draft KV pool layer stride {stride0} matches neither V1's "
                f"page-major slice ({num_pool_layers * layer_block_elems}) nor V2's "
                f"interleaved view ({layer_block_elems})"
            )
            if self._ctx_unified_cache:
                raise ValueError(f"{message}.")
            logger.warning(
                f"{message}; using the private ctx arena rather than guessing block indices."
            )
            return False
        logger.info(
            f"DFlash: ctx block offsets pool_idx={self._ctx_pool_idx}, "
            f"divisor={self._ctx_block_divisor}, block_offset={self._ctx_block_offset}, "
            f"view_pages={pool[0].size(0)}"
        )
        return True

    def _init_ctx_block_tables(self, num_slots: int, max_blocks: int) -> None:
        """Size the per-iteration block table (and the parallel decode's scratch)."""
        self._ctx_block_tables = torch.zeros(
            (num_slots, max_blocks), dtype=torch.int32, device="cuda"
        )
        self._ctx_block_indptr = torch.arange(
            0, (num_slots + 1) * max_blocks, max_blocks, dtype=torch.int32, device="cuda"
        )
        self._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        # Per-(request, column chunk) allocated-block counts of the parallel decode.
        self._ctx_block_partial = torch.zeros(
            num_slots * ((max_blocks + BLOCK_TABLE_CHUNK - 1) // BLOCK_TABLE_CHUNK),
            dtype=torch.int32,
            device="cuda",
        )
        logger.info(f"DFlash: ctx block tables {tuple(self._ctx_block_tables.shape)}")

    @staticmethod
    def _check_block_within_allocation(
        num_ctx: torch.Tensor, allocated_ctx_limit: torch.Tensor, block_size: int
    ) -> None:
        """Raise when a request's drafter block would leave its allocated pages.

        The shared manager sizes the request, so a shortfall there would have the
        block's scratch write land on the last context position, or on another
        request's page. Asynchronous, so it adds no host sync and is safe to
        capture in a CUDA graph; on the device it traps instead of raising.
        """
        torch._assert_async(
            (num_ctx <= allocated_ctx_limit).all(),
            f"DFlash: the drafter's block of {block_size} positions falls outside the "
            "pages the shared KV cache manager allocated for the request",
        )

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """Decode this iteration's draft block table into the persistent buffer.

        In place and fixed-shape so a captured graph keeps reading the same
        tensor. Returns False when there is no table to fill, i.e. every path
        that stayed on the private arena.

        The unified draft cache's pages are rows of the target metadata's
        ``kv_cache_block_offsets`` (the draft layers' pool), a separate draft
        manager's are ``draft_kv_cache_block_offsets``. For the unified cache the
        per-request block counts are also read on the host from the manager's
        caches, so the prefill store need not wait for the device.
        """
        self._ctx_block_counts_host = None
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        offsets_name = (
            "kv_cache_block_offsets" if self._ctx_unified_cache else "draft_kv_cache_block_offsets"
        )
        src = getattr(attn_metadata, offsets_name, None)
        if src is None:
            # The table is bound but unfillable. Leaving it at zeros would send
            # every request's every layer to pool block 0 -- silently.
            raise RuntimeError(
                "DFlash: bound to the draft KV pool but "
                f"{type(attn_metadata).__name__} carries no "
                f"{offsets_name} this iteration."
            )
        # Field 0 of the K/V pair: both name the same block for a kv_factor=2
        # pool, and the drafter addresses blocks, not planes. The block count
        # is taken before the clamp erases the placeholders; it is informative
        # only on the python V2 backend, the cpp one already mapped
        # BAD_PAGE_INDEX to 0. The kernels remove the layer offset in int64
        # and leave the metadata's table unchanged.
        if self._path.tail_v2:
            # path.tail_v2: the same decode with a row's columns spread
            # over programs.
            k3_draft_block_table(
                src[self._ctx_pool_idx, :num_seqs, 0],
                self._ctx_block_divisor,
                self._ctx_block_tables[:num_seqs],
                self._ctx_block_counts[:num_seqs],
                self._ctx_block_partial,
                block_offset=self._ctx_block_offset,
            )
        else:
            k3_draft_block_table(
                src[self._ctx_pool_idx, :num_seqs, 0],
                self._ctx_block_divisor,
                self._ctx_block_tables,
                self._ctx_block_counts,
                block_offset=self._ctx_block_offset,
            )
        # Only the context rows' counts are read on the host (the prefill store).
        num_contexts = min(int(attn_metadata.num_contexts), num_seqs)
        cache_map = getattr(self._ctx_kv_manager, "kv_cache_map", None)
        if (
            self._ctx_unified_cache
            and self._ctx_unified_host_counts
            and cache_map is not None
            and request_ids is not None
            and num_contexts > 0
        ):
            counts = [
                int(cache_map[request_id].num_blocks) if request_id in cache_map else 0
                for request_id in request_ids[:num_contexts]
            ]
            if self._draft_table_checks_left > 0 and not torch.cuda.is_current_stream_capturing():
                self._draft_table_checks_left -= 1
                if not self._check_host_block_counts(counts):
                    return True
            self._ctx_block_counts_host = counts
        return True

    def _check_host_block_counts(self, counts: list[int]) -> bool:
        """Compare the manager's block counts of the context rows with the decoded table's.

        Run on the first eager iterations that carry context requests (it reads
        the device counts back). The decoded count saturates at the table width
        where the manager writes unallocated entries as page 0 and the layer
        offset is 0, which bounds nothing, so that is accepted for any host
        count within the width. On a disagreement the host counts are dropped
        for good and the prefill store reads the device counts instead.
        """
        width = self._ctx_block_tables.size(1)
        device_counts = self._ctx_block_counts[: len(counts)].tolist()
        if all(d == h or (d == width and h <= width) for d, h in zip(device_counts, counts)):
            return True
        logger.error(
            "DFlash: the unified draft cache's host block counts "
            f"{counts[:8]} disagree with the decoded table's {device_counts[:8]}; "
            "the prefill store reads the device counts from now on."
        )
        self._draft_table_checks_left = 0
        self._ctx_unified_host_counts = False
        return False

    def _check_unified_helix(self, draft_kv_cache_manager) -> None:
        """Require the drafter's helix sharding to be the target manager's.

        A unified cache's draft pages follow the target's helix ledger (global
        page g on CP rank g % cp_size, as its local page g // cp_size), which is
        what the drafter's own sharding (``_helix_shard``) assumes of its pages.
        """
        mapping = draft_kv_cache_manager.mapping
        target = (mapping.cp_size, mapping.cp_rank) if mapping.has_cp_helix() else None
        helix = self._helix_mapping
        drafter = None if helix is None else (helix.cp_size, helix.cp_rank)
        if target != drafter:
            raise ValueError(
                "DFlash: the unified draft KV pages are helix-sharded as "
                f"(cp_size, cp_rank)={target}, but the drafter reads them as {drafter}."
            )

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        self._validate_draft_attention_backend(draft_model)
        # The target manager also holding the drafter's layers (the unified draft
        # cache): their pages are rows of the target metadata's block offsets.
        self._ctx_unified_cache = (
            draft_kv_cache_manager is not None
            and draft_kv_cache_manager is getattr(attn_metadata, "kv_cache_manager", None)
        )
        offsets_name = (
            "kv_cache_block_offsets" if self._ctx_unified_cache else "draft_kv_cache_block_offsets"
        )
        # Only TrtllmAttentionMetadata builds the block offsets. Without them the
        # table stays all-zero and every request would read and write pool
        # block 0 -- silently, at acceptance 1.0.
        if draft_kv_cache_manager is not None and not hasattr(attn_metadata, offsets_name):
            if self._ctx_unified_cache:
                raise NotImplementedError(
                    f"DFlash unified KV requires {offsets_name} in attention metadata"
                )
            logger.warning(
                f"DFlash: {type(attn_metadata).__name__} carries no draft KV block offsets; "
                "using the private ctx arena."
            )
            draft_kv_cache_manager = None

        if self._ctx_buf_inited:
            if self._ctx_kv_manager is draft_kv_cache_manager:
                return
            # KV cache estimation swaps a probe manager for the real one.
            # Holding the old pool's tensors across that writes freed memory.
            logger.info("DFlash: draft KV cache manager replaced, rebinding the ctx cache.")
            self._ctx_buf_inited = False

        # Recorded even when the buffers stay private, so that the estimation
        # pass falling back to the arena still rebinds once the real pool lands.
        self._ctx_kv_manager = draft_kv_cache_manager
        self._ctx_block_tables = None
        self._ctx_block_counts = None
        self._ctx_block_partial = None
        self._ctx_block_counts_host = None
        self._ctx_layer_ids = None
        self._ctx_block_offset = 0
        self._ctx_reuse_addressable = False
        self._paged_latent_prepare = False
        max_batch = spec_metadata.max_num_requests

        # ctx_len is 1:1 with the target's positions, so max_seq_len bounds it.
        # max_position_embeddings does not: K3's drafter advertises 1048576,
        # which sizes the arena at hundreds of GiB.
        config = getattr(draft_model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", None) if config else None
        # Take the served length from the KV cache manager, which every backend
        # carries on the metadata base class. A `max_seq_len` attribute is only
        # declared by some metadata classes (TrtllmAttentionMetadata has one --
        # a wrapper returning this same manager value; the base class and
        # FlashInferAttentionMetadata do not), so reading the attribute alone
        # lets max_position_embeddings win the min() below and sizes the arena
        # for the advertised positions instead of the served length. The
        # attribute stays as the fallback for no-cache attention, where it is
        # set by hand and there is no manager to ask.
        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        runtime_max = getattr(cache_manager, "max_seq_len", None)
        if runtime_max is None:
            runtime_max = getattr(attn_metadata, "max_seq_len", None)
        candidates = [c for c in (runtime_max, max_pos) if c is not None]
        self._max_ctx = min(candidates) if candidates else 8192
        # py_executor_creator inflates max_seq_len by at most this much for spec
        # dec (one term is overlap-only), so compare against what is served.
        from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

        spec_slack = 2 * (self.spec_config.tokens_per_gen_step - 1) + get_num_extra_kv_tokens(
            self.spec_config
        )
        if runtime_max is not None and self._max_ctx < runtime_max - spec_slack:
            # Warn, not raise: the per-request skip in _store_prefill_context
            # keeps an oversized request from taking the executor down.
            logger.warning(
                f"DFlash drafter covers {self._max_ctx} positions but the engine serves "
                f"up to {runtime_max - spec_slack}. max_position_embeddings={max_pos} is "
                "the binding constraint, so requests past it will draft nothing."
            )

        dtype = dflash_context_dtype(draft_model)
        # An FP8 drafter context (context_kv_dtype) is held in an FP8 pool.
        pool_dtype = torch.float8_e4m3fn if getattr(draft_model, "context_kv_fp8", False) else dtype

        # Reserve slot index max_batch as a scratch slot for padding/unknown
        # dummies; real requests only draw slots 0..max_batch-1, so dummy
        # writes land here and can't corrupt a real request's context.
        self._dummy_slot = max_batch
        num_slots = max_batch + 1
        # CUDA-graph padding uses ids counting down from
        # CUDA_GRAPH_DUMMY_REQUEST_ID; a floor of one draft block separates them
        # from real ids. Imported lazily to avoid an import cycle.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        self._ctx_len_host = [0] * num_slots
        self._batch_to_slot_buffer = DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.long, device="cuda"),
            pin_memory=prefer_pinned(),
        )
        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._req_ctx_pos = {}

        # checkpoint's trained block width
        self._resolved_block_size = getattr(draft_model, "block_size", None) or (
            self.max_draft_len + 1
        )
        # what the draft forward actually computes
        self._compute_block_size = self._draft_block_width(draft_model)
        if self._compute_block_size > self._resolved_block_size:
            slack = self._compute_block_size - self.max_draft_len
            raise ValueError(
                f"DFlash checkpoint was trained with block_size={self._resolved_block_size}."
                f"Lower max_draft_len to at most {self._resolved_block_size - slack}."
                f"Current max_draft_len={self.max_draft_len}."
            )
        if self._ctx_unified_cache:
            reserved = getattr(draft_kv_cache_manager, "_shared_draft_block_width", 0)
            if self._compute_block_size > reserved:
                raise ValueError(
                    f"Drafter block width {self._compute_block_size} exceeds the "
                    f"{reserved} positions reserved by the shared KV cache manager. "
                    "The manager must reserve the drafter's resolved block width."
                )

        # +block_size slack lets each backend append the per-iteration noise
        # K/V after the accumulated request context.
        assert hasattr(draft_model, "_build_fused_kv_buffers"), (
            "DFlash draft model must define _build_fused_kv_buffers."
        )
        draft_model._build_fused_kv_buffers()
        L = draft_model._num_attn_layers
        nkv = draft_model._num_kv_heads
        hd = draft_model._head_dim
        kv_factor = getattr(draft_model, "_kv_factor", 2)
        self._ctx_kv_factor = kv_factor
        capacity = self._max_ctx + self._compute_block_size
        # Published before any drafter table is built, so a drafter may read it
        # instead of its config cap. _max_ctx is attn_metadata.max_seq_len, which
        # py_executor_creator raises past model_config.max_seq_len and never
        # writes back. _compute_block_size: the block decode's j runs over it.
        draft_model._runtime_position_ceiling = dflash_position_ceiling(
            self._max_ctx, self._compute_block_size, self.max_draft_len
        )
        # A drafter asks for pages on its own (the MLA one does, to get its
        # footprint under free_gpu_memory_fraction) while still reading them
        # eagerly.
        use_paged = self._dflash_attention_backend in _PAGED_ATTENTION_BACKENDS or getattr(
            draft_model, "_paged_ctx_cache", False
        )
        self._ctx_paged = use_paged
        if use_paged:
            pool = self._managed_ctx_pool(draft_kv_cache_manager, L, nkv, hd, pool_dtype, kv_factor)
            # The manager's page size wins when bound to it: its pool is already
            # carved, so the drafter adopts the geometry rather than imposing one.
            page_size = self._ctx_page_size if pool is None else pool[0].size(-2)
            self._ctx_page_size = page_size
            # Settle the block table before committing to the pool: it is the
            # last thing that can rule the pool out, and falling back after
            # taking the pool branch would leave no buffer allocated at all.
            if pool is not None and not self._init_ctx_block_offsets(
                draft_kv_cache_manager, pool, L, nkv, hd, page_size
            ):
                pool = None
            if pool is None:
                if self._ctx_unified_cache:
                    raise ValueError("DFlash could not bind the unified manager's draft KV layers")
                if getattr(draft_model, "context_kv_fp8", False):
                    raise ValueError(
                        "speculative_config.context_kv_dtype='fp8' needs the drafter's context "
                        "KV in an FP8 draft KV cache manager pool."
                    )
                # Per-slot reservation is an arena-only notion; the pool path
                # takes its pages from the manager one iteration at a time.
                self._ctx_pages_per_slot = (capacity + page_size - 1) // page_size
                total_pages = num_slots * self._ctx_pages_per_slot
                self._check_ctx_arena_fits(capacity, num_slots, L, nkv, hd, dtype, kv_factor)
                # HND layout consumed by both FlashInfer's paged append and the
                # TRTLLM-Gen launcher: [L, pages, K/V, Hkv, page, D].
                self._ctx_kv_buf = torch.zeros(
                    (L, total_pages, kv_factor, nkv, page_size, hd),
                    dtype=dtype,
                    device="cuda",
                )
                self._ctx_page_table = torch.arange(
                    total_pages, dtype=torch.int32, device="cuda"
                ).view(num_slots, self._ctx_pages_per_slot)
                self._ctx_kv_indptr = torch.arange(
                    0,
                    total_pages + 1,
                    self._ctx_pages_per_slot,
                    dtype=torch.int32,
                    device="cuda",
                )
            else:
                # Nothing reserved: pages come from the manager's per-request
                # block table one iteration at a time, so the footprint follows
                # the sequences served rather than max_batch x max_seq_len.
                if self._ctx_unified_cache:
                    self._check_unified_helix(draft_kv_cache_manager)
                    self._draft_table_checks_left = _HOST_BLOCK_COUNT_CHECKS
                    self._ctx_unified_host_counts = True
                self._ctx_kv_buf = pool
                self._ctx_kv_ptrs = torch.tensor(
                    [layer.data_ptr() for layer in pool], dtype=torch.int64, device="cuda"
                )
                self._init_ctx_block_tables(num_slots, draft_kv_cache_manager.max_blocks_per_seq)
                # The fused preparation stores single-latent context entries
                # projected from the captured target features through fc and
                # hidden_norm (which every loaded DFlash drafter owns), as the
                # eager path does; LiLiCorr drafters keep the eager path, and so
                # does a helix-sharded drafter (rank-local pages), which only
                # _prepare_gen_drafter_inputs_fused handles.
                self._paged_latent_prepare = (
                    kv_factor == 1
                    and not draft_model._uses_lilicorr
                    and self._helix_mapping is None
                )
                # Only a manager that publishes and matches its own blocks can
                # hand back a reused prefix's drafter K/V; the private arena and
                # an unpaired draft pool both start every request at 0.
                self._ctx_reuse_addressable = self._ctx_unified_cache or bool(
                    getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False)
                )
            self._ctx_kv_last_page_len = torch.full(
                (num_slots,), page_size, dtype=torch.int32, device="cuda"
            )
        else:  # dense private arena (VANILLA, unpaged)
            if self._ctx_unified_cache:
                raise ValueError(
                    "DFlash: the drafter reads a dense context arena, not the unified "
                    "manager's draft KV pages."
                )
            self._check_ctx_arena_fits(capacity, num_slots, L, nkv, hd, dtype, kv_factor)
            kv_shape = (num_slots, L, capacity, nkv, hd)
            self._ctx_k_buf = torch.zeros(kv_shape, dtype=dtype, device="cuda")
            # An MLA drafter stores one latent per token; leaving _ctx_v_buf as
            # None is what tells the write paths below there is no V half.
            self._ctx_v_buf = (
                torch.zeros(kv_shape, dtype=dtype, device="cuda") if kv_factor == 2 else None
            )
        self._ctx_buf_inited = True

        if not self._ctx_reuse_addressable and getattr(
            draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False
        ):
            # Every fallback above (shape mismatch, stride mismatch, pool layer
            # count) logs its own reason and then costs only memory -- except
            # this one, where the scheduler goes on matching prefixes the
            # drafter can no longer read. Silent, and it looks exactly like the
            # unpaired build in every metric but acceptance length.
            logger.warning(
                "DFlash: the draft KV cache manager joins block reuse but the drafter "
                "fell back to a private ctx arena, so a reused prefix's drafter K/V "
                "stays unreachable and acceptance length drops as if unpaired."
            )

        logger.info(
            f"DFlash: allocated ctx buffers: max_batch={max_batch}, "
            f"max_ctx={self._max_ctx}, dtype={dtype}, "
            f"dflash_attention_backend={self._dflash_attention_backend}, "
            f"reuse_addressable={self._ctx_reuse_addressable}, "
            f"unified={self._ctx_unified_cache}"
        )

    def _ctx_paged_index_args(self):
        """(kv_indices, kv_indptr) for the append op, and what ``rows`` means.

        Bound to the manager the page table is per-request and refreshed every
        iteration, so rows are batch positions; on the private arena it is the
        static per-slot table and rows are slots. Both are dense and uniformly
        strided, so kv_indptr is the same arange either way -- only the width
        differs.
        """
        if self._ctx_block_tables is not None:
            table = self._ctx_block_tables
            indptr = self._ctx_block_indptr
        else:
            table = self._ctx_page_table
            indptr = self._ctx_kv_indptr
        return table.flatten().contiguous(), indptr

    def _assign_slot(
        self,
        req_id: int,
        reset: bool = False,
        updates: Optional[dict] = None,
        bases: Optional[dict] = None,
    ):
        """Get (or refresh) this request's context slot; None when none is free.

        ``reset`` recycles an existing slot, which is how a reused request id is
        told apart from a continuation. Pass ``updates`` to defer the length
        resets into a caller's batch instead of writing _ctx_len per request;
        either way the host mirror stays in step with the device tensor.
        ``bases`` is the base-page batch of the engine's ``DFlashSpecMetadata.prepare()``;
        this worker's slots always read from position 0, so it is left untouched.
        """

        def clear(slot: int) -> None:
            if updates is None:
                self._write_ctx_len({slot: 0})
            else:
                updates[slot] = 0

        if reset and req_id in self._req_to_slot:
            old_slot = self._req_to_slot.pop(req_id)
            clear(old_slot)
            self._free_slots.append(old_slot)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                return None
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            clear(slot)
        return self._req_to_slot[req_id]

    def _get_ctx_paged_append(self) -> Callable[..., None]:
        """flashinfer paged append, shared by the TRTLLM and FA4 backends."""
        if self._ctx_paged_append is None:
            self._ctx_paged_append = get_dflash_paged_append()
        return self._ctx_paged_append

    def _store_context_kv_paged(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        owned: Optional[torch.Tensor] = None,
    ) -> None:
        """Write K/V at rank-local ``positions``.

        ``owned`` (helix): False entries rewrite their slot's current value,
        which keeps a fixed-size, CUDA-graph-safe store.
        """
        fp8_cache = self._ctx_kv_buf[0].dtype == torch.float8_e4m3fn
        helix = self._helix_mapping is not None
        if v is None or self._dflash_attention_backend == "VANILLA" or fp8_cache or helix:
            # Single-latent (MLA) cache: one vector per token per layer, so
            # there is no K/V interleave for append_paged_kv_cache to do and the
            # write is a plain scatter into the page the block table names.
            # VANILLA (a unified cache's pages), FP8 and helix-sharded caches
            # write the same way.
            table = (
                self._ctx_block_tables
                if self._ctx_block_tables is not None
                else self._ctx_page_table
            )
            page_size = self._ctx_page_size
            pages = table[rows.long(), positions.long() // page_size].long()
            offsets = positions.long() % page_size
            keep = None if owned is None else owned.view(-1, *([1] * (k.dim() - 2)))
            for layer_idx in range(k.size(1)):
                pool = self._ctx_kv_buf[layer_idx]
                # Byte views for FP8: index_put support varies across builds.
                view = pool.view(torch.uint8) if fp8_cache else pool
                for plane, new in ((0, k), (1, v)):
                    if new is None:
                        continue
                    if fp8_cache and v is None:
                        # The MLA latent saturates (to_draft_fp8), as its block
                        # decode quantizes the query it is read with.
                        new = to_draft_fp8(new[:, layer_idx])
                    else:
                        new = new[:, layer_idx].to(pool.dtype)
                    if fp8_cache:
                        new = new.view(torch.uint8)
                    if keep is not None:
                        new = torch.where(keep, new, view[pages, plane, :, offsets])
                    view[pages, plane, :, offsets] = new
            return
        append_paged_kv_cache = self._get_ctx_paged_append()

        # Convert at most once and reuse for every layer. Calling ``.to()`` in
        # the op call would allocate converted tensors once per layer when the
        # inputs are not already int32.
        rows_i32 = rows.to(torch.int32)
        positions_i32 = positions.to(torch.int32)
        kv_indices, kv_indptr = self._ctx_paged_index_args()
        for layer_idx in range(k.size(1)):
            pool = self._ctx_kv_buf[layer_idx]
            append_paged_kv_cache(
                append_key=k[:, layer_idx].to(pool.dtype).contiguous(),
                append_value=v[:, layer_idx].to(pool.dtype).contiguous(),
                batch_indices=rows_i32,
                positions=positions_i32,
                paged_kv_cache=pool,
                kv_indices=kv_indices,
                kv_indptr=kv_indptr,
                kv_last_page_len=self._ctx_kv_last_page_len,
                kv_layout="HND",
            )

    def _prepare_attn_metadata_for_dflash(self, attn_metadata, spec_metadata):
        """Save attn_metadata fields that DFlash modifies during forward."""
        is_capturing = torch.cuda.is_current_stream_capturing()

        if spec_metadata.is_cuda_graph and not is_capturing:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda", "kv_lens_cuda")
        else:
            attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Adjust kv_lens_cuda so the draft model sees correct RoPE positions."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size

            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1
            self._kv_rewind_pending = True

            attn_metadata.update_for_spec_dec()

    def _apply_kv_rewind_after_draft(self, attn_metadata, spec_metadata):
        """Apply the deferred kv_lens rewind after the draft forward."""
        self._kv_rewind_pending = False
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if is_warmup:
            # kv_lens_cuda was saved by prepare_for_spec_dec in this mode and
            # is restored wholesale, so no rewind is needed.
            return

        if self._kv_rewind_amount is not None and hasattr(attn_metadata, "kv_lens_cuda"):
            nc = self._kv_rewind_nc
            bs = self._kv_rewind_bs
            attn_metadata.kv_lens_cuda[nc:bs] -= self._kv_rewind_amount
            attn_metadata.kv_lens_cuda[nc:bs].clamp_(min=0)

    def _advance_ctx_len(self, slots: torch.Tensor, num_accepted: torch.Tensor) -> None:
        """Add this step's accepted tokens to the context lengths of ``slots``, on the device.

        Padding, warmup and attention-DP dummy rows share the dummy slot, which no request owns
        or frees. Emptying it right after the add keeps it at 0 between steps: those rows attend
        no context, and their next context K / V starts at column 0, inside the padding
        request's own pages (past them, its page-table row maps page 0, another request's).
        The store is to a fixed slot, so a captured step replays it.
        """
        self._ctx_len[slots] += num_accepted
        self._ctx_len.clamp_(max=self._max_ctx)
        self._ctx_len[self._dummy_slot].zero_()

    def _rebuild_batch_to_slot(self, request_ids: list[int]) -> None:
        mapping = [self._req_to_slot.get(rid, self._dummy_slot) for rid in request_ids]
        self._batch_to_slot_buffer.update(torch.tensor(mapping, dtype=torch.long))

    def _seed_generation_slot(self, slot: int, position: int, updates: dict, bases: dict) -> None:
        """Record the context length of a slot newly assigned to a generation row whose first
        token sits at absolute ``position`` (the engine's ``DFlashSpecMetadata.prepare()`` calls this
        for a disaggregated generation request, or one that lost its slot).

        On the unified cache the draft layers travel with the target's prompt KV, so the slot
        holds drafter K/V for every position below ``position``. The private arena holds none
        for such a request, so its slot starts empty. Either way the slot reads from position 0,
        so ``bases`` is left untouched.
        """
        updates[slot] = min(position, self._max_ctx) if self._ctx_unified_cache else 0

    def _write_ctx_len(
        self, updates: dict[int, int], base_updates: Optional[dict[int, int]] = None
    ) -> None:
        """Apply a slot-to-length mapping to _ctx_len in one async scatter.

        ``base_updates`` comes from the engine's ``DFlashSpecMetadata.prepare()``; this worker has no
        base pages (its slots always read from position 0), so it is ignored.
        """
        if not updates:
            return
        for slot, value in updates.items():  # Update host side mirror
            self._ctx_len_host[slot] = value

        slots, values = zip(*updates.items())
        pinned = prefer_pinned()
        slots_pinned = torch.tensor(slots, dtype=torch.long, pin_memory=pinned)
        vals_pinned = torch.tensor(values, dtype=torch.long, pin_memory=pinned)
        self._ctx_len.index_copy_(
            0,
            slots_pinned.to("cuda", non_blocking=True),
            vals_pinned.to("cuda", non_blocking=True),
        )

    def _restore_ctx_len_host(self) -> None:
        """Roll the host shadows back to their pre-forward snapshot."""
        if self._saved_ctx_len_host is not None:
            self._ctx_len_host = list(self._saved_ctx_len_host)
        if self._saved_req_ctx_pos is not None:
            self._req_ctx_pos = dict(self._saved_req_ctx_pos)

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: "DFlashSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Capture prefill hidden states and store as initial accumulated context.

        During prefill (context requests), the target model processes all prompt
        tokens.  We project their captured hidden states through fc + hidden_norm
        and store them per-request so the draft model can use the full prompt
        as cross-attention context on subsequent gen steps.
        """
        if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return
        # Project context tokens through fc + hidden_norm
        ctx_proj = draft_model.project_target_hidden(captured_hs[:num_ctx_tokens])

        # Split by request and store/append accumulated context.
        # Context requests may arrive in chunks (chunked prefill), so we must
        # APPEND successive chunks for the same request rather than overwriting.
        # A chunk is only an append if it starts where the last one ended

        # Scalar (non-rope) token indices on the host. The device tensor holds
        # the same values, but reading it costs a D2H sync per request.
        host_pos = spec_metadata.host_position_ids
        # Slot -> value, uploaded once after the loop: assigning a Python scalar
        # through a CUDA index copies from pageable memory and waits for the
        # stream, i.e. for the whole target forward of this step. A failure
        # mid-loop needs no partial upload: the forward's snapshot restores both.
        ctx_len_updates = {}
        offset = 0
        num_contexts = attn_metadata.num_contexts
        shard = self._helix_shard()
        # A context write addresses pages through row i of the block table, so
        # it must fit that row's allocation as well as the arena's length. The
        # unified draft cache read this iteration's counts from the manager on
        # the host; reading them back from the device would wait for the whole
        # target forward of this step, so only a separate draft manager's table
        # needs one tolist() for the loop. Under helix each local page stands
        # for cp_size global ones.
        cp = 1 if shard is None else shard.cp_size
        page_span = cp * self._ctx_page_size
        host_counts = self._ctx_block_counts_host
        if self._ctx_block_tables is None:
            ctx_alloc = None
        elif host_counts is not None and len(host_counts) >= num_contexts:
            ctx_alloc = [count * page_span for count in host_counts[:num_contexts]]
        else:
            ctx_alloc = (self._ctx_block_counts[:num_contexts] * page_span).tolist()
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            chunk_proj = ctx_proj[offset : offset + slen].detach()
            chunk_pos = position_ids[offset : offset + slen].long().detach()

            if slen == 0:
                first_pos = 0
            elif host_pos is not None:
                first_pos = int(host_pos[offset])
            else:
                first_pos = chunk_pos[0].item()

            # A chunk continues what we already stored only if it starts exactly
            # where the previous one ended. Everything else -- a fresh request, a
            # request id reused after completion, a prefill restarted after
            # preemption -- has to start from a clean slot.
            reset = self._req_ctx_pos.get(req_id) != first_pos
            if self._assign_slot(req_id, reset=reset, updates=ctx_len_updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                offset += slen
                continue

            slot = self._req_to_slot[req_id]
            cur = ctx_len_updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                # Reuse let the target skip [0, first_pos), so only the tail
                # reaches the capture buffer. The prefix K/V are still in the
                # paired pool: precompute_context_kv is per-token in (hidden, p).
                cur = first_pos
            if cur + slen > cap:
                # Request-level, like the no-free-slots path above: truncating
                # would silently draft from a stale prefix, but killing the
                # forward would take every other in-flight request with it.
                logger.warning(
                    f"DFlash: ctx overflow on slot {slot} "
                    f"({cur} + {slen} > {cap}); skipping its context "
                    "store, so this request drafts nothing."
                )
                self._req_to_slot.pop(req_id, None)
                self._req_ctx_pos.pop(req_id, None)
                ctx_len_updates[slot] = 0
                self._free_slots.append(slot)
                offset += slen
                continue
            end = cur + slen
            actual = end - cur
            # Anchor to the true position rather than accumulating, and advance
            # even when the slot is full so the next chunk still reads as a
            # continuation.
            self._req_ctx_pos[req_id] = first_pos + slen
            if actual > 0:
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                if cache_dtype == torch.float8_e4m3fn:
                    # The K/V projection runs in the drafter's weight dtype;
                    # the store converts its outputs to the cache dtype.
                    cache_dtype = chunk_proj.dtype
                chunk_proj_cast = chunk_proj[:actual].to(cache_dtype)
                ctx_len_updates[slot] = end
                # Precompute post-norm/post-RoPE K,V for this prefill chunk
                # so decode iters can read without re-projecting.
                #
                # RoPE positions must be slot-local (cur..end), NOT the
                # target-forward positions in chunk_pos: gen-time appends and
                # draft queries number positions from _ctx_len, so the stored
                # keys must share that numbering. Without KV-block reuse the
                # two coincide, but on a prefix-cache hit chunk_pos starts at
                # the reused-token count, which would offset every stored key
                # from the numbering the queries use (RoPE attention depends
                # only on position differences, so the uniform slot-local
                # renumbering is exact).
                local_pos = torch.arange(cur, end, dtype=torch.long, device="cuda")
                chunk_k, chunk_v = draft_model.precompute_context_kv(chunk_proj_cast, local_pos)
                # chunk_k/v: [actual, L, nkv, hd] → [L, actual, nkv, hd]
                if self._ctx_paged:
                    # Manager block tables are keyed by batch position, the
                    # private arena's by slot. See _ctx_paged_index_args.
                    row = i if self._ctx_block_tables is not None else slot
                    positions, store_k, store_v = local_pos, chunk_k, chunk_v
                    if shard is not None:
                        # Eager prefill: keep only this rank's pages.
                        positions, owned = shard.local_slots(positions)
                        positions, store_k = positions[owned], chunk_k[owned]
                        store_v = None if chunk_v is None else chunk_v[owned]
                    self._store_context_kv_paged(
                        store_k,
                        store_v,
                        torch.full((positions.numel(),), row, dtype=torch.long, device="cuda"),
                        positions,
                    )
                else:  # VANILLA DFlash backend (FlashAttention)
                    self._ctx_k_buf[slot, :, cur:end] = chunk_k.permute(1, 0, 2, 3)
                    if chunk_v is not None:
                        self._ctx_v_buf[slot, :, cur:end] = chunk_v.permute(1, 0, 2, 3)
            offset += slen

        self._write_ctx_len(ctx_len_updates)

    def _check_no_real_generation(
        self, spec_metadata, num_contexts: int, batch_size: int, kv_cache_manager
    ):
        """Reject a generation request on a context-KV-only drafter.

        Only dummies reach a context server's generation rows: warmup and
        attention-DP padding requests (flagged by the target KV cache manager)
        and CUDA-graph padding. A real one means the server is not context-only.
        """
        is_dummy = getattr(kv_cache_manager, "_request_id_to_is_dummy", {})
        real = [
            request_id
            for request_id in spec_metadata.request_ids[num_contexts:batch_size]
            if request_id != ATTENTION_DP_DUMMY_REQUEST_ID
            and request_id < self._graph_dummy_id_floor
            and not is_dummy.get(request_id, False)
        ]
        if real:
            raise RuntimeError(
                f"Generation requests {real[:4]} reached a DSpark drafter built with "
                "speculative_config.context_kv_only, which never drafts; set it only "
                "on disaggregated context servers."
            )

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        # Restore first (in warmup mode kv_lens_cuda was saved and comes back
        # wholesale), then apply any pending rewind for the other modes so a
        # failed draft forward does not leave kv_lens_cuda incremented.
        super()._ensure_spec_dec_state_restored(attn_metadata, spec_metadata)
        if (
            getattr(self, "_kv_rewind_pending", False)
            and attn_metadata is not None
            and spec_metadata is not None
        ):
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)
        if (
            getattr(self, "_ctx_len_restore_pending", False)
            and self._ctx_len is not None
            and self._saved_ctx_len is not None
        ):
            # A failed forward must not keep this iteration's in-place
            # _ctx_len updates: roll back to the pre-forward snapshot.
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
            self._ctx_len_restore_pending = False

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        logits_vocab_shard = False
        K = spec_metadata.runtime_draft_len
        if isinstance(logits, K3VocabShardedLogits):
            if K > 0 and self._vocab_parallel_target_selection_applies(spec_metadata):
                # The target tokens come from the shard: this step returns
                # the shard, which the engine gathers only for a logits
                # post-processor.
                raw_logits = logits.local
                logits_vocab_shard = True
            else:
                logits = raw_logits = logits.gather()

        if K == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        # Lazy init buffers and attach worker reference for prepare()
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        previous_ctx_len = self._ctx_len
        self._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        spec_metadata._dflash_worker = self
        if (
            self._ctx_unified_cache
            and self._ctx_len is not previous_ctx_len
            and spec_metadata.request_ids
        ):
            # New buffers clear request slots; seed them before generation.
            spec_metadata.prepare()
        # Before any store: prefill and decode both address pages through it.
        # Returning False here means an empty batch -- the missing-offsets case
        # raises inside, with the metadata type in the message.
        self._refresh_ctx_block_tables(attn_metadata, batch_size, spec_metadata.request_ids)

        # Save context lengths so both warmup and a failed forward can roll
        # back the in-place _ctx_len updates made during drafting.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            # Never allocate the snapshot while capturing a CUDA graph: the
            # clone would live in the graph memory pool, and its replay-time
            # writes could alias blocks reused by later captures. Rollback is
            # only meaningful for eager/warmup forwards anyway; a failure
            # during capture aborts the graph itself, and captured ops do not
            # mutate _ctx_len until replay.
            self._saved_ctx_len = self._ctx_len.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._ctx_side = self._launch_ctx_side(
            draft_model, spec_metadata, attn_metadata, input_ids.shape[0]
        )

        self._execute_guided_decoder_if_present(logits)

        fused_accept = self._fused_select_accept(logits, attn_metadata, spec_metadata)
        if fused_accept is not None:
            accepted_tokens, num_accepted_tokens, bookkeeping_done = fused_accept
        else:
            bookkeeping_done = False
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                logits, attn_metadata, spec_metadata
            )

        # Opt-in acceptance recording (env-gated; eager-mode measurement
        # runs only). Skipped for CUDA-graph batches (capture/replay/warmup
        # use synthetic requests and forbid the host sync).
        if (
            self._accept_stats is not None
            and num_gens > 0
            and not spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
            and spec_metadata.request_ids is not None
        ):
            self._accept_stats.on_accept(
                spec_metadata.request_ids[num_contexts:batch_size],
                num_accepted_tokens[num_contexts:batch_size].tolist(),
            )

        # Update GDN/Mamba recurrent states to the accepted token's state.
        if (
            not bookkeeping_done
            and num_gens > 0
            and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager)
        ):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        if not bookkeeping_done:
            self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
            self._prepare_kv_for_draft_forward(
                attn_metadata, num_accepted_tokens, num_contexts, batch_size
            )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        # A context-KV-only drafter (disaggregated context server) stores the
        # prompt's drafter context above and never drafts. Main has no
        # speculative_config.context_kv_only yet, so a drafter reports False.
        drafts = not getattr(draft_model, "context_kv_only", False)
        if not drafts and num_gens > 0:
            self._check_no_real_generation(
                spec_metadata, num_contexts, batch_size, attn_metadata.kv_cache_manager
            )
        inputs = (
            self.prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                total_target_tokens=total_target_tokens,
            )
            if drafts
            else None
        )

        vocab_size = None
        if num_gens > 0 and drafts:
            with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
                # The fused preparations place the block in the pages: the
                # generic one as a DraftBlockPlacement, path.spec_tail_fusion's as
                # the MLA block fixup operands.
                fixup_kwargs = (
                    {"block_fixup": inputs["mla_block_fixup"]}
                    if inputs.get("mla_block_fixup") is not None
                    else {}
                )
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    query_positions=inputs["query_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                    ctx_k_cache=inputs["ctx_k_cache"],
                    ctx_v_cache=inputs["ctx_v_cache"],
                    ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                    ctx_kv_cache=inputs["ctx_kv_cache"],
                    ctx_page_table=inputs["ctx_page_table"],
                    block_placement=inputs["block_placement"],
                    **fixup_kwargs,
                )

                # Gather K logits per gen request from the block outputs.
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim].
                # Which block slots carry them is a drafter-family convention,
                # resolved through _draft_slot_ids.
                block_size = self._compute_block_size
                gen_gather_ids = self._block_gather_ids(
                    draft_model, num_gens, block_size, K, hidden_states_out.shape[0]
                )

                # Draft-token selection through the private head.
                draft_head = self._draft_vocab_parallel_head(draft_model, spec_metadata)
                if draft_head is not None:
                    gen_draft_tokens, gen_hidden_states = self._k3_vocab_parallel_draft_tokens(
                        draft_head,
                        draft_model,
                        hidden_states_out,
                        gen_gather_ids,
                        num_gens,
                        K,
                        inputs,
                    )
                    vocab_size = draft_head.vocab_size
                else:
                    gen_hidden_states = hidden_states_out[gen_gather_ids]
                    gen_logits = draft_model.logits_processor(
                        gen_hidden_states, draft_model.lm_head, attn_metadata, True
                    )

                    vocab_size = gen_logits.shape[-1]
                    gen_logits = gen_logits.reshape(num_gens, K, vocab_size)

                    gen_logits = self._refine_block_logits(
                        draft_model, gen_logits, inputs, spec_metadata
                    )

                    # DFlash 2: replace the independent per-position picks with
                    # one coherent path through the block (absent for plain
                    # DFlash).
                    if getattr(draft_model, "lilicorr", None) is not None:
                        gen_logits = self._apply_lilicorr(
                            draft_model,
                            gen_logits,
                            gen_hidden_states.reshape(num_gens, K, -1),
                            inputs["lilicorr_anchor"],
                            spec_metadata,
                        )
                        vocab_size = gen_logits.shape[-1]
                    elif getattr(draft_model, "has_candidate_selector", False):
                        gen_logits = self._apply_dflash2_selector(
                            draft_model,
                            gen_logits,
                            gen_hidden_states.reshape(num_gens, K, -1),
                            inputs["first_prev_tokens"],
                            spec_metadata,
                        )
                        vocab_size = gen_logits.shape[-1]

                    gen_draft_tokens = self.sample_draft_tokens(
                        gen_logits,
                        spec_metadata,
                        batch_size,
                        num_contexts=num_contexts,
                    )

                # Opt-in confidence-calibration recording (env-gated). The
                # confidence values come from an optional provider on the
                # recorder (None here -> no calibration rows collected; the
                # DSpark confidence-scheduled verification MR supplies the
                # real provider, see accept_stats.py). Guarded off for
                # CUDA-graph batches and d2t vocab-mapped drafters.
                if (
                    self._accept_stats is not None
                    and self._accept_stats.confidence_provider is not None
                    and not spec_metadata.is_cuda_graph
                    and not torch.cuda.is_current_stream_capturing()
                    and spec_metadata.request_ids is not None
                    and self._d2t is None
                ):
                    self._accept_stats.record_draft_confidence(
                        spec_metadata.request_ids[num_contexts:batch_size],
                        draft_model,
                        gen_hidden_states.reshape(num_gens, K, -1),
                        inputs["first_prev_tokens"],
                        gen_draft_tokens,
                    )

        else:
            # Warmup and attention-DP padding rows of a context-KV-only drafter
            # get the same zero placeholder as context requests.
            gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a one-hot placeholder so
        # they are a legal distribution when they become gen requests next iter.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, vocab_size)

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        next_new_tokens = None
        fused_rewind = self._fused_kv_rewind_operands(attn_metadata, spec_metadata, batch_size)
        if (
            fused_rewind is not None
            and next_draft_tokens.shape[0] == batch_size
            and next_draft_tokens.is_contiguous()
            and accepted_tokens.stride(-1) == 1
        ):
            # The rewind rides in the next-new-tokens kernel.
            self._kv_rewind_pending = False
            next_new_tokens = k3_next_new_tokens(
                accepted_tokens[:batch_size],
                next_draft_tokens,
                num_accepted_tokens[:batch_size],
                *fused_rewind,
            )
        else:
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        if next_new_tokens is None:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )

        # Restore context lengths after warmup; real runs keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        return {
            # The gathered float32 target logits, or this rank's vocab shard
            # when the target tokens were selected from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _launch_ctx_side(self, draft_model, spec_metadata, attn_metadata, total_target_tokens):
        """``path.tail_v2``: start the context-latent chain on a side stream.

        The step's context latents (TP-sharded fc + all-reduce with hidden_norm,
        the all-layer latent GEMM, RMSNorm + RoPE at ``ctx_len[slot] + j``) only
        need the captured target hidden states, so they run concurrently with
        the target-token selection and acceptance. The all-reduce uses a private
        MNNVL workspace (the main stream's all-reduces run meanwhile). Same
        kernels and inputs as the main-stream path, so the same values.

        Returns ``(latents [L, N, D], positions-read event, done event)``, or None
        when the path does not apply (then the main stream computes them).
        """
        if (
            not self._path.tail_v2
            or draft_model is None
            or getattr(draft_model, "context_kv_only", False)
        ):
            return None
        num_gens = attn_metadata.num_seqs
        if attn_metadata.num_contexts != 0 or num_gens == 0 or num_gens > TAIL_V2_MAX_GENS:
            return None
        if not self._fused_gen_drafter_inputs_applies(draft_model, spec_metadata):
            return None
        # A TP-sharded drafter fc publishes its shard as (weight, begin, end,
        # all_reduce); without one the main stream computes the latents.
        shard = getattr(draft_model, "_fc_shard", None)
        layer_major = getattr(draft_model, "precompute_context_kv_layer_major", None)
        if not shard or layer_major is None:
            return None
        weight, begin, end, all_reduce = shard
        if all_reduce.mnnvl_allreduce is None or draft_model.hidden_norm.eps is None:
            return None
        width = self._draft_tokens_per_req
        tokens = num_gens * width
        hidden = weight.shape[0]
        ar = self._ctx_side_ar
        if ar is None or not ar.fits(tokens, hidden):
            if torch.cuda.is_current_stream_capturing():
                return None
            # Collective: every rank reaches this at the same warmup step.
            ar = PrivateMNNVLAllReduce(all_reduce.mapping, weight.dtype, max(tokens, 256), hidden)
            self._ctx_side_ar = ar
            self._ctx_side_stream = torch.cuda.Stream()
        main = torch.cuda.current_stream()
        side = self._ctx_side_stream
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)
        positions_read = torch.cuda.Event()
        done = torch.cuda.Event()
        with torch.cuda.stream(side):
            positions = k3_dflash_ctx_positions(
                self._ctx_len, self._batch_to_slot_buffer.tensor[:num_gens], width
            )
            positions_read.record(side)
            captured = spec_metadata.get_hidden_states(total_target_tokens)[:tokens]
            captured = captured.to(draft_model.fc.weight.dtype)
            partial = F.linear(captured[:, begin:end], weight)
            projected = ar.rms_norm(
                partial, draft_model.hidden_norm.weight, draft_model.hidden_norm.eps
            )
            projected = projected.to(self._ctx_latent_dtype(draft_model))
            latent = layer_major(projected, positions)
            done.record(side)
        if latent is None:
            main.wait_event(done)
            return None
        # Consumed on the main stream after it waits on ``done``.
        latent.record_stream(main)
        return latent, positions_read, done

    def _fused_select_accept(self, logits, attn_metadata, spec_metadata):
        """``path.tail_v2``: target selection, strict acceptance and the
        drafter's KV-length prep in one kernel.

        Applies to vocab-parallel target selection through the private head
        with strict, uniform-width acceptance.
        It writes what ``sample_and_accept_draft_tokens`` and
        ``_prepare_kv_for_draft_forward`` write, after
        ``_prepare_attn_metadata_for_dflash`` (moved ahead: it only saves
        metadata tensors the acceptance does not touch). Returns ``(accepted,
        num_accepted, bookkeeping_done)``, or None to take the unfused path.

        With a mamba-hybrid target the kernel also writes what
        ``update_mamba_states`` writes when that is only the KDA replay record
        (:meth:`_kda_replay_operands`). Otherwise it fuses only the selection
        and the acceptance (``bookkeeping_done`` False), and
        ``update_mamba_states`` and the KV-length prep run as on the unfused
        path, in the same order.
        """
        self._fused_kv_rewind_armed = False
        if not self._path.tail_v2 or not isinstance(logits, K3VocabShardedLogits):
            return None
        if logits.head.exchange.backend != "mnnvl" and logits.head.exchange.tp_size > 1:
            return None
        # Main's ragged verification packs per-request windows; the kernel
        # reads a uniform K + 1 tokens per generation request.
        if spec_metadata.is_ragged_verify:
            return None
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        if num_gens > TAIL_V2_MAX_GENS:
            return None
        if num_gens > 0 and self._can_use_rejection_sampling(spec_metadata):
            return None
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, logits.device)
        pairs = logits.select_pairs(spec_metadata)
        force = self._force_accept_kernel_args(draft_tokens.shape[-1], spec_metadata, logits.device)
        kda_replay = None
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
            fusable, kda_replay = self._kda_replay_operands(attn_metadata)
            if not fusable:
                accepted, num_accepted, _ = k3_fused_select_accept(
                    pairs, draft_tokens, num_contexts, batch_size, force
                )
                return accepted, num_accepted, False
        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        accepted, num_accepted, rewind = k3_fused_select_accept(
            pairs,
            draft_tokens,
            num_contexts,
            batch_size,
            force,
            kda_replay=kda_replay,
            kv_lens=kv_lens,
        )
        if kv_lens is not None:
            self._kv_rewind_amount = rewind
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
            attn_metadata.update_for_spec_dec()
        self._fused_kv_rewind_armed = kv_lens is not None
        return accepted, num_accepted, True

    @staticmethod
    def _kda_replay_operands(attn_metadata):
        """``(fusable, kda_replay)`` for :func:`fused_select_accept` with a
        mamba-hybrid target.

        Fusable when ``update_mamba_states(attn_metadata, num_accepted,
        state_indices=mamba_metadata.state_indices)`` is
        ``KimiK3HybridCacheManager``'s and writes nothing but
        ``KDAReplayState.record_acceptance``. ``kda_replay`` is then
        ``(prev_num_accepted_tokens, state_indices, dummy_mask)``, or None when
        that call records nothing (a context-only KDA manager, or no local KDA
        layers).
        """
        mgr = attn_metadata.kv_cache_manager
        if (
            not isinstance(mgr, KimiK3HybridCacheManager)
            or type(mgr).update_mamba_states is not KimiK3HybridCacheManager.update_mamba_states
        ):
            return False, None
        if mgr.kda_context_only or mgr.local_num_mamba_layers == 0:
            return True, None
        replay = mgr._speculative_state
        if (
            type(replay) is not KDAReplayState
            or replay.prev_num_accepted_tokens is None
            or mgr._dummy_request_mask is None
        ):
            return False, None
        state_indices = attn_metadata.mamba_metadata.state_indices
        if state_indices is None:
            state_indices = mgr.get_state_indices()
        dummy_mask = mgr._dummy_request_mask
        if state_indices.stride(-1) != 1 or dummy_mask.stride(-1) != 1:
            return False, None
        return True, (replay.prev_num_accepted_tokens, state_indices, dummy_mask)

    def _fused_kv_rewind_operands(self, attn_metadata, spec_metadata, batch_size: int):
        """``(kv_lens, amounts, num_contexts)`` when the rewind after the draft
        forward runs inside the next-new-tokens kernel, else None (the
        ``_apply_kv_rewind_after_draft`` path, which also covers warmup)."""
        if not self._fused_kv_rewind_armed:
            return None
        self._fused_kv_rewind_armed = False
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if (
            is_warmup
            or self._kv_rewind_amount is None
            or not hasattr(attn_metadata, "kv_lens_cuda")
            or self._kv_rewind_bs != batch_size
            or not self._path.spec_tail_fusion
        ):
            return None
        return attn_metadata.kv_lens_cuda, self._kv_rewind_amount, self._kv_rewind_nc

    def _force_accept_kernel_args(
        self, runtime_draft_len: int, spec_metadata, device: torch.device
    ) -> Optional[tuple]:
        """The ``force`` argument of ``fused_strict_accept`` / ``fused_select_accept``
        that reproduces :meth:`_apply_force_accepted_tokens` (same warmup skip
        and draws)."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if (
            spec_metadata is not None
            and spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
        ):
            return None
        int_part = int(self.force_num_accepted_tokens)
        frac_part = self.force_num_accepted_tokens - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if frac_part > 0.0 and base_total < max_total:
            self._ensure_force_accept_rng_state(device)
            return (
                base_total,
                frac_part,
                max_total,
                self._force_accept_rng_pool,
                self._force_accept_rng_counter,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            )
        return (base_total,)

    def _sample_and_accept_draft_tokens_base(
        self,
        logits,
        draft_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        spec_metadata,
    ):
        """``path.spec_tail_fusion``: the strict acceptance and the force
        override in one launch (``fused_strict_accept``), the same values.

        Off, or under ragged verification, the base implementation runs.
        """
        if not self._path.spec_tail_fusion or (
            spec_metadata is not None and spec_metadata.is_ragged_verify
        ):
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if isinstance(logits, torch.Tensor) and logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        return k3_fused_strict_accept(
            target_tokens,
            draft_tokens,
            num_contexts,
            batch_size,
            self._force_accept_kernel_args(
                draft_tokens.shape[-1], spec_metadata, target_tokens.device
            ),
        )

    def _sample_tokens_for_batch(
        self,
        logits,
        spec_metadata,
        num_contexts: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Main's sampler, which also takes the private head's shard logits.

        The worker only passes shard logits through when vocab-parallel
        selection covers this batch's sampling mode.
        """
        if isinstance(logits, K3VocabShardedLogits):
            return logits.select_tokens(spec_metadata)
        return super()._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)

    def _block_gather_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int, num_rows: int
    ) -> torch.Tensor:
        """The block-output rows gathered for the K draft logits per request.

        Shape-only, so with ``path.spec_tail_fusion`` it is built once per
        shape and reused instead of being recomputed every step.
        """
        key = (num_gens, block_size, num_draft_tokens, num_rows)
        if self._path.spec_tail_fusion:
            cache = self.__dict__.setdefault("_block_gather_id_cache", {})
            if key in cache:
                return cache[key]
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        # Shields only the last request: at block_size == K with shift_label
        # off, slots run 1..K, so every request reads the next one's slot 0
        # and the last overruns. Degrades, never raises.
        ids = ids.clamp(max=num_rows - 1)
        if self._path.spec_tail_fusion and not torch.cuda.is_current_stream_capturing():
            self._block_gather_id_cache[key] = ids
        return ids

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        # batch_indices_cuda is the identity here (DFlashSpecMetadata.prepare).
        if (
            self._path.spec_tail_fusion
            and next_draft_tokens.shape[0] == batch_size
            and next_draft_tokens.is_contiguous()
            and accepted_tokens.stride(-1) == 1
        ):
            return k3_next_new_tokens(
                accepted_tokens[:batch_size], next_draft_tokens, num_accepted_tokens[:batch_size]
            )
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )

    def _draft_first_slot(self, draft_model) -> int:
        """The block slot whose hidden state predicts the first draft token
        (``_draft_slot_ids`` takes slots ``first .. first + K - 1`` of each block)."""
        return 1

    def _draft_block_width(self, draft_model) -> int:
        """Block slots the draft forward must compute for K draft tokens.

        Plain DFlash reads slots 1..K, so slot 0 is pure overhead and the
        forward needs K+1. Families whose slot convention differs override
        this alongside :meth:`_draft_slot_ids` — the two must agree, since
        the width bounds the slots the gather is allowed to name.
        """
        return self.max_draft_len + 1

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots whose hidden states produce the K draft logits.

        Plain DFlash uses the K2.7 convention: mask slots 1..K. Drafter
        families with another convention override this — see
        :meth:`K3DSparkWorker._draft_slot_ids` for the shift_label
        variant, which reads slots 0..K-1 instead.
        """
        return dflash_draft_slot_ids(num_gens, block_size, num_draft_tokens, False, device="cuda")

    def _refine_block_logits(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        inputs: dict,
        spec_metadata,
    ) -> torch.Tensor:
        """Refine the block logits between the draft forward and sampling.

        Plain DFlash proposes the backbone's logits unchanged. Drafter
        families carrying extra heads override this — see
        :meth:`K3DSparkWorker._refine_block_logits` for the Markov
        intra-block bias.
        """
        return gen_logits

    def _apply_dflash2_selector(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_tokens: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        """Replace DFlash's independent per-position picks with the DFlash 2
        candidate selector's chosen path.

        Returns full-vocab ``[num_gens, K, vocab]`` logits carrying only each
        position's top-k candidates, scored under the predecessor the walk
        settled on. Downstream sampling is unchanged and stays lossless: each
        row is the distribution its draft token was drawn from.

        The walk always chains through the greedy successor, so under sampling
        a draft token can differ from the predecessor the next position was
        scored under. That costs acceptance rate, not correctness — same
        trade-off as ``_apply_dspark_markov_bias``.
        """
        selector = draft_model.candidate_selector
        candidate_ids, unary_logits, block_logits = self._dflash2_global_top_k(
            gen_logits,
            spec_metadata,
            top_k=selector.top_k,
            full_vocab=selector.predecessor_codebook.shape[0],
        )
        return draft_model.select_candidate_path(
            block_logits,
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_tokens.long(),
        )

    def _apply_lilicorr(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        draft_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        if anchor_hidden is None:
            raise RuntimeError(
                "LiLiCorr requires the projected target row that predicted the anchor"
            )
        full_vocab = draft_model.config.vocab_size
        candidate_ids, unary_logits, block_logits = self._dflash2_global_top_k(
            gen_logits, spec_metadata, draft_model.lilicorr.candidate_topk, full_vocab
        )
        # Candidate features use probabilities normalized over the entire vocabulary,
        # including the mass outside top-k and on other tensor-parallel ranks.
        log_normalizer = torch.logsumexp(gen_logits.float(), dim=-1, keepdim=True)
        if gen_logits.shape[-1] != full_vocab:
            from tensorrt_llm._torch.distributed.ops import allgather

            log_normalizer = torch.logsumexp(
                allgather(log_normalizer, self.mapping, dim=-1), dim=-1, keepdim=True
            )
        proposal = draft_model.select_lilicorr_path(
            candidate_ids, unary_logits.float() - log_normalizer, draft_hidden, anchor_hidden
        )
        block_logits.fill_(float("-inf"))
        block_logits.scatter_(-1, candidate_ids, proposal.to(block_logits.dtype))
        return block_logits

    def _dflash2_global_top_k(
        self,
        gen_logits: torch.Tensor,
        spec_metadata,
        top_k: int,
        full_vocab: int,
    ):
        """Per-position global top-k candidates, and the tensor to score into.

        The selector indexes full-vocab codebooks, so candidate ids must be
        global. A vocab-sharded draft head is reduced by all-gathering only each
        rank's local (id, value) pairs — 2 * top_k per position instead of the
        whole vocabulary. Every rank then writes identical rows, so downstream
        sampling needs no further collective.

        Returns ``(candidate_ids, unary_logits, block_logits)``, where
        ``block_logits`` is ``gen_logits`` rewritten in place when it is already
        full-vocab (the common single-rank case) and a fresh tensor otherwise.
        """
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            unary_logits, candidate_ids = torch.topk(gen_logits, top_k, dim=-1)
            return candidate_ids, unary_logits, gen_logits

        mapping = self.mapping
        if not (
            self._draft_logits_are_sharded(gen_logits, spec_metadata)
            and mapping is not None
            and shard * mapping.tp_size == full_vocab
        ):
            raise NotImplementedError(
                f"DFlash 2 candidate selection needs the drafter's full vocab "
                f"({full_vocab}) or a plain TP column shard of it; got draft "
                f"logits of width {shard}."
            )

        from tensorrt_llm._torch.distributed.ops import allgather

        local_values, local_ids = torch.topk(gen_logits, top_k, dim=-1)
        local_ids = local_ids + mapping.tp_rank * shard
        # Interleaved (id, value) pairs, matching _get_local_max_and_combined.
        combined = torch.stack([local_ids.float(), local_values.float()], dim=-1).flatten(-2)
        gathered = allgather(combined, mapping, dim=-1)
        unary_logits, selected = torch.topk(gathered[..., 1::2], top_k, dim=-1)
        candidate_ids = gathered[..., 0::2].gather(-1, selected).long()
        block_logits = gen_logits.new_full((*gen_logits.shape[:-1], full_vocab), float("-inf"))
        return candidate_ids, unary_logits, block_logits

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """Prepare inputs for DFlash's draft forward.

        For gen requests, builds:
        - noise_embedding: token embeddings for [accepted + mask] tokens
        - query_positions: position IDs for the query tokens
        - num_ctx_per_req: per-request context length in the pool
        - ctx_k_cache / ctx_v_cache / ctx_cache_batch_idx: slot-indexed
          views of the persistent per-layer K/V pool.
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        # Resolve mask_token_id and block_size once, cache for subsequent calls
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        mask_token_id = self._resolved_mask_token_id

        # Get the embed_tokens layer from the draft model
        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )
        lilicorr_anchor = None
        # The drafter's own trained mask row, if it kept one for the mask id in use.
        trained_mask_embedding = (
            getattr(draft_model, "mask_token_embedding", None)
            if mask_token_id == getattr(draft_model, "mask_token_id", None)
            else None
        )

        # One fused preparation at most: path.spec_tail_fusion selects
        # _prepare_gen_drafter_inputs_fused, else the paged-latent one; each falls back to the eager path below.
        if num_gens > 0:
            if self._path.spec_tail_fusion:
                if self._fused_gen_drafter_inputs_applies(draft_model, spec_metadata):
                    return self._prepare_gen_drafter_inputs_fused(
                        accepted_tokens,
                        num_accepted_tokens,
                        attn_metadata,
                        spec_metadata,
                        draft_model,
                        embed_tokens,
                        total_target_tokens,
                        mask_token_id,
                        trained_mask_embedding,
                    )
            elif self._fused_gen_prepare_applies(spec_metadata):
                return self._prepare_paged_latent_gen_inputs(
                    accepted_tokens,
                    num_accepted_tokens,
                    attn_metadata,
                    spec_metadata,
                    draft_model,
                    total_target_tokens,
                    mask_token_id,
                    trained_mask_embedding,
                )

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # K+1
            K = spec_metadata.runtime_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = (
                captured_hs is not None
                and hasattr(draft_model, "fc")
                and hasattr(draft_model, "hidden_norm")
            )

            # Use cached block_size (resolved once on first call)
            block_size = self._compute_block_size
            query_tokens_per_req = block_size

            # Get slots for gen requests from pre-computed mapping
            slots = self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens]
            gen_rows_out = torch.arange(
                num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
            )

            K_plus_1 = K + 1

            bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
            bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()

            ctx_len_gen = self._ctx_len[slots]
            j_block = torch.arange(query_tokens_per_req, dtype=torch.long, device="cuda")
            offsets_kp1 = torch.arange(K_plus_1, dtype=torch.long, device="cuda")

            # _ctx_len is clamped to _max_ctx only AFTER this step's accepted
            # tokens are folded in (see the update below), so the running length
            # used here has to be clamped on its own -- otherwise a request that
            # already sits at the ceiling indexes num_accepted positions past
            # any position the sequence can legitimately reach.
            ctx_len_now = (ctx_len_gen + gen_num_accepted.long()).clamp_(max=self._max_ctx)
            query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)
            ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)

            noise_embed_2d = dflash_noise_block_embedding(
                embed_tokens, bonus, mask_token_id, query_tokens_per_req, trained_mask_embedding
            )

            # Accumulate new accepted features into context buffers
            if has_target_features:
                gen_start = attn_metadata.num_ctx_tokens
                # Target now processes exactly K+1 tokens per gen req, so the
                # captured slice is already the full set we need to project.
                gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
                gen_hs_to_project = gen_hs.reshape(-1, gen_hs.shape[-1])
                projected_to_store = draft_model.project_target_hidden(gen_hs_to_project)
                if getattr(draft_model, "lilicorr", None) is not None:
                    # The last accepted output is predicted by the corresponding input
                    # row; the newly committed token itself has no target row yet.
                    projected_rows = projected_to_store.reshape(num_gens, K_plus_1, -1)
                    lilicorr_anchor = last_accepted_hidden(projected_rows, gen_num_accepted)
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                owned = None
                shard = self._helix_shard()
                if shard is not None:
                    # Helix: rank-local slots; the other ranks' positions
                    # rewrite their slot's current value (see
                    # _store_context_kv_paged).
                    assert K + 1 <= shard.page_size, "helix alias slots must lie outside the write"
                    col_idx, owned = shard.local_slots(col_idx)
                    owned = owned.reshape(-1)
                if self._ctx_block_tables is not None:
                    # Bound by what the manager handed THIS request: the masked
                    # entries below are still written (fixed-size, for graph
                    # safety) and a placeholder page resolves to another's.
                    gen_capacity = self._ctx_block_counts[gen_rows_out] * self._ctx_page_size
                    col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
                if self._ctx_block_tables is None:
                    # The private arena.
                    col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store  # already [num_gens*(K+1), proj_dim]
                pos_flat = ctx_position_ids.long().reshape(-1)
                mask_1d = write_mask.reshape(-1)

                # Fast path: store the pre-projected/pre-RoPE'd K/V.
                # dflash_forward reads these directly via cache_batch_idx.
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                if cache_dtype == torch.float8_e4m3fn:
                    # Project in the drafter's weight dtype; the store converts.
                    cache_dtype = proj_flat.dtype
                k_new, v_new = draft_model.precompute_context_kv(
                    proj_flat.to(cache_dtype), pos_flat
                )
                mask_bc = mask_1d.view(-1, 1, 1, 1).to(k_new.dtype)
                k_new.mul_(mask_bc)
                if v_new is not None:
                    v_new.mul_(mask_bc)
                slot_long = slot_flat.long()
                col_long = col_flat.long()
                if self._ctx_paged:
                    if self._ctx_block_tables is not None:
                        # Batch positions of the gen requests, matching the
                        # per-request block table's row order.
                        rows_long = gen_rows_out.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                    else:
                        rows_long = slot_long
                    self._store_context_kv_paged(k_new, v_new, rows_long, col_long, owned)
                else:  # VANILLA DFlash backend (FlashAttention)
                    self._ctx_k_buf[slot_long, :, col_long] = k_new
                    if v_new is not None:
                        self._ctx_v_buf[slot_long, :, col_long] = v_new

                self._advance_ctx_len(slots, gen_num_accepted_long)

            num_ctx_per_req_t = self._ctx_len[slots]
            if self._ctx_block_tables is not None:
                # The write above clamps columns to the request's allocation, so
                # a context that outruns it has its tail written over the last
                # valid slot. Truncate what is advertised to match, or the read
                # attends that clobbered value -- or another request's page.
                # Under helix local page k holds global page k * cp_size +
                # cp_rank and is the alias of every other page in that span,
                # so k local pages cover k * cp_size global pages.
                shard = self._helix_shard()
                cp = 1 if shard is None else shard.cp_size
                allocated = dflash_allocated_ctx_limit(
                    self._ctx_block_counts * cp, self._ctx_page_size, block_size
                )
                if self._ctx_unified_cache:
                    self._check_block_within_allocation(
                        num_ctx_per_req_t, allocated[gen_rows_out], block_size
                    )
                num_ctx_per_req_t = torch.minimum(num_ctx_per_req_t, allocated[gen_rows_out])
            noise_embedding = noise_embed_2d
            query_positions = query_position_ids.long()

            # Update seq_lens for gen requests to K+1 (the number of tokens
            # the target forward actually processed).
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        else:
            noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
            query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")
            slots = torch.empty(0, dtype=torch.long, device="cuda")
            gen_rows_out = slots
            bonus = torch.empty(0, dtype=torch.long, device="cuda")

        return {
            "lilicorr_anchor": lilicorr_anchor,
            "noise_embedding": noise_embedding,
            "query_positions": query_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            # Slots index the private arena; the manager's block table is keyed
            # by batch position, so the drafter reads its own rows there.
            "ctx_cache_batch_idx": gen_rows_out if self._ctx_block_tables is not None else slots,
            # Anchor token per gen request (block slot 0): last accepted
            # token. The dspark Markov chain conditions its first step on it.
            "first_prev_tokens": bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": (
                self._ctx_block_tables
                if self._ctx_block_tables is not None
                else self._ctx_page_table
            ),
            "block_placement": None,
        }

    def _fused_gen_prepare_applies(self, spec_metadata) -> bool:
        """Whether :meth:`_prepare_paged_latent_gen_inputs` covers this step: a
        drafter it supports (``_paged_latent_prepare``), captured target
        features, drafting from all ``K + 1`` target tokens."""
        return (
            self._paged_latent_prepare
            # Shared allocations require the generic path's strict capacity
            # check; the fused preparation silently caps the context length.
            and not self._ctx_unified_cache
            and spec_metadata.runtime_draft_len + 1 == self._draft_tokens_per_req
            and spec_metadata.captured_hidden_states is not None
        )

    def _prepare_paged_latent_gen_inputs(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor],
    ) -> dict:
        """:meth:`prepare_1st_drafter_inputs` for the generation requests of a
        step (context requests may precede them) whose drafter context is a
        paged single-latent cache.

        Same inputs and side effects (context latents stored, ``_ctx_len``
        advanced, ``seq_lens`` set) with the per-request bookkeeping in one
        kernel and the latent store of every layer in another; also places the
        block's own positions in the pages for the drafter's block decode.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        prep = k3_dflash_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens],
            self._ctx_len,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            num_contexts,
            self._dummy_slot,
            self._max_ctx,
            self._ctx_page_size,
            block_size,
            width,
        )
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = width

        noise_embedding = dflash_noise_block_embedding(
            draft_model.draft_model_full.model.embed_tokens,
            prep.bonus,
            mask_token_id,
            block_size,
            trained_mask_embedding,
        )
        gen_start = attn_metadata.num_ctx_tokens
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
        projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
        latent, _ = draft_model.precompute_context_kv(
            projected.to(self._ctx_kv_buf[0].dtype), prep.ctx_positions
        )
        k3_store_context_latents(
            latent[:, :, 0],
            self._ctx_kv_buf,
            self._ctx_kv_ptrs,
            prep.store_pages,
            prep.store_offsets,
            prep.store_mask,
            mla=False,
        )
        return {
            "lilicorr_anchor": None,
            "noise_embedding": noise_embedding,
            "query_positions": prep.query_positions,
            "num_ctx_per_req": prep.num_ctx,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            "ctx_cache_batch_idx": torch.arange(
                num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
            ),
            "first_prev_tokens": prep.bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": DraftBlockPlacement(
                prep.blk_pages, prep.blk_offsets, prep.blk_seq_lens, prep.blk_kv_bounds
            ),
        }

    def _ctx_latent_dtype(self, draft_model) -> torch.dtype:
        """Dtype the context latents are computed in: the pool's, or the
        projection's activation dtype for an FP8 pool, whose stores quantize them."""
        dtype = self._ctx_kv_buf[0].dtype
        return dflash_context_dtype(draft_model) if dtype == torch.float8_e4m3fn else dtype

    def _fused_gen_drafter_inputs_applies(self, draft_model, spec_metadata) -> bool:
        """Whether ``_prepare_gen_drafter_inputs_fused`` (``path.spec_tail_fusion``)
        covers this step: a generation batch drafting from a single-latent (MLA)
        context that lives in the draft KV manager's paged pool.

        LiLiCorr drafters keep the eager path, which computes their anchor.
        """
        return (
            self._path.spec_tail_fusion
            and self._ctx_paged
            and self._ctx_block_tables is not None
            and getattr(draft_model, "_kv_factor", 2) == 1
            and hasattr(draft_model, "fc")
            and hasattr(draft_model, "hidden_norm")
            and getattr(draft_model, "lilicorr", None) is None
            # The fused latent store writes in the projection's dtype or FP8.
            and self._ctx_kv_buf[0].dtype
            in (dflash_context_dtype(draft_model), torch.float8_e4m3fn)
            and spec_metadata.get_hidden_states(0) is not None
            and spec_metadata.runtime_draft_len + 1 == self._draft_tokens_per_req
            and self._compute_block_size is not None
        )

    def _ctx_kv_pool_ptrs(self) -> torch.Tensor:
        """Device array of the per-layer context-pool base addresses (cached)."""
        ptrs = tuple(buf.data_ptr() for buf in self._ctx_kv_buf)
        if getattr(self, "_ctx_kv_ptrs_key", None) != ptrs:
            self._ctx_kv_ptrs = torch.tensor(ptrs, dtype=torch.int64, device="cuda")
            self._ctx_kv_ptrs_key = ptrs
        return self._ctx_kv_ptrs

    def _fused_prepare_constants(
        self, num_contexts: int, num_gens: int, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The block fixup's upper-triangle mask and the gen rows' batch indices.

        Shape-only. With ``path.tail_v2`` they are built once per shape
        outside graph capture and reused, so no kernel recomputes them per step.
        """
        key = (num_contexts, num_gens, block_size)
        cache = self.__dict__.setdefault("_fused_prepare_constant_cache", {})
        if self._path.tail_v2 and key in cache:
            return cache[key]
        idx = torch.arange(block_size, device="cuda")
        valid = (idx.view(1, block_size) > idx.view(block_size, 1)).view(
            1, block_size, 1, block_size
        )
        batch_idx = torch.arange(
            num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
        )
        if self._path.tail_v2 and not torch.cuda.is_current_stream_capturing():
            cache[key] = (valid, batch_idx)
        return valid, batch_idx

    def _prepare_gen_drafter_inputs_fused(
        self,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        embed_tokens,
        total_target_tokens: int,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        """``prepare_1st_drafter_inputs`` for the paged single-latent context
        (``path.spec_tail_fusion``; off, the generic
        :meth:`_prepare_paged_latent_gen_inputs` runs instead).

        Produces the same drafter inputs and the same side effects (context
        latents stored, ``_ctx_len`` advanced, ``seq_lens`` set) with the
        per-request bookkeeping in one kernel, the noise embedding in one copy
        kernel and the latent store in one scatter kernel. Also returns the
        block-decode fixup operands, which ``dflash_forward`` would otherwise
        rebuild from a gathered copy of the block table.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        slots = self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens]
        side = self._ctx_side
        self._ctx_side = None
        if side is not None:
            # The side stream has read _ctx_len before prep advances it.
            torch.cuda.current_stream().wait_event(side[1])
        shard = self._helix_shard()
        prep = k3_dflash_gen_prepare(
            num_accepted_tokens,
            accepted_tokens,
            slots,
            self._ctx_len,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            num_contexts,
            num_gens,
            self._dummy_slot,
            self._max_ctx,
            int(mask_token_id),
            self._ctx_page_size,
            block_size,
            width,
            cp_size=1 if shard is None else shard.cp_size,
            cp_rank=0 if shard is None else shard.cp_rank,
        )
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = width
        if self._ctx_unified_cache:
            # The kernel caps the advertised context so the block fits the
            # allocation; a shared allocation must not need the cap.
            self._check_block_within_allocation(self._ctx_len[slots], prep["num_ctx"], block_size)

        # A TP-sharded vocab masks its rows and all-reduces (embed_tokens.forward
        # semantics; one lookup kernel when the embedding is vocab-sharded).
        # Owner-rank multicast of each token's row instead of the all-reduce,
        # where it applies.
        embed_ids = prep["embed_ids"]
        combined_embed = draft_embed_site(embed_tokens, embed_ids)
        if combined_embed is None and vocab_shard_embedding_applies(embed_tokens, embed_ids):
            local_rows = k3_vocab_shard_embedding(
                embed_ids, embed_tokens.weight, embed_tokens.vocab_start, embed_tokens.vocab_end
            )
            combined_embed = embed_tokens.all_reduce(local_rows)
        if combined_embed is None:
            combined_embed = embed_tokens(prep["embed_ids"])
        if trained_mask_embedding is not None:
            # The drafter's own trained mask row replaces the shared lookup's
            # (the last row), as in dflash_noise_block_embedding.
            combined_embed = torch.cat(
                (
                    combined_embed[:num_gens],
                    trained_mask_embedding.to(combined_embed.dtype).unsqueeze(0),
                )
            )
        noise_embedding = k3_dflash_noise_embedding(combined_embed, block_size)

        if side is not None:
            # Computed on the side stream from the same captured hidden states
            # and the same positions (ctx_len[slot] + j).
            torch.cuda.current_stream().wait_event(side[2])
            latent = side[0].transpose(0, 1)
        else:
            gen_start = attn_metadata.num_ctx_tokens
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
            projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
            projected = projected.to(self._ctx_latent_dtype(draft_model))
            # Layer-major latents skip the per-layer staging copies when the
            # drafter offers them; the values are the same either way.
            layer_major = getattr(draft_model, "precompute_context_kv_layer_major", None)
            latent = None if layer_major is None else layer_major(projected, prep["ctx_positions"])
            if latent is not None:
                latent = latent.transpose(0, 1)
            else:
                latent = draft_model.precompute_context_kv(projected, prep["ctx_positions"])[0][
                    :, :, 0
                ]
        store_args = (
            latent,
            self._ctx_kv_buf,
            self._ctx_kv_pool_ptrs(),
            prep["store_pages"],
            prep["store_offsets"],
            prep["store_mask"],
        )
        k3_store_context_latents(*store_args, mla=True)

        valid, batch_idx = self._fused_prepare_constants(num_contexts, num_gens, block_size)
        block_fixup = _MLABlockFixup(
            pages=prep["blk_pages"],
            offsets=prep["blk_offsets"],
            valid=valid,
            page_tables_i32=self._ctx_block_tables[num_contexts : num_contexts + num_gens],
            seq_lens_i32=prep["blk_seq_lens"],
            kv_bounds_i32=prep["blk_kv_bounds"],
            owned=None if shard is None else self._blk_owned_mask(prep["blk_owned"]),
        )
        return {
            "lilicorr_anchor": None,
            "noise_embedding": noise_embedding,
            "query_positions": prep["query_positions"],
            "num_ctx_per_req": prep["num_ctx"],
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            "ctx_cache_batch_idx": batch_idx,
            "first_prev_tokens": prep["bonus"],
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": None,
            "mla_block_fixup": block_fixup,
        }

    @staticmethod
    def _blk_owned_mask(blk_owned: torch.Tensor) -> torch.Tensor:
        """The ``[gens, block, 1]`` bool ownership mask of the prep kernel's 0 / 1 int8 rows."""
        return blk_owned.bool().unsqueeze(-1)


class K3DSparkWorker(K3DFlashWorker):
    """Worker for the K3 family's standalone MLA DSpark drafter.

    DSpark is DFlash plus two heads, so the drafting plumbing -- paged context latent, slot
    management, the mask-token block forward -- is :class:`K3DFlashWorker`'s, and only the two
    head-driven policies are overridden here: the block-output slot convention
    (``shift_label``) and the Markov intra-block logit bias. Everything backbone-shaped is
    supplied by the drafter (``K3DSparkDrafter``), which reports its own shapes
    (``_num_attn_layers``, ``_num_heads``, ``_num_kv_heads``, ``_head_dim``) and owns the
    operators (``_build_fused_kv_buffers``, ``precompute_context_kv``, ``dflash_forward``,
    ``apply_markov_chain_logits``, ``project_target_hidden``).
    """

    def set_draft_model(self, draft_model) -> None:
        """Reject an unsupported vocab mapping here rather than mid-decode.

        ``d2t`` is model-static, so a config mistake should surface at load and
        not as a ``NotImplementedError`` raised per decode step, possibly during
        CUDA-graph capture.
        """
        super().set_draft_model(draft_model)
        if self._d2t is not None and getattr(draft_model, "has_markov_head", False):
            raise NotImplementedError(
                "DSpark Markov head requires a shared draft/target vocab "
                "(d2t vocab mapping is not supported); drafter "
                f"{type(draft_model).__name__} declares one."
            )

    def _draft_block_width(self, draft_model) -> int:
        """Block width under the dspark ``shift_label`` convention.

        shift_label reads slots 0..K-1, so K draft tokens fit in K slots and
        the base class' K+1 over-demands by one -- enough to reject a block-7
        checkpoint at max_draft_len=7, which is how both published DSpark
        drafters are meant to run.
        """
        if draft_model.shift_label:
            return self.max_draft_len
        return super()._draft_block_width(draft_model)

    def _draft_first_slot(self, draft_model) -> int:
        return 0 if draft_model.shift_label else 1

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """Block-output slots under the dspark ``shift_label`` convention.

        The drafter checkpoint declares the convention, so it is read off the
        draft model rather than assumed: a DSpark drafter trained with the
        legacy DFlash slot layout keeps the base class' slots 1..K.
        """
        return dflash_draft_slot_ids(
            num_gens, block_size, num_draft_tokens, draft_model.shift_label, device="cuda"
        )

    def _draft_vocab_parallel_head(self, draft_model, spec_metadata):
        """Also require Markov weights the vocab-parallel chain can index:
        contiguous ``[vocab, rank]`` tensors of one dtype."""
        head = super()._draft_vocab_parallel_head(draft_model, spec_metadata)
        if head is None or not getattr(draft_model, "has_markov_head", False):
            return head
        w1, w2 = draft_model.markov_w1, draft_model.markov_w2
        if (
            w1.is_contiguous()
            and w2.is_contiguous()
            and w1.shape == w2.shape
            and w1.dtype == w2.dtype
            and w2.shape[0] == head.vocab_size
        ):
            return head
        return None

    def _vocab_parallel_markov_weights(self, draft_model):
        if not getattr(draft_model, "has_markov_head", False):
            return None, None
        return draft_model.markov_w1, draft_model.markov_w2

    def _refine_block_logits(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        inputs: dict,
        spec_metadata,
    ) -> torch.Tensor:
        """Add the greedy-chained Markov intra-block bias to the block logits.

        A DSpark drafter checkpoint may omit the Markov head (``markov_rank``
        0), which loads as a drafter without one; that case falls through to
        the unmodified backbone logits.
        """
        if not getattr(draft_model, "has_markov_head", False):
            return gen_logits
        return self._apply_dspark_markov_bias(
            draft_model, gen_logits, inputs["first_prev_tokens"], spec_metadata
        )

    def _apply_dspark_markov_bias(
        self,
        draft_model,
        gen_logits: torch.Tensor,
        first_prev_tokens: torch.Tensor,
        spec_metadata=None,
    ) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        Reference (DeepSpec VanillaMarkov.sample_block_tokens, temperature 0):
        step i adds bias = markov_w2 @ markov_w1[prev_i] to the shared-lm_head
        logits, where prev_0 is the anchor (last accepted) token and prev_{i>0}
        is the greedy token from step i-1's biased logits. Greedy per-position
        argmax of the returned logits therefore reproduces the reference
        sampled chain; the rejection-sampling path samples from the same
        biased distributions (proposal conditioned on the greedy chain).

        The logits are full-vocab: the drafter shares the target's gathered
        LM head (vocab-sharded selection runs the chain through
        :meth:`_draft_vocab_parallel_head` instead). Full-vocab logits go through the fused chain
        (``dspark_markov_fused``) where it supports them, instead of the
        drafter's ``apply_markov_chain_logits``; the same chain bit for bit.
        """
        # The d2t guard lives in set_draft_model: it is model-static, so raising
        # it here would surface a load-time config error per decode step.
        # This one keys on the runtime logits width.
        full_vocab = draft_model.markov_w2.shape[0]
        if gen_logits.shape[-1] != full_vocab:
            raise NotImplementedError(
                f"DSpark Markov head: draft logits width {gen_logits.shape[-1]} does not "
                f"match the drafter vocab {full_vocab} (the shared LM head gathers its output)."
            )
        if (
            spec_metadata is None or not self._draft_logits_are_sharded(gen_logits, spec_metadata)
        ) and fused_markov_supported(gen_logits, draft_model.markov_w1, draft_model.markov_w2):
            return k3_fused_markov_chain_logits(
                gen_logits, first_prev_tokens, draft_model.markov_w1, draft_model.markov_w2
            )
        return draft_model.apply_markov_chain_logits(gen_logits, first_prev_tokens)
