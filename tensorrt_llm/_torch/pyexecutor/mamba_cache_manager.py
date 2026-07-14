# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Union

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID, LlmRequest)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager,
    PoolConfiguration, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager import (
    LinearAttentionMetadata, LinearCacheType)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

GB = 1 << 30

# Replay kernels pad the token/window dimension to at least 16 for tensor-core
# tiles, so history sizes below 16 are no faster when not writing and do
# expensive checkpoint writes more often. This minimum tensor-core tile size is
# present in all tensor-core generations, with newer GPUs adding larger tile
# operations, not smaller ones. In the other direction, history sizes above 16
# currently pad to 32, which is substantially slower with the current kernel
# design. Keep the default floor at 16 while still allowing larger T, which
# degenerates to checkpointing every step instead of rejecting the request. If
# larger T becomes common, we could consider padding larger histories to powers
# of 2 or multiples of 16, but this is untested. For smaller T, we could explore
# combining larger histories with new kernel designs that stay efficient when
# the window is only partly full.
MIN_REPLAY_HISTORY_SIZE = 16


def get_tensor_size_bytes(tensor):
    """Calculate tensor size in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.element_size() * tensor.nelement()
    elif isinstance(tensor, list):
        return sum(get_tensor_size_bytes(t) for t in tensor)
    return 0


# Mamba SSM stochastic-rounding Philox seed plumbing.
#
# Both the replay kernel and flashinfer's selective_state_update consume a
# `rand_seed` int64 tensor.  Historically the non-replay paths created this
# tensor via `torch.randint(..., (1,))` on every decode step, which (a) is
# non-deterministic across runs, (b) allocates a fresh CUDA tensor per call
# and is therefore unfriendly to CUDA-graph capture, and (c) has no notion
# of cache-slot identity (so slot reuse cannot rotate the stream).
#
# The functions below produce per-slot int64 seeds via SplitMix64 finalization
# of (counter, slot, rank).  Adjacent inputs yield uncorrelated outputs so
# consecutive cache slots and consecutive request counters do not leave
# structural fingerprints in the Philox input stream.  All outputs live in
# (0, 2**62) so they avoid Philox's degenerate seed=0 case while staying
# within int64.
_MAMBA_SSM_SEED_MASK = (1 << 62) - 1
_MAMBA_SSM_UINT64_MASK = (1 << 64) - 1
_MAMBA_SSM_SEED_BASE = 0x6A09E667F3BCC908
_MAMBA_SSM_SEED_MIX_COUNTER = 0x2545F4914F6CDD1D
_MAMBA_SSM_SEED_MIX_SLOT = 0x1B873593CC9E2D51
_MAMBA_SSM_SEED_MIX_RANK = 0x9E3779B97F4A7C15


def _splitmix64(x: int) -> int:
    """SplitMix64 finalizer; pure function, no torch."""
    x = (x + 0x9E3779B97F4A7C15) & _MAMBA_SSM_UINT64_MASK
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & _MAMBA_SSM_UINT64_MASK
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & _MAMBA_SSM_UINT64_MASK
    x ^= (x >> 31)
    return x & _MAMBA_SSM_UINT64_MASK


def _compute_deterministic_mamba_seed(counter: int, slot: int,
                                      rank_offset: int) -> int:
    """Deterministic int64 seed in (0, 2**62) from (counter, slot, rank).

    Pure function (no RNG, no torch.randint).  Identical inputs across
    process invocations produce identical outputs, which is what the
    acceptance criteria require for cross-run reproducibility.
    """
    folded = (_MAMBA_SSM_SEED_BASE + counter * _MAMBA_SSM_SEED_MIX_COUNTER +
              slot * _MAMBA_SSM_SEED_MIX_SLOT +
              rank_offset * _MAMBA_SSM_SEED_MIX_RANK)
    folded &= _MAMBA_SSM_UINT64_MASK
    value = _splitmix64(folded) & _MAMBA_SSM_SEED_MASK
    if value == 0:
        value = 1
    return value


def _allocate_mamba_seed_buffer(cache_size: int, rank_offset: int,
                                device: torch.device) -> torch.Tensor:
    """Allocate a (cache_size,) int64 CUDA buffer of deterministic seeds.

    counter=0 at allocation time; per-slot reset on fresh request assignment
    bumps a host-side counter and rewrites only the freshly-assigned slot.
    """
    slot_seeds = [
        _compute_deterministic_mamba_seed(0, i, rank_offset)
        for i in range(cache_size)
    ]
    return torch.tensor(slot_seeds, dtype=torch.int64, device=device)


def _mamba_rank_offset(mapping: Mapping) -> int:
    """Distinct seed offset per (tp_rank, pp_rank) so independent ranks
    don't draw identical streams when not coordinated."""
    return (mapping.tp_rank * 1_000_003 + mapping.pp_rank * 1_000_033 +
            mapping.rank * 1_009)


def use_py_mamba_cache_manager() -> bool:
    """Check if PythonMambaCacheManager should be forced (agg mode override).

    Returns True if TRTLLM_USE_PY_MAMBA='1' is set, False otherwise.

    Agg-mode-only override: forces the V1-route MixedMambaHybridCacheManager
    with PythonMambaCacheManager inside instead of the default unified-pool
    CppMambaHybridCacheManager. Disagg mode is unaffected — it already picks
    PythonMambaCacheManager when transceiver_runtime='PYTHON'.
    """
    return os.environ.get('TRTLLM_USE_PY_MAMBA', '0') == '1'


class ReplayStateUpdateMetadata(NamedTuple):
    """Shared tensors and fixed sizes for replay state updates."""
    prev_num_accepted_tokens: torch.Tensor
    cache_buf_idx: torch.Tensor
    replay_step_width: int
    replay_history_size: int


class BaseMambaCacheManager(ABC):
    """Abstract interface for accessing mamba/recurrent state caches."""

    @abstractmethod
    def get_state_indices(self, *args, **kwargs) -> torch.Tensor:
        """Return slot indices of each request.

        Shape: [max_batch_size]
        """
        ...

    def get_replay_state_update_metadata(
            self) -> Optional[ReplayStateUpdateMetadata]:
        """Return replay metadata tensors and fixed replay sizes."""
        return None

    @abstractmethod
    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        """Return conv states for specific layer.

        Shape: [slot_size, conv_dim, d_conv - 1]
        """
        ...

    @abstractmethod
    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        """Return SSM states for specific layer.

        Shape: [slot_size, num_heads, head_dim, d_state]
        """
        ...

    @abstractmethod
    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        ...

    @abstractmethod
    def is_speculative(self) -> bool:
        ...

    @abstractmethod
    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union['PythonMambaCacheManager.State',
               'PythonMambaCacheManager.SpeculativeState', None]:
        ...


class PythonMambaCacheManager(BaseResourceManager):
    """Pure-Python mamba state manager with speculative decoding support.

    Manages only mamba states (conv + SSM) using PyTorch tensors on GPU.
    Supports speculative decoding and disaggregated serving.
    """

    @dataclass(frozen=True, kw_only=True)
    class State:
        """Base state container for Mamba cache."""
        conv: torch.Tensor
        # Represents last state, or "two back" if prev_num_accepted_tokens is > 0
        temporal: torch.Tensor

        # Subclasses override to list fields shared across layers (not indexed by layer).
        _SHARED_FIELDS = frozenset()

        def at_layer_idx(self, layer: int):
            kwargs = {}
            for k, v in vars(self).items():
                if v is None:
                    kwargs[k] = None
                elif k in self._SHARED_FIELDS:
                    kwargs[k] = v
                else:
                    kwargs[k] = v[layer]
            return type(self)(**kwargs)

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        """Speculative state with intermediate states for draft tokens.

        Supports two SSM update paths (only one set of tensors is allocated):
        - Legacy: caches full intermediate SSM states (intermediate_ssm)
        - Replay: compact double-buffered cache (old_x, old_B, old_dt, old_dA_cumsum)
        """
        _SHARED_FIELDS = frozenset({
            "prev_num_accepted_tokens", "cache_buf_idx", "mamba_ssm_rand_seed"
        })

        intermediate_conv_window: torch.Tensor  # always allocated

        # Legacy path: full intermediate SSM states at each step
        intermediate_ssm: torch.Tensor | None = None

        # Replay path: compact double-buffered cache
        # prev_num_accepted_tokens: # accepted tokens (always >= 1 if drafting).
        # 0 means temporal saved state is actually the last state, not two back.
        prev_num_accepted_tokens: torch.Tensor | None = None  # (cache,) int — shared across layers
        cache_buf_idx: torch.Tensor | None = None  # (cache,) int32 — shared across layers
        # Per-cache-slot Philox seeds. Replay bumps them in-place per launch so
        # CUDA graph replay uses fresh SR draws without allocating RNG tensors.
        # (cache,) int64 - shared across layers
        mamba_ssm_rand_seed: torch.Tensor | None = None
        old_x: torch.Tensor | None = None  # (layers, cache, 2, history, nheads, dim)
        old_B: torch.Tensor | None = None  # (layers, cache, 2, history, ngroups, dstate)
        # Processed dt: softplus(raw_dt + dt_bias), clamped to dt_limit.
        old_dt: torch.Tensor | None = None  # (layers, cache, 2, nheads, history) fp32
        old_dA_cumsum: torch.Tensor | None = None  # (layers, cache, 2, nheads, history) fp32

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        spec_state_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        model_type: str = "nemotron_hybrid",
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.spec_state_size = spec_state_size
        self._use_replay_state_update = use_replay_state_update
        self.replay_history_size: Optional[int] = None
        self.replay_step_width: Optional[int] = None
        # When True, allocate the per-slot Philox seed buffer even outside
        # the replay path so the non-replay flashinfer SR kernel reads a
        # persistent deterministic seed instead of a per-call torch.randint.
        self._mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
        self._seed_rank_offset = _mamba_rank_offset(mapping)
        # Host-side counter bumped per fresh cache-slot assignment.  Combined
        # with slot index and rank offset to produce reproducible per-slot
        # seed values.  Starts at 0 so the post-init "reset" stream is
        # disjoint from the counter=0 stream used at allocation time.
        self._seed_request_counter = 0

        # get tp size
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size

        # derive mamba parameters for conv and ssm states
        d_inner = head_dim * num_heads
        conv_dim = d_inner + 2 * n_groups * d_state
        nheads = num_heads

        # check that can be partitioned
        assert nheads % tp_size == 0, "nheads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"

        # partition conv_dim and nheads
        d_inner_local = d_inner // tp_size
        ng_ds_local = n_groups * d_state // tp_size
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        d_inner = d_inner // tp_size

        # Per-section dims for conv_state.
        # Qwen3-Next: [Q | K | V] = [ng*ds, ng*ds, d_inner]
        # Nemotron_hybrid: [x | B | C] = [d_inner, ng*ds, ng*ds]
        if model_type == "qwen3_next":
            self.conv_section_dims = [ng_ds_local, ng_ds_local, d_inner_local]
        elif model_type == "nemotron_hybrid":
            self.conv_section_dims = [d_inner_local, ng_ds_local, ng_ds_local]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # conv and ssm states device
        device = torch.device("cuda")

        pp_layers, num_layers = get_pp_layers(
            num_layers,
            mapping,
            layer_mask=layer_mask,
        )
        num_local_layers = len(pp_layers)
        self.mamba_layer_offsets = {
            idx: offset
            for offset, idx in enumerate(pp_layers)
        }

        conv_state_shape = (conv_dim, d_conv - 1)
        ssm_state_shape = (nheads, head_dim, d_state)

        # create mamba conv and ssm states
        conv_states = torch.zeros(
            size=(num_local_layers, max_batch_size) + conv_state_shape,
            dtype=dtype,
            device=device,
        )

        ssm_states = torch.zeros(
            size=(num_local_layers, max_batch_size) + ssm_state_shape,
            dtype=self.mamba_ssm_cache_dtype,
            device=device,
        )

        # Per-slot Philox seeds.  Allocated whenever stochastic rounding can
        # fire, even outside the replay path and even when MTP/spec is off,
        # so all consumers (replay kernel, MTP non-replay flashinfer,
        # non-MTP flashinfer) read a persistent deterministic seed from the
        # cache manager instead of calling torch.randint per forward.
        self._mamba_ssm_rand_seed: Optional[torch.Tensor] = None
        if (self._use_replay_state_update
                or self._mamba_ssm_stochastic_rounding):
            self._mamba_ssm_rand_seed = _allocate_mamba_seed_buffer(
                max_batch_size, self._seed_rank_offset, device)

        # create state container
        if speculative_num_draft_tokens is not None:
            T = speculative_num_draft_tokens + 1
            self.replay_step_width = T

            # Conv intermediate cache — same for both paths
            intermediate_conv_window_cache = torch.zeros(
                size=(num_local_layers, self.spec_state_size, T) +
                conv_state_shape,
                dtype=dtype,
                device=device,
            )

            # SSM speculative cache — path-specific tensors
            spec_kwargs = {}
            # Share the manager-level seed buffer through SpeculativeState
            # so the MTP path can still read it via layer_cache.
            if self._mamba_ssm_rand_seed is not None:
                spec_kwargs['mamba_ssm_rand_seed'] = self._mamba_ssm_rand_seed
            if self._use_replay_state_update:
                assert n_groups % tp_size == 0, \
                    "replay state update requires n_groups divisible by tp_size"
                n_groups_per_rank = n_groups // tp_size
                self.replay_history_size = max(MIN_REPLAY_HISTORY_SIZE, T)

                # Compact replay cache.
                spec_kwargs['prev_num_accepted_tokens'] = torch.zeros(
                    max_batch_size, dtype=torch.int32, device=device)
                spec_kwargs['cache_buf_idx'] = torch.zeros(max_batch_size,
                                                           dtype=torch.int32,
                                                           device=device)
                spec_kwargs['old_x'] = torch.zeros(num_local_layers,
                                                   max_batch_size,
                                                   2,
                                                   self.replay_history_size,
                                                   nheads,
                                                   head_dim,
                                                   dtype=dtype,
                                                   device=device)
                spec_kwargs['old_B'] = torch.zeros(num_local_layers,
                                                   max_batch_size,
                                                   2,
                                                   self.replay_history_size,
                                                   n_groups_per_rank,
                                                   d_state,
                                                   dtype=dtype,
                                                   device=device)
                spec_kwargs['old_dt'] = torch.zeros(num_local_layers,
                                                    max_batch_size,
                                                    2,
                                                    nheads,
                                                    self.replay_history_size,
                                                    dtype=torch.float32,
                                                    device=device)
                spec_kwargs['old_dA_cumsum'] = torch.zeros(
                    num_local_layers,
                    max_batch_size,
                    2,
                    nheads,
                    self.replay_history_size,
                    dtype=torch.float32,
                    device=device)
                ssm_spec_cache = [
                    spec_kwargs['old_x'], spec_kwargs['old_B'],
                    spec_kwargs['old_dt'], spec_kwargs['old_dA_cumsum']
                ]
                spec_path_label = "replay"
            else:
                # Legacy: full intermediate SSM states at each step
                spec_kwargs['intermediate_ssm'] = torch.zeros(
                    size=(num_local_layers, self.spec_state_size, T) +
                    ssm_state_shape,
                    dtype=self.mamba_ssm_cache_dtype,
                    device=device)
                ssm_spec_cache = [spec_kwargs['intermediate_ssm']]
                spec_path_label = "legacy"

            self.mamba_cache = self.SpeculativeState(
                conv=conv_states,
                temporal=ssm_states,
                intermediate_conv_window=intermediate_conv_window_cache,
                **spec_kwargs,
            )

            logger.info(
                f"Mamba Cache ({spec_path_label}) is allocated. "
                f"max_mamba_cache_size: {max_batch_size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_states) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(ssm_states) / GB:.2f}GB, "
                f"ssm_spec_cache size: {get_tensor_size_bytes(ssm_spec_cache) / GB:.2f}GB, "
                "intermediate_conv_window_cache size: "
                f"{get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB"
            )
        else:
            self.mamba_cache = self.State(
                conv=conv_states,
                temporal=ssm_states,
            )

            logger.info(
                f"Mamba Cache is allocated. "
                f"max_mamba_cache_size: {max_batch_size}, "
                f"conv_state size: {get_tensor_size_bytes(conv_states) / GB:.2f}GB, "
                f"ssm_state size: {get_tensor_size_bytes(ssm_states) / GB:.2f}GB"
            )

        # mamba cache available blocks
        self.mamba_cache_free_blocks = [i for i in range(max_batch_size)]

        # mamba cache index, maps request_id -> state indices
        self.mamba_cache_index: Dict[int, int] = {}
        self._dummy_request_ids: set[int] = set()
        # Batch-order mask aligned with state_indices; duplicate dummy request
        # IDs mark every batch row even when they share one cache slot.
        self._dummy_request_mask = torch.zeros(max_batch_size,
                                               dtype=torch.bool,
                                               device=device)
        self._dummy_request_mask_host = torch.zeros(max_batch_size,
                                                    dtype=torch.bool,
                                                    pin_memory=prefer_pinned())

        # Permanent slot shared by every CUDA-graph padding sentinel id
        # (CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, one per
        # draft length). Pool sizing must include +1 headroom for this;
        # see MixedMambaHybridCacheManager.
        self._padding_slot: int = self.mamba_cache_free_blocks.pop()

        # Reserved slot for the attention-DP padding dummy. Keep it out of the
        # free pool so dummy insertion never consumes real-request capacity.
        self._attention_dp_dummy_slot: Optional[int] = (
            self.mamba_cache_free_blocks.pop()
            if mapping.enable_attention_dp else None)

        # save intermediate state indices for requests
        self.intermediate_state_indices = torch.arange(max_batch_size,
                                                       dtype=torch.int32,
                                                       device=device)

        # Physical tensor rows include reserved dummy slots. Resource capacity
        # is the number of real request slots remaining after those
        # reservations.
        self._max_batch_size = max_batch_size
        self._max_resource_count = len(self.mamba_cache_free_blocks)

    def get_max_resource_count(self) -> int:
        """Return the maximum number of real requests that can be cached."""
        return self._max_resource_count

    def filter_ctx_requests_by_capacity(self, context_requests: list) -> list:
        """Return the prefix of *context_requests* that fits in the
        available Mamba cache blocks.  Requests that already have a
        cached block do not consume a free block."""
        free = len(self.mamba_cache_free_blocks)
        result = []
        for r in context_requests:
            if r.py_request_id in self.mamba_cache_index:
                result.append(r)
            elif free > 0:
                result.append(r)
                free -= 1
            else:
                break
        return result

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        """For Mamba cache manager, we always need one slot per request."""
        return 1

    @torch.inference_mode()
    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        for r in request_ids:
            if r in self.mamba_cache_index:
                continue
            if len(self.mamba_cache_free_blocks) == 0:
                raise RuntimeError("run out of mamba cache blocks")
            block = self.mamba_cache_free_blocks.pop()
            self.mamba_cache_index[r] = block
            if (isinstance(self.mamba_cache, self.SpeculativeState)
                    and self._use_replay_state_update):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
                self.mamba_cache.cache_buf_idx[block] = 0
            if self._mamba_ssm_rand_seed is not None:
                # Deterministic per-slot rotation on fresh assignment.
                # `block` is pulled from mamba_cache_free_blocks, which
                # excludes _padding_slot by construction (see __init__),
                # so padding sentinels never reach this branch.
                self._seed_request_counter += 1
                self._mamba_ssm_rand_seed[block] = (
                    _compute_deterministic_mamba_seed(
                        self._seed_request_counter, block,
                        self._seed_rank_offset))

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        requests = (scheduled_batch.context_requests +
                    scheduled_batch.generation_requests)
        context_ids = [
            i.py_request_id for i in scheduled_batch.context_requests
        ]
        generation_ids = [
            i.py_request_id for i in scheduled_batch.generation_requests
        ]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)
        self._refresh_dummy_request_mask([req.is_dummy for req in requests])

    def _is_padding_sentinel(self, request_id: int) -> bool:
        # cuda_graph_runner caches one dummy per runtime_draft_len value
        # (see _get_padded_batch), so any id in the range of dummy request IDs
        # may be live concurrently.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import \
            CUDA_GRAPH_DUMMY_REQUEST_ID
        max_dl = self.speculative_num_draft_tokens or 0
        return (CUDA_GRAPH_DUMMY_REQUEST_ID - max_dl <= request_id <=
                CUDA_GRAPH_DUMMY_REQUEST_ID)

    @torch.inference_mode()
    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        # Sentinels alias to the shared _padding_slot; non-sentinel
        # dummies (warmup, attention-DP idle padding) get their own
        # slot and are freed individually.
        if not request_ids:
            return
        self._dummy_request_ids.update(request_ids)
        for r in request_ids:
            if r in self.mamba_cache_index:
                block = self.mamba_cache_index[r]
                if (isinstance(self.mamba_cache, self.SpeculativeState)
                        and self._use_replay_state_update):
                    self.mamba_cache.prev_num_accepted_tokens[block] = 0
                    self.mamba_cache.cache_buf_idx[block] = 0
                continue
            if self._is_padding_sentinel(r):
                block = self._padding_slot
            elif (r == ATTENTION_DP_DUMMY_REQUEST_ID
                  and self._attention_dp_dummy_slot is not None):
                block = self._attention_dp_dummy_slot
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise RuntimeError("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
            self.mamba_cache_index[r] = block
            if (isinstance(self.mamba_cache, self.SpeculativeState)
                    and self._use_replay_state_update):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
                self.mamba_cache.cache_buf_idx[block] = 0

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id not in self.mamba_cache_index:
            return
        self._dummy_request_ids.discard(request_id)
        block = self.mamba_cache_index.pop(request_id)
        # Reserved slots must not re-enter the real-request free pool.
        if block != self._padding_slot and \
                block != self._attention_dp_dummy_slot:
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self, request_ids: List[int],
                          is_padding: List[bool]) -> List[int]:
        assert len(request_ids) == len(is_padding)
        indices = [self.mamba_cache_index[rid] for rid in request_ids]
        is_dummy = [
            rid in self._dummy_request_ids or padding
            for rid, padding in zip(request_ids, is_padding)
        ]
        self._refresh_dummy_request_mask(is_dummy)
        return indices

    @torch.inference_mode()
    def _refresh_dummy_request_mask(self, is_dummy: List[bool]) -> None:
        n = len(is_dummy)
        assert n <= self._dummy_request_mask_host.shape[0]
        self._dummy_request_mask_host.zero_()
        if n > 0:
            self._dummy_request_mask_host[:n].copy_(
                torch.as_tensor(is_dummy, dtype=torch.bool))
        self._dummy_request_mask.copy_(self._dummy_request_mask_host,
                                       non_blocking=True)

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).conv

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).temporal

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).intermediate_ssm

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        """Get intermediate conv states for speculative decoding."""
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(
            layer_offset).intermediate_conv_window

    def get_replay_old_x(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.old_x[layer_offset]

    def get_replay_old_B(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.old_B[layer_offset]

    def get_replay_old_dt(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.old_dt[layer_offset]

    def get_replay_old_dA_cumsum(self,
                                 layer_idx: int) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.old_dA_cumsum[layer_offset]

    def get_replay_cache_buf_idx(self) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        return self.mamba_cache.cache_buf_idx

    def get_replay_prev_num_accepted_tokens(self) -> Optional[torch.Tensor]:
        if not self._use_replay_state_update:
            return None
        return self.mamba_cache.prev_num_accepted_tokens

    def is_speculative(self) -> bool:
        return isinstance(self.mamba_cache, self.SpeculativeState)

    def mamba_layer_cache(self,
                          layer_idx: int) -> Union[State, SpeculativeState]:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.mamba_ssm_cache_dtype

    def get_mamba_ssm_rand_seed(self) -> Optional[torch.Tensor]:
        """Return the persistent (cache_size,) int64 Philox seed buffer or
        None when stochastic rounding is not active for this manager.

        Used by mamba2_mixer non-MTP paths that don't hold a SpeculativeState.
        Callers must bump in-place (`.add_(1)` or slice-and-add) and pass a
        view that matches the consuming kernel's expected shape.
        """
        return self._mamba_ssm_rand_seed

    @property
    def use_replay_state_update(self) -> bool:
        return self.get_replay_state_update_metadata() is not None

    def get_replay_state_update_metadata(
            self) -> Optional[ReplayStateUpdateMetadata]:
        if (not self._use_replay_state_update
                or not isinstance(self.mamba_cache, self.SpeculativeState)
                or self.mamba_cache.prev_num_accepted_tokens is None
                or self.mamba_cache.cache_buf_idx is None
                or self.replay_step_width is None
                or self.replay_history_size is None):
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=(
                self.mamba_cache.prev_num_accepted_tokens),
            cache_buf_idx=self.mamba_cache.cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size)

    def shutdown(self):
        """Release tensor memory."""
        # Clear mamba cache states
        empty = torch.tensor([])

        def _drop(tensor):
            return empty if tensor is not None else None

        if isinstance(self.mamba_cache, self.SpeculativeState):
            self.mamba_cache = self.SpeculativeState(
                conv=empty,
                temporal=empty,
                intermediate_conv_window=empty,
                intermediate_ssm=_drop(self.mamba_cache.intermediate_ssm),
                prev_num_accepted_tokens=_drop(
                    self.mamba_cache.prev_num_accepted_tokens),
                cache_buf_idx=_drop(self.mamba_cache.cache_buf_idx),
                mamba_ssm_rand_seed=_drop(self.mamba_cache.mamba_ssm_rand_seed),
                old_x=_drop(self.mamba_cache.old_x),
                old_B=_drop(self.mamba_cache.old_B),
                old_dt=_drop(self.mamba_cache.old_dt),
                old_dA_cumsum=_drop(self.mamba_cache.old_dA_cumsum),
            )
        else:
            self.mamba_cache = self.State(conv=empty, temporal=empty)

        torch.cuda.empty_cache()

    @torch.compile(options={"max-autotune": True})
    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor,
                            state_indices: torch.Tensor):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        state_indices_d = state_indices[num_contexts:num_contexts + num_gens]
        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_replay_state_update:
            # SSM state is handled incrementally by the kernel.  Mirror the
            # kernel's per-slot checkpoint predicate from the previous PNAT and
            # fixed replay step width: checkpoint steps write a fresh history
            # buffer and flip, while no-checkpoint steps append to the active
            # buffer and keep reading from it next step.
            accepted_tokens = num_accepted_tokens[num_contexts:num_contexts +
                                                  num_gens]
            prev_num_accepted_tokens = \
                self.mamba_cache.prev_num_accepted_tokens[state_indices_d]
            wrote_checkpoint = (prev_num_accepted_tokens +
                                self.replay_step_width
                                > self.replay_history_size)
            next_num_accepted_tokens = torch.where(
                wrote_checkpoint, accepted_tokens,
                prev_num_accepted_tokens + accepted_tokens)
            cache_buf_idx = self.mamba_cache.cache_buf_idx[state_indices_d]
            is_dummy_request = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
            next_num_accepted_tokens = torch.where(is_dummy_request,
                                                   prev_num_accepted_tokens,
                                                   next_num_accepted_tokens)
            self.mamba_cache.prev_num_accepted_tokens[state_indices_d] = \
                next_num_accepted_tokens
            self.mamba_cache.cache_buf_idx[state_indices_d] = \
                torch.where(is_dummy_request, cache_buf_idx,
                            torch.where(wrote_checkpoint, 1 - cache_buf_idx,
                                        cache_buf_idx))
        else:
            # Legacy: copy accepted SSM state from intermediate cache.
            ssm_states = self.mamba_cache.temporal
            intermediate_ssm_cache = self.mamba_cache.intermediate_ssm
            accepted_ssm_state = intermediate_ssm_cache[:, src_state_indices,
                                                        num_accepted_draft_tokens]
            ssm_states[:, state_indices_d, :] = accepted_ssm_state

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        conv_states = self.mamba_cache.conv
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window
        accepted_conv_state = intermediate_conv_window_cache[:,
                                                             src_state_indices,
                                                             num_accepted_draft_tokens]
        conv_states[:, state_indices_d, :] = accepted_conv_state


class MambaCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Facade for standalone mamba state management (no KV cache).

    Delegates to PythonMambaCacheManager.
    """

    def __init__(
        self,
        d_state: int,
        d_conv: int,
        num_heads: int,
        n_groups: int,
        head_dim: int,
        num_layers: int,
        max_batch_size: int,
        spec_state_size: int,
        mapping: Mapping,
        dtype: torch.dtype,
        ssm_cache_dtype: torch.dtype,
        layer_mask: Optional[List[bool]] = None,
        stream: Optional[torch.cuda.Stream] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        model_type: str = "nemotron_hybrid",
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
    ) -> None:
        max_num_sequences = max_batch_size * mapping.pp_size

        self._impl = PythonMambaCacheManager(
            d_state=d_state,
            d_conv=d_conv,
            num_heads=num_heads,
            n_groups=n_groups,
            head_dim=head_dim,
            num_layers=num_layers,
            max_batch_size=max_num_sequences,
            spec_state_size=spec_state_size,
            mapping=mapping,
            dtype=dtype,
            ssm_cache_dtype=ssm_cache_dtype,
            layer_mask=layer_mask,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            model_type=model_type,
            use_replay_state_update=use_replay_state_update,
            mamba_ssm_stochastic_rounding=mamba_ssm_stochastic_rounding,
        )

    def get_max_resource_count(self) -> int:
        return self._impl.get_max_resource_count()

    def filter_ctx_requests_by_capacity(self, context_requests: list) -> list:
        return self._impl.filter_ctx_requests_by_capacity(context_requests)

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return self._impl.get_needed_resource_to_completion(request)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        self._impl.prepare_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        self._impl.free_resources(request)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        self._impl.add_dummy_requests(request_ids, **kwargs)

    def get_state_indices(
        self,
        request_ids: Optional[List[int]] = None,
        is_padding: Optional[List[bool]] = None
    ) -> Union[torch.Tensor, List[int]]:
        return self._impl.get_state_indices(request_ids, is_padding)

    @property
    def mamba_cache_free_blocks(self) -> List[int]:
        return self._impl.mamba_cache_free_blocks

    @property
    def mamba_cache_index(self) -> Dict[int, int]:
        return self._impl.mamba_cache_index

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self._impl.get_conv_states(layer_idx)

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self._impl.get_ssm_states(layer_idx)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self._impl.get_mamba_ssm_cache_dtype()

    def get_mamba_ssm_rand_seed(self) -> Optional[torch.Tensor]:
        """Delegate to the underlying Python manager."""
        return self._impl.get_mamba_ssm_rand_seed()

    @property
    def use_replay_state_update(self) -> bool:
        return self.get_replay_state_update_metadata() is not None

    def get_replay_state_update_metadata(
            self) -> Optional[ReplayStateUpdateMetadata]:
        return self._impl.get_replay_state_update_metadata()

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_intermediate_ssm_states(layer_idx)

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_intermediate_conv_states(layer_idx)

    def get_replay_old_x(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_x(layer_idx)

    def get_replay_old_B(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_B(layer_idx)

    def get_replay_old_dt(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_dt(layer_idx)

    def get_replay_old_dA_cumsum(self,
                                 layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_dA_cumsum(layer_idx)

    def get_replay_cache_buf_idx(self) -> Optional[torch.Tensor]:
        return self._impl.get_replay_cache_buf_idx()

    def get_replay_prev_num_accepted_tokens(self) -> Optional[torch.Tensor]:
        return self._impl.get_replay_prev_num_accepted_tokens()

    def is_speculative(self) -> bool:
        return self._impl.is_speculative()

    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union[PythonMambaCacheManager.State,
               PythonMambaCacheManager.SpeculativeState, None]:
        return self._impl.mamba_layer_cache(layer_idx)

    def shutdown(self):
        self._impl.shutdown()

    def update_mamba_states(self, attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor,
                            state_indices: torch.Tensor):
        # Non-speculative configs don't allocate intermediate state; the
        # promotion is a clean no-op.
        if not self._impl.is_speculative():
            return
        self._impl.update_mamba_states(attn_metadata, num_accepted_tokens,
                                       state_indices)


class MambaHybridCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Marker base class for hybrid mamba cache manager implementations.

    Used purely for ``isinstance`` / type-hint purposes so callers can refer
    to the family without caring about the concrete implementation. Concrete
    selection (Mixed vs Cpp) lives in ``_util.py:_get_model_kv_cache_manager_cls``.
    """


def _get_mamba_hybrid_pool_size(max_batch_size: int, mapping: Mapping) -> int:
    """Return the internal Mamba state pool size for MixedMambaHybridCacheManager."""
    pool_size = max_batch_size
    # One permanent slot is shared by every CUDA-graph padding sentinel.
    pool_size += 1
    if mapping.enable_attention_dp:
        # Attention-DP can insert a transient dummy request on an otherwise
        # idle rank. Keep this headroom internal so scheduler-visible
        # max_batch_size still limits real requests.
        pool_size += 1
    return pool_size


class MixedMambaHybridCacheManager(KVCacheManager, MambaCacheManager,
                                   MambaHybridCacheManager):
    """Hybrid cache manager combining separate KVCacheManager and MambaCacheManager.

    Manages KV cache and mamba states in independent pools, with support of
    speculative decoding and disaggregated serving.
    Does not support block reuse / prefix caching for mamba states.
    """

    def __init__(
        self,
        # mamba cache parameters
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_num_heads: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        mamba_ssm_cache_dtype: torch.dtype,
        # kv cache parameters
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        layer_mask: List[bool],
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        is_estimating_kv_cache: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        model_type: str = "nemotron_hybrid",
        is_draft: bool = False,
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        # Per-pool configurations forwarded to the C++ KVCacheManager ctor.
        # Lets a single manager host pools with mixed shapes (e.g. Gemma4
        # hybrid attention). See KVCacheManager.__init__.
        pool_configurations: Optional[List[PoolConfiguration]] = None,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, (
            "mamba hybrid cache requires block reuse to be disabled in KV cache config"
        )

        pool_size = _get_mamba_hybrid_pool_size(max_batch_size, mapping)

        MambaCacheManager.__init__(
            self,
            mamba_d_state,
            mamba_d_conv,
            mamba_num_heads,
            mamba_n_groups,
            mamba_head_dim,
            mamba_num_layers,
            pool_size,
            max_batch_size,
            mapping,
            mamba_cache_dtype,
            mamba_ssm_cache_dtype,
            mamba_layer_mask,
            execution_stream,
            speculative_num_draft_tokens=(spec_config.tokens_per_gen_step - 1
                                          if spec_config is not None else None),
            model_type=model_type,
            use_replay_state_update=use_replay_state_update,
            mamba_ssm_stochastic_rounding=mamba_ssm_stochastic_rounding,
        )

        # initialize kv cache manager
        KVCacheManager.__init__(
            self,
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            is_estimating_kv_cache=is_estimating_kv_cache,
            execution_stream=execution_stream,
            is_draft=is_draft,
            pool_configurations=pool_configurations,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        MambaCacheManager.prepare_resources(self, scheduled_batch)
        KVCacheManager.prepare_resources(self, scheduled_batch)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        MambaCacheManager.free_resources(self, request)
        KVCacheManager.free_resources(self, request, pin_on_release)

    def add_dummy_requests(self, request_ids: List[int], **kwargs):
        MambaCacheManager.add_dummy_requests(self, request_ids)
        return KVCacheManager.add_dummy_requests(self, request_ids, **kwargs)

    def shutdown(self):
        MambaCacheManager.shutdown(self)
        KVCacheManager.shutdown(self)

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        KVCacheManager.update_resources(self, scheduled_batch, attn_metadata,
                                        kv_cache_dtype_byte_size)


def calc_context_stop_positions(prompt_len: int,
                                tokens_per_block: int,
                                mamba_state_cache_interval: int,
                                save_last_snapshot: bool = False) -> list[int]:
    """Compute token positions at which mamba state snapshots should be saved.

    Returns positions spaced by ``mamba_state_cache_interval`` plus the final
    prompt length (and optionally the last block-aligned position).
    """
    stop_positions = list(
        range(mamba_state_cache_interval, prompt_len,
              mamba_state_cache_interval))
    last_ckpt = prompt_len // tokens_per_block * tokens_per_block
    if save_last_snapshot and (last_ckpt not in stop_positions):
        stop_positions.append(last_ckpt)
    if prompt_len not in stop_positions:
        stop_positions.append(prompt_len)
    return stop_positions


@triton.jit
def _promote_mamba_state_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    acc_ptr,
    blk_ptr,
    num_gens,
    count,
    src_s_layer,
    src_s_row,
    src_s_step,
    dst_s_layer,
    dst_s_block,
    BLOCK: tl.constexpr,
):
    # One program copies a BLOCK-sized tile of the contiguous inner state for a
    # single (layer, gen) pair. grid = (num_layers * num_gens, ceil(count/BLOCK)).
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    row = tl.load(src_idx_ptr + g).to(tl.int64)
    acc = tl.load(acc_ptr + g).to(tl.int64)
    blk = tl.load(blk_ptr + g).to(tl.int64)
    # int64 throughout: per-layer strides are O(1e8), so layer*stride overflows
    # int32 and would corrupt addresses / fault.
    src_base = layer * src_s_layer + row * src_s_row + acc * src_s_step
    dst_base = layer * dst_s_layer + blk * dst_s_block
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    v = tl.load(src_ptr + src_base + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, v, mask=mask)


def _promote_mamba_state_triton(dst: torch.Tensor,
                                intermediate: torch.Tensor,
                                src_state_indices: torch.Tensor,
                                accepted_draft: torch.Tensor,
                                dst_state_indices: torch.Tensor,
                                BLOCK: int = 2048) -> None:
    """Scatter each generation request's accepted draft-step recurrent state
    from the per-request intermediate buffer into the unified C++ pool view.

    Args:
        dst: ``[num_layers, num_blocks, *state_shape]`` view of the C++ pool.
            The block dim (dim 1) is strided (the pool interleaves
            ``ssm_bytes | conv_bytes`` per block), so ``dst`` is non-contiguous.
            The kernel writes through ``dst``'s real strides, touching only this
            state's bytes -- the case that defeats torch.compile / aot_autograd
            (dtype-view mutation) and Inductor codegen (``XBLOCK`` on the uint8
            pool).
        intermediate: ``[num_layers, max_batch_size, T, *state_shape]`` dense.
        src_state_indices, accepted_draft, dst_state_indices: ``[num_gens]`` int.

    For each gen ``g``::

        dst[:, dst_state_indices[g]] = intermediate[:, src_state_indices[g], accepted_draft[g]]

    Pure gather->scatter copy; bandwidth-bound (~85% of HBM peak), one launch.
    """
    num_layers = dst.shape[0]
    num_gens = src_state_indices.shape[0]
    if num_gens == 0:
        return
    count = 1
    for s in dst.shape[2:]:
        count *= s
    # Grid dim order matters: dim 0 -> CUDA grid.x (limit 2^31-1), dim 1 ->
    # grid.y (limit 65535). Put the (layer, gen) pairs in dim 0 since that is
    # what grows with batch size (num_layers*num_gens stays far below 2^31 for
    # any real config); the tile count in dim 1 is small (~count/BLOCK). Do NOT
    # swap them: pairs would hit the 65535 y-limit at num_layers*num_gens>65535.
    grid = (num_layers * num_gens, triton.cdiv(count, BLOCK))
    _promote_mamba_state_kernel[grid](
        intermediate,
        dst,
        src_state_indices,
        accepted_draft,
        dst_state_indices,
        num_gens,
        count,
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        dst.stride(0),
        dst.stride(1),
        BLOCK=BLOCK,
    )


class CppMambaHybridCacheManager(KVCacheManager, MambaHybridCacheManager):
    """Hybrid cache manager storing mamba states inside the KVCacheManager pool.

    Both KV cache blocks and recurrent state blocks are managed by the unified
    C++ KVCacheManager, enabling block reuse / prefix caching across attention
    and mamba layers. This is the default hybrid manager.

    """

    def __init__(
        self,
        # mamba cache parameters
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_num_heads: int,
        mamba_n_groups: int,
        mamba_head_dim: int,
        mamba_num_layers: int,
        mamba_layer_mask: List[bool],
        mamba_cache_dtype: torch.dtype,
        mamba_ssm_cache_dtype: torch.dtype,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[
            List[bool]] = None,  # this is the full attention layer mask
        is_estimating_kv_cache: bool = False,
        is_draft: bool = False,
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        model_type: str = "nemotron_hybrid",
        **kwargs,
    ) -> None:
        # 3 kinds of layers:
        # 1) Mamba layers (mamba_layer_mask is True)
        # 2) Full attention layers (full_attention_layer_mask is True)
        # 3) Not managed layers (both masks are False)
        total_layers = len(mamba_layer_mask)
        if layer_mask is None:
            full_attention_layer_mask = [False] * total_layers
        elif len(layer_mask) != total_layers:
            raise ValueError(
                f"layer_mask length ({len(layer_mask)}) must match "
                f"mamba_layer_mask length ({total_layers})")
        else:
            full_attention_layer_mask = list(layer_mask)
        layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i]
            for i in range(total_layers)
        ]
        # PP sharding is done across all layers.
        # This is called again in the super().__init__, but we want it to run first
        # to set up mtp states before the C++ backend is initialized.
        self.pp_layers, _ = get_pp_layers(
            mamba_num_layers + num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.mamba_pp_layers = [
            layer_idx for layer_idx in self.pp_layers
            if mamba_layer_mask[layer_idx]
        ]
        self.local_num_mamba_layers = len(self.mamba_pp_layers)
        self.requests = []
        # Seed externally visible mamba fields before any early return so that
        # accessors (get_mamba_ssm_cache_dtype, use_replay_state_update) work
        # on ranks with no local mamba layers.
        self._use_replay_state_update = use_replay_state_update
        self.replay_step_width: Optional[int] = (
            spec_config.tokens_per_gen_step
            if spec_config is not None and use_replay_state_update else None)
        self.replay_history_size: Optional[int] = (
            max(MIN_REPLAY_HISTORY_SIZE, self.replay_step_width)
            if self.replay_step_width is not None else None)
        # Same allocation gate as PythonMambaCacheManager: the rand_seed
        # buffer must exist whenever SR can fire, not only on the replay path.
        self._mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
        self._seed_rank_offset = _mamba_rank_offset(mapping)
        # Host-side counter bumped per fresh context-slot assignment; combined
        # with the slot index and rank offset to produce reproducible per-slot
        # seed values without any torch.randint.
        self._seed_request_counter = 0
        self.ssm_state_dtype = mamba_ssm_cache_dtype

        if self.local_num_mamba_layers == 0:
            logger.info(
                "No local mamba layers for this rank, skipping mamba cache initialization"
            )
            super().__init__(
                kv_cache_config,
                kv_cache_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                tokens_per_block=tokens_per_block,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                mapping=mapping,
                dtype=dtype,
                spec_config=spec_config,
                layer_mask=full_attention_layer_mask,
                is_estimating_kv_cache=is_estimating_kv_cache,
                is_draft=is_draft,
            )
            return

        # Derive ssm_state_shape and conv_state_shape from mamba params (same as MambaCacheManager)
        tp_size = mapping.tp_size if not mapping.enable_attention_dp else 1
        d_inner = mamba_head_dim * mamba_num_heads
        conv_dim = d_inner + 2 * mamba_n_groups * mamba_d_state
        nheads = mamba_num_heads
        assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
        if use_replay_state_update:
            assert mamba_n_groups % tp_size == 0, \
                "replay state update requires mamba_n_groups divisible by tp_size"
        self._n_groups_per_rank = mamba_n_groups // tp_size
        conv_dim = conv_dim // tp_size
        nheads = nheads // tp_size
        self.conv_state_shape = [conv_dim, mamba_d_conv - 1]
        self.ssm_state_shape = [nheads, mamba_head_dim, mamba_d_state]
        self.conv_state_dtype = mamba_cache_dtype

        # Store GLOBAL (pre-TP-division) mamba params for disagg RnnModelConfig.
        # d_inner is computed at the top of __init__ before TP division.
        d_inner_global = d_inner  # = mamba_head_dim * mamba_num_heads (GLOBAL)
        conv_dim_global = d_inner_global + 2 * mamba_n_groups * mamba_d_state
        self._rnn_d_state = mamba_d_state
        self._rnn_d_conv = mamba_d_conv
        self._rnn_num_heads = mamba_num_heads  # GLOBAL
        self._rnn_n_groups = mamba_n_groups
        self._rnn_head_dim = mamba_head_dim
        self._rnn_hidden_size = d_inner_global  # GLOBAL = head_dim * num_heads
        self._rnn_conv_dim_size = conv_dim_global  # GLOBAL conv dim
        self._rnn_num_layers = mamba_num_layers

        # Conv section layout for section-aware split/concat in TP mismatch.
        # Section dims are derived from mHiddenSize and mNGroups*mDState in C++;
        # we just need to tell C++ which ordering to use.
        self._rnn_conv_section_layout = model_type  # "nemotron_hybrid" or "qwen3_next"
        self.ssm_count = math.prod(self.ssm_state_shape)
        self.conv_count = math.prod(self.conv_state_shape)
        self.ssm_bytes = self.ssm_count * self.ssm_state_dtype.itemsize
        self.conv_bytes = self.conv_count * self.conv_state_dtype.itemsize

        total_bytes = self.ssm_bytes + self.conv_bytes
        if total_bytes % self.ssm_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"ssm_state_dtype size ({self.ssm_state_dtype.itemsize})")
        if total_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})")
        if self.ssm_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"SSM state bytes ({self.ssm_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})")
        self.linear_attention_metadata = LinearAttentionMetadata()
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = self.ssm_bytes + self.conv_bytes
        self.linear_attention_metadata.states_snapshot_interval = (
            kv_cache_config.mamba_state_cache_interval
            if kv_cache_config.enable_block_reuse else 0)
        # RNN model params for disagg TP-mismatch split/concat.
        conv_section_map = {"nemotron_hybrid": 1, "qwen3_next": 2}
        self.linear_attention_metadata.rnn_num_heads = self._rnn_num_heads
        self.linear_attention_metadata.rnn_head_dim = self._rnn_head_dim
        self.linear_attention_metadata.rnn_d_state = self._rnn_d_state
        self.linear_attention_metadata.rnn_d_conv = self._rnn_d_conv
        self.linear_attention_metadata.rnn_n_groups = self._rnn_n_groups
        self.linear_attention_metadata.rnn_conv_section_layout = conv_section_map.get(
            self._rnn_conv_section_layout, 0)
        self.linear_attention_metadata.rnn_ssm_bytes = self.ssm_bytes
        self.linear_attention_metadata.rnn_ssm_dtype_size = self.ssm_state_dtype.itemsize
        self.linear_attention_metadata.rnn_conv_dtype_size = self.conv_state_dtype.itemsize
        kv_cache_config = kv_cache_config.model_copy(deep=True)
        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for mamba hybrid models, disabling partial reuse"
            )
            kv_cache_config.enable_partial_reuse = False

        kv_cache_config.max_attention_window = []
        for i in range(len(layer_mask)):
            if layer_mask[i]:
                kv_cache_config.max_attention_window.append(
                    LinearCacheType.RECURRENT_STATES.
                    value if mamba_layer_mask[i] else max_seq_len)

        # Normalize num_kv_heads to a per-layer list and zero out mamba
        # layer positions: those layers carry SSM/conv state instead of KV
        # heads, so the parent KV cache should not allocate KV head storage
        # for them.
        if isinstance(num_kv_heads, int):
            per_layer_kv_heads = [num_kv_heads] * total_layers
        else:
            if len(num_kv_heads) != total_layers:
                raise ValueError(
                    f"num_kv_heads list length ({len(num_kv_heads)}) does not "
                    f"match total layers ({total_layers})")
            per_layer_kv_heads = list(num_kv_heads)
        for i, is_mamba in enumerate(mamba_layer_mask):
            if is_mamba:
                per_layer_kv_heads[i] = 0

        self._setup_mtp_intermediate_states(spec_config, max_batch_size)

        # pass remaining arguments to super class
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=mamba_num_layers + num_layers,
            num_kv_heads=per_layer_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype,
            spec_config=spec_config,
            layer_mask=layer_mask,
            is_estimating_kv_cache=is_estimating_kv_cache,
            is_draft=is_draft,
            linear_attention_metadata=self.linear_attention_metadata,
            **kwargs,
        )

        assert self.local_num_mamba_layers > 0, "At least one mamba layer is required"
        self.mamba_layer_offsets = {}
        for idx, layer_id in enumerate(self.mamba_pp_layers):
            self.mamba_layer_offsets[layer_id] = idx

        self.host_block_offsets = torch.zeros([
            self.impl.num_pools, self.max_batch_size, 2, self.max_blocks_per_seq
        ],
                                              dtype=torch.int32,
                                              device="cpu")
        self.recurrent_states_pool_index = self.kv_cache_pool_mapping[
            self.layer_offsets[self.mamba_pp_layers[0]]][0]

        self.cuda_state_indices = torch.zeros([self.max_batch_size],
                                              dtype=torch.int32,
                                              device="cuda")
        self._host_state_indices = torch.zeros([self.max_batch_size],
                                               dtype=torch.int32,
                                               pin_memory=prefer_pinned())
        self._row_indices = torch.arange(self.max_batch_size,
                                         dtype=torch.long,
                                         device="cpu")
        self._request_id_to_state_index = {}
        self._request_id_to_is_dummy = {}
        # Batch-order mask aligned with state_indices; duplicate dummy request
        # IDs mark every batch row even when they share one cache slot.
        self._dummy_request_mask = None
        self._dummy_request_mask_host = None
        self.kv_cache_config = kv_cache_config
        self.is_estimating_kv_cache = is_estimating_kv_cache

        self._setup_states()
        self._setup_replay_buffers(spec_config)

    @staticmethod
    def get_cache_size_per_token(
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        """Affine memory model for the unified hybrid KV pool.

        Returns ``(slope_bytes_per_token, intercept_bytes)``:

        * ``slope`` = attention KV bytes per token (parent's formula) plus
          the amortized regular-snapshot bytes per token from mamba layers.
        * ``intercept`` = ``max_batch_size * num_mamba_layers_per_rank *
          state_bytes_per_layer * STATIC_SLOTS_PER_REQUEST``.

        Memory budget -> max tokens then becomes
        ``T = (budget - intercept) // slope`` instead of plain
        ``T = budget // bytes_per_token``.
        """
        # Lazy import to avoid pulling config_utils into module import order.
        from tensorrt_llm._torch.pyexecutor.config_utils import \
            extract_mamba_kv_cache_params

        # Attention slope from the parent's existing formula.
        attention_slope = KVCacheManager.get_cache_size_per_token(
            model_config, mapping, num_layers=num_layers, **kwargs)

        params = extract_mamba_kv_cache_params(
            model_config.pretrained_config,
            quant_config=model_config.quant_config,
        )

        state_bytes_per_layer = params.get_states_bytes_per_layer(mapping)

        # This not precise since pp layers are sharded by their order in model, not by their types.
        # e.g. the upper half are all mamba layers while the lower half are all attention layers.
        # But that's close enough for real world models where mamba and attention layers are interleaved
        # and we don't have access with layer_masks at this point.
        num_mamba_layers_per_rank = len(
            mapping.pp_layers(params.num_mamba_layers))
        state_bytes_per_rank = num_mamba_layers_per_rank * state_bytes_per_layer

        # Per-request fixed cost. STATIC_SLOTS_PER_REQUEST = 1 today (the
        # live mamba state); fixed-position snapshots are not yet
        # implemented and would simply increment this constant. With
        # pipeline parallelism, multiple microbatches are in-flight
        # concurrently on the same rank, so each rank holds Mamba state
        # for up to ``max_batch_size * pp_size`` concurrent sequences.
        STATIC_SLOTS_PER_REQUEST = 1
        pp_size = mapping.pp_size if mapping is not None else 1
        intercept = (max_batch_size * pp_size * state_bytes_per_rank *
                     STATIC_SLOTS_PER_REQUEST)

        # Regular-snapshot bytes per token. None / non-positive intervals
        # mean "no regular snapshots", so the mamba contribution is zero.
        interval = kv_cache_config.mamba_state_cache_interval if kv_cache_config.enable_block_reuse else 0
        if interval is None or interval <= 0:
            mamba_slope = 0
        else:
            mamba_slope = state_bytes_per_rank // interval
        # heuristic: When block reuse is enabled, we assume the mamba snapshots are dominant instead of active states,
        # otherwise we may run out of kv cache blocks prior to mamba blocks due to the large number of max_batch_size.
        # So we ignore intercept and only calculate max_tokens based on slope
        # This can be improved by a more accurate max_batch_size and ISL/OSL estimation in the future.
        if mamba_slope > 0:
            intercept = 0
        return attention_slope + mamba_slope, intercept

    def shutdown(self):
        # Release tensor views into the pool before the pool memory is freed,
        # so their deleters don't see stale pointers.
        self.all_ssm_states = None
        self.all_conv_states = None
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.mamba_ssm_rand_seed = None
        self._dummy_request_mask = None
        self._dummy_request_mask_host = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None
        super().shutdown()

    def add_dummy_requests(
        self,
        request_ids: List[int],
        # Note that token_nums should be past_kv_len + input_len (without
        # spec decoding). The draft tokens will be added in this function,
        # so we don't need to take care of it in the caller. When preparing
        # token_nums, we should not take the draft tokens into account, so
        # don't use the kv_cache_manager.max_seq_len, which includes both
        # extra tokens and draft tokens.
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        kv_reserve_draft_tokens: Optional[int] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        encoder_output_lens: Optional[List[int]] = None,
        # For capturable drafting loops. During normal inference, the draft model always
        # has enough KV cache space to fit all of our draft tokens. During warmup, however,
        # we need to make the KV cache manager aware that multiple autoregressive steps will
        # occur.
        num_extra_decoding_steps: int = 0,
        draft_kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> List[LlmRequest]:
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            kv_reserve_draft_tokens=kv_reserve_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
            encoder_output_lens=encoder_output_lens,
            num_extra_decoding_steps=num_extra_decoding_steps,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )
        if requests:
            self.requests.extend(requests)
            # Only process the newly added requests, not all of self.requests.
            # self.requests may contain stale entries from the previous
            # _prepare_resources call (e.g., disagg transfer-pending requests)
            # that would exceed max_batch_size and cause out-of-bounds access.
            self._setup_state_indices(requests)
        return requests

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        super().update_resources(scheduled_batch, attn_metadata,
                                 kv_cache_dtype_byte_size)

    @nvtx_range("hybrid_prepare_resources")
    def _prepare_resources(self, scheduled_batch: ScheduledRequests):
        self.requests = scheduled_batch.context_requests + \
            scheduled_batch.generation_requests
        # Issue all async block onboards; defer the syncTransfers() (refresh_blocks)
        # until just before forward so the CPU-side prep (attn metadata, input
        # tensor assembly, draft model prep, etc.) can overlap with the in-flight
        # cudaMemcpyAsync calls instead of being serialized behind them.
        # copy_linear_attention_block returns True iff it actually issued a copy;
        # we skip refresh_blocks entirely when nothing was scheduled.
        self._pending_state_transfers = self.impl.copy_linear_attention_block_batch(
            self.requests)
        self._setup_state_indices()
        # Reset replay double-buffer state for fresh context blocks. A reused
        # block (prefix-cache hit or block recycled across requests) may carry
        # stale prev_num_accepted_tokens / cache_buf_idx values from a prior
        # owner; the replay kernel reads these on the first decode step.
        num_contexts = len(scheduled_batch.context_requests)
        if num_contexts > 0:
            ctx_slots = self.cuda_state_indices[:num_contexts].long()
            if (self._use_replay_state_update
                    and self.prev_num_accepted_tokens is not None
                    and self.cache_buf_idx is not None):
                self.prev_num_accepted_tokens[ctx_slots] = 0
                self.cache_buf_idx[ctx_slots] = 0
                if self.old_x is not None:
                    self.old_x[:, ctx_slots] = 0
                if self.old_B is not None:
                    self.old_B[:, ctx_slots] = 0
                if self.old_dt is not None:
                    self.old_dt[:, ctx_slots] = 0
                if self.old_dA_cumsum is not None:
                    self.old_dA_cumsum[:, ctx_slots] = 0
            # Deterministic per-context-slot seed rotation.  Runs whenever
            # the seed buffer exists, including the non-replay SR path.
            # Bump the host counter once per batch and write one new seed
            # per fresh context slot from a pure function of
            # (counter, slot, rank).  No torch.randint involved.
            if self.mamba_ssm_rand_seed is not None:
                self._seed_request_counter += 1
                counter = self._seed_request_counter
                rank_offset = self._seed_rank_offset
                host_slots = ctx_slots.cpu().tolist()
                new_seeds = [
                    _compute_deterministic_mamba_seed(counter, slot,
                                                      rank_offset)
                    for slot in host_slots
                ]
                seed_tensor = torch.tensor(
                    new_seeds,
                    dtype=torch.int64,
                    device=self.mamba_ssm_rand_seed.device,
                )
                self.mamba_ssm_rand_seed[ctx_slots] = seed_tensor

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        if self.local_num_mamba_layers == 0:
            return
        self._prepare_resources(scheduled_batch)

    @nvtx_range("hybrid_flush_state_transfers")
    def flush_state_transfers(self) -> None:
        """Complete any deferred state-block onboards scheduled by
        prepare_resources(). Must be called before forward() reads recurrent
        state blocks. Cheap no-op when nothing was scheduled.
        """
        if getattr(self, "_pending_state_transfers", False):
            self.impl.refresh_blocks()
            self._pending_state_transfers = False

    def is_speculative(self) -> bool:
        return self.spec_config is not None

    @nvtx_range("hybrid_update_mamba_states")
    def update_mamba_states(self,
                            attn_metadata: "AttentionMetadata",
                            num_accepted_tokens: torch.Tensor,
                            state_indices: Optional[torch.Tensor] = None):
        if self.local_num_mamba_layers == 0:
            return
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = (
            num_accepted_tokens[num_contexts:num_contexts + num_gens] - 1).to(
                torch.int32)
        # Match the API of MambaCacheManager.update_mamba_states: callers
        # may pass per-request state slot indices explicitly (e.g. MTP via
        # attn_metadata.mamba_metadata.state_indices). Fall back to this
        # manager's own slot mapping when not provided.
        if state_indices is None:
            state_indices = self.get_state_indices()
        state_indices_d = state_indices[num_contexts:num_contexts +
                                        num_gens].to(torch.int32)
        src_state_indices = self.intermediate_state_indices[:num_gens]

        # The accepted SSM/conv promotion is a bandwidth-bound gather->scatter
        # into the dtype-reinterpreted, strided C++ pool view. torch.compile
        # can't handle the dtype-view mutation (and Inductor chokes on the
        # uint8 pool with "XBLOCK too large"), so a dedicated Triton kernel
        # writes through the view's real strides (~85% of HBM peak, one launch
        # per state).
        if self._use_replay_state_update:
            # SSM state is handled incrementally by the kernel. Mirror the
            # kernel's checkpoint predicate from the previous PNAT and fixed
            # replay step width: checkpoint steps flip buffers, while no-write
            # steps append to the active history.
            slots = state_indices_d.long()
            accepted = num_accepted_tokens[num_contexts:num_contexts +
                                           num_gens].to(
                                               self.prev_num_accepted_tokens.
                                               dtype)
            prev_num_accepted_tokens = self.prev_num_accepted_tokens[slots]
            wrote_checkpoint = (prev_num_accepted_tokens +
                                self.replay_step_width
                                > self.replay_history_size)
            next_num_accepted_tokens = torch.where(
                wrote_checkpoint, accepted, prev_num_accepted_tokens + accepted)
            cache_buf_idx = self.cache_buf_idx[slots]
            assert self._dummy_request_mask is not None
            is_dummy_request = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
            next_num_accepted_tokens = torch.where(is_dummy_request,
                                                   prev_num_accepted_tokens,
                                                   next_num_accepted_tokens)
            self.prev_num_accepted_tokens[slots] = next_num_accepted_tokens
            self.cache_buf_idx[slots] = torch.where(
                is_dummy_request, cache_buf_idx,
                torch.where(wrote_checkpoint, 1 - cache_buf_idx, cache_buf_idx))
        else:
            # Legacy: copy the accepted SSM state from the intermediate buffer.
            _promote_mamba_state_triton(self.all_ssm_states,
                                        self.intermediate_ssm_states,
                                        src_state_indices,
                                        num_accepted_draft_tokens,
                                        state_indices_d)

        # Conv: both paths save all intermediate conv windows, carry over the
        # accepted one.
        _promote_mamba_state_triton(self.all_conv_states,
                                    self.intermediate_conv_states,
                                    src_state_indices,
                                    num_accepted_draft_tokens, state_indices_d)

    @torch.inference_mode()
    def _refresh_dummy_request_mask(self, is_dummy: List[bool]) -> None:
        if self._dummy_request_mask is None:
            return

        n = len(is_dummy)
        assert n <= self._dummy_request_mask_host.shape[0]
        self._dummy_request_mask_host.zero_()
        if n > 0:
            self._dummy_request_mask_host[:n].copy_(
                torch.tensor(is_dummy, dtype=torch.bool))
        self._dummy_request_mask.copy_(self._dummy_request_mask_host,
                                       non_blocking=True)

    def get_num_available_tokens(self,
                                 token_num_upper_bound: int,
                                 max_num_draft_tokens: int = 0,
                                 **kwargs) -> int:
        # Base bound from attention KV cache pool (the parent's behaviour).
        result = super().get_num_available_tokens(token_num_upper_bound,
                                                  max_num_draft_tokens,
                                                  **kwargs)
        # Also bound by the recurrent-state pool capacity: each request needs
        # roughly ceil(N / states_snapshot_interval) recurrent-state blocks
        # (+1 for the corner case where N is a multiple of tokens_per_block).
        # When block reuse is disabled, only one snapshot is needed per
        # request, so no additional capping is required here.
        interval = (self.linear_attention_metadata.states_snapshot_interval
                    if self.linear_attention_metadata is not None else 0)
        if interval and interval > 0:
            stats = self.impl.get_kv_cache_stats()
            rs_free = stats.num_free_blocks_per_window_size.get(
                LinearCacheType.RECURRENT_STATES.value, 0)
            # Reserve 1 block for the always-allocated last block (corner case
            # / final live state) so we don't promise more tokens than the
            # pool can actually back at allocation time.
            usable_rs_blocks = max(0, rs_free - 1)
            rs_token_cap = usable_rs_blocks * interval
            result = min(result, rs_token_cap)
        return max(result, 0)

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_ssm_states[self.mamba_layer_offsets[layer_idx]]

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_conv_states[self.mamba_layer_offsets[layer_idx]]

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_ssm_states is None:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.intermediate_ssm_states[layer_offset]

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_conv_states is None:
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.intermediate_conv_states[layer_offset]

    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union[PythonMambaCacheManager.State,
               PythonMambaCacheManager.SpeculativeState, None]:
        conv = self.get_conv_states(layer_idx)
        ssm = self.get_ssm_states(layer_idx)
        if self.spec_config is not None:
            layer_offset = self.mamba_layer_offsets[layer_idx]
            spec_kwargs = {}
            # Per-cache-slot Philox seed buffer is shared across replay and
            # non-replay MTP paths.  The mixer asserts non-None on both
            # branches when SR is enabled, so pass it through whenever it
            # exists — not just on the replay branch.
            if self.mamba_ssm_rand_seed is not None:
                spec_kwargs['mamba_ssm_rand_seed'] = self.mamba_ssm_rand_seed
            if self._use_replay_state_update:
                # Per-layer slices for the replay kernel; shared 1D tensors
                # (cache_buf_idx, prev_num_accepted_tokens) are passed
                # untouched via the SpeculativeState._SHARED_FIELDS contract.
                spec_kwargs['old_x'] = self.old_x[layer_offset]
                spec_kwargs['old_B'] = self.old_B[layer_offset]
                spec_kwargs['old_dt'] = self.old_dt[layer_offset]
                spec_kwargs['old_dA_cumsum'] = self.old_dA_cumsum[layer_offset]
                spec_kwargs['cache_buf_idx'] = self.cache_buf_idx
                spec_kwargs['prev_num_accepted_tokens'] = (
                    self.prev_num_accepted_tokens)
            else:
                spec_kwargs['intermediate_ssm'] = self.intermediate_ssm_states[
                    layer_offset]
            return PythonMambaCacheManager.SpeculativeState(
                conv=conv,
                temporal=ssm,
                intermediate_conv_window=self.
                intermediate_conv_states[layer_offset],
                **spec_kwargs,
            )
        return PythonMambaCacheManager.State(conv=conv, temporal=ssm)

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        if request in self.requests:
            self.requests.remove(request)
            self._request_id_to_state_index.pop(request.py_request_id, None)
            self._request_id_to_is_dummy.pop(request.py_request_id, None)
        super().free_resources(request, pin_on_release)

    def _setup_state_indices(self, requests=None) -> None:
        if self.local_num_mamba_layers == 0:
            return
        if requests is None:
            requests = self.requests
        block_indices = []
        for req in requests:
            if req.is_context_finished:
                next_step = self.get_num_tokens(req) - 1
            elif self.kv_cache_config.enable_block_reuse:
                next_step = (req.context_current_position - 1 +
                             req.context_chunk_size)
            else:
                next_step = req.prompt_len - 1
            block_indices.append(next_step // self.tokens_per_block)
        self.impl.copy_batch_block_offsets(
            self.host_block_offsets, [req.py_request_id for req in requests], 1,
            0)
        max_blocks = self.blocks_per_window[
            LinearCacheType.RECURRENT_STATES.value][0]
        n = len(requests)
        self._host_state_indices.zero_()
        if n > 0:
            # Vectorized gather: replace per-element Python loop with a single
            # tensor index op. block_indices is a small Python list so
            # torch.tensor() conversion here is O(n) at C level, much cheaper
            # than n round-trips through Python indexing.
            bi = torch.tensor(block_indices, dtype=torch.long)
            rows = self._row_indices[:n]
            # host_block_offsets: [num_pools, max_batch_size, 2, max_blocks_per_seq]
            values = self.host_block_offsets[self.recurrent_states_pool_index,
                                             rows, 0, bi]
            invalid_mask = (values < 0) | (values >= max_blocks)
            if invalid_mask.any():
                bad_i = int(invalid_mask.nonzero(as_tuple=False)[0, 0])
                req = requests[bad_i]
                value = int(values[bad_i])
                raise RuntimeError(
                    f"Invalid recurrent state block index {value} "
                    f"(expected 0 <= index < {max_blocks}) for request {bad_i}, "
                    f"prompt_len={req.prompt_len}, "
                    f"is_context_finished={req.is_context_finished}, "
                    f"context_current_position={req.context_current_position}, "
                    f"prepopulated_token_num={req.prepopulated_prompt_len}, "
                    f"context_chunk_size={req.context_chunk_size if not req.is_context_finished else 'N/A'}, "
                    f"block_index for next step is {block_indices[bad_i]}, "
                    "\nblock_ids="
                    f"{self.impl.get_cache_block_ids(req.py_request_id, LinearCacheType.RECURRENT_STATES.value)}"
                )
            self._host_state_indices[:n] = values

        self.cuda_state_indices.copy_(self._host_state_indices,
                                      non_blocking=True)
        self._refresh_dummy_request_mask(
            [req.is_dummy for req in self.requests])

        # Build request_id → pool block offset mapping so that
        # get_state_indices can return indices in arbitrary request order.
        for i, req in enumerate(requests):
            self._request_id_to_state_index[
                req.py_request_id] = self._host_state_indices[i].item()
            self._request_id_to_is_dummy[req.py_request_id] = req.is_dummy

    def get_state_indices(self,
                          request_ids: Optional[List[int]] = None,
                          is_padding: Optional[List[bool]] = None) -> list:
        if request_ids is not None:
            # Return indices in the order of the caller's request_ids,
            # not the internal self.requests order.  This is critical when
            # the batch is reordered after prepare_resources (e.g. disagg
            # serving sorts generation_requests by py_batch_idx).
            indices = [
                self._request_id_to_state_index[rid] for rid in request_ids
            ]
            if is_padding is None:
                is_padding = [False] * len(request_ids)
            assert len(request_ids) == len(is_padding)
            is_dummy = [
                self._request_id_to_is_dummy.get(rid, False) or padding
                for rid, padding in zip(request_ids, is_padding)
            ]
            self._refresh_dummy_request_mask(is_dummy)
            return indices
        return self.cuda_state_indices

    def calc_next_context_chunk_size(self, request: LlmRequest) -> int:
        """Compute the next prefill chunk size for a context request when block reuse is enabled.

        When kv_cache_config.enable_block_reuse is True, context prefill must stop exactly at
        the positions returned by calc_context_stop_positions (mamba_state_cache_interval boundaries
        and block boundaries). This returns the chunk_size to use for the next prefill step so
        that the next stop position is not exceeded.

        Args:
            request: Context request with prompt_len and context_current_position set.

        Returns:
            Number of tokens to prefill in the next step (0 if context is already complete).
        """
        prompt_len = request.prompt_len
        current = request.context_current_position
        if current >= prompt_len:
            return 0
        if not self.kv_cache_config.enable_block_reuse:
            assert current == 0, (
                "Expected context_current_position to be 0 when block reuse is "
                f"disabled, but got {current}")
            return prompt_len - current
        step = self.linear_attention_metadata.states_snapshot_interval
        stop_positions = calc_context_stop_positions(prompt_len,
                                                     self.tokens_per_block,
                                                     step)
        stop_positions = sorted(set(stop_positions))
        for pos in stop_positions:
            if pos > current:
                return pos - current
        return prompt_len - current

    def _setup_states(self) -> None:
        # Pool layout: {numLocalLayers, numBlocks, ssm_bytes + conv_bytes} (as uint8)
        pool: torch.Tensor = self.impl.get_recurrent_states_pool().view(
            torch.uint8).reshape(self.local_num_mamba_layers, -1,
                                 self.ssm_bytes + self.conv_bytes)
        num_blocks_in_pool = pool.shape[1]
        self.all_ssm_states = pool[:, :, :self.ssm_bytes].view(
            self.ssm_state_dtype).view(
                [self.local_num_mamba_layers, num_blocks_in_pool] +
                self.ssm_state_shape)
        self.all_conv_states = pool[:, :, self.ssm_bytes:self.ssm_bytes +
                                    self.conv_bytes].view(
                                        self.conv_state_dtype).view([
                                            self.local_num_mamba_layers,
                                            num_blocks_in_pool
                                        ] + self.conv_state_shape)
        self.all_ssm_states.zero_()
        self.all_conv_states.zero_()

    def _setup_mtp_intermediate_states(self, spec_config,
                                       max_batch_size) -> None:
        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        if self.spec_config is not None:
            # DFlash/PARD use 2K query tokens per gen, so size by tokens_per_gen_step.
            speculative_num_draft_tokens = self.spec_config.tokens_per_gen_step - 1
            num_local_mamba_layers = len(self.mamba_pp_layers)

            # Legacy SSM intermediate buffer is only needed when replay is
            # disabled; replay reads from the per-block double-buffered cache
            # set up in _setup_replay_buffers instead.
            if not self._use_replay_state_update:
                self.intermediate_ssm_states = torch.zeros(
                    size=[
                        num_local_mamba_layers, max_batch_size,
                        speculative_num_draft_tokens + 1
                    ] + self.ssm_state_shape,
                    dtype=self.ssm_state_dtype,
                    device="cuda",
                )

            self.intermediate_conv_states = torch.zeros(
                size=[
                    num_local_mamba_layers, max_batch_size,
                    speculative_num_draft_tokens + 1
                ] + self.conv_state_shape,
                dtype=self.conv_state_dtype,
                device="cuda",
            )

            self.intermediate_state_indices = torch.arange(max_batch_size,
                                                           dtype=torch.int32,
                                                           device="cuda")

    def _setup_replay_buffers(self, spec_config) -> None:
        """Allocate per-pool-block replay buffers used by replay_selective_state_update.

        Unlike the Mixed cache manager (where slots are 0..max_batch_size-1),
        the unified C++ KV pool assigns recurrent-state block indices up to
        ``num_blocks_in_pool``. The replay kernel indexes ``cache_buf_idx`` and
        ``prev_num_accepted_tokens`` by these block indices, so the buffers
        must match the pool extent rather than ``max_batch_size``.
        """
        # Replay tensors require spec_config + replay path enabled.  The
        # rand_seed buffer is separable from replay and must also be
        # allocated for non-replay SR so the flashinfer path has a
        # persistent deterministic seed source.
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.mamba_ssm_rand_seed = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None

        if (not self._use_replay_state_update
                and not self._mamba_ssm_stochastic_rounding):
            return

        cache_size = self.all_ssm_states.shape[1]
        device = self.all_ssm_states.device
        # Always-available deterministic seed buffer when SR (or replay)
        # is on.  Works for non-MTP runs because we don't depend on
        # spec_config to allocate it.
        self.mamba_ssm_rand_seed = _allocate_mamba_seed_buffer(
            cache_size, self._seed_rank_offset, device)

        if spec_config is None or not self._use_replay_state_update:
            # Without spec_config or replay we still keep the seed buffer
            # (above) so the non-MTP flashinfer SR path has a persistent
            # rand_seed source.
            self.prev_num_accepted_tokens = None
            self.cache_buf_idx = None
            self.old_x = None
            self.old_B = None
            self.old_dt = None
            self.old_dA_cumsum = None
            self._dummy_request_mask = None
            self._dummy_request_mask_host = None
            return

        history_size = self.replay_history_size
        num_local_mamba_layers = self.local_num_mamba_layers
        nheads, head_dim, d_state = self.ssm_state_shape
        n_groups_per_rank = self._n_groups_per_rank

        # Shared across layers (consumed by the replay kernel via slot index).
        self.prev_num_accepted_tokens = torch.zeros(cache_size,
                                                    dtype=torch.int32,
                                                    device=device)
        self.cache_buf_idx = torch.zeros(cache_size,
                                         dtype=torch.int32,
                                         device=device)
        self._dummy_request_mask = torch.zeros(self.max_batch_size,
                                               dtype=torch.bool,
                                               device=device)
        self._dummy_request_mask_host = torch.zeros(self.max_batch_size,
                                                    dtype=torch.bool,
                                                    pin_memory=prefer_pinned())
        self.old_x = torch.zeros(num_local_mamba_layers,
                                 cache_size,
                                 2,
                                 history_size,
                                 nheads,
                                 head_dim,
                                 dtype=self.conv_state_dtype,
                                 device=device)
        # Per-layer double-buffered caches.
        self.old_B = torch.zeros(num_local_mamba_layers,
                                 cache_size,
                                 2,
                                 history_size,
                                 n_groups_per_rank,
                                 d_state,
                                 dtype=self.conv_state_dtype,
                                 device=device)
        self.old_dt = torch.zeros(num_local_mamba_layers,
                                  cache_size,
                                  2,
                                  nheads,
                                  history_size,
                                  dtype=torch.float32,
                                  device=device)
        self.old_dA_cumsum = torch.zeros(num_local_mamba_layers,
                                         cache_size,
                                         2,
                                         nheads,
                                         history_size,
                                         dtype=torch.float32,
                                         device=device)

    @property
    def use_replay_state_update(self) -> bool:
        return self.get_replay_state_update_metadata() is not None

    def get_replay_state_update_metadata(
            self) -> Optional[ReplayStateUpdateMetadata]:
        prev_num_accepted_tokens = getattr(self, 'prev_num_accepted_tokens',
                                           None)
        cache_buf_idx = getattr(self, 'cache_buf_idx', None)
        if (not self._use_replay_state_update
                or prev_num_accepted_tokens is None or cache_buf_idx is None
                or self.replay_step_width is None
                or self.replay_history_size is None):
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            cache_buf_idx=cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size)

    def get_mamba_ssm_cache_dtype(self) -> torch.dtype:
        return self.ssm_state_dtype

    def get_mamba_ssm_rand_seed(self) -> Optional[torch.Tensor]:
        """Return the persistent (cache_size,) int64 Philox seed buffer or
        None when stochastic rounding is not active for this manager."""
        return getattr(self, 'mamba_ssm_rand_seed', None)
