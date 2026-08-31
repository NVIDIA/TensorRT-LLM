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
from dataclasses import dataclass, replace
from typing import (TYPE_CHECKING, Dict, Iterable, List, Literal, NamedTuple,
                    Optional, Tuple, Union)

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
    from tensorrt_llm.sampling_params import SamplingParams

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    BlockReusePolicy, KVCacheManagerV2, Role)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    ATTENTION_DP_DUMMY_REQUEST_ID, LlmRequest)
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager, CacheTypeCpp, DataType, KVCacheManager,
    PoolConfiguration, get_pp_layers)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import (TensorWrapper, convert_to_torch_tensor,
                                 nvtx_range, prefer_pinned,
                                 torch_dtype_to_binding)
from tensorrt_llm.bindings.internal.batch_manager import (
    LinearAttentionMetadata, LinearCacheType)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import (DEFAULT_BEAM_INDEX,
                                                      BatchDesc, BufferConfig,
                                                      DataRole, KVCacheDesc)
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import (LayerId, PageIndexMode,
                                                      SsmLayerConfig)

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


def _get_num_cuda_graph_padding_dummy_slots(
    spec_config: Optional["DecodingBaseConfig"],
    max_batch_size: int,
) -> int:
    """Return the number of persistent CUDA-graph padding dummy IDs.

    This is computed before ``ModelEngine`` exists and covers draft lengths
    reachable at every batch size, including the zero-length acceptance-rate
    fallback. ``ModelEngine._compute_dynamic_draft_len_mapping`` is created
    later and covers only configured CUDA-graph batch sizes, so it cannot size
    this persistent ID set.
    """
    if spec_config is None:
        return 1

    draft_len_schedule = getattr(spec_config, "draft_len_schedule", None)
    spec_dec_mode = getattr(spec_config, "spec_dec_mode", None)
    supports_dynamic_draft_len = (spec_dec_mode is not None and hasattr(
        spec_dec_mode, "support_dynamic_draft_len")
                                  and spec_dec_mode.support_dynamic_draft_len())
    if draft_len_schedule and supports_dynamic_draft_len:
        runtime_draft_lengths = set()
        first_uncovered_batch_size = 1
        for batch_size_threshold, draft_len in draft_len_schedule.items():
            if first_uncovered_batch_size > max_batch_size:
                break
            if batch_size_threshold >= first_uncovered_batch_size:
                runtime_draft_lengths.add(draft_len)
                first_uncovered_batch_size = batch_size_threshold + 1
        if first_uncovered_batch_size <= max_batch_size:
            runtime_draft_lengths.add(0)
    else:
        max_draft_len = getattr(spec_config, "max_draft_len", 0) or 0
        max_total_draft_tokens = (getattr(spec_config, "max_total_draft_tokens",
                                          0) or 0)
        is_linear_tree = getattr(
            spec_config,
            "is_linear_tree",
            max_draft_len == max_total_draft_tokens,
        )
        static_draft_len = (max_draft_len
                            if is_linear_tree else max_total_draft_tokens)
        runtime_draft_lengths = {static_draft_len or 0}

    if ((getattr(spec_config, "acceptance_rate_window_size", 0) or 0) > 0 and
        (getattr(spec_config, "acceptance_rate_threshold", 0) or 0) > 0):
        runtime_draft_lengths.add(0)
    return len(runtime_draft_lengths)


class MambaRole:
    """V2 buffer roles owned only by the hybrid Mamba manager."""

    SSM_STATE = DataRole("ssm_state")
    CONV_STATE = DataRole("conv_state")


def _mamba_effective_tp_size(mapping: Mapping) -> int:
    """TP degree for sizing per-rank mamba/KDA state pools.

    Attention-DP replicates the state and takes precedence; helix
    repurposes CP ranks as plain TP for recurrent-state layers.
    """
    if mapping.enable_attention_dp:
        return 1
    if mapping.has_cp_helix():
        return mapping.tp_size * mapping.cp_size
    return mapping.tp_size


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
    with PythonMambaCacheManager inside instead of the configured manager.
    Disagg mode is unaffected — its compatibility routing selects Mixed or
    Cpp based on the transceiver configuration.
    """
    return os.environ.get('TRTLLM_USE_PY_MAMBA', '0') == '1'


class ReplayStateUpdateMetadata(NamedTuple):
    """Shared tensors and fixed sizes for replay state updates."""
    prev_num_accepted_tokens: torch.Tensor
    cache_buf_idx: torch.Tensor
    replay_step_width: int
    replay_history_size: int


def _advance_replay_state(
    replay_metadata: ReplayStateUpdateMetadata,
    state_indices: torch.Tensor,
    accepted_tokens: torch.Tensor,
    is_dummy_request: Optional[torch.Tensor] = None,
) -> None:
    """Advance replay bookkeeping after a speculative generation step."""
    slots = state_indices.long()
    accepted_tokens = accepted_tokens.to(
        replay_metadata.prev_num_accepted_tokens.dtype)
    prev_num_accepted_tokens = replay_metadata.prev_num_accepted_tokens[slots]
    wrote_checkpoint = (prev_num_accepted_tokens +
                        replay_metadata.replay_step_width
                        > replay_metadata.replay_history_size)
    next_num_accepted_tokens = torch.where(
        wrote_checkpoint,
        accepted_tokens,
        prev_num_accepted_tokens + accepted_tokens,
    )
    cache_buf_idx = replay_metadata.cache_buf_idx[slots]
    next_cache_buf_idx = torch.where(wrote_checkpoint, 1 - cache_buf_idx,
                                     cache_buf_idx)
    if is_dummy_request is not None:
        next_num_accepted_tokens = torch.where(
            is_dummy_request,
            prev_num_accepted_tokens,
            next_num_accepted_tokens,
        )
        next_cache_buf_idx = torch.where(is_dummy_request, cache_buf_idx,
                                         next_cache_buf_idx)
    replay_metadata.prev_num_accepted_tokens[slots] = next_num_accepted_tokens
    replay_metadata.cache_buf_idx[slots] = next_cache_buf_idx


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

        @property
        def has_kda_replay_caches(self) -> bool:
            """Whether the Kimi K3 KDA fused-verify replay caches are
            allocated. The base (non-speculative) state never has them; the
            speculative state overrides this."""
            return False

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        """Speculative state with intermediate states for draft tokens.

        Supports three SSM update paths (only one set of tensors is
        allocated):
        - Legacy: caches full intermediate SSM states (intermediate_ssm)
        - Replay: compact double-buffered cache (old_x, old_B, old_dt, old_dA_cumsum)
        - KDA replay: per-slot draft-token caches consumed by the fused
          ``trtllm::kda_mtp_decode`` verify kernel, which replays accepted
          drafts and commits states in place (kda_conv_*, kda_*_cache)
        """
        _SHARED_FIELDS = frozenset({
            "prev_num_accepted_tokens", "cache_buf_idx", "mamba_ssm_rand_seed"
        })

        # Allocated for the legacy and Mamba2-replay paths; None for the
        # KDA replay path (the kernel commits conv windows in place).
        intermediate_conv_window: torch.Tensor | None = None

        # Legacy path: full intermediate SSM states at each step
        intermediate_ssm: torch.Tensor | None = None

        # KDA replay path (fused multi-token verify, kimi_linear).
        # Pool invariant under this path: `temporal` holds the state after
        # the LAST GOLDEN token; the accepted drafts recorded in
        # prev_num_accepted_tokens are pending in these caches and are
        # replayed by the kernel at the start of the next verify round.
        # Extended conv caches [layers, slots, dim, (W-1) + num_spec] fp32,
        # dim-contiguous (stride(dim) == 1): columns [0, W-1) are the
        # committed raw-input window, tail columns the pending drafts' raw
        # inputs.
        kda_conv_q: torch.Tensor | None = None
        kda_conv_k: torch.Tensor | None = None
        kda_conv_v: torch.Tensor | None = None
        # Post-processed per-draft quantities for replay:
        # [layers, slots, num_spec, 3, H*K] (q/k/gate), [.., num_spec, H*V],
        # [.., num_spec, H] — all fp32.
        kda_qkg_cache: torch.Tensor | None = None
        kda_v_cache: torch.Tensor | None = None
        kda_beta_cache: torch.Tensor | None = None

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

        @property
        def has_kda_replay_caches(self) -> bool:
            """True when the KDA fused-verify replay caches were allocated
            (the fused ``trtllm::kda_mtp_decode`` verify path)."""
            return self.kda_qkg_cache is not None

        def commit_conv_window(self, slot_indices: torch.Tensor,
                               conv_pool: torch.Tensor) -> None:
            """Seed the KDA replay caches from the live ``W - 1`` conv pool."""
            from ..modules.kimi_kda._kda_kernels import \
                copy_kda_replay_conv_window

            copy_kda_replay_conv_window(
                conv_pool,
                self.kda_conv_q,
                self.kda_conv_k,
                self.kda_conv_v,
                slot_indices,
            )

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
        kda_replay_num_spec: Optional[int] = None,
    ) -> None:

        self.mamba_ssm_cache_dtype = ssm_cache_dtype
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.spec_state_size = spec_state_size
        self._use_replay_state_update = use_replay_state_update
        # KDA replay path (kimi_linear fused multi-token verify). Mutually
        # exclusive with use_replay_state_update; requires speculative mode.
        self._kda_replay_num_spec = kda_replay_num_spec
        self._use_kda_replay_update = kda_replay_num_spec is not None
        if self._use_kda_replay_update:
            assert not use_replay_state_update, (
                "kda_replay_num_spec and use_replay_state_update are "
                "mutually exclusive")
            assert speculative_num_draft_tokens is not None, (
                "KDA replay caches require speculative decoding")
            assert kda_replay_num_spec == speculative_num_draft_tokens, (
                f"KDA replay cache width ({kda_replay_num_spec}) must match "
                f"the draft length ({speculative_num_draft_tokens}): the "
                "fused verify kernel is compiled with a static NUM_SPEC")
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
        tp_size = _mamba_effective_tp_size(mapping)

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

            # Conv intermediate cache — legacy and Mamba2-replay paths only.
            # The KDA replay kernel commits conv windows in place, so the
            # per-step window scratch is not needed.
            intermediate_conv_window_cache = None
            if not self._use_kda_replay_update:
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
            if self._use_kda_replay_update:
                # KDA replay caches for the fused multi-token verify kernel
                # (trtllm::kda_mtp_decode). Per cache slot (persistent
                # across rounds — the next round replays from them), unlike
                # the batch-row-indexed intermediate buffers.
                M = self._kda_replay_num_spec
                # conv_dim covers the [q | k | v] short-conv sections; the
                # kernel consumes them as three dim-contiguous caches.
                assert conv_dim % 3 == 0, (
                    "KDA replay caches expect the [q | k | v] conv-state "
                    "sectioning (3 equal sections)")
                section_dim = conv_dim // 3
                # d_conv is KDA's convolution width; the live pool stores
                # its W - 1 committed input columns.
                w_kernel = d_conv
                extended_s = w_kernel - 1 + M

                def _dim_contiguous_conv_cache():
                    return torch.zeros(num_local_layers,
                                       max_batch_size,
                                       extended_s,
                                       section_dim,
                                       dtype=torch.float32,
                                       device=device).transpose(-1, -2)

                spec_kwargs['prev_num_accepted_tokens'] = torch.zeros(
                    max_batch_size, dtype=torch.int32, device=device)
                spec_kwargs['kda_conv_q'] = _dim_contiguous_conv_cache()
                spec_kwargs['kda_conv_k'] = _dim_contiguous_conv_cache()
                spec_kwargs['kda_conv_v'] = _dim_contiguous_conv_cache()
                spec_kwargs['kda_qkg_cache'] = torch.zeros(num_local_layers,
                                                           max_batch_size,
                                                           M,
                                                           3,
                                                           section_dim,
                                                           dtype=torch.float32,
                                                           device=device)
                spec_kwargs['kda_v_cache'] = torch.zeros(num_local_layers,
                                                         max_batch_size,
                                                         M,
                                                         section_dim,
                                                         dtype=torch.float32,
                                                         device=device)
                spec_kwargs['kda_beta_cache'] = torch.zeros(num_local_layers,
                                                            max_batch_size,
                                                            M,
                                                            nheads,
                                                            dtype=torch.float32,
                                                            device=device)
                ssm_spec_cache = [
                    spec_kwargs['kda_conv_q'], spec_kwargs['kda_conv_k'],
                    spec_kwargs['kda_conv_v'], spec_kwargs['kda_qkg_cache'],
                    spec_kwargs['kda_v_cache'], spec_kwargs['kda_beta_cache']
                ]
                spec_path_label = "kda-replay"
            elif self._use_replay_state_update:
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
            elif (isinstance(self.mamba_cache, self.SpeculativeState)
                  and self._use_kda_replay_update):
                # Fresh request: no drafts pending in the replay caches.
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
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

    @torch.inference_mode()
    def seed_kda_replay_caches_for_disagg_gen(self,
                                              request_ids: List[int]) -> None:
        """Seed the fused-verify KDA replay conv caches from the conv pool.

        On a disaggregated generation server the ctx->gen transfer populates
        only the base ``conv`` / ``temporal`` pools (see
        ``disaggregation/resource/page.py``); the per-slot ``kda_conv_*``
        replay caches consumed by ``trtllm::kda_mtp_decode`` are normally
        seeded by a *local* prefill/decode via
        ``_sync_kda_replay_conv_window`` and would otherwise hold zeros (or a
        previous occupant's window) for a transferred request — corrupting
        the recurrent state from the first verify step onward. Call this
        after the state transfer completes, before the first generation
        forward. Mirrors ``_sync_kda_replay_conv_window``: the conv pool row
        stores the committed ``W - 1`` raw inputs directly. Pending-draft
        scratch is cleared (no drafts are pending for a freshly transferred
        request).
        """
        if not (self._use_kda_replay_update
                and isinstance(self.mamba_cache, self.SpeculativeState)):
            return
        blocks = [
            self.mamba_cache_index[rid] for rid in request_ids
            if rid in self.mamba_cache_index
        ]
        if not blocks:
            return
        conv = self.mamba_cache.conv  # [L, slots, 3D, W - 1]
        idx = torch.tensor(sorted(set(blocks)),
                           dtype=torch.long,
                           device=conv.device)
        d = conv.shape[2] // 3
        committed = conv.shape[3]
        cs = conv.index_select(1, idx)
        for cache, section in (
            (self.mamba_cache.kda_conv_q, cs[:, :, :d]),
            (self.mamba_cache.kda_conv_k, cs[:, :, d:2 * d]),
            (self.mamba_cache.kda_conv_v, cs[:, :, 2 * d:]),
        ):
            # cache: [L, slots, D, committed + num_spec]; zero the draft
            # tail columns and seed the committed window in one copy.
            seeded = torch.zeros(
                (cache.shape[0], idx.numel()) + cache.shape[2:],
                dtype=cache.dtype,
                device=cache.device)
            seeded[:, :, :, :committed] = section.to(cache.dtype)
            cache.index_copy_(1, idx, seeded)
        for buf in (self.mamba_cache.kda_qkg_cache,
                    self.mamba_cache.kda_v_cache,
                    self.mamba_cache.kda_beta_cache):
            buf.index_fill_(1, idx, 0)
        self.mamba_cache.prev_num_accepted_tokens[idx] = 0

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
                elif (isinstance(self.mamba_cache, self.SpeculativeState)
                      and self._use_kda_replay_update):
                    self.mamba_cache.prev_num_accepted_tokens[block] = 0
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
            elif (isinstance(self.mamba_cache, self.SpeculativeState)
                  and self._use_kda_replay_update):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0

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
                intermediate_conv_window=_drop(
                    self.mamba_cache.intermediate_conv_window),
                intermediate_ssm=_drop(self.mamba_cache.intermediate_ssm),
                prev_num_accepted_tokens=_drop(
                    self.mamba_cache.prev_num_accepted_tokens),
                cache_buf_idx=_drop(self.mamba_cache.cache_buf_idx),
                mamba_ssm_rand_seed=_drop(self.mamba_cache.mamba_ssm_rand_seed),
                old_x=_drop(self.mamba_cache.old_x),
                old_B=_drop(self.mamba_cache.old_B),
                old_dt=_drop(self.mamba_cache.old_dt),
                old_dA_cumsum=_drop(self.mamba_cache.old_dA_cumsum),
                kda_conv_q=_drop(self.mamba_cache.kda_conv_q),
                kda_conv_k=_drop(self.mamba_cache.kda_conv_k),
                kda_conv_v=_drop(self.mamba_cache.kda_conv_v),
                kda_qkg_cache=_drop(self.mamba_cache.kda_qkg_cache),
                kda_v_cache=_drop(self.mamba_cache.kda_v_cache),
                kda_beta_cache=_drop(self.mamba_cache.kda_beta_cache),
            )
        else:
            self.mamba_cache = self.State(conv=empty, temporal=empty)

        torch.cuda.empty_cache()

    @torch.compile(options={"max-autotune": True})
    def update_mamba_states(
            self,
            attn_metadata: "AttentionMetadata",
            num_accepted_tokens: torch.Tensor,
            state_indices: torch.Tensor,
            accepted_leaf_positions: Optional[torch.Tensor] = None):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[
            num_contexts:num_contexts + num_gens] - 1
        # Dynamic tree passes tree-node leaf positions; linear MTP uses depth.
        accepted_positions = (accepted_leaf_positions if accepted_leaf_positions
                              is not None else num_accepted_draft_tokens)
        state_indices_d = state_indices[num_contexts:num_contexts + num_gens]
        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_kda_replay_update:
            # KDA replay: the fused verify kernel already committed the SSM
            # state and conv windows (in place, after the golden token) and
            # cached this round's drafts. All that remains is recording how
            # many of those drafts the sampler accepted, so the next round's
            # kernel launch replays exactly that prefix.
            is_dummy_request = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
            prev = self.mamba_cache.prev_num_accepted_tokens
            current = prev[state_indices_d]
            accepted = num_accepted_draft_tokens.to(torch.int32).clamp(min=0)
            prev[state_indices_d] = torch.where(is_dummy_request, current,
                                                accepted)
            return

        if self._use_replay_state_update:
            is_dummy_request = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
            replay_metadata = self.get_replay_state_update_metadata()
            assert replay_metadata is not None
            _advance_replay_state(
                replay_metadata,
                state_indices_d,
                num_accepted_tokens[num_contexts:num_contexts + num_gens],
                is_dummy_request,
            )
        else:
            # Legacy: copy accepted SSM state from intermediate cache.
            ssm_states = self.mamba_cache.temporal
            intermediate_ssm_cache = self.mamba_cache.intermediate_ssm
            accepted_ssm_state = intermediate_ssm_cache[:, src_state_indices,
                                                        accepted_positions]
            ssm_states[:, state_indices_d, :] = accepted_ssm_state

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        conv_states = self.mamba_cache.conv
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window
        accepted_conv_state = intermediate_conv_window_cache[:,
                                                             src_state_indices,
                                                             accepted_positions]
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
        kda_replay_num_spec: Optional[int] = None,
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
            kda_replay_num_spec=kda_replay_num_spec,
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

    def seed_kda_replay_caches_for_disagg_gen(self,
                                              request_ids: List[int]) -> None:
        self._impl.seed_kda_replay_caches_for_disagg_gen(request_ids)

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

    def update_mamba_states(
            self,
            attn_metadata: "AttentionMetadata",
            num_accepted_tokens: torch.Tensor,
            state_indices: torch.Tensor,
            accepted_leaf_positions: Optional[torch.Tensor] = None):
        # Non-speculative configs don't allocate intermediate state; the
        # promotion is a clean no-op.
        if not self._impl.is_speculative():
            return
        self._impl.update_mamba_states(attn_metadata, num_accepted_tokens,
                                       state_indices, accepted_leaf_positions)


class MambaHybridCacheManager(BaseResourceManager, BaseMambaCacheManager):
    """Shared interface and state plumbing for hybrid Mamba managers.

    Concrete storage, state-index, and resource lifecycles remain owned by the
    Cpp and V2 implementations.
    """

    _supports_additional_snapshot_offsets = False

    def _setup_mtp_intermediate_states(self, spec_config,
                                       max_batch_size: int) -> None:
        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        if spec_config is None or self.local_num_mamba_layers == 0:
            return

        tokens_per_gen_step = spec_config.tokens_per_gen_step
        if not self._use_replay_state_update:
            self.intermediate_ssm_states = torch.zeros(
                size=[
                    self.local_num_mamba_layers, max_batch_size,
                    tokens_per_gen_step
                ] + self.ssm_state_shape,
                dtype=self.ssm_state_dtype,
                device="cuda",
            )
        self.intermediate_conv_states = torch.zeros(
            size=[
                self.local_num_mamba_layers, max_batch_size, tokens_per_gen_step
            ] + self.conv_state_shape,
            dtype=self.conv_state_dtype,
            device="cuda",
        )
        self.intermediate_state_indices = torch.arange(max_batch_size,
                                                       dtype=torch.int32,
                                                       device="cuda")

    def _allocate_pool_replay_buffers(
        self,
        spec_config,
        cache_size: int,
        device: Optional[torch.device],
    ) -> bool:
        """Allocate replay tensors shared by the Cpp and V2 state pools."""
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.mamba_ssm_rand_seed = None
        self._dummy_request_mask = None
        self._dummy_request_mask_host = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None

        if (self.local_num_mamba_layers == 0
                or (not self._use_replay_state_update
                    and not self._mamba_ssm_stochastic_rounding)):
            return False

        assert device is not None
        self.mamba_ssm_rand_seed = _allocate_mamba_seed_buffer(
            cache_size, self._seed_rank_offset, device)
        if spec_config is None or not self._use_replay_state_update:
            return False

        history_size = self.replay_history_size
        assert history_size is not None
        nheads, head_dim, d_state = self.ssm_state_shape
        common_shape = [self.local_num_mamba_layers, cache_size, 2]
        self.prev_num_accepted_tokens = torch.zeros(cache_size,
                                                    dtype=torch.int32,
                                                    device=device)
        self.cache_buf_idx = torch.zeros(cache_size,
                                         dtype=torch.int32,
                                         device=device)
        self.old_x = torch.zeros(
            common_shape + [history_size, nheads, head_dim],
            dtype=self.conv_state_dtype,
            device=device,
        )
        self.old_B = torch.zeros(
            common_shape + [history_size, self._n_groups_per_rank, d_state],
            dtype=self.conv_state_dtype,
            device=device,
        )
        self.old_dt = torch.zeros(
            common_shape + [nheads, history_size],
            dtype=torch.float32,
            device=device,
        )
        self.old_dA_cumsum = torch.zeros(
            common_shape + [nheads, history_size],
            dtype=torch.float32,
            device=device,
        )
        return True

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

    def _reset_context_mamba_slots(self, num_contexts: int) -> None:
        if num_contexts == 0:
            return

        context_slots = self.cuda_state_indices[:num_contexts].long()
        if (self._use_replay_state_update
                and self.prev_num_accepted_tokens is not None
                and self.cache_buf_idx is not None):
            self.prev_num_accepted_tokens[context_slots] = 0
            self.cache_buf_idx[context_slots] = 0
            if self.old_x is not None:
                self.old_x[:, context_slots] = 0
            if self.old_B is not None:
                self.old_B[:, context_slots] = 0
            if self.old_dt is not None:
                self.old_dt[:, context_slots] = 0
            if self.old_dA_cumsum is not None:
                self.old_dA_cumsum[:, context_slots] = 0

        if self.mamba_ssm_rand_seed is None:
            return
        self._seed_request_counter += 1
        counter = self._seed_request_counter
        rank_offset = self._seed_rank_offset
        host_slots = self._host_state_indices[:num_contexts].tolist()
        new_seeds = [
            _compute_deterministic_mamba_seed(counter, slot, rank_offset)
            for slot in host_slots
        ]
        self.mamba_ssm_rand_seed[context_slots] = torch.tensor(
            new_seeds,
            dtype=torch.int64,
            device=self.mamba_ssm_rand_seed.device,
        )

    def prepare_expect_snapshot_points(self,
                                       requests: List[LlmRequest]) -> None:
        """Set reusable Mamba snapshot boundaries before scheduling."""
        if not self.enable_block_reuse:
            for request in requests:
                request.expect_snapshot_points = []
            return

        state_config = self.kv_cache_config.mamba_state_config
        interval = state_config.periodic_snapshot_interval
        for request in requests:
            snapshot_points = set()
            if interval is not None and interval > 0:
                snapshot_points.update(
                    range(interval, request.prompt_len + 1, interval))
            if self._supports_additional_snapshot_offsets:
                for offset in state_config.additional_snapshot_offsets_from_start:
                    if offset <= request.prompt_len:
                        snapshot_points.add(offset)
                for offset in state_config.additional_snapshot_offsets_from_end:
                    point = request.prompt_len - offset
                    if point > 0:
                        snapshot_points.add(point)
            request.expect_snapshot_points = sorted(snapshot_points)

    def is_speculative(self) -> bool:
        return self.spec_config is not None

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_ssm_states[self.mamba_layer_offsets[layer_idx]]

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        return self.all_conv_states[self.mamba_layer_offsets[layer_idx]]

    def get_intermediate_ssm_states(self,
                                    layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_ssm_states is None:
            return None
        return self.intermediate_ssm_states[self.mamba_layer_offsets[layer_idx]]

    def get_intermediate_conv_states(self,
                                     layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_conv_states is None:
            return None
        return self.intermediate_conv_states[
            self.mamba_layer_offsets[layer_idx]]

    def mamba_layer_cache(
        self, layer_idx: int
    ) -> Union[PythonMambaCacheManager.State,
               PythonMambaCacheManager.SpeculativeState, None]:
        conv = self.get_conv_states(layer_idx)
        ssm = self.get_ssm_states(layer_idx)
        if self.spec_config is not None:
            layer_offset = self.mamba_layer_offsets[layer_idx]
            spec_kwargs = {}
            if self.mamba_ssm_rand_seed is not None:
                spec_kwargs['mamba_ssm_rand_seed'] = self.mamba_ssm_rand_seed
            if self._use_replay_state_update:
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
        return getattr(self, 'mamba_ssm_rand_seed', None)


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
        max_num_tokens: int = 8192,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        is_estimating_kv_cache: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        model_type: str = "nemotron_hybrid",
        is_draft: bool = False,
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        kda_replay_num_spec: Optional[int] = None,
        # Per-pool configurations forwarded to the C++ KVCacheManager ctor.
        # Lets a single manager host pools with mixed shapes (e.g. Gemma4
        # hybrid attention). See KVCacheManager.__init__.
        pool_configurations: Optional[List[PoolConfiguration]] = None,
    ) -> None:

        # mamba hybrid cache requires block reuse to be disabled in KV cache config
        assert not kv_cache_config.enable_block_reuse, (
            "mamba hybrid cache requires block reuse to be disabled in KV cache config"
        )

        # Host-drafter spec modes (NGram) have no spec worker to call
        # update_mamba_states; this manager promotes accepted states itself
        # in update_resources. One-model modes (MTP/Eagle/DFlash) and the
        # suffix-automaton worker promote from their spec workers and must
        # NOT be promoted twice.
        self._promote_states_in_update_resources = (
            spec_config is not None
            and getattr(spec_config, "decoding_type", None) == "NGram")

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
            kda_replay_num_spec=kda_replay_num_spec,
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
            max_num_tokens=max_num_tokens,
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
        self._maybe_promote_drafter_states(scheduled_batch, attn_metadata)

    def _maybe_promote_drafter_states(self, scheduled_batch, attn_metadata):
        """Commit accepted verification states for host-drafter spec modes.

        One-model spec workers (MTP/Eagle/DFlash) and the suffix-automaton
        worker call update_mamba_states themselves right after on-device
        acceptance. Host-drafter modes (NGram) have no spec worker:
        acceptance lands on the requests as
        ``py_num_accepted_draft_tokens`` during sampler update, and this
        hook — running right after, alongside the KV rewind — promotes the
        accepted step's intermediate state into the live pools (or, on the
        KDA replay path, records the accepted-draft count for the next
        round's replay). Without it, the pools would keep the
        pre-verification state and the next forward would resume from a
        stale prefix.
        """
        if not self._promote_states_in_update_resources:
            return
        if not self.is_speculative() or attn_metadata is None:
            return
        gen_requests = scheduled_batch.generation_requests
        if not gen_requests:
            return
        drafted = [
            r for r in gen_requests
            if r.py_draft_tokens is not None and len(r.py_draft_tokens) > 0
        ]
        if not drafted:
            # Drafter skipped this step (e.g. speculation gated off): the
            # forward ran the plain in-place decode path; nothing to promote.
            return
        assert len(drafted) == len(gen_requests), (
            "mixed drafted/undrafted generation batch is not supported for "
            "hybrid state promotion (drafts are padded to the static max)")
        device = self._impl.mamba_cache.temporal.device
        num_contexts = len(scheduled_batch.context_requests)
        num_accepted = torch.tensor(
            [0] * num_contexts +
            [r.py_num_accepted_draft_tokens + 1 for r in gen_requests],
            dtype=torch.int32,
            device=device)
        # Batch-ordered slots (contexts then gens), matching the ordering
        # the forward used for the intermediate scratch buffers. Requests
        # that finished this step were already freed by response handling
        # (which runs before update_resources) and are gone from the index;
        # their rows must stay in place for alignment, so redirect them to
        # the reserved padding slot (a harmless scratch write).
        slot_index = self.mamba_cache_index
        padding_slot = self._impl._padding_slot
        state_indices = torch.tensor([
            slot_index.get(r.py_request_id, padding_slot)
            for r in scheduled_batch.context_requests + gen_requests
        ],
                                     dtype=torch.int32,
                                     device=device)
        self.update_mamba_states(attn_metadata, num_accepted, state_indices)


@triton.jit
def _promote_mamba_state_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    accepted_position_source_ptr,
    num_accepted_tokens_ptr,
    blk_ptr,
    replay_pnat_ptr,
    replay_cache_buf_idx_ptr,
    dummy_request_mask_ptr,
    num_gens,
    count,
    src_s_layer,
    src_s_row,
    src_s_step,
    dst_s_layer,
    dst_s_block,
    POSITION_SOURCE_IS_TOKEN_COUNT: tl.constexpr,
    REPLAY_STEP_WIDTH: tl.constexpr,
    REPLAY_HISTORY_SIZE: tl.constexpr,
    UPDATE_REPLAY_BOOKKEEPING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program copies a BLOCK-sized tile of the contiguous inner state for a
    # single (layer, gen) pair. grid = (num_layers * num_gens, ceil(count/BLOCK)).
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    row = tl.load(src_idx_ptr + g).to(tl.int64)
    accepted_position = tl.load(accepted_position_source_ptr + g)
    if POSITION_SOURCE_IS_TOKEN_COUNT:
        accepted_position -= 1
    acc = accepted_position.to(tl.int64)
    blk = tl.load(blk_ptr + g).to(tl.int64)

    # The replay metadata update is batch-scoped, so layer 0's first copy tile
    # owns it. This folds PNAT and active-buffer maintenance into the existing
    # accepted-convolution-state promotion launch.
    if UPDATE_REPLAY_BOOKKEEPING:
        if (layer == 0) & (tile == 0):
            num_accepted_tokens = tl.load(num_accepted_tokens_ptr + g)
            is_dummy_request = tl.load(dummy_request_mask_ptr + g)
            pnat = tl.load(replay_pnat_ptr + blk)
            cache_buf_idx = tl.load(replay_cache_buf_idx_ptr + blk)
            wrote_checkpoint = pnat + REPLAY_STEP_WIDTH > REPLAY_HISTORY_SIZE
            next_pnat = tl.where(wrote_checkpoint, num_accepted_tokens,
                                 pnat + num_accepted_tokens)
            next_cache_buf_idx = tl.where(wrote_checkpoint, 1 - cache_buf_idx,
                                          cache_buf_idx)
            tl.store(replay_pnat_ptr + blk, next_pnat, mask=~is_dummy_request)
            tl.store(replay_cache_buf_idx_ptr + blk,
                     next_cache_buf_idx,
                     mask=~is_dummy_request)

    # int64 throughout: per-layer strides are O(1e8), so layer*stride overflows
    # int32 and would corrupt addresses / fault.
    src_base = layer * src_s_layer + row * src_s_row + acc * src_s_step
    dst_base = layer * dst_s_layer + blk * dst_s_block
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    v = tl.load(src_ptr + src_base + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, v, mask=mask)


def _promote_mamba_state_triton(
        dst: torch.Tensor,
        intermediate: torch.Tensor,
        src_state_indices: torch.Tensor,
        accepted_position_source: torch.Tensor,
        dst_state_indices: torch.Tensor,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        position_source_is_token_count: bool = False,
        replay_pnat: Optional[torch.Tensor] = None,
        replay_cache_buf_idx: Optional[torch.Tensor] = None,
        dummy_request_mask: Optional[torch.Tensor] = None,
        replay_step_width: int = 0,
        replay_history_size: int = 0,
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
        src_state_indices, accepted_position_source, dst_state_indices:
            ``[num_gens]`` int.
        num_accepted_tokens: ``[num_gens]`` accepted-token counts. Required
            when replay bookkeeping is enabled.
        position_source_is_token_count: Whether ``accepted_position_source``
            contains accepted-token counts instead of zero-based intermediate
            state positions. When true, the kernel subtracts one while
            computing the source position.
        replay_pnat, replay_cache_buf_idx, dummy_request_mask: Optional replay
            metadata. When supplied together, the kernel advances PNAT and the
            active history buffer while promoting the convolution state.

    For each gen ``g``::

        accepted_position = (
            accepted_position_source[g] - 1
            if position_source_is_token_count
            else accepted_position_source[g]
        )
        dst[:, dst_state_indices[g]] = intermediate[
            :, src_state_indices[g], accepted_position
        ]

    Pure gather->scatter copy; bandwidth-bound (~85% of HBM peak), one launch.
    """
    num_layers = dst.shape[0]
    num_gens = src_state_indices.shape[0]
    if num_gens == 0:
        return
    update_replay_bookkeeping = replay_pnat is not None
    if update_replay_bookkeeping:
        assert num_accepted_tokens is not None
        assert replay_cache_buf_idx is not None
        assert dummy_request_mask is not None
        assert replay_step_width > 0
        assert replay_history_size >= replay_step_width
        assert replay_pnat.dtype == torch.int32
        assert replay_cache_buf_idx.dtype == torch.int32
        assert dummy_request_mask.dtype == torch.bool
        assert dummy_request_mask.numel() >= num_gens
    else:
        # Compile-time-disabled pointer arguments still require tensors.
        num_accepted_tokens = accepted_position_source
        assert replay_cache_buf_idx is None
        assert dummy_request_mask is None
        replay_pnat = dst_state_indices
        replay_cache_buf_idx = dst_state_indices
        dummy_request_mask = dst_state_indices
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
        accepted_position_source,
        num_accepted_tokens,
        dst_state_indices,
        replay_pnat,
        replay_cache_buf_idx,
        dummy_request_mask,
        num_gens,
        count,
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        dst.stride(0),
        dst.stride(1),
        POSITION_SOURCE_IS_TOKEN_COUNT=position_source_is_token_count,
        REPLAY_STEP_WIDTH=replay_step_width,
        REPLAY_HISTORY_SIZE=replay_history_size,
        UPDATE_REPLAY_BOOKKEEPING=update_replay_bookkeeping,
        BLOCK=BLOCK,
    )


def _mamba_snapshot_rule_counts(
    kv_cache_config: KvCacheConfig,
    max_seq_len: Optional[int],
    tokens_per_block: int,
) -> Tuple[int, int]:
    """Return reachable fixed rules and their partial-block upper bound."""
    if not kv_cache_config.enable_block_reuse:
        return 0, 0

    num_rules = 0
    num_unaligned_rules = 0
    state_config = kv_cache_config.mamba_state_config
    for offset in set(state_config.additional_snapshot_offsets_from_start):
        if max_seq_len is not None and offset > max_seq_len:
            continue
        num_rules += 1
        num_unaligned_rules += int(offset % tokens_per_block != 0)
    for offset in set(state_config.additional_snapshot_offsets_from_end):
        # A from-end offset is reachable iff some valid prompt is longer than
        # the offset. Its absolute alignment depends on that prompt.
        if max_seq_len is not None and offset >= max_seq_len:
            continue
        num_rules += 1
        num_unaligned_rules += 1
    return num_rules, num_unaligned_rules


def _mamba_regular_snapshot_interval(
    kv_cache_config: KvCacheConfig,
    max_seq_len: Optional[int],
) -> Optional[int]:
    if not kv_cache_config.enable_block_reuse:
        return None
    interval = kv_cache_config.mamba_state_config.periodic_snapshot_interval
    if interval is None or interval <= 0:
        return None
    if max_seq_len is not None and interval > max_seq_len:
        return None
    return interval


def _get_local_mamba_cache_layout(
    model_config,
    mapping: Mapping,
    *,
    spec_config=None,
    is_draft: bool = False,
    use_separate_draft_kv_cache: bool = False,
):
    """Return normalized params and local Mamba/attention layer counts.

    Cache construction and affine sizing must follow the model's PP layout:
    partition base layers first, then place appended speculative layers on the
    last PP rank. The normalized params retain target masks and the appended
    draft-layer count so estimation selects the same combined or per-manager
    layout as runtime.
    """
    from tensorrt_llm._torch.pyexecutor.config_utils import \
        extract_mamba_kv_cache_params

    params = extract_mamba_kv_cache_params(
        model_config.pretrained_config,
        spec_config=spec_config,
        quant_config=model_config.quant_config,
    )
    mamba_layer_mask, full_attention_layer_mask = params.get_layer_masks(
        is_draft=is_draft,
        use_separate_draft_kv_cache=use_separate_draft_kv_cache,
    )
    combined_layer_mask = [
        is_mamba or is_attention for is_mamba, is_attention in zip(
            mamba_layer_mask, full_attention_layer_mask)
    ]
    local_layer_indices, _ = get_pp_layers(
        sum(combined_layer_mask),
        mapping,
        spec_config=spec_config,
        layer_mask=combined_layer_mask,
    )
    local_mamba_layers = sum(mamba_layer_mask[layer_idx]
                             for layer_idx in local_layer_indices)
    local_attention_layers = sum(full_attention_layer_mask[layer_idx]
                                 for layer_idx in local_layer_indices)
    return params, local_mamba_layers, local_attention_layers


def _estimate_mamba_hybrid_cache_cost(
    model_config,
    mapping: Mapping,
    *,
    max_batch_size: int,
    kv_cache_config: KvCacheConfig,
    tokens_per_block: int,
    max_seq_len: Optional[int],
    num_reserved_dummy_slots: int,
    include_explicit_snapshots: bool,
    cap_partial_attention_snapshots: bool,
    is_draft: bool = False,
    use_separate_draft_kv_cache: bool = False,
    **kwargs,
) -> Tuple[int, int]:
    spec_config = kwargs.get("spec_config")
    params, local_mamba_layers, local_attention_layers = (
        _get_local_mamba_cache_layout(
            model_config,
            mapping,
            spec_config=spec_config,
            is_draft=is_draft,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache,
        ))
    attention_slope = (KVCacheManager.get_cache_size_per_token(
        model_config,
        mapping,
        num_layers=local_attention_layers,
        **kwargs,
    ) if local_attention_layers > 0 else 0)
    state_bytes_per_rank = (local_mamba_layers *
                            params.get_states_bytes_per_layer(mapping))
    max_resident_sequences = max_batch_size * mapping.pp_size

    if include_explicit_snapshots:
        fixed_rules, unaligned_fixed_rules = _mamba_snapshot_rule_counts(
            kv_cache_config, max_seq_len, tokens_per_block)
    else:
        fixed_rules = 0
        unaligned_fixed_rules = 0
    fixed_state_slots = (max_resident_sequences + num_reserved_dummy_slots +
                         max_resident_sequences * fixed_rules)
    attention_block_bytes = attention_slope * tokens_per_block

    interval = _mamba_regular_snapshot_interval(kv_cache_config, max_seq_len)
    has_unaligned_periodic_snapshot = (interval is not None
                                       and interval % tokens_per_block != 0)
    if cap_partial_attention_snapshots:
        # Snapshot alignment is unknown while estimating cache cost. Once a
        # snapshot is possible, reserve one retained partial attention page
        # per resident lineage. Dummy requests carry no attention capacity.
        has_non_live_ssm_capacity = fixed_rules > 0 or interval is not None
        partial_attention_slots = (max_resident_sequences
                                   if has_non_live_ssm_capacity else 0)
    else:
        partial_attention_slots = (max_resident_sequences *
                                   unaligned_fixed_rules)
    intercept = (fixed_state_slots * state_bytes_per_rank +
                 partial_attention_slots * attention_block_bytes)

    if interval is None:
        regular_slope = 0
    else:
        regular_slope = math.ceil(state_bytes_per_rank / interval)
        if (has_unaligned_periodic_snapshot
                and not cap_partial_attention_snapshots):
            regular_slope += math.ceil(attention_block_bytes / interval)
    return attention_slope + regular_slope, intercept


class CppMambaHybridCacheManager(KVCacheManager, MambaHybridCacheManager):
    """Hybrid cache manager storing mamba states inside the KVCacheManager pool.

    Both KV cache blocks and recurrent state blocks are managed by the unified
    C++ KVCacheManager, enabling block reuse / prefix caching across attention
    and mamba layers. This compatibility manager remains available through the
    manager preference override and legacy disaggregated routing.

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
        self._use_gdn_cached_replay_all_layer_commit = (
            use_replay_state_update and model_type == "qwen3_next"
            and self.local_num_mamba_layers > 0)
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
        # Keep the shared Mamba interface valid on PP ranks that do not own a
        # local Mamba layer.
        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None

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
                **kwargs,
            )
            # PP ranks replay the same scheduling decisions, so a rank without
            # local Mamba layers must still publish the configured boundaries.
            self.kv_cache_config = kv_cache_config
            self.linear_attention_metadata = LinearAttentionMetadata()
            self.linear_attention_metadata.states_snapshot_interval = (
                kv_cache_config.mamba_state_config.periodic_snapshot_interval
                if kv_cache_config.enable_block_reuse else 0)
            return

        # Derive ssm_state_shape and conv_state_shape from mamba params (same as MambaCacheManager)
        tp_size = _mamba_effective_tp_size(mapping)
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
        self.ssm_bytes = (math.prod(self.ssm_state_shape) *
                          self.ssm_state_dtype.itemsize)
        self.conv_bytes = (math.prod(self.conv_state_shape) *
                           self.conv_state_dtype.itemsize)

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
            kv_cache_config.mamba_state_config.periodic_snapshot_interval
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

        recurrent_states_window = LinearCacheType.RECURRENT_STATES.value
        local_windows = {
            recurrent_states_window
            if mamba_layer_mask[layer_idx] else max_seq_len
            for layer_idx in self.pp_layers
        }
        kwargs["pool_configurations"] = [
            PoolConfiguration(
                window_size=window_size,
                head_dim=head_dim,
                dtype=torch_dtype_to_binding(self.ssm_state_dtype)
                if window_size == recurrent_states_window else dtype,
            ) for window_size in sorted(local_windows)
        ]

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
        self.kv_cache_config = kv_cache_config
        self.is_estimating_kv_cache = is_estimating_kv_cache

        self._setup_states()
        self._setup_replay_buffers(spec_config)
        if use_replay_state_update and model_type == "qwen3_next":
            logger.info_once(
                "Configured GDN cached replay commit mode: small-batch fused, "
                "large-batch all-layer",
                key="gdn_cached_replay_commit_mode_fused",
            )

    @staticmethod
    def get_cache_size_per_token(
        model_config,
        mapping: Mapping,
        *,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        num_layers: Optional[int] = None,
        tokens_per_block: int = 32,
        max_seq_len: Optional[int] = None,
        **kwargs,
    ):
        """Affine memory model for the unified hybrid KV pool.

        Returns ``(slope_bytes_per_token, intercept_bytes)``:

        * ``slope`` = attention KV bytes per token plus conservative periodic
          Mamba-state and partial-attention-snapshot costs.
        * ``intercept`` = live and CUDA-graph dummy Mamba state.

        Memory budget -> max tokens then becomes
        ``T = (budget - intercept) // slope`` instead of plain
        ``T = budget // bytes_per_token``.
        """
        return _estimate_mamba_hybrid_cache_cost(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            num_reserved_dummy_slots=1,
            include_explicit_snapshots=False,
            cap_partial_attention_snapshots=False,
            **kwargs,
        )

    @property
    def use_gdn_cached_replay_all_layer_commit(self) -> bool:
        return self._use_gdn_cached_replay_all_layer_commit

    def _commit_gdn_cached_replay_history_layers(
        self,
        attn_metadata: "AttentionMetadata",
        num_decodes: int,
    ) -> None:
        """Synchronously advance every local GDN checkpoint in one launch."""
        from tensorrt_llm._torch.modules.fla.cached_replay import (
            CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
            commit_gdn_cached_replay_history_layers)

        if (not self._use_gdn_cached_replay_all_layer_commit
                or num_decodes < CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE):
            return
        if (self.all_ssm_states is None or self.old_x is None
                or self.old_B is None or self.old_dt is None
                or self.replay_history_size is None):
            raise RuntimeError(
                "GDN cached replay all-layer commit requires replay state buffers."
            )

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}.")
        commit_gdn_cached_replay_history_layers(
            ssm_states=self.all_ssm_states,
            old_u=self.old_x,
            old_k=self.old_B,
            old_G=self.old_dt,
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            n_writes=mamba_metadata.replay_n_writes,
            history_size=self.replay_history_size,
        )

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
        draft_kv_cache_manager: Optional[KVCacheManager] = None,
        capture_sampling_params: Optional["SamplingParams"] = None,
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
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
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
        self._reset_context_mamba_slots(len(scheduled_batch.context_requests))

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

    @nvtx_range("hybrid_update_mamba_states")
    def update_mamba_states(
            self,
            attn_metadata: "AttentionMetadata",
            num_accepted_tokens: torch.Tensor,
            state_indices: Optional[torch.Tensor] = None,
            accepted_leaf_positions: Optional[torch.Tensor] = None):
        if self.local_num_mamba_layers == 0:
            return
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        accepted_tokens = num_accepted_tokens[num_contexts:num_contexts +
                                              num_gens]
        # Dynamic tree supplies explicit tree-node leaf positions. Linear MTP
        # passes accepted-token counts and lets the promotion kernel subtract
        # one, avoiding a separate GPU subtraction/cast.
        if accepted_leaf_positions is None:
            accepted_position_source = accepted_tokens
            position_source_is_token_count = True
        else:
            accepted_position_source = accepted_leaf_positions.to(torch.int32)
            position_source_is_token_count = False
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
            # Every GDN layer has finished reading the old checkpoint and
            # writing its candidate history. Advance all local checkpoints
            # in one launch before PNAT and the active history buffer change.
            self._commit_gdn_cached_replay_history_layers(
                attn_metadata, num_gens)
            assert self._dummy_request_mask is not None
        else:
            # Legacy: copy the accepted SSM state from the intermediate buffer.
            _promote_mamba_state_triton(
                self.all_ssm_states,
                self.intermediate_ssm_states,
                src_state_indices,
                accepted_position_source,
                state_indices_d,
                position_source_is_token_count=position_source_is_token_count)

        # Conv: both paths save all intermediate conv windows, carry over the
        # accepted one. The replay path also advances PNAT and the active cache
        # buffer in this same launch.
        replay_dummy_request_mask = None
        if self._use_replay_state_update:
            replay_dummy_request_mask = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
        _promote_mamba_state_triton(
            self.all_conv_states,
            self.intermediate_conv_states,
            src_state_indices,
            accepted_position_source,
            state_indices_d,
            num_accepted_tokens=accepted_tokens,
            position_source_is_token_count=position_source_is_token_count,
            replay_pnat=(self.prev_num_accepted_tokens
                         if self._use_replay_state_update else None),
            replay_cache_buf_idx=(self.cache_buf_idx
                                  if self._use_replay_state_update else None),
            dummy_request_mask=replay_dummy_request_mask,
            replay_step_width=(self.replay_step_width
                               if self._use_replay_state_update else 0),
            replay_history_size=(self.replay_history_size
                                 if self._use_replay_state_update else 0))

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
        # Attention-only PP ranks keep the interval so every rank publishes
        # identical scheduling boundaries, but they have no recurrent-state
        # pool whose capacity should constrain their attention KV cache.
        if self.local_num_mamba_layers > 0 and interval and interval > 0:
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
            # The C++ recurrent-state manager uses null page-table entries for
            # logical blocks that are not snapshot boundaries.  Usually a
            # context chunk ends exactly at an allocated snapshot (or at the
            # final live-state block), but the scheduler may shorten a chunk
            # further when KV capacity is tight.  In that case the Mamba
            # kernel must keep accumulating into the next allocated snapshot
            # or final block, matching copyLinearAttentionBlock(), which also
            # skips placeholders when it advances the live state.
            for bad_i in invalid_mask.nonzero(
                    as_tuple=False).flatten().tolist():
                req = requests[bad_i]
                if req.is_context_finished:
                    continue
                last_prompt_block = (req.prompt_len -
                                     1) // self.tokens_per_block
                row = self.host_block_offsets[self.recurrent_states_pool_index,
                                              rows[bad_i], 0]
                candidates = row[block_indices[bad_i]:last_prompt_block + 1]
                valid_candidates = ((candidates >= 0)
                                    & (candidates < max_blocks)).nonzero(
                                        as_tuple=False)
                if valid_candidates.numel() > 0:
                    values[bad_i] = candidates[valid_candidates[0, 0]]

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
        is_dummy = [req.is_dummy for req in requests]
        self._refresh_dummy_request_mask(is_dummy)

        # Build request_id → pool block offset mapping so that
        # get_state_indices can return indices in arbitrary request order.
        # Bulk tolist avoids a per-request tensor-index + .item() round-trip.
        state_values = self._host_state_indices[:n].tolist()
        for req, value, dummy in zip(requests, state_values, is_dummy):
            self._request_id_to_state_index[req.py_request_id] = value
            self._request_id_to_is_dummy[req.py_request_id] = dummy

    def get_state_indices(self,
                          request_ids: Optional[List[int]] = None,
                          is_padding: Optional[List[bool]] = None) -> list:
        if self.local_num_mamba_layers == 0:
            # Mamba metadata is prepared on every PP rank even when this rank
            # owns only attention layers. No local kernel consumes these
            # indices, so avoid consulting state that is intentionally absent.
            return [0] * len(request_ids) if request_ids is not None else []
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

    def _setup_replay_buffers(self, spec_config) -> None:
        cache_size = self.all_ssm_states.shape[1]
        device = self.all_ssm_states.device
        if not self._allocate_pool_replay_buffers(spec_config, cache_size,
                                                  device):
            return

        self._dummy_request_mask = torch.zeros(self.max_batch_size,
                                               dtype=torch.bool,
                                               device=device)
        self._dummy_request_mask_host = torch.zeros(
            self.max_batch_size,
            dtype=torch.bool,
            pin_memory=prefer_pinned(),
        )


class MambaHybridCacheManagerV2(KVCacheManagerV2, MambaHybridCacheManager):
    """Hybrid Mamba cache manager backed by KVCacheManagerV2.

    Attention KV pages and Mamba recurrent-state pages are both owned by the
    Python V2 cache manager.  Mamba layers are represented as V2 SSM layers,
    while this wrapper exposes the state tensors and slot indices expected by
    the PyTorch Mamba kernels.
    """

    _supports_additional_snapshot_offsets = True

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
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        is_estimating_kv_cache: bool = False,
        is_draft: bool = False,
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        conv_state_layout: Literal["x_b_c", "q_k_v"] = "x_b_c",
        **kwargs,
    ) -> None:
        if conv_state_layout not in ("x_b_c", "q_k_v"):
            raise ValueError(
                f"Unsupported convolution state layout: {conv_state_layout!r}")
        if "model_type" in kwargs:
            # The V1 managers select the conv-state layout by model_type; this
            # class takes it explicitly. Silently absorbing model_type here
            # means a caller's layout request would be dropped on the floor.
            raise TypeError(
                "MambaHybridCacheManagerV2 does not accept 'model_type' "
                f"(got {kwargs['model_type']!r}); pass "
                "conv_state_layout='x_b_c' or 'q_k_v' instead")
        total_layers = len(mamba_layer_mask)
        if layer_mask is None:
            full_attention_layer_mask = [False] * total_layers
        elif len(layer_mask) != total_layers:
            raise ValueError(
                f"layer_mask length ({len(layer_mask)}) must match "
                f"mamba_layer_mask length ({total_layers})")
        else:
            full_attention_layer_mask = list(layer_mask)

        combined_layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i]
            for i in range(total_layers)
        ]

        self._mamba_layer_mask = list(mamba_layer_mask)
        self._use_replay_state_update = use_replay_state_update
        self._use_gdn_cached_replay_all_layer_commit = (use_replay_state_update
                                                        and conv_state_layout
                                                        == "q_k_v")
        self._gdn_cached_replay_state_descriptors: Optional[torch.Tensor] = None
        self._gdn_cached_replay_state_strides: Optional[Tuple[int, int,
                                                              int]] = None
        self.replay_step_width: Optional[int] = (
            spec_config.tokens_per_gen_step
            if spec_config is not None and use_replay_state_update else None)
        self.replay_history_size: Optional[int] = (
            max(MIN_REPLAY_HISTORY_SIZE, self.replay_step_width)
            if self.replay_step_width is not None else None)
        self._mamba_ssm_stochastic_rounding = mamba_ssm_stochastic_rounding
        self._seed_rank_offset = _mamba_rank_offset(mapping)
        self._seed_request_counter = 0
        num_cuda_graph_padding_dummy_slots = (
            _get_num_cuda_graph_padding_dummy_slots(spec_config,
                                                    max_batch_size))
        self._num_reserved_dummy_slots = (num_cuda_graph_padding_dummy_slots +
                                          int(mapping.enable_attention_dp))
        self.ssm_state_dtype = (mamba_ssm_cache_dtype if mamba_ssm_cache_dtype
                                is not None else mamba_cache_dtype)
        self.conv_state_dtype = mamba_cache_dtype

        self.pp_layers, _ = get_pp_layers(
            mamba_num_layers + num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=combined_layer_mask,
        )
        self.mamba_pp_layers = [
            layer_idx for layer_idx in self.pp_layers
            if mamba_layer_mask[layer_idx]
        ]
        self.local_num_mamba_layers = len(self.mamba_pp_layers)
        self._use_gdn_cached_replay_all_layer_commit = (
            self._use_gdn_cached_replay_all_layer_commit
            and self.local_num_mamba_layers > 0)

        if self.local_num_mamba_layers > 0:
            tp_size = _mamba_effective_tp_size(mapping)
            d_inner = mamba_head_dim * mamba_num_heads
            grouped_state_dim = mamba_n_groups * mamba_d_state
            conv_dim = d_inner + 2 * grouped_state_dim
            nheads = mamba_num_heads
            assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
            assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
            if kwargs.get("is_disagg",
                          False) and grouped_state_dim % tp_size != 0:
                raise ValueError(
                    "Disaggregated Mamba transfer requires each convolution "
                    "state section to be divisible by tp_size")
            if use_replay_state_update:
                assert mamba_n_groups % tp_size == 0, \
                    "replay state update requires mamba_n_groups divisible by tp_size"
            self._n_groups_per_rank = mamba_n_groups // tp_size
            d_inner_local = d_inner // tp_size
            grouped_state_dim_local = grouped_state_dim // tp_size
            conv_dim = conv_dim // tp_size
            nheads = nheads // tp_size
            self.conv_state_shape = [conv_dim, mamba_d_conv - 1]
            self.ssm_state_shape = [nheads, mamba_head_dim, mamba_d_state]
            # TP-mismatch disaggregated transfers must split the flat
            # convolution state at its true semantic boundaries. Mamba2 stores
            # [x | B | C], while GDN stores [Q | K | V]. The large section is
            # therefore first for Mamba2 and last for GDN.
            if conv_state_layout == "x_b_c":
                self.conv_section_dims = [
                    d_inner_local,
                    grouped_state_dim_local,
                    grouped_state_dim_local,
                ]
            else:
                self.conv_section_dims = [
                    grouped_state_dim_local,
                    grouped_state_dim_local,
                    d_inner_local,
                ]
            self.ssm_bytes = (math.prod(self.ssm_state_shape) *
                              self.ssm_state_dtype.itemsize)
            self.conv_bytes = (math.prod(self.conv_state_shape) *
                               self.conv_state_dtype.itemsize)
        else:
            logger.info(
                "No local mamba layers for this rank, skipping mamba state views"
            )
            self._n_groups_per_rank = 0
            self.conv_state_shape = []
            self.ssm_state_shape = []
            self.conv_section_dims = []
            self.ssm_bytes = 0
            self.conv_bytes = 0

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

        kv_cache_config = kv_cache_config.model_copy(deep=True)
        if any(mamba_layer_mask) and kv_cache_config.enable_block_reuse:
            block_reuse_config = kv_cache_config.block_reuse_config
            block_reuse_policy = BlockReusePolicy(block_reuse_config.policy)
            if block_reuse_policy == BlockReusePolicy.ALL_REUSABLE:
                # SSM reuse is valid only at explicit snapshot boundaries.
                kv_cache_config.block_reuse_config = block_reuse_config.model_copy(
                    update={"policy": BlockReusePolicy.PER_REQUEST.value})
        self.kv_cache_config = kv_cache_config

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
            layer_mask=combined_layer_mask,
            is_draft=is_draft,
            is_estimating_kv_cache=is_estimating_kv_cache,
            num_reserved_index_slots=self._num_reserved_dummy_slots,
            **kwargs,
        )

        self.mamba_layer_offsets = {
            layer_id: offset
            for offset, layer_id in enumerate(self.mamba_pp_layers)
        }
        self._request_id_to_state_index = {}
        self._request_id_to_is_dummy = {}

        state_index_capacity = (self.max_batch_size +
                                self._num_reserved_dummy_slots)
        self.cuda_state_indices = torch.zeros([state_index_capacity],
                                              dtype=torch.int32,
                                              device="cuda")
        self._host_state_indices = torch.zeros([state_index_capacity],
                                               dtype=torch.int32,
                                               pin_memory=prefer_pinned())

        if self.local_num_mamba_layers > 0:
            first_mamba_local_layer = self.layer_offsets[
                self.mamba_pp_layers[0]]
            self.ssm_layer_group_id = self.impl.get_layer_group_id(
                LayerId(first_mamba_local_layer))
            self._ssm_page_index_scale = self.impl.get_page_index_scale(
                LayerId(first_mamba_local_layer), MambaRole.SSM_STATE)
            num_ssm_pages = self.impl.get_page_index_upper_bound(
                LayerId(first_mamba_local_layer), MambaRole.SSM_STATE)
            num_ssm_slots = ((num_ssm_pages + self._ssm_page_index_scale - 1) //
                             self._ssm_page_index_scale)
            required_live_slots = (self._max_resident_sequences() +
                                   self._num_reserved_dummy_slots)
            if num_ssm_slots < required_live_slots:
                KVCacheManagerV2.shutdown(self)
                raise ValueError(
                    "The V2 Mamba state pool has only "
                    f"{num_ssm_slots} slots but needs at least "
                    f"{required_live_slots} live/dummy slots. Increase the "
                    "KV cache budget or allocate a larger Mamba pool_ratio.")
            self._setup_states()
            self._setup_replay_buffers(spec_config)
            if self._use_gdn_cached_replay_all_layer_commit:
                state_layout = ("affine"
                                if self._gdn_cached_replay_state_strides
                                is not None else "indirect")
                logger.info_once(
                    "Configured GDN cached replay commit mode for V2: "
                    "small-batch fused, large-batch all-layer; "
                    f"state layout: {state_layout}",
                    key="gdn_cached_replay_v2_commit_mode_fused",
                )
        else:
            self.ssm_layer_group_id = None
            self._ssm_page_index_scale = 1
            self.all_ssm_states = []
            self.all_conv_states = []
            self._setup_replay_buffers(spec_config)

    @property
    def use_gdn_cached_replay_all_layer_commit(self) -> bool:
        return getattr(self, "_use_gdn_cached_replay_all_layer_commit", False)

    def _commit_gdn_cached_replay_history_layers(
        self,
        attn_metadata: "AttentionMetadata",
        num_decodes: int,
    ) -> None:
        """Advance all V2 GDN checkpoints through their state views."""
        from tensorrt_llm._torch.modules.fla.cached_replay import (
            CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
            commit_gdn_cached_replay_history_layers)

        if (not getattr(self, "_use_gdn_cached_replay_all_layer_commit", False)
                or num_decodes < CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE):
            return
        if (not self.all_ssm_states
                or (self._gdn_cached_replay_state_descriptors is None
                    and self._gdn_cached_replay_state_strides is None)
                or self.old_x is None or self.old_B is None
                or self.old_dt is None or self.replay_history_size is None):
            raise RuntimeError(
                "GDN cached replay all-layer commit requires V2 replay state buffers."
            )

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}.")
        state_strides = self._gdn_cached_replay_state_strides
        commit_gdn_cached_replay_history_layers(
            ssm_states=self.all_ssm_states[0],
            ssm_state_descriptors=(self._gdn_cached_replay_state_descriptors),
            ssm_state_num_layers=(state_strides[0]
                                  if state_strides is not None else None),
            ssm_state_layer_stride=(state_strides[1]
                                    if state_strides is not None else None),
            ssm_state_slot_stride=(state_strides[2]
                                   if state_strides is not None else None),
            old_u=self.old_x,
            old_k=self.old_B,
            old_G=self.old_dt,
            replay_work_items=mamba_metadata.replay_work_items[:num_decodes],
            n_writes=mamba_metadata.replay_n_writes,
            history_size=self.replay_history_size,
        )

    @staticmethod
    def get_cache_size_per_token(model_config,
                                 mapping: Mapping,
                                 *,
                                 max_batch_size: int,
                                 kv_cache_config: KvCacheConfig,
                                 num_layers: Optional[int] = None,
                                 tokens_per_block: int = 32,
                                 max_seq_len: Optional[int] = None,
                                 **kwargs):
        spec_config = kwargs.get("spec_config")
        num_reserved_dummy_slots = (_get_num_cuda_graph_padding_dummy_slots(
            spec_config, max_batch_size) + int(mapping.enable_attention_dp))
        return _estimate_mamba_hybrid_cache_cost(
            model_config,
            mapping,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            num_reserved_dummy_slots=num_reserved_dummy_slots,
            include_explicit_snapshots=True,
            cap_partial_attention_snapshots=True,
            **kwargs,
        )

    def _is_local_mamba_layer(self, local_layer_idx: int) -> bool:
        return self._mamba_layer_mask[self.pp_layers[local_layer_idx]]

    def _get_pool_roles(self,
                        pool_id: int) -> Tuple[DataRole, Optional[DataRole]]:
        layer_id = int(self.impl.layer_grouping[pool_id][0])
        if self._is_local_mamba_layer(layer_id):
            return MambaRole.SSM_STATE, None
        return super()._get_pool_roles(pool_id)

    def _max_resident_sequences(self) -> int:
        return self.max_batch_size * self.mapping.pp_size

    def _mamba_state_bytes_per_slot(self) -> int:
        return self.local_num_mamba_layers * (self.ssm_bytes + self.conv_bytes)

    def _num_ssm_snapshots_for_capacity(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        if capacity <= 0 or not kv_cache_config.enable_block_reuse:
            return 0

        fixed_rules, _ = _mamba_snapshot_rule_counts(kv_cache_config,
                                                     self.max_seq_len,
                                                     self.tokens_per_block)
        interval = _mamba_regular_snapshot_interval(kv_cache_config,
                                                    self.max_seq_len)
        regular_snapshots = capacity // interval if interval is not None else 0
        return (self._max_resident_sequences() * fixed_rules +
                regular_snapshots)

    def _num_ssm_states_per_typical_request(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        fixed_rules, _ = _mamba_snapshot_rule_counts(
            kv_cache_config,
            capacity,
            self.tokens_per_block,
        )
        # Additional snapshots are stable boundaries that must remain alive.
        # Periodic snapshots are evictable cache entries and therefore do not
        # increase the guaranteed state count represented by BatchDesc.
        return 1 + fixed_rules

    def _typical_request_descs(
        self,
        capacity: int,
        kv_cache_config: KvCacheConfig,
    ) -> List[KVCacheDesc]:
        """Model one request with one descriptor per live SSM state."""
        num_states = self._num_ssm_states_per_typical_request(
            capacity, kv_cache_config)
        capacity_per_state, capacity_remainder = divmod(capacity, num_states)
        capacities = [
            capacity_per_state + int(i < capacity_remainder)
            for i in range(num_states)
        ]
        return [
            KVCacheDesc(
                capacity=state_capacity,
                history_length=max(0, state_capacity - 1),
            ) for state_capacity in capacities
        ]

    def _get_typical_request_capacity(
        self,
        kv_cache_config: KvCacheConfig,
    ) -> int:
        if kv_cache_config.avg_seq_len is not None:
            return kv_cache_config.avg_seq_len

        fallback_capacity = max(1, self.max_seq_len // 2)
        logger.warning(
            "'kv_cache_config.avg_seq_len' is not set for a hybrid Mamba "
            "model using KV cache manager V2. Falling back to "
            f"max_seq_len / 2={fallback_capacity} for cache-pool sizing. Set "
            "'kv_cache_config.avg_seq_len' in the YAML configuration to the "
            "workload's average total sequence length for an accurate KV/SSM "
            "pool ratio.")
        return fallback_capacity

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        attention_quota = super()._get_quota_from_max_tokens(max_tokens)
        num_request_lineages = self._max_resident_sequences()
        snapshot_slots = self._num_ssm_snapshots_for_capacity(
            max_tokens, self.kv_cache_config)
        state_slots = (num_request_lineages + self._num_reserved_dummy_slots +
                       snapshot_slots)
        state_quota = state_slots * self._mamba_state_bytes_per_slot()
        # Once the plan contains any non-live SSM capacity, reserve one partial
        # attention page per request lineage. This remains conservative when
        # the plan contains fewer than one non-live slot per lineage.
        extra_attention_quota = (num_request_lineages *
                                 self._attention_cache_bytes_per_token() *
                                 self.tokens_per_block
                                 if snapshot_slots > 0 else 0)
        return attention_quota + state_quota + extra_attention_quota

    def _get_max_tokens_from_quota(self, quota: int) -> float:
        if self._get_quota_from_max_tokens(0) > quota:
            return 0

        low = 0
        high = 1
        while self._get_quota_from_max_tokens(high) <= quota:
            low = high
            high *= 2
            if high >= 1 << 62:
                return float("inf")

        while low + 1 < high:
            mid = (low + high) // 2
            if self._get_quota_from_max_tokens(mid) <= quota:
                low = mid
            else:
                high = mid
        return low

    def _minimum_live_gpu_quota(self) -> int:
        """Return the minimum quota for live states and one attention page."""
        attention_block_quota = (self._attention_cache_bytes_per_token() *
                                 self.tokens_per_block)
        num_state_slots = (self._max_resident_sequences() +
                           self._num_reserved_dummy_slots)
        state_quota = num_state_slots * self._mamba_state_bytes_per_slot()
        return max(
            self._get_quota_from_max_tokens(0),
            state_quota + attention_block_quota,
        )

    def _build_cache_config(
            self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        kv_cache_config = self.kv_cache_config
        cache_tiers = config.cache_tiers
        gpu_quota = cache_tiers[0].quota
        minimum_live_quota = self._minimum_live_gpu_quota()
        if minimum_live_quota > gpu_quota:
            raise ValueError(
                "The V2 Mamba GPU cache quota is too small for live recurrent "
                f"states and attention pages: got {gpu_quota} bytes, need at "
                f"least {minimum_live_quota} bytes.")
        # _build_base_config already constructed every attention layer,
        # including dtype-specific scale and subclass-provided side buffers.
        # Preserve those configs and replace only the local Mamba layers.
        layers = list(config.layers)
        for local_layer_idx, global_layer_idx in enumerate(self.pp_layers):
            if self._mamba_layer_mask[global_layer_idx]:
                layer_id = LayerId(local_layer_idx)
                layers[local_layer_idx] = SsmLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(role=MambaRole.SSM_STATE,
                                     size=self.ssm_bytes),
                        BufferConfig(role=MambaRole.CONV_STATE,
                                     size=self.conv_bytes),
                    ],
                )

        dummy_requests = [
            KVCacheDesc(capacity=0, history_length=0)
            for _ in range(self._num_reserved_dummy_slots)
        ]
        constraints = [
            replace(
                batch,
                kv_caches=[*batch.kv_caches, *dummy_requests],
            ) for batch in config.constraints
        ]

        typical_step = config.typical_step
        if config.initial_pool_ratio is None:
            typical_capacity = self._get_typical_request_capacity(
                kv_cache_config)
            request_descs = self._typical_request_descs(typical_capacity,
                                                        kv_cache_config)
            typical_step = BatchDesc(request_descs *
                                     self._max_resident_sequences() +
                                     dummy_requests)
        # The recurrent (SSM) state pool must hold one slot per resident
        # sequence plus every reserved dummy slot. Unlike attention pages, a
        # Mamba state is fixed-size per sequence, so this floor is independent
        # of sequence length. The base config only emits constraints when
        # ``avg_seq_len`` is set, and speculative decoding inflates the reserved
        # dummy slots (CUDA-graph padding), so without an explicit floor the SSM
        # pool can be undersized (see the live/dummy-slot check in _setup_states
        # / __init__). Add a min-slots constraint of zero-capacity requests:
        # these cost no attention pages but reserve one SSM slot each.
        if any(isinstance(layer, SsmLayerConfig) for layer in layers):
            ssm_floor_slots = (self._max_resident_sequences() +
                               self._num_reserved_dummy_slots)
            constraints = [
                *constraints,
                BatchDesc([
                    KVCacheDesc(capacity=0, history_length=0)
                    for _ in range(ssm_floor_slots)
                ]),
            ]
        return replace(
            config,
            layers=layers,
            typical_step=typical_step,
            constraints=constraints,
            # SSM lifecycles require minimum-snapshot commit semantics. The
            # flag is harmless when reuse is disabled because no commits are
            # attempted, while the runtime config still needs the invariant.
            commit_min_snapshot=True,
        )

    def _get_state_buffer(self, local_layer_idx: int, role, dtype: torch.dtype,
                          state_shape: List[int]) -> torch.Tensor:
        addr = self.impl.get_mem_pool_base_address(LayerId(local_layer_idx),
                                                   role, PageIndexMode.SHARED)
        num_pages = self.impl.get_page_index_upper_bound(
            LayerId(local_layer_idx), role)
        raw = convert_to_torch_tensor(
            TensorWrapper(addr, dtype, [num_pages] + state_shape))
        page_index_scale = self.impl.get_page_index_scale(
            LayerId(local_layer_idx), role)
        num_slots = (num_pages + page_index_scale - 1) // page_index_scale
        # V2 coalesces same-size per-layer buffers inside each slot.  Kernels
        # index Mamba states by logical slot id, so expose only this layer's
        # sub-page from each coalesced slot instead of the raw page-index view.
        return raw.as_strided(
            [num_slots] + state_shape,
            [raw.stride(0) * page_index_scale] + list(raw.stride()[1:]),
        )

    def _setup_states(self) -> None:
        local_layer_ids = [
            self.layer_offsets[layer_id] for layer_id in self.mamba_pp_layers
        ]
        self.all_ssm_states = [
            self._get_state_buffer(local_layer_idx, MambaRole.SSM_STATE,
                                   self.ssm_state_dtype, self.ssm_state_shape)
            for local_layer_idx in local_layer_ids
        ]
        self.all_conv_states = [
            self._get_state_buffer(local_layer_idx, MambaRole.CONV_STATE,
                                   self.conv_state_dtype, self.conv_state_shape)
            for local_layer_idx in local_layer_ids
        ]
        if self._use_gdn_cached_replay_all_layer_commit:
            expected_inner_strides = (
                self.ssm_state_shape[1] * self.ssm_state_shape[2],
                self.ssm_state_shape[2],
                1,
            )
            reference = self.all_ssm_states[0]
            for state in self.all_ssm_states:
                if (state.dtype != reference.dtype
                        or state.device != reference.device
                        or list(state.shape[1:]) != self.ssm_state_shape
                        or state.stride()[1:] != expected_inner_strides):
                    raise RuntimeError(
                        "GDN cached replay requires V2 SSM layers with matching "
                        "dtype, device, shape, and dense inner dimensions.")
            pointers = [state.data_ptr() for state in self.all_ssm_states]
            slot_stride = reference.stride(0)
            layer_stride_bytes = (pointers[1] -
                                  pointers[0] if len(pointers) > 1 else 0)
            has_affine_layout = (
                layer_stride_bytes % reference.element_size() == 0 and all(
                    state.stride(0) == slot_stride
                    for state in self.all_ssm_states)
                and all(pointer == pointers[0] + layer * layer_stride_bytes
                        for layer, pointer in enumerate(pointers)))
            if has_affine_layout:
                self._gdn_cached_replay_state_strides = (
                    len(pointers),
                    layer_stride_bytes // reference.element_size(),
                    slot_stride,
                )
            else:
                logger.warning_once(
                    "V2 GDN state views are not affine; using indirect replay "
                    "checkpoint addressing",
                    key="gdn_cached_replay_v2_indirect_state_layout",
                )
                self._gdn_cached_replay_state_descriptors = torch.tensor(
                    [(state.data_ptr(), state.stride(0))
                     for state in self.all_ssm_states],
                    dtype=torch.int64,
                    device=reference.device,
                )

    def _setup_replay_buffers(self, spec_config) -> None:
        cache_size = 0
        device = None
        if self.local_num_mamba_layers > 0:
            cache_size = self.all_ssm_states[0].shape[0]
            assert all(t.shape[0] == cache_size for t in self.all_ssm_states)
            device = self.all_ssm_states[0].device
        if not self._allocate_pool_replay_buffers(spec_config, cache_size,
                                                  device):
            return

        mask_capacity = self._host_state_indices.shape[0]
        self._dummy_request_mask = torch.zeros(mask_capacity,
                                               dtype=torch.bool,
                                               device=device)
        self._dummy_request_mask_host = torch.zeros(
            mask_capacity,
            dtype=torch.bool,
            pin_memory=prefer_pinned(),
        )

    def _attention_cache_bytes_per_token(self) -> int:
        # Mamba layers have zero KV heads, so the generic calculation naturally
        # returns only bytes owned by local attention layers.
        return super().get_cache_bytes_per_token()

    def get_cache_bytes_per_token(self) -> int:
        cache_bytes = self._attention_cache_bytes_per_token()

        interval = (
            self.kv_cache_config.mamba_state_config.periodic_snapshot_interval)
        if (self.kv_cache_config.enable_block_reuse and interval is not None
                and interval > 0):
            cache_bytes += self._mamba_state_bytes_per_slot() // interval
        if cache_bytes == 0 and self.local_num_mamba_layers > 0:
            cache_bytes = self._mamba_state_bytes_per_slot()
        return max(1, cache_bytes)

    def get_num_free_blocks(self) -> int:
        assert len(self.kv_cache_map) == 0, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        attention_pages = []
        ssm_pages = []
        for local_layer_idx in range(self.num_local_layers):
            layer_id = LayerId(local_layer_idx)
            if self._is_local_mamba_layer(local_layer_idx):
                ssm_pages.append(
                    self.impl.get_page_index_upper_bound(
                        layer_id, MambaRole.SSM_STATE) //
                    self._ssm_page_index_scale)
            else:
                attention_pages.append(
                    self.impl.get_page_index_upper_bound(layer_id, Role.KEY) //
                    self.kv_factor)
        if attention_pages:
            return max(attention_pages)
        return max(ssm_pages) if ssm_pages else 0

    @property
    def blocks_in_primary_pool(self) -> int:
        for local_layer_idx in range(self.num_local_layers):
            if self._is_local_mamba_layer(local_layer_idx):
                continue
            return self.impl.get_page_index_upper_bound(
                LayerId(local_layer_idx), Role.KEY)
        return 0

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        local_layer_idx = self.layer_offsets[layer_idx]
        if self._is_local_mamba_layer(local_layer_idx):
            return None
        return super().get_buffers(layer_idx, kv_layout)

    def _iter_cache_buffers_for_invalid_check(self) -> Iterable[torch.Tensor]:
        for global_layer_id, local_layer_id in self.layer_offsets.items():
            if self._is_local_mamba_layer(local_layer_id):
                continue
            # A layer group is a lifecycle, not a physical memory pool.
            # Differently sized attention buffers can share one lifecycle,
            # so scan every attention layer in this diagnostic path.
            yield KVCacheManagerV2.get_buffers(self, global_layer_id)

        yield from self.all_ssm_states
        yield from self.all_conv_states

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        kv_reserve_draft_tokens: Optional[int] = None,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        encoder_output_lens: Optional[List[int]] = None,
        draft_kv_cache_manager: Optional[BaseResourceManager] = None,
        capture_sampling_params: Optional["SamplingParams"] = None,
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
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
        )
        if requests and prepare_resource:
            self._setup_state_indices(requests)
        return requests

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is not None and kv_cache.is_active:
            self.try_commit_blocks(request, kv_cache)
        self._request_id_to_state_index.pop(request.py_request_id, None)
        self._request_id_to_is_dummy.pop(request.py_request_id, None)
        super().free_resources(request, pin_on_release)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        if self.local_num_mamba_layers == 0:
            return
        requests = (scheduled_batch.context_requests +
                    scheduled_batch.generation_requests)
        self._setup_state_indices(requests)
        num_contexts = len(scheduled_batch.context_requests)
        self._reset_context_mamba_slots(num_contexts)

    def _setup_state_indices(self, requests: List[LlmRequest]) -> None:
        if self.local_num_mamba_layers == 0:
            return
        n = len(requests)
        assert n <= self._host_state_indices.shape[0], (
            f"State-index batch size {n} exceeds max_batch_size "
            f"{self._host_state_indices.shape[0]}")
        self._host_state_indices.zero_()
        if n > 0:
            for i, req in enumerate(requests):
                kv_cache = self.kv_cache_map.get(req.py_request_id)
                if kv_cache is None:
                    raise RuntimeError(
                        f"Missing V2 KV cache for request {req.py_request_id}")
                base_index = kv_cache.get_ssm_block_base_index(
                    self.ssm_layer_group_id)
                if base_index < 0:
                    raise RuntimeError(
                        f"Invalid SSM state block index {base_index} for "
                        f"request {req.py_request_id}")
                self._host_state_indices[i] = base_index

        self.cuda_state_indices.copy_(self._host_state_indices,
                                      non_blocking=True)
        is_dummy = [req.is_dummy for req in requests]
        self._refresh_dummy_request_mask(is_dummy)
        state_values = self._host_state_indices[:n].tolist()
        for req, value, dummy in zip(requests, state_values, is_dummy):
            self._request_id_to_state_index[req.py_request_id] = value
            self._request_id_to_is_dummy[req.py_request_id] = dummy

    def get_state_indices(self,
                          request_ids: Optional[List[int]] = None,
                          is_padding: Optional[List[bool]] = None):
        if self.local_num_mamba_layers == 0:
            # Mamba metadata is still prepared on attention-only PP ranks,
            # but no local kernel consumes these indices. Return harmless
            # placeholders instead of consulting an intentionally empty map.
            if request_ids is not None:
                return [0] * len(request_ids)
            return self.cuda_state_indices
        if request_ids is not None:
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

    def get_max_resource_count(self) -> int:
        return self.max_batch_size

    def update_mamba_states(
            self,
            attn_metadata: "AttentionMetadata",
            num_accepted_tokens: torch.Tensor,
            state_indices: Optional[torch.Tensor] = None,
            accepted_leaf_positions: Optional[torch.Tensor] = None):
        if self.local_num_mamba_layers == 0:
            return
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = (
            num_accepted_tokens[num_contexts:num_contexts + num_gens] - 1).to(
                torch.int32)
        # Dynamic tree selects a tree node rather than a linear draft depth.
        accepted_positions = (accepted_leaf_positions.to(torch.int32)
                              if accepted_leaf_positions is not None else
                              num_accepted_draft_tokens)
        if state_indices is None:
            state_indices = self.get_state_indices()
        state_indices_d = state_indices[num_contexts:num_contexts +
                                        num_gens].to(torch.int32)
        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_replay_state_update:
            self._commit_gdn_cached_replay_history_layers(
                attn_metadata, num_gens)
            assert self._dummy_request_mask is not None
            is_dummy_request = self._dummy_request_mask[
                num_contexts:num_contexts + num_gens]
            replay_metadata = self.get_replay_state_update_metadata()
            assert replay_metadata is not None
            _advance_replay_state(
                replay_metadata,
                state_indices_d,
                num_accepted_tokens[num_contexts:num_contexts + num_gens],
                is_dummy_request,
            )
        else:
            for layer_offset, dst in enumerate(self.all_ssm_states):
                _promote_mamba_state_triton(
                    dst.unsqueeze(0),
                    self.intermediate_ssm_states[layer_offset:layer_offset + 1],
                    src_state_indices,
                    accepted_positions,
                    state_indices_d,
                )

        for layer_offset, dst in enumerate(self.all_conv_states):
            _promote_mamba_state_triton(
                dst.unsqueeze(0),
                self.intermediate_conv_states[layer_offset:layer_offset + 1],
                src_state_indices,
                accepted_positions,
                state_indices_d,
            )

    def _mark_context_position_as_history(self, request: LlmRequest,
                                          kv_cache) -> None:
        """Advance history without making later recurrent state reusable."""
        history_length = request.context_current_position
        if history_length <= kv_cache.history_length:
            return
        capacity = max(kv_cache.capacity, history_length)
        if not kv_cache.resize(capacity, history_length=history_length):
            raise ValueError(
                "Failed to resize history length of V2 Mamba cache for "
                f"request {request.py_request_id} to {history_length} tokens")

    def try_commit_blocks(self, request: LlmRequest, kv_cache=None) -> None:
        should_block_reuse = (self.enable_block_reuse and not self.is_draft
                              and not request.is_dummy_request)
        if not should_block_reuse:
            return

        if kv_cache is None:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
        if kv_cache is None:
            return

        snapshot_points = request.expect_snapshot_points
        commit_limit = (min(max(snapshot_points), request.prompt_len)
                        if snapshot_points else request.prompt_len)
        commit_end = min(request.context_current_position, commit_limit)
        if (request.context_current_position in request.expect_snapshot_points
                and commit_end > kv_cache.num_committed_tokens):
            tokens = self._augment_tokens_for_block_reuse(
                request.get_tokens(DEFAULT_BEAM_INDEX),
                request,
                start=kv_cache.num_committed_tokens,
                end=commit_end,
            )
            kv_cache.commit(tokens)
        if request.context_current_position >= commit_limit:
            self._mark_context_position_as_history(request, kv_cache)
        if request.context_remaining_length == 0:
            kv_cache.stop_committing()

    def update_context_resources(self,
                                 scheduled_batch: ScheduledRequests) -> None:
        for request in scheduled_batch.context_requests:
            kv_cache = self.kv_cache_map.get(request.py_request_id)
            if kv_cache is None or not kv_cache.is_active:
                continue

            should_block_reuse = (self.enable_block_reuse and not self.is_draft
                                  and not request.is_dummy_request)
            is_all_reusable = (
                self.block_reuse_policy == BlockReusePolicy.ALL_REUSABLE)
            is_snapshot_boundary = (request.context_current_position
                                    in request.expect_snapshot_points)
            has_pending_snapshot = any(
                point > request.context_current_position
                for point in request.expect_snapshot_points)
            should_resize = (not should_block_reuse or
                             (not is_all_reusable and not has_pending_snapshot))
            should_commit = (is_all_reusable or is_snapshot_boundary
                             or request.context_remaining_length == 0)

            if should_resize and not kv_cache.resize(
                    None, request.context_current_position):
                raise ValueError(
                    "Failed to resize history length of V2 Mamba cache for "
                    f"request {request.py_request_id} to "
                    f"{request.context_current_position} tokens at context "
                    "update")
            if should_commit:
                self.try_commit_blocks(request, kv_cache)
            if request.context_remaining_length == 0:
                if self.conversation_manager is not None:
                    self.conversation_manager.save_drop_plan(request, kv_cache)
                kv_cache.enable_swa_scratch_reuse = False

    def shutdown(self):
        self._gdn_cached_replay_state_descriptors = None
        self._gdn_cached_replay_state_strides = None
        self.all_ssm_states = []
        self.all_conv_states = []
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
