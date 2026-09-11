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
from dataclasses import fields
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import torch

from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayLayerCache,
    allocate_kda_replay_fields,
    seed_kda_replay_after_state_transfer,
)
from tensorrt_llm._torch.modules.mamba.cache_manager import Mamba2ReplayLayerCache
from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID, LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    CacheTypeCpp,
    DataType,
    KVCacheManager,
    PoolConfiguration,
    get_pp_layers,
)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, prefer_pinned, torch_dtype_to_binding
from tensorrt_llm.bindings.internal.batch_manager import LinearAttentionMetadata, LinearCacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .common import (
    MIN_REPLAY_HISTORY_SIZE,
    BaseMambaCacheManager,
    MambaHybridCacheManager,
    MambaLayerCache,
    ReplayStateUpdateMetadata,
    SpeculativeMambaLayerCache,
    _advance_replay_state,
    _allocate_mamba_seed_buffer,
    _compute_deterministic_mamba_seed,
    _estimate_mamba_hybrid_cache_cost,
    _get_mamba_hybrid_pool_size,
    _mamba_effective_tp_size,
    _mamba_rank_offset,
    _promote_mamba_state_triton,
    get_tensor_size_bytes,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
    from tensorrt_llm.sampling_params import SamplingParams

GB = 1 << 30


def warn_legacy_mamba_cache_manager(
    implementation: Literal["python", "cpp"], manager_name: str
) -> None:
    """Warn once when a V1 implementation is selected or constructed."""
    display_name = "Python" if implementation == "python" else "C++"
    logger.warning_once(
        f"{manager_name} uses the deprecated {display_name} Mamba cache "
        "manager V1 path. Set kv_cache_config.use_kv_cache_manager_v2=True; "
        "V1 remains available only for compatibility.",
        key=f"deprecated_mamba_cache_manager_{implementation}_v1",
    )


class PythonMambaCacheManager(BaseResourceManager):
    """Pure-Python mamba state manager with speculative decoding support.

    Manages only mamba states (conv + SSM) using PyTorch tensors on GPU.
    Supports speculative decoding and disaggregated serving.
    """

    State = MambaLayerCache
    SpeculativeState = SpeculativeMambaLayerCache

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
        warn_legacy_mamba_cache_manager("python", type(self).__name__)

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
                "kda_replay_num_spec and use_replay_state_update are mutually exclusive"
            )
            assert speculative_num_draft_tokens is not None, (
                "KDA replay caches require speculative decoding"
            )
            assert kda_replay_num_spec == speculative_num_draft_tokens, (
                f"KDA replay cache width ({kda_replay_num_spec}) must match "
                f"the draft length ({speculative_num_draft_tokens}): the "
                "fused verify kernel is compiled with a static NUM_SPEC"
            )
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
        self.mamba_layer_offsets = {idx: offset for offset, idx in enumerate(pp_layers)}

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
        if self._use_replay_state_update or self._mamba_ssm_stochastic_rounding:
            self._mamba_ssm_rand_seed = _allocate_mamba_seed_buffer(
                max_batch_size, self._seed_rank_offset, device
            )

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
                    size=(num_local_layers, self.spec_state_size, T) + conv_state_shape,
                    dtype=dtype,
                    device=device,
                )

            # SSM speculative cache — path-specific tensors
            spec_kwargs = {}
            # Share the manager-level seed buffer through SpeculativeState
            # so the MTP path can still read it via layer_cache.
            if self._mamba_ssm_rand_seed is not None:
                spec_kwargs["mamba_ssm_rand_seed"] = self._mamba_ssm_rand_seed
            if self._use_kda_replay_update:
                kda_fields, ssm_spec_cache = allocate_kda_replay_fields(
                    num_local_layers=num_local_layers,
                    cache_size=max_batch_size,
                    num_speculative_tokens=self._kda_replay_num_spec,
                    conv_dim=conv_dim,
                    conv_kernel_size=d_conv,
                    num_heads=nheads,
                    device=device,
                )
                spec_kwargs.update(kda_fields)
                spec_path_label = "kda-replay"
            elif self._use_replay_state_update:
                assert n_groups % tp_size == 0, (
                    "replay state update requires n_groups divisible by tp_size"
                )
                n_groups_per_rank = n_groups // tp_size
                self.replay_history_size = max(MIN_REPLAY_HISTORY_SIZE, T)

                # Compact replay cache.
                spec_kwargs["prev_num_accepted_tokens"] = torch.zeros(
                    max_batch_size, dtype=torch.int32, device=device
                )
                spec_kwargs["cache_buf_idx"] = torch.zeros(
                    max_batch_size, dtype=torch.int32, device=device
                )
                spec_kwargs["old_x"] = torch.zeros(
                    num_local_layers,
                    max_batch_size,
                    2,
                    self.replay_history_size,
                    nheads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                )
                spec_kwargs["old_B"] = torch.zeros(
                    num_local_layers,
                    max_batch_size,
                    2,
                    self.replay_history_size,
                    n_groups_per_rank,
                    d_state,
                    dtype=dtype,
                    device=device,
                )
                spec_kwargs["old_dt"] = torch.zeros(
                    num_local_layers,
                    max_batch_size,
                    2,
                    nheads,
                    self.replay_history_size,
                    dtype=torch.float32,
                    device=device,
                )
                spec_kwargs["old_dA_cumsum"] = torch.zeros(
                    num_local_layers,
                    max_batch_size,
                    2,
                    nheads,
                    self.replay_history_size,
                    dtype=torch.float32,
                    device=device,
                )
                ssm_spec_cache = [
                    spec_kwargs["old_x"],
                    spec_kwargs["old_B"],
                    spec_kwargs["old_dt"],
                    spec_kwargs["old_dA_cumsum"],
                ]
                spec_path_label = "replay"
            else:
                # Legacy: full intermediate SSM states at each step
                spec_kwargs["intermediate_ssm"] = torch.zeros(
                    size=(num_local_layers, self.spec_state_size, T) + ssm_state_shape,
                    dtype=self.mamba_ssm_cache_dtype,
                    device=device,
                )
                ssm_spec_cache = [spec_kwargs["intermediate_ssm"]]
                spec_path_label = "legacy"

            if self._use_kda_replay_update:
                cache_type = KDAReplayLayerCache
            elif self._use_replay_state_update:
                cache_type = Mamba2ReplayLayerCache
            else:
                cache_type = self.SpeculativeState
            self.mamba_cache = cache_type(
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
        self._dummy_request_mask = torch.zeros(max_batch_size, dtype=torch.bool, device=device)
        self._dummy_request_mask_host = torch.zeros(
            max_batch_size, dtype=torch.bool, pin_memory=prefer_pinned()
        )

        # Permanent slot shared by every CUDA-graph padding sentinel id
        # (CUDA_GRAPH_DUMMY_REQUEST_ID - runtime_draft_len, one per
        # draft length). Pool sizing must include +1 headroom for this;
        # see MixedMambaHybridCacheManager.
        self._padding_slot: int = self.mamba_cache_free_blocks.pop()

        # Reserved slot for the attention-DP padding dummy. Keep it out of the
        # free pool so dummy insertion never consumes real-request capacity.
        self._attention_dp_dummy_slot: Optional[int] = (
            self.mamba_cache_free_blocks.pop() if mapping.enable_attention_dp else None
        )

        # save intermediate state indices for requests
        self.intermediate_state_indices = torch.arange(
            max_batch_size, dtype=torch.int32, device=device
        )

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

    @property
    def use_kda_replay_update(self) -> bool:
        """Whether the deprecated manager installed KDA replay buffers."""
        return self._use_kda_replay_update

    @torch.inference_mode()
    def _prepare_mamba_cache_blocks(self, request_ids: List[int]):
        for r in request_ids:
            if r in self.mamba_cache_index:
                continue
            if len(self.mamba_cache_free_blocks) == 0:
                raise RuntimeError("run out of mamba cache blocks")
            block = self.mamba_cache_free_blocks.pop()
            self.mamba_cache_index[r] = block
            if (
                isinstance(self.mamba_cache, self.SpeculativeState)
                and self._use_replay_state_update
            ):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
                self.mamba_cache.cache_buf_idx[block] = 0
            elif (
                isinstance(self.mamba_cache, self.SpeculativeState) and self._use_kda_replay_update
            ):
                # Fresh request: no drafts pending in the replay caches.
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
            if self._mamba_ssm_rand_seed is not None:
                # Deterministic per-slot rotation on fresh assignment.
                # `block` is pulled from mamba_cache_free_blocks, which
                # excludes _padding_slot by construction (see __init__),
                # so padding sentinels never reach this branch.
                self._seed_request_counter += 1
                self._mamba_ssm_rand_seed[block] = _compute_deterministic_mamba_seed(
                    self._seed_request_counter, block, self._seed_rank_offset
                )

    @torch.inference_mode()
    def on_state_transfer_complete(self, request_ids: List[int]) -> None:
        """Prepare owner-specific scratch after base recurrent-state transfer."""
        if not isinstance(self.mamba_cache, KDAReplayLayerCache):
            return
        blocks = [
            self.mamba_cache_index[rid] for rid in request_ids if rid in self.mamba_cache_index
        ]
        if not blocks:
            return
        idx = torch.tensor(
            sorted(set(blocks)), dtype=torch.long, device=self.mamba_cache.conv.device
        )
        seed_kda_replay_after_state_transfer(self.mamba_cache, idx)

    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: List[int]) -> None:
        """Compatibility alias for the generic state-transfer hook."""
        self.on_state_transfer_complete(request_ids)

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        requests = scheduled_batch.context_requests + scheduled_batch.generation_requests
        context_ids = [i.py_request_id for i in scheduled_batch.context_requests]
        generation_ids = [i.py_request_id for i in scheduled_batch.generation_requests]
        request_ids = context_ids + generation_ids
        self._prepare_mamba_cache_blocks(request_ids)
        self._refresh_dummy_request_mask([req.is_dummy for req in requests])

    def _is_padding_sentinel(self, request_id: int) -> bool:
        # cuda_graph_runner caches one dummy per runtime_draft_len value
        # (see _get_padded_batch), so any id in the range of dummy request IDs
        # may be live concurrently.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        max_dl = self.speculative_num_draft_tokens or 0
        return CUDA_GRAPH_DUMMY_REQUEST_ID - max_dl <= request_id <= CUDA_GRAPH_DUMMY_REQUEST_ID

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
                if (
                    isinstance(self.mamba_cache, self.SpeculativeState)
                    and self._use_replay_state_update
                ):
                    self.mamba_cache.prev_num_accepted_tokens[block] = 0
                    self.mamba_cache.cache_buf_idx[block] = 0
                elif (
                    isinstance(self.mamba_cache, self.SpeculativeState)
                    and self._use_kda_replay_update
                ):
                    self.mamba_cache.prev_num_accepted_tokens[block] = 0
                continue
            if self._is_padding_sentinel(r):
                block = self._padding_slot
            elif r == ATTENTION_DP_DUMMY_REQUEST_ID and self._attention_dp_dummy_slot is not None:
                block = self._attention_dp_dummy_slot
            else:
                if len(self.mamba_cache_free_blocks) == 0:
                    raise RuntimeError("run out of mamba cache blocks")
                block = self.mamba_cache_free_blocks.pop()
            self.mamba_cache_index[r] = block
            if (
                isinstance(self.mamba_cache, self.SpeculativeState)
                and self._use_replay_state_update
            ):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0
                self.mamba_cache.cache_buf_idx[block] = 0
            elif (
                isinstance(self.mamba_cache, self.SpeculativeState) and self._use_kda_replay_update
            ):
                self.mamba_cache.prev_num_accepted_tokens[block] = 0

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        if request_id not in self.mamba_cache_index:
            return
        self._dummy_request_ids.discard(request_id)
        block = self.mamba_cache_index.pop(request_id)
        # Reserved slots must not re-enter the real-request free pool.
        if block != self._padding_slot and block != self._attention_dp_dummy_slot:
            self.mamba_cache_free_blocks.append(block)

    def get_state_indices(self, request_ids: List[int], is_padding: List[bool]) -> List[int]:
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
            self._dummy_request_mask_host[:n].copy_(torch.as_tensor(is_dummy, dtype=torch.bool))
        self._dummy_request_mask.copy_(self._dummy_request_mask_host, non_blocking=True)

    def get_conv_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).conv

    def get_ssm_states(self, layer_idx: int) -> torch.Tensor:
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).temporal

    def get_intermediate_ssm_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).intermediate_ssm

    def get_intermediate_conv_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get intermediate conv states for speculative decoding."""
        if not isinstance(self.mamba_cache, self.SpeculativeState):
            return None
        layer_offset = self.mamba_layer_offsets[layer_idx]
        return self.mamba_cache.at_layer_idx(layer_offset).intermediate_conv_window

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

    def get_replay_old_dA_cumsum(self, layer_idx: int) -> Optional[torch.Tensor]:
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

    def mamba_layer_cache(self, layer_idx: int) -> Union[State, SpeculativeState]:
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

    def get_replay_state_update_metadata(self) -> Optional[ReplayStateUpdateMetadata]:
        if (
            not self._use_replay_state_update
            or not isinstance(self.mamba_cache, self.SpeculativeState)
            or self.mamba_cache.prev_num_accepted_tokens is None
            or self.mamba_cache.cache_buf_idx is None
            or self.replay_step_width is None
            or self.replay_history_size is None
        ):
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=(self.mamba_cache.prev_num_accepted_tokens),
            cache_buf_idx=self.mamba_cache.cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size,
        )

    def shutdown(self):
        """Release tensor memory."""
        # Clear mamba cache states
        empty = torch.tensor([])

        def _drop(tensor):
            return empty if tensor is not None else None

        if isinstance(self.mamba_cache, self.SpeculativeState):
            cache_type = type(self.mamba_cache)
            values = {
                field.name: (
                    empty
                    if field.name in ("conv", "temporal")
                    else _drop(getattr(self.mamba_cache, field.name))
                )
                for field in fields(cache_type)
            }
            self.mamba_cache = cache_type(**values)
        else:
            self.mamba_cache = self.State(conv=empty, temporal=empty)

        torch.cuda.empty_cache()

    @torch.compile(options={"max-autotune": True})
    def update_mamba_states(
        self,
        attn_metadata: "AttentionMetadata",
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        accepted_leaf_positions: Optional[torch.Tensor] = None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        num_accepted_draft_tokens = num_accepted_tokens[num_contexts : num_contexts + num_gens] - 1
        # Dynamic tree passes tree-node leaf positions; linear MTP uses depth.
        accepted_positions = (
            accepted_leaf_positions
            if accepted_leaf_positions is not None
            else num_accepted_draft_tokens
        )
        state_indices_d = state_indices[num_contexts : num_contexts + num_gens]
        src_state_indices = self.intermediate_state_indices[:num_gens]

        if self._use_kda_replay_update:
            # KDA replay: the fused verify kernel already committed the SSM
            # state and conv windows (in place, after the golden token) and
            # cached this round's drafts. All that remains is recording how
            # many of those drafts the sampler accepted, so the next round's
            # kernel launch replays exactly that prefix.
            is_dummy_request = self._dummy_request_mask[num_contexts : num_contexts + num_gens]
            prev = self.mamba_cache.prev_num_accepted_tokens
            current = prev[state_indices_d]
            accepted = num_accepted_draft_tokens.to(torch.int32).clamp(min=0)
            prev[state_indices_d] = torch.where(is_dummy_request, current, accepted)
            return

        if self._use_replay_state_update:
            is_dummy_request = self._dummy_request_mask[num_contexts : num_contexts + num_gens]
            replay_metadata = self.get_replay_state_update_metadata()
            assert replay_metadata is not None
            _advance_replay_state(
                replay_metadata,
                state_indices_d,
                num_accepted_tokens[num_contexts : num_contexts + num_gens],
                is_dummy_request,
            )
        else:
            # Legacy: copy accepted SSM state from intermediate cache.
            ssm_states = self.mamba_cache.temporal
            intermediate_ssm_cache = self.mamba_cache.intermediate_ssm
            accepted_ssm_state = intermediate_ssm_cache[:, src_state_indices, accepted_positions]
            ssm_states[:, state_indices_d, :] = accepted_ssm_state

        # Conv: both paths save all intermediate conv windows, carry over the accepted one.
        conv_states = self.mamba_cache.conv
        intermediate_conv_window_cache = self.mamba_cache.intermediate_conv_window
        accepted_conv_state = intermediate_conv_window_cache[
            :, src_state_indices, accepted_positions
        ]
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
        warn_legacy_mamba_cache_manager("python", type(self).__name__)
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
        self, request_ids: Optional[List[int]] = None, is_padding: Optional[List[bool]] = None
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

    def seed_kda_replay_caches_for_disagg_gen(self, request_ids: List[int]) -> None:
        self.on_state_transfer_complete(request_ids)

    def on_state_transfer_complete(self, request_ids: List[int]) -> None:
        self._impl.on_state_transfer_complete(request_ids)

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

    @property
    def use_kda_replay_update(self) -> bool:
        """Delegate the legacy KDA replay capability for metadata compatibility."""
        return self._impl.use_kda_replay_update

    def get_replay_state_update_metadata(self) -> Optional[ReplayStateUpdateMetadata]:
        return self._impl.get_replay_state_update_metadata()

    def get_intermediate_ssm_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_intermediate_ssm_states(layer_idx)

    def get_intermediate_conv_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_intermediate_conv_states(layer_idx)

    def get_replay_old_x(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_x(layer_idx)

    def get_replay_old_B(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_B(layer_idx)

    def get_replay_old_dt(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_dt(layer_idx)

    def get_replay_old_dA_cumsum(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._impl.get_replay_old_dA_cumsum(layer_idx)

    def get_replay_cache_buf_idx(self) -> Optional[torch.Tensor]:
        return self._impl.get_replay_cache_buf_idx()

    def get_replay_prev_num_accepted_tokens(self) -> Optional[torch.Tensor]:
        return self._impl.get_replay_prev_num_accepted_tokens()

    def is_speculative(self) -> bool:
        return self._impl.is_speculative()

    def mamba_layer_cache(self, layer_idx: int) -> MambaLayerCache | None:
        return self._impl.mamba_layer_cache(layer_idx)

    def shutdown(self):
        self._impl.shutdown()

    def update_mamba_states(
        self,
        attn_metadata: "AttentionMetadata",
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        accepted_leaf_positions: Optional[torch.Tensor] = None,
    ):
        # Non-speculative configs don't allocate intermediate state; the
        # promotion is a clean no-op.
        if not self._impl.is_speculative():
            return
        self._impl.update_mamba_states(
            attn_metadata, num_accepted_tokens, state_indices, accepted_leaf_positions
        )


class MixedMambaHybridCacheManager(KVCacheManager, MambaCacheManager, MambaHybridCacheManager):
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
        warn_legacy_mamba_cache_manager("python", type(self).__name__)

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
            spec_config is not None and getattr(spec_config, "decoding_type", None) == "NGram"
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
            speculative_num_draft_tokens=(
                spec_config.tokens_per_gen_step - 1 if spec_config is not None else None
            ),
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

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata" = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        KVCacheManager.update_resources(
            self, scheduled_batch, attn_metadata, kv_cache_dtype_byte_size
        )
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
            r for r in gen_requests if r.py_draft_tokens is not None and len(r.py_draft_tokens) > 0
        ]
        if not drafted:
            # Drafter skipped this step (e.g. speculation gated off): the
            # forward ran the plain in-place decode path; nothing to promote.
            return
        assert len(drafted) == len(gen_requests), (
            "mixed drafted/undrafted generation batch is not supported for "
            "hybrid state promotion (drafts are padded to the static max)"
        )
        device = self._impl.mamba_cache.temporal.device
        num_contexts = len(scheduled_batch.context_requests)
        num_accepted = torch.tensor(
            [0] * num_contexts + [r.py_num_accepted_draft_tokens + 1 for r in gen_requests],
            dtype=torch.int32,
            device=device,
        )
        # Batch-ordered slots (contexts then gens), matching the ordering
        # the forward used for the intermediate scratch buffers. Requests
        # that finished this step were already freed by response handling
        # (which runs before update_resources) and are gone from the index;
        # their rows must stay in place for alignment, so redirect them to
        # the reserved padding slot (a harmless scratch write).
        slot_index = self.mamba_cache_index
        padding_slot = self._impl._padding_slot
        state_indices = torch.tensor(
            [
                slot_index.get(r.py_request_id, padding_slot)
                for r in scheduled_batch.context_requests + gen_requests
            ],
            dtype=torch.int32,
            device=device,
        )
        self.update_mamba_states(attn_metadata, num_accepted, state_indices)


class CppMambaHybridCacheManager(KVCacheManager, MambaHybridCacheManager):
    """Hybrid cache manager storing mamba states inside the KVCacheManager pool.

    Both KV cache blocks and recurrent state blocks are managed by the unified
    C++ KVCacheManager, enabling block reuse / prefix caching across attention
    and mamba layers. This compatibility manager remains available through the
    manager preference override and legacy disaggregated routing.

    """

    def _setup_mtp_intermediate_states(self, spec_config, max_batch_size: int) -> None:
        self.spec_config = spec_config
        self.intermediate_ssm_states = None
        self.intermediate_conv_states = None
        self.intermediate_state_indices = None
        if spec_config is None or self.local_num_mamba_layers == 0:
            return
        tokens_per_gen_step = spec_config.tokens_per_gen_step
        if not self._use_replay_state_update:
            self.intermediate_ssm_states = torch.zeros(
                [
                    self.local_num_mamba_layers,
                    max_batch_size,
                    tokens_per_gen_step,
                    *self.ssm_state_shape,
                ],
                dtype=self.ssm_state_dtype,
                device="cuda",
            )
        self.intermediate_conv_states = torch.zeros(
            [
                self.local_num_mamba_layers,
                max_batch_size,
                tokens_per_gen_step,
                *self.conv_state_shape,
            ],
            dtype=self.conv_state_dtype,
            device="cuda",
        )
        self.intermediate_state_indices = torch.arange(
            max_batch_size, dtype=torch.int32, device="cuda"
        )

    def _allocate_pool_replay_buffers(
        self, spec_config, cache_size: int, device: Optional[torch.device]
    ) -> bool:
        self.prev_num_accepted_tokens = None
        self.cache_buf_idx = None
        self.mamba_ssm_rand_seed = None
        self._dummy_request_mask = None
        self._dummy_request_mask_host = None
        self.old_x = None
        self.old_B = None
        self.old_dt = None
        self.old_dA_cumsum = None
        if self.local_num_mamba_layers == 0 or (
            not self._use_replay_state_update and not self._mamba_ssm_stochastic_rounding
        ):
            return False
        assert device is not None
        from tensorrt_llm._torch.modules.mamba.cache_manager import allocate_mamba_seed_buffer

        self.mamba_ssm_rand_seed = allocate_mamba_seed_buffer(
            cache_size, self._seed_rank_offset, device
        )
        if spec_config is None or not self._use_replay_state_update:
            return False
        nheads, head_dim, d_state = self.ssm_state_shape
        common_shape = [self.local_num_mamba_layers, cache_size, 2]
        self.prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
        self.old_x = torch.zeros(
            common_shape + [self.replay_history_size, nheads, head_dim],
            dtype=self.conv_state_dtype,
            device=device,
        )
        self.old_B = torch.zeros(
            common_shape + [self.replay_history_size, self._n_groups_per_rank, d_state],
            dtype=self.conv_state_dtype,
            device=device,
        )
        self.old_dt = torch.zeros(
            common_shape + [nheads, self.replay_history_size],
            dtype=torch.float32,
            device=device,
        )
        self.old_dA_cumsum = torch.zeros_like(self.old_dt)
        return True

    @torch.inference_mode()
    def _refresh_dummy_request_mask(self, is_dummy: List[bool]) -> None:
        if self._dummy_request_mask is None:
            return
        count = len(is_dummy)
        assert count <= self._dummy_request_mask_host.shape[0]
        self._dummy_request_mask_host.zero_()
        if count:
            self._dummy_request_mask_host[:count].copy_(torch.tensor(is_dummy, dtype=torch.bool))
        self._dummy_request_mask.copy_(self._dummy_request_mask_host, non_blocking=True)

    def _reset_context_mamba_slots(self, num_contexts: int) -> None:
        if num_contexts == 0:
            return
        from tensorrt_llm._torch.modules.mamba.cache_manager import compute_deterministic_mamba_seed

        slots = self.cuda_state_indices[:num_contexts].long()
        host_slots = self._host_state_indices[:num_contexts].tolist()
        if (
            self._use_replay_state_update
            and self.prev_num_accepted_tokens is not None
            and self.cache_buf_idx is not None
        ):
            self.prev_num_accepted_tokens[slots] = 0
            self.cache_buf_idx[slots] = 0
            for buffer in (self.old_x, self.old_B, self.old_dt, self.old_dA_cumsum):
                if buffer is not None:
                    buffer[:, slots] = 0
        if self.mamba_ssm_rand_seed is None:
            return
        self._seed_request_counter += 1
        seeds = [
            compute_deterministic_mamba_seed(
                self._seed_request_counter, slot, self._seed_rank_offset
            )
            for slot in host_slots
        ]
        self.mamba_ssm_rand_seed[slots] = torch.tensor(
            seeds, dtype=torch.int64, device=self.mamba_ssm_rand_seed.device
        )

    def get_intermediate_ssm_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_ssm_states is None:
            return None
        return self.intermediate_ssm_states[self.mamba_layer_offsets[layer_idx]]

    def get_intermediate_conv_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        if self.intermediate_conv_states is None:
            return None
        return self.intermediate_conv_states[self.mamba_layer_offsets[layer_idx]]

    def mamba_layer_cache(self, layer_idx: int) -> MambaLayerCache | SpeculativeMambaLayerCache:
        conv = self.get_conv_states(layer_idx)
        temporal = self.get_ssm_states(layer_idx)
        if self.spec_config is None:
            return MambaLayerCache(conv=conv, temporal=temporal)
        layer_offset = self.mamba_layer_offsets[layer_idx]
        fields = {"mamba_ssm_rand_seed": self.mamba_ssm_rand_seed}
        if self._use_replay_state_update:
            fields.update(
                old_x=self.old_x[layer_offset],
                old_B=self.old_B[layer_offset],
                old_dt=self.old_dt[layer_offset],
                old_dA_cumsum=self.old_dA_cumsum[layer_offset],
                cache_buf_idx=self.cache_buf_idx,
                prev_num_accepted_tokens=self.prev_num_accepted_tokens,
            )
        else:
            fields["intermediate_ssm"] = self.intermediate_ssm_states[layer_offset]
        cache_type = (
            Mamba2ReplayLayerCache if self._use_replay_state_update else SpeculativeMambaLayerCache
        )
        return cache_type(
            conv=conv,
            temporal=temporal,
            intermediate_conv_window=self.intermediate_conv_states[layer_offset],
            **fields,
        )

    @property
    def use_replay_state_update(self) -> bool:
        return self.get_replay_state_update_metadata() is not None

    def get_replay_state_update_metadata(self) -> Optional[ReplayStateUpdateMetadata]:
        prev_num_accepted_tokens = getattr(self, "prev_num_accepted_tokens", None)
        cache_buf_idx = getattr(self, "cache_buf_idx", None)
        if (
            not self._use_replay_state_update
            or prev_num_accepted_tokens is None
            or cache_buf_idx is None
            or self.replay_step_width is None
            or self.replay_history_size is None
        ):
            return None
        return ReplayStateUpdateMetadata(
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            cache_buf_idx=cache_buf_idx,
            replay_step_width=self.replay_step_width,
            replay_history_size=self.replay_history_size,
        )

    def get_mamba_ssm_rand_seed(self) -> Optional[torch.Tensor]:
        return getattr(self, "mamba_ssm_rand_seed", None)

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
        layer_mask: Optional[List[bool]] = None,  # this is the full attention layer mask
        is_estimating_kv_cache: bool = False,
        is_draft: bool = False,
        use_replay_state_update: bool = False,
        mamba_ssm_stochastic_rounding: bool = False,
        model_type: str = "nemotron_hybrid",
        **kwargs,
    ) -> None:
        warn_legacy_mamba_cache_manager("cpp", type(self).__name__)
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
                f"mamba_layer_mask length ({total_layers})"
            )
        else:
            full_attention_layer_mask = list(layer_mask)
        layer_mask = [
            mamba_layer_mask[i] or full_attention_layer_mask[i] for i in range(total_layers)
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
            layer_idx for layer_idx in self.pp_layers if mamba_layer_mask[layer_idx]
        ]
        self.local_num_mamba_layers = len(self.mamba_pp_layers)
        self.requests = []
        # Seed externally visible mamba fields before any early return so that
        # accessors (get_mamba_ssm_cache_dtype, use_replay_state_update) work
        # on ranks with no local mamba layers.
        self._use_replay_state_update = use_replay_state_update
        self._use_gdn_cached_replay_all_layer_commit = (
            use_replay_state_update
            and model_type == "qwen3_next"
            and self.local_num_mamba_layers > 0
        )
        self.replay_step_width: Optional[int] = (
            spec_config.tokens_per_gen_step
            if spec_config is not None and use_replay_state_update
            else None
        )
        self.replay_history_size: Optional[int] = (
            max(MIN_REPLAY_HISTORY_SIZE, self.replay_step_width)
            if self.replay_step_width is not None
            else None
        )
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
            logger.info("No local mamba layers for this rank, skipping mamba cache initialization")
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
                if kv_cache_config.enable_block_reuse
                else 0
            )
            return

        # Derive ssm_state_shape and conv_state_shape from mamba params (same as MambaCacheManager)
        tp_size = _mamba_effective_tp_size(mapping)
        d_inner = mamba_head_dim * mamba_num_heads
        conv_dim = d_inner + 2 * mamba_n_groups * mamba_d_state
        nheads = mamba_num_heads
        assert nheads % tp_size == 0, "mamba_num_heads must be divisible by tp_size"
        assert conv_dim % tp_size == 0, "conv_dim must be divisible by tp_size"
        if use_replay_state_update:
            assert mamba_n_groups % tp_size == 0, (
                "replay state update requires mamba_n_groups divisible by tp_size"
            )
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
        self.ssm_bytes = math.prod(self.ssm_state_shape) * self.ssm_state_dtype.itemsize
        self.conv_bytes = math.prod(self.conv_state_shape) * self.conv_state_dtype.itemsize

        total_bytes = self.ssm_bytes + self.conv_bytes
        if total_bytes % self.ssm_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"ssm_state_dtype size ({self.ssm_state_dtype.itemsize})"
            )
        if total_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"Total state bytes ({total_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})"
            )
        if self.ssm_bytes % self.conv_state_dtype.itemsize != 0:
            raise RuntimeError(
                f"SSM state bytes ({self.ssm_bytes}) not divisible by "
                f"conv_state_dtype size ({self.conv_state_dtype.itemsize})"
            )
        self.linear_attention_metadata = LinearAttentionMetadata()
        self.linear_attention_metadata.cache_type = LinearCacheType.RECURRENT_STATES.value
        self.linear_attention_metadata.all_recurrent_states_bytes = self.ssm_bytes + self.conv_bytes
        self.linear_attention_metadata.states_snapshot_interval = (
            kv_cache_config.mamba_state_config.periodic_snapshot_interval
            if kv_cache_config.enable_block_reuse
            else 0
        )
        # RNN model params for disagg TP-mismatch split/concat.
        conv_section_map = {"nemotron_hybrid": 1, "qwen3_next": 2}
        self.linear_attention_metadata.rnn_num_heads = self._rnn_num_heads
        self.linear_attention_metadata.rnn_head_dim = self._rnn_head_dim
        self.linear_attention_metadata.rnn_d_state = self._rnn_d_state
        self.linear_attention_metadata.rnn_d_conv = self._rnn_d_conv
        self.linear_attention_metadata.rnn_n_groups = self._rnn_n_groups
        self.linear_attention_metadata.rnn_conv_section_layout = conv_section_map.get(
            self._rnn_conv_section_layout, 0
        )
        self.linear_attention_metadata.rnn_ssm_bytes = self.ssm_bytes
        self.linear_attention_metadata.rnn_ssm_dtype_size = self.ssm_state_dtype.itemsize
        self.linear_attention_metadata.rnn_conv_dtype_size = self.conv_state_dtype.itemsize
        kv_cache_config = kv_cache_config.model_copy(deep=True)
        if kv_cache_config.enable_partial_reuse:
            logger.warning(
                "Partial reuse is not supported for mamba hybrid models, disabling partial reuse"
            )
            kv_cache_config.enable_partial_reuse = False

        # Keep the vector in physical global-layer order. Disabled entries are
        # placeholders that are never projected into this manager.
        kv_cache_config.max_attention_window = [
            LinearCacheType.RECURRENT_STATES.value if mamba_layer_mask[layer_idx] else max_seq_len
            for layer_idx in range(len(layer_mask))
        ]

        recurrent_states_window = LinearCacheType.RECURRENT_STATES.value
        local_windows = {
            recurrent_states_window if mamba_layer_mask[layer_idx] else max_seq_len
            for layer_idx in self.pp_layers
        }
        kwargs["pool_configurations"] = [
            PoolConfiguration(
                window_size=window_size,
                head_dim=head_dim,
                dtype=torch_dtype_to_binding(self.ssm_state_dtype)
                if window_size == recurrent_states_window
                else dtype,
            )
            for window_size in sorted(local_windows)
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
                    f"match total layers ({total_layers})"
                )
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

        self.host_block_offsets = torch.zeros(
            [self.impl.num_pools, self.max_batch_size, 2, self.max_blocks_per_seq],
            dtype=torch.int32,
            device="cpu",
        )
        self.recurrent_states_pool_index = self.kv_cache_pool_mapping[
            self.layer_offsets[self.mamba_pp_layers[0]]
        ][0]

        self.cuda_state_indices = torch.zeros(
            [self.max_batch_size], dtype=torch.int32, device="cuda"
        )
        self._host_state_indices = torch.zeros(
            [self.max_batch_size], dtype=torch.int32, pin_memory=prefer_pinned()
        )
        self._row_indices = torch.arange(self.max_batch_size, dtype=torch.long, device="cpu")
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
            commit_gdn_cached_replay_history_layers,
        )

        if (
            not self._use_gdn_cached_replay_all_layer_commit
            or num_decodes < CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
        ):
            return
        if (
            self.all_ssm_states is None
            or self.old_x is None
            or self.old_B is None
            or self.old_dt is None
            or self.replay_history_size is None
        ):
            raise RuntimeError("GDN cached replay all-layer commit requires replay state buffers.")

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.replay_num_decodes != num_decodes:
            raise RuntimeError(
                "GDN replay metadata contains "
                f"{mamba_metadata.replay_num_decodes} decode requests, "
                f"but state update received {num_decodes}."
            )
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

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata" = None,
        kv_cache_dtype_byte_size: float = None,
    ):
        super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    @nvtx_range("hybrid_prepare_resources")
    def _prepare_resources(self, scheduled_batch: ScheduledRequests):
        self.requests = scheduled_batch.context_requests + scheduled_batch.generation_requests
        # Issue all async block onboards; defer the syncTransfers() (refresh_blocks)
        # until just before forward so the CPU-side prep (attn metadata, input
        # tensor assembly, draft model prep, etc.) can overlap with the in-flight
        # cudaMemcpyAsync calls instead of being serialized behind them.
        # copy_linear_attention_block returns True iff it actually issued a copy;
        # we skip refresh_blocks entirely when nothing was scheduled.
        self._pending_state_transfers = self.impl.copy_linear_attention_block_batch(self.requests)
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
        accepted_leaf_positions: Optional[torch.Tensor] = None,
    ):
        if self.local_num_mamba_layers == 0:
            return
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        accepted_tokens = num_accepted_tokens[num_contexts : num_contexts + num_gens]
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
        state_indices_d = state_indices[num_contexts : num_contexts + num_gens].to(torch.int32)
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
            self._commit_gdn_cached_replay_history_layers(attn_metadata, num_gens)
            assert self._dummy_request_mask is not None
        else:
            # Legacy: copy the accepted SSM state from the intermediate buffer.
            _promote_mamba_state_triton(
                self.all_ssm_states,
                self.intermediate_ssm_states,
                src_state_indices,
                accepted_position_source,
                state_indices_d,
                position_source_is_token_count=position_source_is_token_count,
            )

        # Conv: both paths save all intermediate conv windows, carry over the
        # accepted one. The replay path also advances PNAT and the active cache
        # buffer in this same launch.
        replay_dummy_request_mask = None
        if self._use_replay_state_update:
            replay_dummy_request_mask = self._dummy_request_mask[
                num_contexts : num_contexts + num_gens
            ]
        _promote_mamba_state_triton(
            self.all_conv_states,
            self.intermediate_conv_states,
            src_state_indices,
            accepted_position_source,
            state_indices_d,
            num_accepted_tokens=accepted_tokens,
            position_source_is_token_count=position_source_is_token_count,
            replay_pnat=(self.prev_num_accepted_tokens if self._use_replay_state_update else None),
            replay_cache_buf_idx=(self.cache_buf_idx if self._use_replay_state_update else None),
            dummy_request_mask=replay_dummy_request_mask,
            replay_step_width=(self.replay_step_width if self._use_replay_state_update else 0),
            replay_history_size=(self.replay_history_size if self._use_replay_state_update else 0),
        )

    def get_num_available_tokens(
        self, token_num_upper_bound: int, max_num_draft_tokens: int = 0, **kwargs
    ) -> int:
        # Base bound from attention KV cache pool (the parent's behaviour).
        result = super().get_num_available_tokens(
            token_num_upper_bound, max_num_draft_tokens, **kwargs
        )
        # Also bound by the recurrent-state pool capacity: each request needs
        # roughly ceil(N / states_snapshot_interval) recurrent-state blocks
        # (+1 for the corner case where N is a multiple of tokens_per_block).
        # When block reuse is disabled, only one snapshot is needed per
        # request, so no additional capping is required here.
        interval = (
            self.linear_attention_metadata.states_snapshot_interval
            if self.linear_attention_metadata is not None
            else 0
        )
        # Attention-only PP ranks keep the interval so every rank publishes
        # identical scheduling boundaries, but they have no recurrent-state
        # pool whose capacity should constrain their attention KV cache.
        if self.local_num_mamba_layers > 0 and interval and interval > 0:
            stats = self.impl.get_kv_cache_stats()
            rs_free = stats.num_free_blocks_per_window_size.get(
                LinearCacheType.RECURRENT_STATES.value, 0
            )
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
                next_step = req.context_current_position - 1 + req.context_chunk_size
            else:
                next_step = req.prompt_len - 1
            block_indices.append(next_step // self.tokens_per_block)
        self.impl.copy_batch_block_offsets(
            self.host_block_offsets, [req.py_request_id for req in requests], 1, 0
        )
        max_blocks = self.blocks_per_window[LinearCacheType.RECURRENT_STATES.value][0]
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
            values = self.host_block_offsets[self.recurrent_states_pool_index, rows, 0, bi]
            invalid_mask = (values < 0) | (values >= max_blocks)
            # The C++ recurrent-state manager uses null page-table entries for
            # logical blocks that are not snapshot boundaries.  Usually a
            # context chunk ends exactly at an allocated snapshot (or at the
            # final live-state block), but the scheduler may shorten a chunk
            # further when KV capacity is tight.  In that case the Mamba
            # kernel must keep accumulating into the next allocated snapshot
            # or final block, matching copyLinearAttentionBlock(), which also
            # skips placeholders when it advances the live state.
            for bad_i in invalid_mask.nonzero(as_tuple=False).flatten().tolist():
                req = requests[bad_i]
                if req.is_context_finished:
                    continue
                last_prompt_block = (req.prompt_len - 1) // self.tokens_per_block
                row = self.host_block_offsets[self.recurrent_states_pool_index, rows[bad_i], 0]
                candidates = row[block_indices[bad_i] : last_prompt_block + 1]
                valid_candidates = ((candidates >= 0) & (candidates < max_blocks)).nonzero(
                    as_tuple=False
                )
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

        self.cuda_state_indices.copy_(self._host_state_indices, non_blocking=True)
        is_dummy = [req.is_dummy for req in requests]
        self._refresh_dummy_request_mask(is_dummy)

        # Build request_id → pool block offset mapping so that
        # get_state_indices can return indices in arbitrary request order.
        # Bulk tolist avoids a per-request tensor-index + .item() round-trip.
        state_values = self._host_state_indices[:n].tolist()
        for req, value, dummy in zip(requests, state_values, is_dummy):
            self._request_id_to_state_index[req.py_request_id] = value
            self._request_id_to_is_dummy[req.py_request_id] = dummy

    def get_state_indices(
        self, request_ids: Optional[List[int]] = None, is_padding: Optional[List[bool]] = None
    ) -> list:
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
            indices = [self._request_id_to_state_index[rid] for rid in request_ids]
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
        pool: torch.Tensor = (
            self.impl.get_recurrent_states_pool()
            .view(torch.uint8)
            .reshape(self.local_num_mamba_layers, -1, self.ssm_bytes + self.conv_bytes)
        )
        num_blocks_in_pool = pool.shape[1]
        self.all_ssm_states = (
            pool[:, :, : self.ssm_bytes]
            .view(self.ssm_state_dtype)
            .view([self.local_num_mamba_layers, num_blocks_in_pool] + self.ssm_state_shape)
        )
        self.all_conv_states = (
            pool[:, :, self.ssm_bytes : self.ssm_bytes + self.conv_bytes]
            .view(self.conv_state_dtype)
            .view([self.local_num_mamba_layers, num_blocks_in_pool] + self.conv_state_shape)
        )
        self.all_ssm_states.zero_()
        self.all_conv_states.zero_()

    def _setup_replay_buffers(self, spec_config) -> None:
        cache_size = self.all_ssm_states.shape[1]
        device = self.all_ssm_states.device
        if not self._allocate_pool_replay_buffers(spec_config, cache_size, device):
            return

        self._dummy_request_mask = torch.zeros(self.max_batch_size, dtype=torch.bool, device=device)
        self._dummy_request_mask_host = torch.zeros(
            self.max_batch_size,
            dtype=torch.bool,
            pin_memory=prefer_pinned(),
        )
