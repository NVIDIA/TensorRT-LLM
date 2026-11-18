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
"""CSA2 sparse attention (DeepSeek-V4.1): the core of its backend and its request state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import torch

from ..backends.sparse.csa2.decoder_replay import apply_decoder_swa_floor
from ..backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from ..backends.sparse.csa2.params import CSA2MetadataParams
from .base import _Registry
from .core import TrtllmCoreMetadata
from .sparse import SparseMetadata

if TYPE_CHECKING:
    from ..backends.sparse.csa2.decoder_replay import DecoderReplayPlan
    from .batch_inputs import BatchInputs


@dataclass(frozen=True)
class CSA2StepInputs:
    """What a CSA2 step needs beyond the batch: ``BatchInputs.backend_inputs``."""

    # The step's context requests, in row order; their replay plans decide
    # which sliding-window rows the step reads and writes.
    context_requests: Sequence[Any] = ()
    # Whether a context chunk may keep only the final window of its history.
    allow_final_window: bool = True


class CSA2CoreMetadata(TrtllmCoreMetadata):
    """The core of the CSA2 attention backend.

    The CSA2 cache manager publishes the sliding-window page tables it
    converts while the block offsets are copied, and the CSA2 request state
    reuses them. A step is therefore prepared inside the manager's publication
    scope, which this core opens around the whole set.
    """

    # CSA2 attends over its own staged selections; the MLA context tables of
    # the dense path are not built.
    enable_context_mla_with_cached_kv = False

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        # The sliding-window tables published while this step is prepared.
        self.swa_publication = None

    def create_cuda_graph_metadata(self, *args, **kwargs) -> "CSA2CoreMetadata":
        graph_core = super().create_cuda_graph_metadata(*args, **kwargs)
        if graph_core is not self:
            self.get_metadata(CSA2Metadata).share_state_with(graph_core.get_metadata(CSA2Metadata))
        return graph_core

    def prepare_all(self, inputs: "BatchInputs") -> None:
        if not inputs.stage_host:
            super().prepare_all(inputs)
            return
        with self.kv_cache_manager._swa_publication_scope() as publication:
            self.swa_publication = publication
            try:
                super().prepare_all(inputs)
            finally:
                self.swa_publication = None

    def get_adp_token_counts(self) -> List[int]:
        """Input rows, decoder rows, and whether this rank's replay plan must run."""
        plan = self.decoder_replay_plan
        input_tokens = self.padded_num_tokens or self.num_tokens
        decoder_tokens = (
            plan.num_replay_tokens if plan and plan.replays_local_tokens else input_tokens
        )
        return [input_tokens, decoder_tokens, int(plan is not None and plan.updates_local_metadata)]

    def set_adp_token_counts(self, counts: List[List[int]]) -> List[int]:
        """Keep or drop the replay plan by one decision every attention-DP rank shares."""
        input_counts, decoder_counts, required = counts
        plan = self.decoder_replay_plan
        if plan is None:
            if any(required):
                raise ValueError(
                    "An attention-DP peer enters the decoder replay but this rank has no "
                    "replay plan; the decoder-half collectives would diverge"
                )
            return input_counts
        plan.saved_all_rank_num_tokens = input_counts
        plan.replay_all_rank_num_tokens = decoder_counts
        if not any(required) and input_counts == decoder_counts:
            self.decoder_replay_plan = None
        return input_counts


# What CSA2TrtllmMetadata defines that the request state does not take over:
# the hooks of the old metadata, what the core holds, and the dataclass plumbing.
_NOT_TAKEN = frozenset(
    {
        "prepare",
        "on_update_kv_lens",
        "prepare_context_mla_with_cached_kv",
        "prepare_for_draft_forward",
        "restore_after_draft_forward",
        "get_adp_token_counts",
        "set_adp_token_counts",
        "tokens_per_block",
        "host_kv_cache_pool_pointers",
        "host_kv_cache_pool_mapping",
        "for_query_tile",
        "_bind_runtime_views",
        "_prepare_deferred",
    }
)


class CSA2Metadata(SparseMetadata):
    """The CSA2 request state: page tables, slots, indexer plans and routing.

    Its logic is CSA2TrtllmMetadata's, taken over method by method; the
    lengths and the cache layout it reads resolve to the core. The query
    tiles the attention kernel runs on stay CSA2TrtllmMetadata objects.
    """

    requires = (CSA2CoreMetadata,)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        params = config.sparse_metadata_params
        if not isinstance(params, CSA2MetadataParams):
            raise ValueError("CSA2 sparse attention metadata params are not set")
        self.indexer_max_chunk_size = params.indexer_max_chunk_size
        self.indexer_q_split_threshold = params.q_split_threshold
        self.decoder_replay_plan = None
        self.decoder_context_ends = None
        self.csa2_decoder_capture_lens = None
        self.csa2_replay_mode = None
        # Set per step by the engine of disaggregated serving, which the
        # feature-split metadata do not cover.
        self.csa2_remote_tail_mode = None
        self.csa2_remote_tail_starts: List[int] = []
        self._csa2_steady_snapshot = None
        self._csa2_ready_for_kv_update = False
        # The step's description while the decoder layers run on replayed rows.
        self._csa2_step_before_replay = None

    def share_state_with(self, graph_metadata: "CSA2Metadata") -> None:
        """Start the metadata of a CUDA graph batch size from this request state.

        The graph metadata of every batch size then share the buffer banks,
        query tiles and indexer arenas created here, as they reserve graph
        memory once.
        """
        for name, value in vars(self).items():
            if name not in ("_registry", "config"):
                setattr(graph_metadata, name, value)
        # A graph metadata prepares for itself.
        graph_metadata._csa2_deferred_decode_outputs = False
        graph_metadata._csa2_steady_snapshot = None
        graph_metadata.decoder_replay_plan = None
        graph_metadata.decoder_context_ends = None

    def prepare(self, inputs: "BatchInputs") -> None:
        if inputs.stage_host:
            step = inputs.backend_inputs
            if step is not None:
                self.prepare_context_replay(
                    list(step.context_requests), allow_final_window=step.allow_final_window
                )
            self._csa2_deferred_decode_outputs = False
            self._csa2_context_geometry = None
            self._csa2_ready_for_kv_update = False
            # The whole request state is prepared here; nothing is left to
            # the first CSA2 layer.
            self._csa2_pending_prepare = None
            self.prepare_csa2(swa_publication=self.swa_publication)
        if self._csa2_ready_for_kv_update:
            self._refresh_live_endpoints()

    def enter_decoder_replay(self, plan: "DecoderReplayPlan", first_layer: Optional[int]) -> None:
        """Re-describe the step for the decoder layers, which run on the replayed rows.

        Called inside the forward, between the encoder and the decoder layers.
        The replayed rows are a second description of the same step, so the
        whole set is prepared again with it; what the encoder layers published
        for the decoder is carried across, narrowed to the replayed rows.
        ``exit_decoder_replay`` restores the step's rows.
        """
        if plan.replay_all_rank_num_tokens is not None:
            self.all_rank_num_tokens = plan.replay_all_rank_num_tokens
        if not plan.updates_local_metadata:
            return
        core = self.get_metadata(CSA2CoreMetadata)
        step = core.inputs
        self._csa2_step_before_replay = step
        saved_indices = dict(self.csa2_indices)
        saved_candidates = {
            name: dict(getattr(self, name))
            for name in ("csa2_candidates", "csa2_candidate_blocks", "csa2_candidate_counts")
        }
        saved_last_layer = self._csa2_last_layer
        precomputed = set(self.csa2_precomputed_kv_layers)
        num_cached = self.kv_cache_params.num_cached_tokens_per_seq
        num_cached[: len(plan.replay_num_cached)] = plan.replay_num_cached
        self.prompt_lens = plan.replay_prompt_lens
        core._set_rows(plan.replay_seq_lens, core.num_contexts)
        if plan.num_replay_tokens == 0:
            # No local attention runs: keep what was published and the live
            # KV lengths, and prepare nothing for an empty query batch.
            self.reset_routing()
            self.csa2_precomputed_kv_layers = precomputed
            self.csa2_replay_query_rows = plan.rows
            return
        if first_layer is not None:
            self.set_decoder_query_boundary(first_layer)
        padding = step.padding
        if padding is not None and plan.replay_all_rank_num_tokens is not None:
            padding = replace(padding, all_rank_num_tokens=plan.replay_all_rank_num_tokens)
        # The step's device corrections of the KV lengths stay in the
        # description, so the replayed rows keep their live endpoints.
        core.prepare_all(
            replace(
                step,
                seq_lens=plan.replay_seq_lens,
                prompt_lens=plan.replay_prompt_lens,
                padding=padding,
                backend_inputs=None,
                stage_host=True,
            )
        )
        self.csa2_precomputed_kv_layers = precomputed
        self.csa2_replay_query_rows = plan.rows
        for layer, indices in saved_indices.items():
            self.csa2_indices[layer] = indices.index_select(0, plan.rows)
        for name, published in saved_candidates.items():
            target = getattr(self, name)
            for layer, value in published.items():
                target[layer] = value.index_select(0, plan.rows)
        self._csa2_last_layer = saved_last_layer
        # Only the sliding window of the context rows is truncated. Generation
        # rows and the global visibility keep their positions and history.
        apply_decoder_swa_floor(
            self,
            self._copy_csa2_tensor(
                "decoder_starts", torch.tensor(plan.swa_floors, dtype=torch.int64, device="cpu")
            ),
        )

    def exit_decoder_replay(self, plan: "DecoderReplayPlan") -> None:
        """Restore the rows of the step ``enter_decoder_replay`` re-described.

        Only the host description is restored. The device buffers keep the
        replay values: the next step prepares them again.
        """
        if plan.saved_all_rank_num_tokens is not None:
            self.all_rank_num_tokens = plan.saved_all_rank_num_tokens
        if not plan.updates_local_metadata:
            return
        core = self.get_metadata(CSA2CoreMetadata)
        num_cached = self.kv_cache_params.num_cached_tokens_per_seq
        num_cached[: len(plan.saved_num_cached)] = plan.saved_num_cached
        self.prompt_lens = plan.saved_prompt_lens
        core._set_rows(plan.saved_seq_lens, core.num_contexts)
        core.inputs = self._csa2_step_before_replay
        self._csa2_step_before_replay = None
        self.csa2_replay_query_rows = None
        # What was published is indexed by the replayed rows.
        self.reset_routing()

    @staticmethod
    def for_query_tile(*args, **kwargs) -> CSA2TrtllmMetadata:
        """A compute tile for the attention kernel; it is not request metadata."""
        return CSA2TrtllmMetadata.for_query_tile(*args, **kwargs)


def _take_over_request_state() -> None:
    """Give CSA2Metadata the request-state logic and defaults of CSA2TrtllmMetadata."""
    # FIXME: move these functions from CSA2TrtllmMetadata into CSA2Metadata once the
    # feature-split metadata is the only path, and build the query tiles from them.
    for name, value in vars(CSA2TrtllmMetadata).items():
        if name.startswith("__") or name in _NOT_TAKEN or name in vars(CSA2Metadata):
            continue
        # Methods and properties, and the class-level defaults every instance starts from.
        setattr(CSA2Metadata, name, value)


_take_over_request_state()
