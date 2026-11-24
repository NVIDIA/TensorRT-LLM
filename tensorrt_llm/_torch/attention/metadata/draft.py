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
"""Draft loop of one-model speculative decoding."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional

import torch

from .base import Metadata, UnsupportedMetadataConfiguration, _Registry, build_metadata
from .core import TrtllmCoreMetadata
from .sparse import SparseMetadata
from .spec_decoding import SpecDecodingMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs


class DraftMetadata(Metadata):
    """Base of what the draft loop of one-model speculative decoding reads.

    The spec worker calls four hooks around the loop, each through its
    ``_all`` form, which calls that hook on every attached draft metadata,
    dependencies first. A hook updates what its own metadata is responsible
    for and calls no other hook. The target's metadata are never modified by
    the loop.
    """

    requires = (TrtllmCoreMetadata,)

    def begin_draft_loop(self, num_accepted: torch.Tensor) -> None:
        """Before the first draft forward, once the acceptance result is known.

        num_accepted holds, on the device, the tokens each request
        accepted in the target step, [num_seqs].
        """

    def begin_draft_step(self, step: int) -> None:
        """Before draft forward step."""

    def end_draft_step(self, step: int) -> None:
        """After draft forward step and its sampling."""

    def end_draft_loop(self) -> None:
        """After the last draft forward."""

    def begin_draft_loop_all(self, num_accepted: torch.Tensor) -> None:
        """Call begin_draft_loop on every attached draft metadata."""
        for metadata in self._attached_draft_metadata():
            metadata.begin_draft_loop(num_accepted)

    def begin_draft_step_all(self, step: int) -> None:
        """Call begin_draft_step on every attached draft metadata."""
        for metadata in self._attached_draft_metadata():
            metadata.begin_draft_step(step)

    def end_draft_step_all(self, step: int) -> None:
        """Call end_draft_step on every attached draft metadata, in reverse order."""
        for metadata in reversed(self._attached_draft_metadata()):
            metadata.end_draft_step(step)

    def end_draft_loop_all(self) -> None:
        """Call end_draft_loop on every attached draft metadata, in reverse order."""
        for metadata in reversed(self._attached_draft_metadata()):
            metadata.end_draft_loop()

    def _attached_draft_metadata(self) -> List["DraftMetadata"]:
        return [
            metadata for metadata in self._registry.ordered if isinstance(metadata, DraftMetadata)
        ]


def _build_draft_set(metadata: DraftMetadata, config: "BatchInputs") -> TrtllmCoreMetadata:
    """The types attached alongside metadata, built from config: the core first."""
    draft_types = [
        metadata_type
        for metadata_type in metadata.attached_types()
        if not issubclass(metadata_type, DraftMetadata)
    ]
    draft_types.sort(key=lambda metadata_type: not issubclass(metadata_type, TrtllmCoreMetadata))
    return build_metadata(draft_types, config)


class LinearDraftMetadata(DraftMetadata):
    """The metadata of a linear draft chain of the target's own layers over a separate draft KV cache (MTP-Eagle).

    The draft layers run the attention backend on a second set of metadata,
    draft: the target's types, built over the draft KV cache. The engine
    prepares it with each target step that drafts; the other draft metadata
    update it in place for each draft step.
    """

    # Constructed after the other metadata, whose types the draft set copies.
    builds_metadata = True

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is None or draft_manager is config.kv_cache_manager:
            raise ValueError("LinearDraftMetadata requires a separate draft KV cache")
        # The core of the metadata the draft layers receive as attn_metadata.
        self.draft = _build_draft_set(
            self,
            dataclasses.replace(
                config,
                kv_cache_manager=draft_manager,
                draft_kv_cache_manager=None,
                buffer_name_prefix="draft_",
            ),
        )


class SharedKVCacheLinearDraftMetadata(DraftMetadata):
    """The metadata of a linear draft chain of the target's own layers over the target's KV cache (MTP-Eagle).

    The draft layers are extra layers of the target's KV cache. They run the
    attention backend on a second set of metadata, draft: the target's types,
    built over that same cache with lengths of their own, so the draft steps
    leave the target's metadata untouched. The engine prepares it with each
    target step that drafts; the other draft metadata update it in place for
    each draft step.
    """

    # Constructed after the other metadata, whose types the draft set copies.
    builds_metadata = True

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is not None and draft_manager is not config.kv_cache_manager:
            raise ValueError("SharedKVCacheLinearDraftMetadata requires a shared draft KV cache")
        # The core of the metadata the draft layers receive as attn_metadata.
        self.draft = _build_draft_set(
            self,
            dataclasses.replace(config, draft_kv_cache_manager=None, buffer_name_prefix="draft_"),
        )


class DenseLinearDraftMetadata(DraftMetadata):
    """The metadata of a linear draft chain of a dense draft model over a separate draft KV cache (Eagle3).

    The draft layers are dense attention layers, whatever the target
    attaches. They run the attention backend on a second set of metadata,
    draft: the dense types, built over the draft KV cache. The engine
    prepares it with each target step that drafts; the other draft metadata
    update it in place for each draft step.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is None or draft_manager is config.kv_cache_manager:
            raise ValueError("DenseLinearDraftMetadata requires a separate draft KV cache")
        # The core of the metadata the draft layers receive as attn_metadata.
        self.draft = build_metadata(
            [TrtllmCoreMetadata, SpecDecodingMetadata, SparseMetadata],
            dataclasses.replace(
                config,
                kv_cache_manager=draft_manager,
                draft_kv_cache_manager=None,
                sparse_metadata_params=None,
                buffer_name_prefix="draft_",
            ),
        )


class SharedKVCacheDenseLinearDraftMetadata(DraftMetadata):
    """The metadata of a linear draft chain of a dense draft model over the target's KV cache (Eagle3).

    The draft layers are dense attention layers, whatever the target
    attaches, and extra layers of the target's KV cache. They run the
    attention backend on a second set of metadata, draft: the dense types,
    built over that same cache with lengths of their own, so the draft steps
    leave the target's metadata untouched. The engine prepares it with each
    target step that drafts; the other draft metadata update it in place for
    each draft step.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is not None and draft_manager is not config.kv_cache_manager:
            raise ValueError(
                "SharedKVCacheDenseLinearDraftMetadata requires a shared draft KV cache"
            )
        # The core of the metadata the draft layers receive as attn_metadata.
        self.draft = build_metadata(
            [TrtllmCoreMetadata, SpecDecodingMetadata, SparseMetadata],
            dataclasses.replace(
                config,
                draft_kv_cache_manager=None,
                sparse_metadata_params=None,
                buffer_name_prefix="draft_",
            ),
        )


class LinearDraftLengthsMetadata(DraftMetadata):
    """Rows and lengths of the draft steps of a linear chain.

    Step 0 runs the target step's rows. Each later step runs one token per
    sequence, every row as a generation row, over a KV cache rewound to the
    accepted tokens. Before each step this metadata writes those rows and
    lengths into the core of the draft set, on the device only, so the loop
    can run inside a captured CUDA graph.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        num_rows = config.max_num_sequences
        # KV-length offsets of the steps after the first: the target step's
        # own offsets plus the rewind to the accepted tokens.
        self.rewind_kv_lens_offsets = self.get_buffer(
            (num_rows,), torch.int, "rewind_kv_lens_offsets"
        )
        self._one_token_per_seq = torch.ones((num_rows,), dtype=torch.int)
        self._target_inputs: Optional["BatchInputs"] = None

    def prepare(self, inputs: "BatchInputs") -> None:
        # The target's latest layout, also after a device-only prepare.
        self._target_inputs = inputs

    def begin_draft_loop(self, num_accepted: torch.Tensor) -> None:
        target = self._target_inputs
        if target.spec.num_extend_ctx_requests > 0:
            raise UnsupportedMetadataConfiguration("a draft loop after extend-context rows")
        num_seqs = target.num_seqs
        num_contexts = target.num_contexts
        core = self.get_metadata(TrtllmCoreMetadata)
        rewind = self.rewind_kv_lens_offsets
        # A context request appends its first draft token. A generation
        # request keeps its accepted tokens and appends one.
        rewind[:num_contexts] = 1
        rewind[num_contexts:num_seqs] = (
            num_accepted[num_contexts:num_seqs] - core.base_seq_lens_cuda[num_contexts:num_seqs] + 1
        )
        if target.kv_lens_offsets is not None:
            rewind[num_contexts:num_seqs] += target.kv_lens_offsets[: num_seqs - num_contexts]

    def begin_draft_step(self, step: int) -> None:
        target = self._target_inputs
        draft = self.draft
        num_seqs = target.num_seqs
        if step == 0:
            num_contexts = target.num_contexts
            draft._set_rows(target.seq_lens, num_contexts)
            draft.seq_lens_cuda.copy_(draft.base_seq_lens_cuda[:num_seqs])
        else:
            num_contexts = 0
            draft._set_rows(self._one_token_per_seq[:num_seqs], num_contexts)
            draft.seq_lens_cuda.fill_(1)
        draft.invalidate_step_caches()
        # Recomputed from the staged lengths at every step, so nothing accumulates.
        kv_lens_cuda = draft.kv_lens_cuda[:num_seqs]
        kv_lens_cuda.copy_(draft.base_kv_lens_cuda[:num_seqs])
        if step == 0:
            if target.kv_lens_offsets is not None and num_seqs > num_contexts:
                kv_lens_cuda[num_contexts:] += target.kv_lens_offsets[: num_seqs - num_contexts]
        else:
            kv_lens_cuda += self.rewind_kv_lens_offsets[:num_seqs]
            if step > 1:
                kv_lens_cuda += step - 1
        draft.host_request_types[:num_contexts].fill_(0)
        draft.host_request_types[num_contexts:num_seqs].fill_(1)


class EmbeddedDSparkDraftMetadata(DraftMetadata):
    """The DSpark drafter embedded in its target model.

    The drafter runs once per target step, on the generation rows, with its
    own attention over a rolling window the spec worker keeps. It reads no
    attention metadata and has no draft KV cache, so this mode holds nothing
    and its hooks do nothing.
    """


class StandaloneDSparkDraftMetadata(DraftMetadata):
    """A standalone DSpark drafter over a draft KV cache of its own.

    The drafter runs once per target step, on the generation rows, with its
    own attention over pages of the draft KV cache. It addresses those pages
    through the block offsets held here, in the target step's row order, and
    takes positions and lengths from the spec worker's own state.
    """

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        num_pools = getattr(draft_manager, "num_attention_op_pools", draft_manager.num_pools)
        # Block offsets of the step's rows in the draft KV cache,
        # [num_pools, num_rows, 2, max_blocks_per_seq].
        self.draft_kv_cache_block_offsets = self.get_buffer(
            (num_pools, config.max_num_sequences, 2, draft_manager.max_blocks_per_seq),
            torch.int32,
            "draft_kv_cache_block_offsets",
        )

    def prepare(self, inputs: "BatchInputs") -> None:
        if not inputs.stage_host:
            return
        self.config.draft_kv_cache_manager.copy_batch_block_offsets(
            self.draft_kv_cache_block_offsets,
            inputs.request_ids,
            inputs.beam_width,
            inputs.num_contexts,
            inputs.num_seqs,
        )


class SuffixAutomatonDraftMetadata(DraftMetadata):
    """Drafting from a suffix automaton over the tokens generated so far.

    The draft is a pattern match, not a model forward: it reads no attention
    metadata and has no draft KV cache, so this mode holds nothing and its
    hooks do nothing.
    """


class SharedKVCacheStandaloneDSparkDraftMetadata(DraftMetadata):
    """A standalone DSpark drafter without a draft KV cache of its own.

    The drafter runs once per target step, on the generation rows, with its
    own attention. Its layers are extra layers of the target's KV cache, which
    it addresses through the target's block offsets, or it keeps its context
    in a buffer of the spec worker. Either way it reads nothing that the
    target's metadata do not already hold, so this mode holds nothing.
    """
