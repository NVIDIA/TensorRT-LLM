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
"""Draft loop of one-model speculative decoding (Eagle3 / MTP-Eagle)."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

from .base import Metadata, UnsupportedMetadataConfiguration, _Registry, build_metadata
from .batch_inputs import DraftStepInputs
from .core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs


class DraftMetadata(Metadata):
    """The draft steps' metadata, over the draft KV cache.

    draft is the core of a second set of metadata: the target's types,
    built over the draft KV cache. The draft layers receive it as
    attn_metadata; the target's metadata are never modified by the loop.

    Step 0 runs the target step's rows. Each later step runs one token per
    sequence, every row as a generation row. The spec worker calls the four
    hooks around the loop; each step is prepared on the device only, so the
    loop can run inside a captured CUDA graph.
    """

    requires = (TrtllmCoreMetadata,)
    # Constructed after the other metadata, whose types the draft set copies.
    builds_metadata = True

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        draft_manager = config.draft_kv_cache_manager
        if draft_manager is None or draft_manager is config.kv_cache_manager:
            raise UnsupportedMetadataConfiguration("a shared draft KV cache")
        draft_types = [
            metadata_type
            for metadata_type in self.attached_types()
            if not issubclass(metadata_type, DraftMetadata)
        ]
        draft_types.sort(
            key=lambda metadata_type: not issubclass(metadata_type, TrtllmCoreMetadata)
        )
        self.draft: TrtllmCoreMetadata = build_metadata(
            draft_types,
            dataclasses.replace(
                config,
                kv_cache_manager=draft_manager,
                draft_kv_cache_manager=None,
                buffer_name_prefix="draft_",
            ),
        )
        num_rows = config.max_num_sequences
        # KV-length offsets of the steps after the first: the target step's
        # own offsets plus the rewind to the accepted tokens.
        self.rewind_kv_lens_offsets = self.get_buffer(
            (num_rows,), torch.int, "rewind_kv_lens_offsets"
        )
        self._one_token_per_seq = torch.ones((num_rows,), dtype=torch.int)
        self._target_inputs: Optional["BatchInputs"] = None
        self._num_accepted: Optional[torch.Tensor] = None

    def prepare(self, inputs: "BatchInputs") -> None:
        # The target's latest layout, also after a device-only prepare.
        self._target_inputs = inputs
        if not inputs.stage_host:
            return
        if inputs.spec is not None and inputs.spec.runtime_draft_len > 0:
            self.draft.prepare_all(inputs)

    def begin_draft_loop(self, num_accepted: torch.Tensor) -> None:
        """Before the first draft forward, once the acceptance result is known.

        num_accepted holds, on the device, the tokens each request
        accepted in the target step, [num_seqs].
        """
        self._num_accepted = num_accepted
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
        """Before draft forward step: prepare the draft metadata for it."""
        target = self._target_inputs
        spec = dataclasses.replace(
            target.spec, draft=DraftStepInputs(step=step, num_accepted=self._num_accepted)
        )
        if step == 0:
            inputs = dataclasses.replace(target, spec=spec, stage_host=False)
        else:
            inputs = dataclasses.replace(
                target,
                seq_lens=self._one_token_per_seq[: target.num_seqs],
                num_contexts=0,
                spec=spec,
                kv_lens_offsets=self.rewind_kv_lens_offsets,
                kv_lens_offset_delta=step - 1,
                stage_host=False,
            )
        self.draft.prepare_all(inputs)

    def end_draft_step(self, step: int) -> None:
        """After draft forward step and its sampling."""

    def end_draft_loop(self) -> None:
        """After the last draft forward."""
