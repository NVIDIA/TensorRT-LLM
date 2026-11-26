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
"""Rows and lengths of the draft steps of a linear chain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..base import UnsupportedMetadataConfiguration, _Registry
from ..core import TrtllmCoreMetadata
from .base import DraftMetadata

if TYPE_CHECKING:
    from ..batch_inputs import BatchInputs


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
        draft = self.draft_set
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
