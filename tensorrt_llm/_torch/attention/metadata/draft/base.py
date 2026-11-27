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
"""Base of the draft metadata of one-model speculative decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from ..base import Metadata, build_metadata
from ..core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from ..batch_inputs import BatchInputs


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


def build_draft_set(metadata: DraftMetadata, config: "BatchInputs") -> TrtllmCoreMetadata:
    """The types attached alongside metadata, built from config: the core first."""
    draft_types = [
        metadata_type
        for metadata_type in metadata.attached_types()
        if not issubclass(metadata_type, DraftMetadata)
    ]
    draft_types.sort(key=lambda metadata_type: not issubclass(metadata_type, TrtllmCoreMetadata))
    return build_metadata(draft_types, config)
