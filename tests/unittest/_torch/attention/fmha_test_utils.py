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

from types import SimpleNamespace
from typing import Callable

import torch

from tensorrt_llm._torch.attention.backends.features import HelixMetadata
from tensorrt_llm._torch.attention.backends.features.spec_decoding import SpecDecodingMetadata
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs


class FakeAttention:
    def __init__(self, local_layer_idx: int = 0) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = 4
        self.num_heads = 1
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.has_fp8_kv_cache = False
        self.skip_correction_threshold = 0.0
        self.local_layer_idx = local_layer_idx


class FakePhasedFmha(PhasedFmha):
    def __init__(
        self,
        attn: FakeAttention,
        supported_phases: set[FmhaPhase | None],
        name: str,
        events: list[tuple],
        workspace_size: int = 0,
        support_predicate: Callable[[object, FmhaPhase | None], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._supported_phases = supported_phases
        self._name = name
        self._events = events
        self._workspace_size = workspace_size
        self._support_predicate = support_predicate

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return phase in self._supported_phases and (
            self._support_predicate is None or self._support_predicate(metadata, phase)
        )

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        self._events.append(("prepare", self._name))
        if workspace.numel() < self._workspace_size:
            workspace.resize_(self._workspace_size)

    def run_context(self, params: FmhaParams) -> None:
        self._events.append(
            (
                "run",
                self._name,
                FmhaPhase.CONTEXT,
                params.num_tokens,
                params.batch_size,
                params.num_requests,
            )
        )

    def run_generation(self, params: FmhaParams) -> None:
        self._events.append(
            (
                "run",
                self._name,
                FmhaPhase.GENERATION,
                params.num_tokens,
                params.batch_size,
                params.num_requests,
            )
        )


class FakeFmha(Fmha):
    def __init__(
        self,
        attn: FakeAttention,
        name: str,
        events: list[tuple],
        support_predicate: Callable[[AttentionForwardArgs], bool] | None = None,
        request_support_predicate: Callable[[torch.Tensor, object], bool] | None = None,
    ) -> None:
        super().__init__(attn)
        self._name = name
        self._events = events
        self._support_predicate = support_predicate
        self._request_support_predicate = request_support_predicate

    def _is_supported(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
        *,
        phase: FmhaPhase | None = None,
    ) -> bool:
        self._events.append(("support", self._name, phase))
        return (self._support_predicate is None or self._support_predicate(forward_args)) and (
            self._request_support_predicate is None or self._request_support_predicate(q, metadata)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: object,
        forward_args: AttentionForwardArgs,
    ) -> None:
        self._events.append(("forward", self._name))


_SPEC_FIELDS = (
    "is_spec_decoding_enabled",
    "use_spec_decoding",
    "is_spec_dec_tree",
    "is_spec_dec_dynamic_tree",
    "max_total_draft_tokens",
)
_HELIX_FIELDS = {
    "helix_position_offsets": "position_offsets",
    "helix_is_inactive_rank": "is_inactive_rank",
    "helix_kv_bounds": "kv_bounds",
    "helix_local_slots": "local_slots",
    "_helix_spec_tokens_valid": "spec_tokens_valid",
}


def metadata_stub(**fields) -> SimpleNamespace:
    """A duck-typed metadata serving its spec-dec and Helix fields through extensions.

    The spec-dec extension is always attached, as on TRTLLM metadata; Helix only
    when one of its fields is set.
    """
    spec = SpecDecodingMetadata(None)
    for name in _SPEC_FIELDS:
        if name in fields:
            setattr(spec, name, fields.pop(name))
    extensions = {SpecDecodingMetadata: spec}
    helix_fields = {
        _HELIX_FIELDS[name]: fields.pop(name) for name in list(fields) if name in _HELIX_FIELDS
    }
    if any(value is not None and value is not False for value in helix_fields.values()):
        helix = HelixMetadata(None)
        for name, value in helix_fields.items():
            setattr(helix, name, value)
        extensions[HelixMetadata] = helix
    metadata = SimpleNamespace(**fields)
    metadata.find_ext = extensions.get
    metadata.ext = extensions.__getitem__
    return metadata
