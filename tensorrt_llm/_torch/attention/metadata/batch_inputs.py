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
"""The per-step input of every attention metadata ``prepare``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Literal, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SpeculativeDecodingMode
    from tensorrt_llm.mapping import Mapping

    from ...metadata import KVCacheParams
    from ..backends.interface import AttentionRuntimeFeatures
    from ..backends.sparse.params import SparseMetadataParams


@dataclass(frozen=True)
class TokenPadding:
    """Token counts of attention data parallelism.

    ``all_rank_num_seqs`` is set with one-model speculation, whose later draft
    steps run one token per sequence.
    """

    padded_num_tokens: Optional[int] = None
    all_rank_num_tokens: Optional[List[int]] = None
    all_rank_num_seqs: Optional[List[int]] = None


@dataclass(frozen=True)
class DraftStepInputs:
    """One step of the one-model draft loop."""

    step: int
    # Tokens each request accepted in the target step: device tensor at a
    # fixed address, [num_seqs].
    num_accepted: torch.Tensor


@dataclass(frozen=True)
class SpecStepInputs:
    """Speculative decoding state of the step."""

    spec_dec_mode: "SpeculativeDecodingMode"
    # Draft tokens verified per generation request in this step.
    runtime_draft_len: int
    max_draft_len: int
    max_total_draft_tokens: int
    # Whether the attention kernels run in multi-query spec-dec mode.
    attention_spec_dec_mode: bool = False
    is_spec_dec_tree: bool = False
    is_spec_dec_dynamic_tree: bool = False
    # Query tokens of each generation request; the verify layout is uniform.
    tokens_per_gen_step: int = 1
    # Generation requests that run as the trailing context rows of the step.
    num_extend_ctx_requests: int = 0
    # Set only on a draft step.
    draft: Optional[DraftStepInputs] = None


@dataclass(frozen=True)
class BatchInputs:
    """Frozen description of one decoder step.

    The metadata are constructed from the first description, which sizes
    their buffers, and prepared with one description per step. The engine
    fields below the step fields are the same in every description.
    """

    # The step. Query length of each sequence, context requests first: host
    # int tensor [num_seqs].
    seq_lens: torch.Tensor
    num_contexts: int
    request_ids: List[int]
    prompt_lens: List[int]
    kv_cache_params: "KVCacheParams"
    beam_width: int = 1
    padding: Optional[TokenPadding] = None
    spec: Optional[SpecStepInputs] = None
    # The runner's TRTLLM-Gen FMHA JIT warmup pass.
    trtllm_gen_jit_warmup: bool = False
    # Device-side correction of the KV lengths: a device tensor added to the
    # extend rows when the step has any, otherwise to the generation rows
    # (indexed from 0 over those rows), plus a host scalar.
    kv_lens_offsets: Optional[torch.Tensor] = None
    kv_lens_offset_delta: int = 0
    # Whether prepare copies this step's host values into the device buffers.
    # False inside a forward (possibly inside a captured CUDA graph): prepare
    # then only derives on the device, from the buffers staged before.
    stage_host: bool = True

    # The engine.
    max_num_requests: int = 0
    max_num_tokens: int = 0
    max_num_sequences: int = 0
    kv_cache_manager: Any = None
    draft_kv_cache_manager: Any = None
    mapping: Optional["Mapping"] = None
    runtime_features: Optional["AttentionRuntimeFeatures"] = None
    cache_indirection: Optional[torch.Tensor] = None
    max_num_heads_per_kv: int = 1
    sparse_metadata_params: Optional["SparseMetadataParams"] = None
    enable_context_mla_with_cached_kv: bool = False
    # Draft tokens verified per generation request, at most.
    max_draft_tokens: int = 0
    kv_layout: Literal["NHD", "HND"] = "HND"
    # A CUDA graph captures the addresses of the metadata buffers, which
    # then come from ``cuda_graph_buffers`` and are never reallocated.
    is_cuda_graph: bool = False
    cuda_graph_buffers: Any = None
    # Prefix of the buffer names in ``cuda_graph_buffers``, which shares
    # buffers of one name between the CUDA graph batch sizes.
    buffer_name_prefix: str = ""

    @property
    def num_seqs(self) -> int:
        return int(self.seq_lens.shape[0])

    @property
    def draft(self) -> Optional[DraftStepInputs]:
        return self.spec.draft if self.spec is not None else None
