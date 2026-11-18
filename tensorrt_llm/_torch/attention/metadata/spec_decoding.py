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
"""Speculative decoding state of the attention kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version, is_sm_100f

from .base import Metadata, UnsupportedMetadataConfiguration, _Registry
from .core import TrtllmCoreMetadata

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SpeculativeDecodingMode

    from .batch_inputs import BatchInputs


class SpecDecodingMetadata(Metadata):
    """Spec-dec mode, tree flags, draft-token bound and verify layout.

    On the SM100 family the TRTLLM-Gen kernels verify a linear draft with
    ordinary multi-token queries, so the multi-query spec-dec mode of the
    attention op stays off and its mask buffers are not allocated.
    """

    requires = (TrtllmCoreMetadata,)

    def __init__(self, registry: _Registry, config: "BatchInputs") -> None:
        super().__init__(registry, config)
        self.spec_dec_mode: Optional["SpeculativeDecodingMode"] = None
        # Whether the attention op supports spec-dec mode for the whole run,
        # and whether this step and layer runs in it.
        self.is_spec_decoding_enabled = False
        self.use_spec_decoding = False
        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False
        self.force_prepare_spec_dec_tree_mask = False
        self.max_total_draft_tokens: Optional[int] = None
        # Query tokens of each generation request in this step.
        self.tokens_per_gen_step = 1
        self.runtime_draft_len = 0
        # Mask and position buffers of the multi-query spec-dec mode.
        self.spec_decoding_position_offsets: Optional[torch.Tensor] = None
        self.spec_decoding_query_len = 0
        self.spec_decoding_packed_mask: Optional[torch.Tensor] = None
        self.spec_decoding_generation_lengths: Optional[torch.Tensor] = None
        self.spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor] = None
        self.spec_decoding_bl_tree_mask: Optional[torch.Tensor] = None
        self.spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor] = None

    def prepare(self, inputs: "BatchInputs") -> None:
        spec = inputs.spec
        if spec is None:
            return
        if spec.is_spec_dec_tree or spec.is_spec_dec_dynamic_tree:
            raise UnsupportedMetadataConfiguration("tree speculative decoding")
        if spec.attention_spec_dec_mode and not is_sm_100f(get_sm_version()):
            raise UnsupportedMetadataConfiguration(
                f"the multi-query spec-dec attention kernels of SM{get_sm_version()}"
            )
        self.spec_dec_mode = spec.spec_dec_mode
        self.max_total_draft_tokens = spec.max_total_draft_tokens
        self.tokens_per_gen_step = spec.tokens_per_gen_step
        self.runtime_draft_len = spec.runtime_draft_len
