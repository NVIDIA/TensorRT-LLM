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
"""The ULL generation target's DSA metadata (``rubin_ull_dsa_metadata.py``): the Helix host slot
mappings of its generation rows (``helix_host_gen_slot_rows``), which the engine indexer uploads
with its own rows. CPU stand-ins, no GPU."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_metadata import (  # noqa: E501
    RubinUllDSAMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    IndexerParams,
    _compute_slot_mappings,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata

pytestmark = pytest.mark.cpu_only


def test_helix_host_gen_slot_rows_are_row_aligned() -> None:
    tpb, head_dim, qbs = 64, 128, 128
    block_offsets = torch.tensor([[3, 9, -1], [5, 11, -1], [2, 4, -1]], dtype=torch.int32)
    cached = [70, 10, 64]
    params = IndexerParams(
        num_contexts=0,
        num_generations=3,
        num_ctx_tokens=0,
        head_dim=head_dim,
        quant_block_size=qbs,
        tokens_per_block=tpb,
        compress_ratio=1,
        request_ids=[0, 1, 2],
        num_past_tokens=cached,
        seq_lens=torch.tensor([1, 1, 1]),
    )
    n = 3
    md = SimpleNamespace(
        num_ctx_tokens=0,
        num_tokens=n,
        num_contexts=0,
        num_seqs=n,
        num_generations=n,
        _helix_spec_tokens_valid=False,
        helix_is_inactive_rank_cpu=torch.tensor([False, True, False]),
        host_indexer_k_cache_block_offsets=block_offsets,
        host_slot_mapping_fp8=torch.full((8,), -7, dtype=torch.int64),
        host_slot_mapping_scale=torch.full((8,), -7, dtype=torch.int64),
        slot_mapping_fp8=torch.full((8,), -7, dtype=torch.int64),
        slot_mapping_scale=torch.full((8,), -7, dtype=torch.int64),
    )
    # The host rows are rewritten and returned; the indexer uploads them once, with the rows of
    # the scatter they overlap, so the device mappings are untouched here.
    assert RubinUllDSAMetadata.helix_host_gen_slot_rows(md, params) == slice(0, n)
    assert md.slot_mapping_fp8.tolist() == [-7] * 8
    expected_fp8, expected_scale = _compute_slot_mappings(
        torch.tensor([70, 0, 64]), block_offsets, torch.arange(3), head_dim, tpb, qbs
    )
    # One slot per generation row; the inactive rank's row is skipped (-1).
    assert md.host_slot_mapping_fp8[:n].tolist() == [
        int(expected_fp8[0]),
        -1,
        int(expected_fp8[2]),
    ]
    assert md.host_slot_mapping_scale[:n].tolist() == [
        int(expected_scale[0]),
        -1,
        int(expected_scale[2]),
    ]
    assert md.host_slot_mapping_fp8[n:].tolist() == [-7] * (8 - n)

    # Verify groups: host rows are placeholders that on_update_kv_lens() rewrites.
    md._helix_spec_tokens_valid = True
    md.num_tokens = 6
    md.num_generations = md.num_seqs = 1
    assert RubinUllDSAMetadata.helix_host_gen_slot_rows(md, params) == slice(0, 6)
    assert md.host_slot_mapping_fp8[:6].tolist() == [-1] * 6


def test_the_engine_metadata_keeps_the_generic_mapping() -> None:
    assert DSAtrtllmAttentionMetadata.helix_host_gen_slot_rows(SimpleNamespace(), None) is None
