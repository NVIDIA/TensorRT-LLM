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

"""Architecture-level regression tests for page-sparse MHA computation.

The tests supply static block indices and per-request offsets, invoke
``TrtllmAttention.forward``, and compare the result with an equivalent
token-level PyTorch reference.

Prefill attention is dense on this path; page-sparse MHA computation starts
during generation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm._torch.attention_backend.sparse.params import SparseParams
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

from ._sparse_mha_test_utils import (
    SUPPORTED_MHA_HEAD_DIMS,
    SUPPORTED_MODEL_DTYPES,
    MhaGenerationScenario,
    create_generation_inputs,
    fp8_qdq,
    quant_config,
    read_paged_kv_cache,
    reference_generation_attention,
)

ATOL = 2e-2
RTOL = 2e-2
FP8_ATOL = 8e-2
FP8_RTOL = 4e-2
TESTED_MHA_HEAD_COUNTS = (1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128)
TESTED_KV_PAGE_SIZES = (8, 16, 32, 64, 128, 256, 512)
SUPPORTED_SM_VERSIONS = (100, 103)

pytestmark = pytest.mark.skipif(
    getSMVersion() not in SUPPORTED_SM_VERSIONS,
    reason="Page-sparse MHA requires SM100 or SM103",
)


@pytest.fixture(autouse=True)
def _force_trtllm_gen_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test on the TRTLLM-Gen fallback path."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")


# Page-sparse MHA support matrix:
#
#   GPU architecture       SM100 and SM103
#   Sparse compute phase   Single-token and linear draft-token generation
#   Attention type         MHA; num_heads == num_kv_heads
#   Q heads per KV head    1
#   Number of MHA heads    No discrete source restriction; tests cover
#                          1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, and 128
#   Model QKV input        BF16 or FP16
#   Model QKV layout       Fused QKV
#   Kernel output          Model dtype for H64/H80/H128/H256; E4M3 FP8 for
#                          H64/H128/H256 with an FP8 KV cache
#   KV-cache dtype         Model dtype for H64/H80/H128/H256; E4M3 FP8 for
#                          H64/H128/H256
#   Q/K/V head dimension  Equal dimensions: 64, 80, 128, or 256
#   KV-cache layout        Paged; page sizes 8, 16, 32, 64, 128, 256, and 512
#   Selection granularity  Block indices expanded to KV-cache pages
#   Sparse indices         int32 block indices plus int32 request offsets
#                          Per-head patterns and variable request offsets
#   Sparse index block     Blocks may cross KV-page boundaries; sizes
#                          1, 2, 3, 4, 5, 8, 16, 24, 32, and 48 are tested
#   Attention semantics    Causal self-attention


@dataclass(kw_only=True, frozen=True)
class PageSparseMhaScenario:
    """Page-sparse MHA geometry layered on generation inputs."""

    attention: MhaGenerationScenario
    sparse_index_block_size: int = 4
    num_selected_sparse_blocks: int = 2

    def __post_init__(self) -> None:
        if self.attention.num_heads != self.attention.num_kv_heads:
            raise ValueError("PageSparseMhaScenario requires MHA head geometry")
        if self.sparse_index_block_size <= 0:
            raise ValueError("sparse_index_block_size must be positive")
        if self.num_selected_sparse_blocks <= 0:
            raise ValueError("num_selected_sparse_blocks must be positive")


@dataclass(kw_only=True, frozen=True)
class _PageSparseMhaParams(SparseParams):
    """Sparse parameters selecting block/page-granular attention."""

    sparse_index_block_size: int
    algorithm: str = "test_page_sparse_mha"

    @property
    def indices_block_size(self) -> int:
        return self.sparse_index_block_size


class _StaticPageSparseMhaAttention(TrtllmAttention):
    """MHA backend adapter returning predetermined page selections."""

    def __init__(
        self,
        *args,
        sparse_index_block_size: int,
        sparse_attn_indices: torch.Tensor,
        sparse_attn_offsets: torch.Tensor,
        **kwargs,
    ) -> None:
        kwargs["sparse_params"] = _PageSparseMhaParams(
            sparse_index_block_size=sparse_index_block_size
        )
        kwargs["pos_embd_params"] = None
        super().__init__(*args, **kwargs)
        self._sparse_attn_indices = sparse_attn_indices
        self._sparse_attn_offsets = sparse_attn_offsets

    def sparse_kv_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return None, None

    def sparse_attn_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_attn_indices, self._sparse_attn_offsets


def _selected_sparse_blocks(
    head_idx: int,
    num_sparse_blocks: int,
    num_selected_blocks: int,
    page_size: int,
    sparse_block_size: int,
) -> Tuple[int, ...]:
    """Choose head-dependent blocks and retain the newest sparse block."""
    newest_block = num_sparse_blocks - 1
    if num_selected_blocks == 1:
        return (newest_block,)

    older_blocks = list(range(newest_block))
    boundary_block = page_size // sparse_block_size
    if head_idx % 2 and boundary_block in older_blocks:
        older_blocks.remove(boundary_block)
        older_blocks.insert(0, boundary_block)
    selected = older_blocks[: num_selected_blocks - 1] + [newest_block]
    if head_idx % 2:
        selected.reverse()
    return tuple(selected)


def _make_page_sparse_pattern(
    scenario: PageSparseMhaScenario,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build page indices plus equivalent request-local token indices for the reference."""
    attention = scenario.attention
    sparse_indices_by_head = [[] for _ in range(attention.num_kv_heads)]
    sparse_offsets = [0]
    max_reference_tokens = max(attention.past_kv_lens) + attention.query_len
    reference_indices = torch.full(
        (attention.num_kv_heads, attention.nnz_q, max_reference_tokens),
        -1,
        dtype=torch.int32,
        device=device,
    )

    query_offset = 0
    for past_kv_len in attention.past_kv_lens:
        total_kv_len = past_kv_len + attention.query_len
        num_sparse_blocks = math.ceil(total_kv_len / scenario.sparse_index_block_size)
        num_selected_sparse_blocks = min(scenario.num_selected_sparse_blocks, num_sparse_blocks)
        for head_idx in range(attention.num_kv_heads):
            sparse_blocks = _selected_sparse_blocks(
                head_idx,
                num_sparse_blocks,
                num_selected_sparse_blocks,
                attention.page_size,
                scenario.sparse_index_block_size,
            )
            sparse_indices_by_head[head_idx].extend(sparse_blocks)

            for query_idx in range(attention.query_len):
                available_kv_len = past_kv_len + query_idx + 1
                touched_pages = set()
                for sparse_block_idx in sparse_blocks:
                    block_start = sparse_block_idx * scenario.sparse_index_block_size
                    block_end = min(
                        block_start + scenario.sparse_index_block_size,
                        available_kv_len,
                    )
                    if block_start >= block_end:
                        continue
                    first_page = block_start // attention.page_size
                    last_page = (block_end - 1) // attention.page_size
                    touched_pages.update(range(first_page, last_page + 1))
                selected_tokens = []
                for page_idx in sorted(touched_pages):
                    page_start = page_idx * attention.page_size
                    page_end = min(page_start + attention.page_size, available_kv_len)
                    selected_tokens.extend(range(page_start, page_end))
                reference_indices[
                    head_idx,
                    query_offset + query_idx,
                    : len(selected_tokens),
                ] = torch.tensor(selected_tokens, dtype=torch.int32, device=device)

        sparse_offsets.append(sparse_offsets[-1] + num_selected_sparse_blocks)
        query_offset += attention.query_len

    sparse_attn_indices = torch.tensor(
        sparse_indices_by_head,
        dtype=torch.int32,
        device=device,
    )
    sparse_attn_offsets = torch.tensor(sparse_offsets, dtype=torch.int32, device=device)
    return sparse_attn_indices, sparse_attn_offsets, reference_indices


def _run_page_sparse_mha(scenario: PageSparseMhaScenario) -> None:
    """Compare page-sparse generation with an equivalent PyTorch token reference."""
    attention_scenario = scenario.attention
    inputs = create_generation_inputs(attention_scenario)
    try:
        sparse_attn_indices, sparse_attn_offsets, reference_indices = _make_page_sparse_pattern(
            scenario, inputs.q.device
        )
        attention = _StaticPageSparseMhaAttention(
            layer_idx=0,
            num_heads=attention_scenario.num_heads,
            head_dim=attention_scenario.head_dim,
            num_kv_heads=attention_scenario.num_kv_heads,
            quant_config=quant_config(attention_scenario),
            sparse_index_block_size=scenario.sparse_index_block_size,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
        )

        kv_caches = read_paged_kv_cache(inputs, attention_scenario)
        reference_q = inputs.q
        reference_k_new = inputs.k_new
        reference_v_new = inputs.v_new
        if attention_scenario.kvcache_dtype == torch.float8_e4m3fn:
            reference_q = fp8_qdq(reference_q)
            reference_k_new = fp8_qdq(reference_k_new)
            reference_v_new = fp8_qdq(reference_v_new)
        reference_output = reference_generation_attention(
            reference_q,
            kv_caches,
            reference_k_new,
            reference_v_new,
            reference_indices,
            attention_scenario,
        )

        forward_args: Optional[AttentionForwardArgs] = None
        if attention_scenario.fp8_output:
            forward_args = AttentionForwardArgs(
                out_scale=torch.ones(1, dtype=torch.float32, device=inputs.q.device)
            )
        output = attention.forward(
            inputs.fused_qkv,
            None,
            None,
            inputs.metadata,
            forward_args=forward_args,
        )

        expected_shape = (
            attention_scenario.nnz_q,
            attention_scenario.num_heads * attention_scenario.head_dim,
        )
        assert output.shape == expected_shape
        expected_output_dtype = (
            torch.float8_e4m3fn if attention_scenario.fp8_output else attention_scenario.dtype
        )
        assert output.dtype == expected_output_dtype
        uses_fp8 = (
            attention_scenario.kvcache_dtype == torch.float8_e4m3fn or attention_scenario.fp8_output
        )
        output_for_comparison = output.float() if uses_fp8 else output
        if attention_scenario.fp8_output:
            reference_output = reference_output.to(torch.float8_e4m3fn)
        reference_for_comparison = reference_output.float() if uses_fp8 else reference_output
        assert torch.isfinite(output_for_comparison).all()
        torch.testing.assert_close(
            output_for_comparison,
            reference_for_comparison,
            atol=FP8_ATOL if uses_fp8 else ATOL,
            rtol=FP8_RTOL if uses_fp8 else RTOL,
        )
    finally:
        inputs.kv_cache_manager.shutdown()


_NUM_MHA_HEADS = 8

_PAGE_SPARSE_MHA_CASES = (
    [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    dtype=dtype,
                    kvcache_dtype=dtype,
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    head_dim=head_dim,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                )
            ),
            id=f"{str(dtype).removeprefix('torch.')}_h{head_dim}",
        )
        for dtype in SUPPORTED_MODEL_DTYPES
        for head_dim in SUPPORTED_MHA_HEAD_DIMS
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    page_size=kv_page_size,
                    batch_size=1,
                    past_kv_lens=(max(96, 3 * kv_page_size),),
                    num_pages=math.ceil((max(96, 3 * kv_page_size) + 1) / kv_page_size),
                )
            ),
            id=f"kv_page_size_{kv_page_size}",
        )
        for kv_page_size in TESTED_KV_PAGE_SIZES
        if kv_page_size != 32
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                ),
                sparse_index_block_size=sparse_index_block_size,
            ),
            id=f"sparse_index_block_size_{sparse_index_block_size}",
        )
        for sparse_index_block_size in (1, 2, 3, 5, 8, 16, 24, 32, 48)
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                ),
                num_selected_sparse_blocks=num_selected_sparse_blocks,
            ),
            id=f"selected_sparse_blocks_{num_selected_sparse_blocks}",
        )
        for num_selected_sparse_blocks in (1, 3)
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=2,
                    past_kv_lens=(32, 160),
                    num_pages=8,
                ),
                num_selected_sparse_blocks=3,
            ),
            id="batch2_var_kv_and_offsets",
        ),
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(95,),
                    num_pages=4,
                )
            ),
            id="non_page_aligned_kv_length",
        ),
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    query_len=4,
                    num_pages=4,
                )
            ),
            id="linear_3_draft_tokens",
        ),
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    dtype=dtype,
                    kvcache_dtype=torch.float8_e4m3fn,
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    head_dim=head_dim,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                    fp8_output=fp8_output,
                )
            ),
            id=(
                f"{str(dtype).removeprefix('torch.')}_h{head_dim}_fp8_kv_"
                f"{'fp8_output' if fp8_output else 'model_output'}"
            ),
        )
        for dtype in SUPPORTED_MODEL_DTYPES
        for head_dim in SUPPORTED_MHA_HEAD_DIMS
        for fp8_output in (False, True)
        if head_dim != 80
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=num_mha_heads,
                    num_kv_heads=num_mha_heads,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                )
            ),
            id=f"mha_{num_mha_heads}_heads",
        )
        for num_mha_heads in TESTED_MHA_HEAD_COUNTS
        if num_mha_heads != _NUM_MHA_HEADS
    ]
)


@pytest.mark.parametrize("scenario", _PAGE_SPARSE_MHA_CASES)
def test_generation_page_sparse_mha(scenario: PageSparseMhaScenario) -> None:
    """Static page selections drive sparse MHA generation computation."""
    _run_page_sparse_mha(scenario)
