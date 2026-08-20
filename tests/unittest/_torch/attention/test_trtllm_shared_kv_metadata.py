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

import copy
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from kv_cache_utils import fill_kv_cache_logical

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionMetadata
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping


class _SelectorAttention:
    def __init__(self, head_dim: int) -> None:
        self.is_mla_enable = False
        self.kv_lora_rank = None
        self.v_head_dim = None
        self.head_dim = head_dim
        self.num_heads = 8
        self.num_kv_heads = 1
        self.predicted_tokens_per_seq = 1
        self.sparse_params = None
        self.position_embedding_type = 0
        self.quant_mode = 0


def _make_target_metadata() -> TrtllmAttentionMetadata:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata.max_num_requests = 4
    metadata.max_num_sequences = 4
    metadata.max_num_tokens = 16
    metadata._seq_lens = torch.tensor([4, 2, 3], dtype=torch.int32)
    metadata._seq_lens_cuda = metadata._seq_lens.clone()
    metadata._seq_lens_kv = None
    metadata._seq_lens_kv_cuda = None
    metadata._num_contexts = 1
    metadata._num_generations = 2
    metadata._num_ctx_tokens = 4
    metadata._num_tokens = 9
    metadata.cross = None

    metadata.kv_cache_manager = SimpleNamespace(kv_factor=2)
    metadata.draft_kv_cache_manager = None
    metadata.kv_cache_params = object()
    metadata.kv_cache_block_offsets = torch.arange(24).view(1, 3, 2, 4)
    metadata.host_kv_cache_block_offsets = object()
    metadata.draft_kv_cache_block_offsets = None
    metadata.workspace = torch.empty(7, dtype=torch.int8)
    metadata.cuda_graph_workspace = torch.empty(11, dtype=torch.int8)
    metadata.kv_lens_cuda = torch.empty(4, dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = torch.tensor([10, 20, 30], dtype=torch.int32)
    metadata.kv_lens_runtime = torch.tensor([10, 20, 30], dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.tensor([4, 8, 9], dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = torch.tensor([4, 8, 9], dtype=torch.int32)
    metadata.host_request_types = torch.tensor([0, 1, 1, -1], dtype=torch.int32)
    metadata.host_request_types_runtime = metadata.host_request_types[:3]
    metadata.host_total_kv_lens = torch.tensor([10, 50], dtype=torch.int32)

    metadata.request_ids = [101, 102, 103]
    metadata.prompt_lens = [4, 8, 9]
    metadata.all_rank_num_tokens = [9]
    metadata.mapping = object()
    metadata.sparse_metadata_params = None
    metadata.num_sparse_topk = 0
    metadata.beam_width = 1
    metadata.padded_num_tokens = 9
    metadata.cuda_graph_buffers = None
    metadata._saved_tensors = {"target": torch.tensor(1)}

    metadata.is_spec_decoding_enabled = True
    metadata.use_spec_decoding = True
    metadata.is_spec_dec_tree = True
    metadata.is_spec_dec_dynamic_tree = True
    metadata.force_prepare_spec_dec_tree_mask = True
    metadata.max_total_draft_tokens = 4
    metadata.spec_decoding_position_offsets = torch.tensor([0])
    metadata.spec_decoding_position_offsets_cpp = torch.tensor([[0]])
    metadata.spec_decoding_packed_mask = torch.tensor([1])
    metadata.spec_decoding_generation_lengths = torch.tensor([1])
    metadata.spec_decoding_bl_tree_mask_offset = torch.tensor([0])
    metadata.spec_decoding_bl_tree_mask = torch.tensor([1])
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = torch.tensor([0])
    metadata.position_offsets_stride = 1

    metadata._draft_metadata = None
    metadata._draft_kv_runtime_lens = None
    metadata._shared_kv_draft_seq_lens = None
    metadata._shared_kv_draft_seq_lens_cuda = None
    metadata._shared_kv_draft_request_types = None
    return metadata


def test_shared_kv_draft_view_uses_accepted_prefix_without_target_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.trtllm.prefer_pinned",
        lambda: False,
    )
    target = _make_target_metadata()
    target_kv_lens = target.kv_lens_cuda_runtime.clone()
    target_seq_lens = target.seq_lens_cuda.clone()
    target_request_types = target.host_request_types.clone()

    draft = target.get_shared_kv_draft_metadata(
        torch.tensor([4, 1, 2], dtype=torch.int32),
        num_contexts=1,
    )

    torch.testing.assert_close(
        draft.kv_lens_cuda_runtime,
        torch.tensor([10, 19, 29], dtype=torch.int32),
    )
    torch.testing.assert_close(draft.seq_lens, torch.ones(3, dtype=torch.int32))
    torch.testing.assert_close(draft.seq_lens_cuda, torch.ones(3, dtype=torch.int32))
    torch.testing.assert_close(draft.host_request_types_runtime, torch.ones(3, dtype=torch.int32))
    assert draft.num_contexts == 0
    assert draft.num_generations == 3
    assert draft.num_ctx_tokens == 0
    assert draft.num_tokens == 3
    assert not draft.is_cross

    assert draft.kv_cache_manager is target.kv_cache_manager
    assert draft.kv_cache_params is target.kv_cache_params
    assert draft.kv_cache_block_offsets is target.kv_cache_block_offsets
    assert draft.host_kv_cache_block_offsets is target.host_kv_cache_block_offsets
    assert draft.workspace is target.workspace
    assert draft.cuda_graph_workspace is target.cuda_graph_workspace
    assert draft.mapping is target.mapping
    assert draft.kv_lens_runtime.data_ptr() == target.kv_lens_runtime.data_ptr()
    assert not draft.is_spec_decoding_enabled
    assert not draft.use_spec_decoding
    assert draft.spec_decoding_packed_mask is None
    assert draft.padded_num_tokens is None

    torch.testing.assert_close(target.kv_lens_cuda_runtime, target_kv_lens)
    torch.testing.assert_close(target.seq_lens_cuda, target_seq_lens)
    torch.testing.assert_close(target.host_request_types, target_request_types)
    assert target.num_contexts == 1
    assert target.use_spec_decoding
    assert target._saved_tensors


def test_shared_kv_draft_view_reuses_stable_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention_backend.trtllm.prefer_pinned",
        lambda: False,
    )
    target = _make_target_metadata()
    draft = target.get_shared_kv_draft_metadata(
        torch.tensor([4, 1, 2], dtype=torch.int32),
        num_contexts=1,
    )
    storage_pointers = (
        draft._draft_kv_runtime_lens.data_ptr(),
        draft._shared_kv_draft_seq_lens.data_ptr(),
        draft._shared_kv_draft_seq_lens_cuda.data_ptr(),
        draft._shared_kv_draft_request_types.data_ptr(),
    )

    target.kv_lens_cuda_runtime.copy_(torch.tensor([11, 22, 33]))
    target.kv_lens_runtime.copy_(torch.tensor([11, 22, 33]))
    second = target.get_shared_kv_draft_metadata(
        torch.tensor([4, 2, 1], dtype=torch.int32),
        num_contexts=1,
    )

    assert second is draft
    assert storage_pointers == (
        draft._draft_kv_runtime_lens.data_ptr(),
        draft._shared_kv_draft_seq_lens.data_ptr(),
        draft._shared_kv_draft_seq_lens_cuda.data_ptr(),
        draft._shared_kv_draft_request_types.data_ptr(),
    )
    torch.testing.assert_close(
        draft.kv_lens_cuda_runtime,
        torch.tensor([11, 22, 31], dtype=torch.int32),
    )
    torch.testing.assert_close(
        target.kv_lens_cuda_runtime,
        torch.tensor([11, 22, 33], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("beam_width", 2, "beam search"),
        ("draft_kv_cache_manager", object(), "separate draft KV cache"),
        ("sparse_metadata_params", object(), "sparse attention"),
    ],
)
def test_shared_kv_draft_view_rejects_unsupported_modes(
    field: str,
    value: object,
    error: str,
) -> None:
    target = _make_target_metadata()
    setattr(target, field, value)

    with pytest.raises(ValueError, match=error):
        target.get_shared_kv_draft_metadata(torch.ones(3, dtype=torch.int32), num_contexts=1)


def test_shared_kv_draft_view_rejects_mla_and_cross_attention() -> None:
    target = _make_target_metadata()
    target.kv_cache_manager.kv_factor = 1
    with pytest.raises(ValueError, match="MLA"):
        target.get_shared_kv_draft_metadata(torch.ones(3, dtype=torch.int32), num_contexts=1)

    target = _make_target_metadata()
    target._seq_lens_kv = target._seq_lens.clone()
    target._seq_lens_kv_cuda = target._seq_lens_cuda.clone()
    with pytest.raises(ValueError, match="cross attention"):
        target.get_shared_kv_draft_metadata(torch.ones(3, dtype=torch.int32), num_contexts=1)


def test_cuda_graph_metadata_does_not_inherit_eager_draft_view() -> None:
    target = _make_target_metadata()
    target._draft_metadata = object()
    graph_metadata = copy.copy(target)

    with mock.patch.object(
        AttentionMetadata,
        "create_cuda_graph_metadata",
        return_value=graph_metadata,
    ):
        result = target.create_cuda_graph_metadata(max_batch_size=4)

    assert result is graph_metadata
    assert result._draft_metadata is None
    assert target._draft_metadata is not None


def test_cuda_graph_metadata_preserves_its_draft_view() -> None:
    graph_metadata = _make_target_metadata()
    graph_metadata.is_cuda_graph = True
    draft_metadata = object()
    graph_metadata._draft_metadata = draft_metadata

    result = graph_metadata.create_cuda_graph_metadata(max_batch_size=4)

    assert result is graph_metadata
    assert result._draft_metadata is draft_metadata


@pytest.mark.parametrize("head_dim", [256, 512])
def test_q_only_generation_selects_flashinfer_trtllm_gen(head_dim: int) -> None:
    attention = _SelectorAttention(head_dim)
    fmha = FlashInferTrtllmGenFmha(attention)
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=2,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(
            dtype=DataType.BF16,
            blocks_in_primary_pool=8,
            num_local_layers=1,
            impl=None,
        ),
        is_cross=False,
        tokens_per_block=32,
        beam_width=1,
    )
    q = torch.empty((2, attention.num_heads * head_dim), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        is_fused_qkv=False,
        update_kv_cache=False,
    )

    assert fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    backend = SimpleNamespace(fmha_libs=[fmha], combined_fmha=None)
    selected = TrtllmAttention._select_non_mla_fmha(
        backend,
        q,
        None,
        None,
        metadata,
        forward_args,
    )

    assert selected is fmha


def _shared_kv_reference(
    q: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    kv_lens: list[int],
) -> torch.Tensor:
    num_heads = q.shape[1] // keys[0].shape[-1]
    num_kv_heads = keys[0].shape[1]
    repeats = num_heads // num_kv_heads
    outputs = []
    for row, kv_len in enumerate(kv_lens):
        q_row = q[row].view(num_heads, -1).float()
        k_row = keys[row][:kv_len].repeat_interleave(repeats, dim=1).float()
        v_row = values[row][:kv_len].repeat_interleave(repeats, dim=1).float()
        scores = torch.einsum("hd,lhd->hl", q_row, k_row) / math.sqrt(q_row.shape[-1])
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hl,lhd->hd", probs, v_row))
    return torch.stack(outputs).to(q.dtype).view_as(q)


@pytest.mark.parametrize("head_dim", [256, 512])
def test_q_only_shared_kv_forward_preserves_target_cache(head_dim: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the shared-KV forward test")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("FlashInfer trtllm-gen requires SM100 or SM103")
    if not IS_FLASHINFER_AVAILABLE:
        pytest.skip("FlashInfer is required for the trtllm-gen forward test")

    num_heads = 8
    num_kv_heads = 1
    target_q_lens = [3, 3]
    cached_lens = [5, 7]
    target_kv_lens = [
        cached + query for cached, query in zip(cached_lens, target_q_lens, strict=True)
    ]
    accepted_lens = [2, 1]
    draft_kv_lens = [
        target - query + accepted
        for target, query, accepted in zip(
            target_kv_lens,
            target_q_lens,
            accepted_lens,
            strict=True,
        )
    ]
    request_ids = [0, 1]
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    manager = KVCacheManager(
        KvCacheConfig(max_tokens=64),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=32,
        max_seq_len=32,
        max_batch_size=len(request_ids),
        mapping=mapping,
        dtype=DataType.BF16,
    )
    try:
        manager.add_dummy_requests(request_ids, target_kv_lens)
        generator = torch.Generator(device="cuda").manual_seed(2026 + head_dim)
        keys = [
            torch.randn(
                length,
                num_kv_heads,
                head_dim,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for length in target_kv_lens
        ]
        values = [
            torch.randn(
                length,
                num_kv_heads,
                head_dim,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for length in target_kv_lens
        ]
        fill_kv_cache_logical(
            manager,
            0,
            request_ids,
            keys,
            values,
            kv_layout="HND",
        )

        target_metadata = TrtllmAttentionMetadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_lens,
            ),
            seq_lens=torch.tensor(target_q_lens, dtype=torch.int32),
            max_num_requests=len(request_ids),
            max_num_tokens=16,
            kv_cache_manager=manager,
            request_ids=request_ids,
            prompt_lens=target_kv_lens,
            kv_layout="HND",
        )
        target_metadata.prepare()
        draft_metadata = target_metadata.get_shared_kv_draft_metadata(
            torch.tensor(accepted_lens, dtype=torch.int32, device="cuda"),
            num_contexts=0,
        )

        attention = TrtllmAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
        assert FlashInferTrtllmGenFmha.is_available(attention)
        fmha = FlashInferTrtllmGenFmha(attention)
        attention.fmha_libs = [fmha]
        attention.phased_fmha_libs = [fmha]
        attention.non_phased_fmha_libs = []

        q = torch.randn(
            len(request_ids),
            num_heads * head_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = _shared_kv_reference(q, keys, values, draft_kv_lens)
        cache = manager.get_buffers(0, kv_layout="HND")
        cache_before = cache.clone()

        actual = attention.forward(q, None, None, draft_metadata)

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-3)
        torch.testing.assert_close(cache, cache_before, atol=0, rtol=0)
    finally:
        manager.shutdown()
