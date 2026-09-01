# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import flashinfer_sparse_mla
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
)
from tensorrt_llm._torch.attention_backend.sparse import dsa_flashinfer, inline_scale_kv
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.kernels import (
    deepseek_v4_local_to_global_indices,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.sparse.flashinfer_utils import (
    allocate_sparse_mla_split_workspace,
)
from tensorrt_llm._torch.attention_backend.sparse.params import SparseRuntimeParams
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping


def test_sparse_mla_split_workspace_follows_kernel_threshold() -> None:
    mid_out, mid_lse = allocate_sparse_mla_split_workspace(
        num_tokens=64,
        num_heads=8,
        num_splits=3,
        value_dim=512,
        device=torch.device("cpu"),
    )

    assert mid_out is not None and mid_out.shape == (64, 8, 3, 512)
    assert mid_lse is not None and mid_lse.shape == (64, 8, 3)
    assert mid_out.dtype == torch.bfloat16
    assert mid_lse.dtype == torch.float32

    assert allocate_sparse_mla_split_workspace(
        num_tokens=65,
        num_heads=8,
        num_splits=3,
        value_dim=512,
        device=torch.device("cpu"),
    ) == (None, None)


def test_flashinfer_sparse_mla_missing_private_op_warns() -> None:
    with (
        patch.object(flashinfer_sparse_mla, "get_sm_version", return_value=120),
        patch.object(
            flashinfer_sparse_mla,
            "get_sparse_mla_op",
            side_effect=ImportError("missing private op"),
        ),
        patch.object(flashinfer_sparse_mla.logger, "warning") as warning,
    ):
        assert flashinfer_sparse_mla.is_flashinfer_sparse_mla_enabled("dsa") is False

    warning.assert_called_once()
    assert "_sparse_mla_sm120_paged_attention" in warning.call_args.args[0]


def test_split_extra_rejects_empty_compressed_output() -> None:
    req_id = torch.zeros(1, dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    indices = torch.zeros((1, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match=r"split_extra.*num_compressed_indices"):
        deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table,
            swa_local_indices=indices,
            swa_pool_base_ptr=0,
            swa_buffer_ptr=0,
            tokens_per_block=64,
            token_stride=1,
            block_table_compressed=block_table,
            compressed_local_indices=indices,
            compress_ratio=4,
            num_compressed_indices=0,
            split_extra=True,
        )


def test_dsa_cache_size_estimation_allows_missing_kv_cache_config() -> None:
    class FakeModelConfig:
        pretrained_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        sparse_attention_config = DeepSeekSparseAttentionConfig(index_head_dim=128)
        quant_config = None

        @staticmethod
        def get_num_attention_layers() -> int:
            return 2

    assert (
        DSACacheManager.get_cache_size_per_token(
            FakeModelConfig(),
            Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        )
        > 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_inline_scale_kv_quant_scatter_layout() -> None:
    torch.manual_seed(0)
    rows = torch.randn(2, 576, dtype=torch.bfloat16, device="cuda") / 10
    rows[0].zero_()
    locations = torch.tensor([0, 63], dtype=torch.int32, device="cuda")
    pool = torch.zeros((1, inline_scale_kv.PAGE_BYTES), dtype=torch.uint8, device="cuda")

    inline_scale_kv.quant_scatter(pool, locations, rows)

    packed = pool.view(inline_scale_kv.PAGE_SIZE, inline_scale_kv.TOKEN_BYTES)[locations.long()]
    scales = (
        packed[:, inline_scale_kv.DIM_NOPE : inline_scale_kv.ROPE_OFFSET]
        .contiguous()
        .view(torch.float32)
    )
    expected_scales = (
        rows[:, : inline_scale_kv.DIM_NOPE]
        .float()
        .view(2, inline_scale_kv.NUM_NOPE_TILES, inline_scale_kv.QUANT_TILE)
        .abs()
        .amax(dim=-1)
        .clamp_min(1e-8)
        / torch.finfo(torch.float8_e4m3fn).max
    )
    rope = packed[:, inline_scale_kv.ROPE_OFFSET :].contiguous().view(torch.bfloat16)

    torch.testing.assert_close(scales, expected_scales)
    torch.testing.assert_close(rope, rows[:, inline_scale_kv.DIM_NOPE :], rtol=0, atol=0)
    torch.testing.assert_close(
        inline_scale_kv.dequant_gather(pool, locations),
        rows,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (120, 121),
    reason="FlashInfer sparse MLA requires SM 120 or SM 121",
)
def test_dsa_sparse_mla_cuda_graph_capture_replay() -> None:
    torch.manual_seed(1)
    num_tokens, num_heads, topk = 2, 8, 128
    cache_rows = torch.randn(topk, 576, dtype=torch.bfloat16, device="cuda") / 10
    locations = torch.arange(topk, dtype=torch.int32, device="cuda")
    pool = torch.zeros(
        (topk // inline_scale_kv.PAGE_SIZE, inline_scale_kv.PAGE_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    inline_scale_kv.quant_scatter(pool, locations, cache_rows)
    dequantized = inline_scale_kv.dequant_gather(pool, locations)
    indices = locations.expand(num_tokens, -1).contiguous()

    metadata = AttentionMetadata(
        max_num_requests=num_tokens,
        max_num_tokens=num_tokens,
        seq_lens=None,
        seq_lens_kv=None,
        num_contexts=0,
        is_cuda_graph=True,
    )
    metadata._num_tokens = num_tokens
    metadata._num_ctx_tokens = 0
    metadata.max_draft_tokens = 0
    metadata.token_positions_cuda = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    metadata.kv_cache_manager = SimpleNamespace(
        tokens_per_block=inline_scale_kv.PAGE_SIZE,
        get_unique_primary_pool=lambda: pool.view(pool.shape[0], 1, -1),
    )
    attn = SimpleNamespace(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=576,
        q_scaling=1.0,
        mla_params=SimpleNamespace(
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
        ),
    )
    q = torch.randn(num_tokens, num_heads, 576, dtype=torch.bfloat16, device="cuda") / 10
    output = torch.empty(num_tokens, num_heads * 512, dtype=torch.bfloat16, device="cuda")
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        output=output,
        update_kv_cache=False,
        skip_mla_rope_generation=True,
        sparse_runtime_params=SparseRuntimeParams(sparse_attn_indices=indices),
    )

    def run() -> None:
        dsa_flashinfer.run_flashinfer_sparse_mla(attn, q, metadata, forward_args, None)

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run()
        run()
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    q.copy_(torch.randn_like(q) / 10)
    graph.replay()
    torch.cuda.synchronize()

    scores = torch.einsum("bhd,sd->bhs", q.float(), dequantized.float())
    expected = torch.einsum(
        "bhs,sd->bhd",
        torch.softmax(scores / math.sqrt(576), dim=-1),
        dequantized[:, :512].float(),
    ).to(torch.bfloat16)
    torch.testing.assert_close(output.view_as(expected), expected, rtol=5e-2, atol=5e-2)
