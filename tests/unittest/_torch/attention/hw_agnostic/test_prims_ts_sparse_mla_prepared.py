# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for the prepared sparse MLA launch contract."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode

pytestmark = pytest.mark.cpu_only


def _assert_same_argument(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.data_ptr() == expected.data_ptr()
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_same_argument(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize("extra_pool", [False, True])
@pytest.mark.parametrize("has_workspace", [False, True])
def test_prepared_sparse_mla_matches_full_launch_abi(extra_pool: bool, has_workspace: bool) -> None:
    rows, heads, topk = 2, 64, 4
    extra_topk = 3 if extra_pool else 0
    query = torch.zeros(rows, 1, heads, 512, dtype=torch.bfloat16)
    out = torch.empty_like(query)
    indices = torch.zeros(rows, topk, dtype=torch.int32)
    lengths = torch.full((rows,), topk, dtype=torch.int32)
    extra_indices = torch.zeros(rows, max(extra_topk, 1), dtype=torch.int32)
    extra_lengths = torch.full((rows,), extra_topk, dtype=torch.int32)
    primary = torch.ones(8, 1, 512, dtype=torch.bfloat16)
    secondary = torch.full_like(primary, 2) if extra_pool else primary
    core = torch.empty(1024 if has_workspace else 0, dtype=torch.int8)
    lse = torch.empty(rows, 1, heads, dtype=torch.float32)
    sinks = torch.zeros(heads)
    split_kv = 2 if has_workspace else 1
    calls = []

    def compiled(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_offsets: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
        workspace: torch.Tensor | None,
        splits: int,
        cache_seqs: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        sparse_inputs: tuple,
        softmax_scale: float,
        output_scale: float,
        output_quant_args: tuple | None,
        max_q_len: int,
        softmax_stats: torch.Tensor | None,
        kv_bounds: torch.Tensor | None,
    ) -> None:
        # The exported TVM-FFI callable has no Python defaults for these slots.
        calls.append(
            (
                q_latent,
                q_rope,
                c_latent,
                c_rope,
                page_offsets,
                output,
                softmax_lse,
                workspace,
                splits,
                cache_seqs,
                cu_seqlens_q,
                sparse_inputs,
                softmax_scale,
                output_scale,
                output_quant_args,
                max_q_len,
                softmax_stats,
                kv_bounds,
            )
        )

    wrapper = object.__new__(mla_decode.BatchSparseMLADecodePagedTSWrapper)
    wrapper._state = dict(
        batch_size=rows,
        num_qo_heads=heads,
        device=query.device,
        q_dtype=query.dtype,
        requested_splits=split_kv,
        buffers={"core": core, "public_lse": lse},
        compiled=compiled,
    )
    prepared = dict(
        ks=topk,
        kc=extra_topk,
        dummy_indices=extra_indices,
        default_cl=extra_lengths,
        c_latent=primary.permute(1, 2, 0),
        c_rope=secondary.permute(1, 2, 0),
        sinks=sinks,
        bmm=(0.5, 1.0),
    )
    assert (
        wrapper._run_prepared(
            prepared,
            query,
            indices,
            lengths,
            extra_indices if extra_pool else None,
            extra_lengths if extra_pool else None,
            out,
        )
        is out
    )
    assert len(calls) == 1
    assert calls[0][-4:] == (None, 1, None, None)

    runtime = mla_decode._MLARuntime(
        query=query,
        normalized_cache=primary,
        out=out,
        num_physical_pages=primary.shape[0],
        softmax_scale=0.5,
        output_scale=1.0,
        extra_cache=secondary,
        sparse_inputs=(indices, extra_indices, lengths, extra_lengths, sinks, 1.0),
    )
    assert (
        mla_decode._launch_mla_decode(
            runtime,
            block_tables=indices,
            seq_lens_kv=lengths,
            qo_indptr=None,
            packed_query=False,
            kv_lora_rank=512,
            split_kv=split_kv,
            max_q_len=1,
            workspace=mla_decode._MLAWorkspaceViews(core if has_workspace else None, lse),
            compiled=compiled,
        )
        is out
    )
    assert len(calls) == 2
    _assert_same_argument(calls[0], calls[1])
