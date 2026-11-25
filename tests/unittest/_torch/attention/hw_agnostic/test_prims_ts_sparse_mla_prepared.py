# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only check of the sparse MLA exported kernel calling convention."""

from unittest.mock import Mock

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
def test_trusted_sparse_mla_matches_exported_launch_abi(
    extra_pool: bool, has_workspace: bool
) -> None:
    rows, heads, topk = 2, 64, 4
    extra_topk = 3 if extra_pool else 0
    query = torch.zeros(rows, 1, heads, 512, dtype=torch.bfloat16)
    out = torch.empty_like(query)
    indices = torch.zeros(rows, topk, dtype=torch.int32)
    lengths = torch.full((rows,), topk, dtype=torch.int32)
    extra_indices = torch.zeros(rows, max(extra_topk, 1), dtype=torch.int32)
    extra_lengths = torch.full((rows,), extra_topk, dtype=torch.int32)
    primary = torch.ones(8, 1, 512, dtype=torch.bfloat16)
    secondary = torch.full_like(primary, 2)
    core = torch.empty(1024 if has_workspace else 0, dtype=torch.int8)
    lse = torch.empty(rows, 1, heads, dtype=torch.float32)
    sinks = torch.zeros(heads)
    split_kv = 2 if has_workspace else 1
    compiled = Mock()

    wrapper = object.__new__(mla_decode.BatchSparseMLADecodePagedTSWrapper)
    wrapper._state = dict(
        device=query.device,
        num_qo_heads=heads,
        packed_query=False,
        kv_cache_format="native",
        extra_kv_cache_format="native",
        cache_dim=512,
        extra_cache_dim=512,
        max_topk=topk,
        max_extra_topk=extra_topk,
        use_attention_sinks=True,
        softmax_scale=0.5,
        output_scale=1.0,
        kv_cache_scale=1.0,
        output_quant=None,
        return_lse=False,
        reload_d576=False,
        requested_splits=split_kv,
        buffers={"core": core, "public_lse": lse},
        dummy_cache=secondary,
        dummy_indices=extra_indices,
        default_cl=extra_lengths,
        view_cache={
            name: mla_decode._TensorViewCache()
            for name in ("query", "kv_cache", "extra_kv_cache", "indices", "out", "lse")
        },
        view_context=None,
        compiled=compiled,
    )
    q_view = query.permute(2, 3, 1, 0)
    expected = (
        q_view,
        q_view,
        primary.permute(1, 2, 0),
        secondary.permute(1, 2, 0),
        indices.T,
        out.permute(2, 3, 1, 0),
        lse.permute(2, 1, 0),
        core if has_workspace else None,
        split_kv,
        lengths,
        None,
        (indices, extra_indices, lengths, extra_lengths, sinks, 1.0),
        0.5,
        1.0,
        None,
        1,
        0,
    )
    # The third launch reuses bindings admitted on the second observation.
    for _ in range(3):
        result = wrapper.run(
            query,
            primary,
            indices,
            lengths,
            secondary if extra_pool else None,
            extra_indices=extra_indices if extra_pool else None,
            extra_lengths=extra_lengths if extra_pool else None,
            attention_sinks=sinks,
            out=out,
            validate=False,
        )
        assert result is out
        assert not compiled.call_args.kwargs
        _assert_same_argument(compiled.call_args.args, expected)
    assert compiled.call_count == 3
