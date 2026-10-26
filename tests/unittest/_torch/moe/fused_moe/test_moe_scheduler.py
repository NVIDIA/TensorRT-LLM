# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of external-communication MoE chunk scheduling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("enable_alltoall", [False, True])
@pytest.mark.parametrize("with_ids", [False, True])
@pytest.mark.parametrize(
    "counts,max_tokens,expected_work",
    [
        ([0, 640], 2048, 641),
        ([0, 640], 1024, 642),
        ([0] * 7 + [4096], 1024, 4320),
        ([1, 640], 1024, 642),
    ],
)
def test_empty_ranks_use_minimal_dummy_rows(
    counts, max_tokens, expected_work, with_ids, enable_alltoall
):
    """Keep collective sizes consistent, bound dummy work, and restore real outputs."""
    peer_sizes = []
    dispatched_rows = 0
    for rank, local_tokens in enumerate(counts):
        moe = SimpleNamespace(
            use_dp=True,
            comm=object(),
            enable_alltoall=enable_alltoall,
            parallel_size=len(counts),
            rank=rank,
            mapping=SimpleNamespace(tp_rank=rank),
            moe_max_num_tokens=max_tokens,
            backend=SimpleNamespace(
                capabilities=SimpleNamespace(supports_moe_lora=False),
                input_requirement=SimpleNamespace(requires_run_moe_workspace=False),
            ),
            aux_stream=None,
            repeat_idx=0,
            repeat_count=1,
            determine_communication_method=Mock(),
            _dp_padded_num_rows=ConfigurableMoE._dp_padded_num_rows,
        )
        moe.calculate_num_chunks = lambda sizes: ConfigurableMoE.calculate_num_chunks(moe, sizes)
        moe.split_chunk = lambda tokens, chunks: ConfigurableMoE.split_chunk(moe, tokens, chunks)
        scheduler = ExternalCommMoEScheduler(moe)
        x = torch.arange(local_tokens * 3, dtype=torch.float32).view(local_tokens, 3)
        logits = torch.zeros(local_tokens, 4)
        ids = torch.arange(local_tokens, dtype=torch.int32) if with_ids else None
        routing_aux = torch.arange(local_tokens * 2).view(local_tokens, 2)
        calls = []

        def run_chunk(x_chunk, logits_chunk, dtype, sizes, dp_padding, *args, **kwargs):
            ids_chunk = kwargs["input_ids"]
            aux_chunk = kwargs["routing_aux"]
            assert not dp_padding
            assert all(size > 0 for size in sizes)
            assert x_chunk.shape[0] == logits_chunk.shape[0] == aux_chunk.shape[0] == sizes[rank]
            assert ids_chunk is None or ids_chunk.shape[0] == x_chunk.shape[0]
            if local_tokens == 0:
                assert x_chunk.shape[0] == 1
                assert not x_chunk.count_nonzero()
                assert not logits_chunk.count_nonzero()
                assert not aux_chunk.count_nonzero()
                assert ids_chunk is None or not ids_chunk.count_nonzero()
            calls.append(list(sizes))
            return x_chunk + (ids_chunk[:, None] if ids_chunk is not None else 1)

        scheduler._forward_chunk_impl = run_chunk
        result = scheduler.forward(
            x,
            logits,
            do_finalize=True,
            output_dtype=x.dtype,
            all_rank_num_tokens=counts,
            use_dp_padding=False,
            input_ids=ids,
            routing_aux=routing_aux,
        )
        expected = x + (ids[:, None] if ids is not None else 1)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert result.shape == x.shape
        assert x.shape[0] == local_tokens
        assert ids is None or ids.shape[0] == local_tokens
        assert routing_aux.shape[0] == local_tokens
        num_chunks = moe.calculate_num_chunks(counts)
        moe.determine_communication_method.assert_called_once_with(
            [max(1, n) for n in counts], num_chunks
        )
        assert len(calls) == num_chunks
        peer_sizes.append(calls)
        dispatched_rows += sum(sizes[rank] for sizes in calls)
    assert all(sizes == peer_sizes[0] for sizes in peer_sizes)
    assert dispatched_rows == expected_work
