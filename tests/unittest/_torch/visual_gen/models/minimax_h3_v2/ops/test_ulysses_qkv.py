# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _check_exchange(rank: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.ulysses_qkv import (
        ulysses_qkv_all_to_all,
        ulysses_unscatter_qkv,
    )

    dist.init_process_group(
        "nccl", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=90)
    )
    try:
        for cell in ulysses_qkv_all_to_all.CELLS:
            spec = cell.spec
            per_rank = []
            for source in range(2):
                generator = torch.Generator().manual_seed(source)
                data = torch.randn(3, 2, 5, spec["heads"], spec["dim"], generator=generator)
                q, k, v = data.to(device=rank, dtype=spec["dtype"]).unbind(0)
                per_rank.append((q, k, v))
            q, k, v = per_rank[rank]
            for hnd in (False, True):
                kernel = torch.ops.trtllm.ulysses_packed_qkv_post_unscatter
                with patch.object(
                    torch.ops.trtllm, "ulysses_packed_qkv_post_unscatter", wraps=kernel
                ) as fused:
                    actual = ulysses_qkv_all_to_all(q, k, v, dist.group.WORLD, hnd)
                assert fused.called == spec["fused"]
                expected = ulysses_qkv_all_to_all.reference(per_rank, rank, hnd)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)

        send = torch.randn(2, 2, 5, 3, 8, dtype=torch.bfloat16, device=rank)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        ulysses_unscatter_qkv.is_valid(recv, recv, recv, hnd=True, aligned_receives=True)
        kernel = torch.ops.trtllm.ulysses_post_unscatter_qkv
        with patch.object(torch.ops.trtllm, "ulysses_post_unscatter_qkv", wraps=kernel) as fused:
            actual = ulysses_unscatter_qkv(recv, recv, recv, hnd=True, aligned_receives=True)
        fused.assert_called_once()
        expected = ulysses_unscatter_qkv.reference(recv, recv, recv, hnd=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.no_xdist
def test_ulysses_qkv_all_to_all(tmp_path, monkeypatch):
    if torch.cuda.device_count() < 2:
        pytest.skip("Ulysses head exchange needs two GPUs")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    mp.spawn(_check_exchange, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)


def test_ulysses_unscatter_qkv():
    from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.ulysses_qkv import (
        ulysses_unscatter_qkv,
    )

    for cell in ulysses_unscatter_qkv.CELLS:
        spec = cell.spec
        stride = spec.get("stride", 1)
        q, k, v = (
            torch.randn(
                2, 2, rows, spec["heads"], spec["dim"] * stride, dtype=spec["dtype"], device="cuda"
            )[..., ::stride]
            for rows in (5, 3, 3)
        )
        for hnd in (False, True):
            kernel = torch.ops.trtllm.ulysses_post_unscatter_qkv
            with patch.object(
                torch.ops.trtllm, "ulysses_post_unscatter_qkv", wraps=kernel
            ) as fused:
                ulysses_unscatter_qkv.is_valid(q, k, v, hnd, aligned_receives=True)
                actual = ulysses_unscatter_qkv(q, k, v, hnd, aligned_receives=True)
            assert fused.called == spec["fused"]
            expected = ulysses_unscatter_qkv.reference(q, k, v, hnd)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for out in actual:
                assert out.is_contiguous() == (not hnd or not spec["fused"])


def test_ulysses_unscatter_misaligned_buffer():
    from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.ops.ulysses_qkv import (
        ulysses_unscatter_qkv,
    )

    x = torch.arange(17, device="cuda", dtype=torch.bfloat16)[1:].view(2, 1, 1, 1, 8)
    with patch.object(torch.ops.trtllm, "ulysses_post_unscatter_qkv") as fused:
        actual = ulysses_unscatter_qkv(x, x, x, hnd=False)
    fused.assert_not_called()
    torch.testing.assert_close(
        actual, ulysses_unscatter_qkv.reference(x, x, x, hnd=False), rtol=0, atol=0
    )
    with pytest.raises(AssertionError, match="16-byte"):
        ulysses_unscatter_qkv.is_valid(x, x, x, hnd=False, aligned_receives=True)
