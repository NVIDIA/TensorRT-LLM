# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM fp8_paged_mqa_logits entry.

The keys live in a real INDEX_KEY pool of 64-token pages, scattered over random physical pages; the
schedule comes from the production schedule op (``paged_mqa_logits_schedule``), as at c1.

Most cells use integer data: Q and K are small integers (exact in E4M3), the key scales powers of
two and the weights small integers times 2^-4. Every product and partial sum is then an exactly
representable fp32 value, whatever the accumulation order, so the kernel must equal the fp64
reference bit for bit. One cell uses UE8M0-quantized random keys and real-valued weights, where the
accumulation order is visible and a band applies.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa.fp8_paged_mqa_logits import (  # noqa: E501
    fp8_paged_mqa_logits as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.dsa.paged_mqa_logits_schedule import (  # noqa: E501
    paged_mqa_logits_schedule,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "."]
import _indexer_inputs as inputs  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's entries are certified on sm_107"
)

assert torch.cuda.is_available(), "fp8_paged_mqa_logits requires a CUDA device"

_SPLIT = 256


def _sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _keys(count: int, data: str, gen: torch.Generator):
    """(E4M3 keys [count, 128], fp32 scales [count])."""
    if data == "int":
        keys = torch.randint(-3, 4, (count, inputs.HEAD_DIM), generator=gen).float()
        scales = torch.pow(2.0, torch.randint(-6, 2, (count,), generator=gen).float())
        return keys.to(torch.float8_e4m3fn), scales
    raw = torch.randn(count, inputs.HEAD_DIM, generator=gen)
    amax = raw.abs().amax(dim=1).clamp_min(1e-4)
    scales = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return (raw / scales[:, None]).to(torch.float8_e4m3fn), scales


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    lens = list(spec["context_lens"])
    batch, next_n, heads, d = len(lens), spec["next_n"], spec["num_heads"], spec["head_dim"]
    page = spec["tokens_per_block"]
    max_ctx = spec["max_context_len"]
    width = -(-max_ctx // page)
    pages_per_request = [-(-n // page) for n in lens]
    pool_pages = sum(pages_per_request) + 3
    cache, whole = inputs.k_cache_pool(pool_pages, spec["pool_layout"])
    order = torch.randperm(pool_pages, generator=gen).to(torch.int32)
    table = torch.zeros(batch, width, dtype=torch.int32)
    used = 0
    flat_pages = cache.reshape(pool_pages, -1)
    for b, n in enumerate(lens):
        count = pages_per_request[b]
        table[b, :count] = order[used : used + count]
        used += count
        keys, scales = _keys(count * page, spec["data"], gen)
        page_bytes = torch.cat(
            [
                keys.view(torch.uint8).reshape(count, page * d),
                scales.view(torch.uint8).reshape(count, page * 4),
            ],
            dim=1,
        )
        flat_pages[table[b, :count].long().cuda()] = page_bytes.cuda()
    if spec["data"] == "int":
        q = torch.randint(-3, 4, (batch, next_n, heads, d), generator=gen).float()
        weights = torch.randint(-3, 4, (batch * next_n, heads), generator=gen).float() * 2.0**-4
    else:
        q = torch.randn(batch, next_n, heads, d, generator=gen)
        weights = torch.randn(batch * next_n, heads, generator=gen) * 0.05
    context_lens = torch.tensor(lens, dtype=torch.int32).cuda()
    schedule = torch.empty(_sms() + 1, 2, dtype=torch.int32, device="cuda")
    paged_mqa_logits_schedule(context_lens, page, _sms(), schedule)
    logits = torch.full(
        (batch * next_n, -(-max_ctx // _SPLIT) * _SPLIT), float("nan"), device="cuda"
    )
    return dict(
        q=q.to(torch.float8_e4m3fn).cuda(),
        kv_cache=cache,
        weights=weights.cuda(),
        context_lens=context_lens,
        block_table=table.cuda(),
        schedule_meta=schedule,
        logits=logits,
        max_context_len=max_ctx,
        early_kv=spec["early_kv"],
    )


def _compare(out: torch.Tensor, ref: torch.Tensor, lens, next_n: int, exact: bool) -> None:
    """Request by request over its keys; the columns past a request's keys are unspecified."""
    for b, n in enumerate(lens):
        got = out[b * next_n : (b + 1) * next_n, :n]
        want = ref[b * next_n : (b + 1) * next_n, :n]
        if exact:
            assert torch.equal(got, want), f"request {b}: logits differ from the exact reference"
        else:
            # fp32 accumulation in another order: products of E4M3 values are exact, so the
            # difference is the rounding of the 128-term dots and of the 32-term weighted sum,
            # a few ulps of the partial sums. 64 ulps of each row's largest logit leave an order of
            # magnitude of margin; a wrong head, weight or key is off by the logits themselves.
            assert_within_ulp(got.contiguous(), want.contiguous(), element_ulp=64.0, rms_ulp=8.0)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**args)
    out = op(**args)
    torch.cuda.synchronize()
    assert out.data_ptr() == args["logits"].data_ptr(), "the op writes the caller's buffer"
    assert out.shape == (args["logits"].shape[0], cell.spec["max_context_len"])
    _compare(out, ref, cell.spec["context_lens"], cell.spec["next_n"], cell.spec["data"] == "int")


def test_early_kv_kernel_equals_the_default_kernel_bitwise():
    """The N split and the early KV loads move work, not arithmetic: same bits on real values."""
    spec = next(c.spec for c in op.CELLS if c.spec["data"] == "randn")
    args = _build(spec, seed=51)
    early = op(**dict(args, early_kv=True)).clone()
    default = op(**dict(args, early_kv=False))
    torch.cuda.synchronize()
    n = spec["context_lens"][0]
    assert torch.equal(early[:, :n], default[:, :n])


def test_cuda_graph_replay_reads_new_queries_and_keys():
    """The decode path is captured: a replay follows Q, the weights and the cache in place."""
    spec = op.CELLS[0].spec
    args = _build(spec, seed=52)
    op(**args)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(**args)
    fresh = _build(spec, seed=53)
    args["q"].copy_(fresh["q"])
    args["weights"].copy_(fresh["weights"])
    args["kv_cache"].copy_(fresh["kv_cache"])
    args["block_table"].copy_(fresh["block_table"])
    ref = op.reference(**args)
    graph.replay()
    torch.cuda.synchronize()
    _compare(args["logits"], ref, spec["context_lens"], spec["next_n"], exact=True)


def test_guard_refuses_a_block_table_too_narrow_for_the_keys():
    args = _build(op.CELLS[2].spec, seed=54)
    narrow = dict(
        args,
        context_lens=torch.tensor([5_000], dtype=torch.int32, device="cuda"),
        max_context_len=5_000,
        block_table=args["block_table"][:, :10].contiguous(),
    )
    with pytest.raises(AssertionError, match="cannot cover"):
        with validating(op):
            op(**narrow)


def test_guard_refuses_a_schedule_of_another_sm_count():
    args = _build(op.CELLS[2].spec, seed=55)
    schedule = torch.zeros(_sms(), 2, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="schedule_meta"):
        with validating(op):
            op(**dict(args, schedule_meta=schedule))


def test_guard_refuses_logits_without_room_for_whole_splits():
    args = _build(op.CELLS[2].spec, seed=56)
    narrow = torch.empty(args["logits"].shape[0], args["max_context_len"] - 64, device="cuda")
    with pytest.raises(AssertionError, match="whole"):
        with validating(op):
            op(**dict(args, logits=narrow))


def test_guard_refuses_strided_weights():
    args = _build(op.CELLS[2].spec, seed=57)
    strided = args["weights"].t().contiguous().t()
    assert not strided.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(**dict(args, weights=strided))
