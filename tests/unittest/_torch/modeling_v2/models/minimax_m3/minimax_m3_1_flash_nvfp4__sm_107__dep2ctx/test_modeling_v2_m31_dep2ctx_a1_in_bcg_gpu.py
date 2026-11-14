# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_A1_FUSED_IN_BCG at the kernel level: the fused QKV GEMM + NVFP4 producer + KV insert captured in a
CUDA graph at a bucket's rows replays bit for bit what it runs eagerly. GPU (sm_107).

The served path captures the kernel with the step metadata's persistent out-cache-loc buffer and the KV pool views;
each replay reads the step's slots from that buffer, whose tail past the live rows is -1 (prepare()). So:

- a replay equals an eager run at the same rows and slots on every output byte (q, index-q, the NVFP4 K/V data and
  scale pages, the FP8 index-K pages), for two different live counts / slot maps replayed from one capture;
- pages no live slot names keep their canary bytes (pad rows' -1 slots store nothing);
- the live rows' q / index-q equal an eager run over the live rows alone (rows are independent in the 2-CTA M256
  mainloop and the tile config is the same for every M).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401

_SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
pytestmark = pytest.mark.skipif(_SM != (10, 7), reason="the A1 fused QKV kernel is sm_107 only")

HQ, HKV, HI, K = 64, 4, 4, 6144
N = (HQ + 2 * HKV + HI + 1) * 128
PAGES = 160
CANARY = 0x5A
MAX_POS = 32768


def _setup(rows: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, K), generator=gen, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn((N, K), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    weight, weight_sf = torch.ops.trtllm.mxfp8_quantize(w, True)
    angles = torch.rand((MAX_POS, 32), generator=gen, device="cuda") * 6.28
    rope = torch.stack((angles.cos(), angles.sin()), dim=1).contiguous()
    norms = tuple(
        (torch.randn(128, generator=gen, device="cuda") * 0.1).to(torch.bfloat16) for _ in range(4)
    )
    return dict(
        act=act.view(rows, -1),
        act_sf=act_sf,
        weight=weight.view(N, K),
        weight_sf=weight_sf,
        alpha=torch.ones(1, device="cuda"),
        kv_quant_scale=torch.tensor([1.0, 0.5, 0.25], device="cuda"),
        rope=rope,
        positions=(torch.arange(rows, device="cuda", dtype=torch.int32) * 7) % MAX_POS,
        norms=torch.cat(norms),
    )


def _caches():
    kv = torch.full((PAGES, 2, HKV, 128, 64), CANARY, dtype=torch.uint8, device="cuda")
    kv_scale = torch.full((PAGES, 2, HKV, 128, 8), CANARY, dtype=torch.uint8, device="cuda")
    idx_k = torch.full((PAGES, 1, 128, 128), CANARY, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )
    return kv, kv_scale, idx_k


def _slots(rows: int, live: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm((PAGES - 8) * 128, generator=gen)[:live].to(torch.int32)
    loc = torch.full((rows,), -1, dtype=torch.int32)
    loc[:live] = slots
    return loc.cuda()


def _args(runner, inp, rows, q, iq, caches, loc, positions=None):
    kv, kv_scale, idx_k = caches
    return runner.make_args(
        inp["act"][:rows],
        inp["act_sf"],
        inp["weight"],
        inp["weight_sf"],
        inp["alpha"],
        q,
        iq,
        kv,
        kv_scale,
        idx_k,
        loc,
        inp["kv_quant_scale"],
        inp["rope"],
        inp["positions"][:rows] if positions is None else positions,
        inp["norms"],
        1e-6,
    )


def _eager(runner, inp, rows, loc):
    q = torch.empty((rows, HQ, 128), dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.empty((rows, HI, 128), dtype=torch.float8_e4m3fn, device="cuda")
    caches = _caches()
    runner.run_args(_args(runner, inp, rows, q, iq, caches, loc))
    torch.cuda.synchronize()
    return q, iq, caches


def _bytes(t):
    return t.view(torch.uint8)


@pytest.fixture(scope="module")
def runner():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.kernels.qkv_fusion.kernel import (  # noqa: E501
        M3QkvFusedRunner,
    )

    return M3QkvFusedRunner(num_heads_q=HQ, num_heads_kv=HKV, num_heads_index=HI)


@pytest.mark.parametrize(("bucket", "lives"), [(2048, (1500, 2048)), (16384, (9000, 4097))])
def test_a_captured_producer_replays_eager_bit_for_bit(runner, bucket, lives):
    inp = _setup(bucket)
    # The captured graph's persistent buffers: outputs, caches, the out-cache-loc slots.
    q = torch.empty((bucket, HQ, 128), dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.empty((bucket, HI, 128), dtype=torch.float8_e4m3fn, device="cuda")
    caches = _caches()
    loc = _slots(bucket, bucket, seed=1)
    args = _args(runner, inp, bucket, q, iq, caches, loc)
    runner.run_args(args)  # the bucket's eager warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        runner.run_args(args)
    torch.cuda.current_stream().wait_stream(stream)

    for seed, live in enumerate(lives, start=2):
        step_loc = _slots(bucket, live, seed=seed)
        loc.copy_(step_loc)  # prepare(): live slots, -1 tail
        for t in caches:
            _bytes(t).fill_(CANARY)
        graph.replay()
        torch.cuda.synchronize()
        want_q, want_iq, want_caches = _eager(runner, inp, bucket, step_loc)
        assert torch.equal(_bytes(q), _bytes(want_q))
        assert torch.equal(_bytes(iq), _bytes(want_iq))
        for got, want in zip(caches, want_caches):
            assert torch.equal(_bytes(got), _bytes(want))
        # Pages no live slot names were not written.
        used = torch.zeros(PAGES, dtype=torch.bool, device="cuda")
        used[(step_loc[:live] // 128).long()] = True
        for got in caches:
            assert bool((_bytes(got)[~used] == CANARY).all())
        # The live rows equal an eager run over the live rows alone.
        live_q, live_iq, _ = _eager(runner, inp, live, step_loc[:live].contiguous())
        assert torch.equal(_bytes(q[:live]), _bytes(live_q))
        assert torch.equal(_bytes(iq[:live]), _bytes(live_iq))
