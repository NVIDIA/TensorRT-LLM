# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_megamoe on the CPU: the bucket ladder, the stateless staging, the direct path, priming;
and rubin_megamoe_in_bucket, the launch on a bucket its producers filled.

The family launcher runs with the engine's MegaMoE op and symmetric-buffer provider replaced by
fakes that record what the kernel reads at each launch, and every tensor write between launches
is recorded too (a write is a kernel launch on the GPU). The entry's reference is checked against
the shared NVFP4 MoE reference on experts laid out as the weight table stores them. The cells run
across ranks in ``rubin_multirank/test_rubin_megamoe_ranks.py`` and
``rubin_multirank/test_rubin_megamoe_in_bucket_ranks.py``.
"""

from __future__ import annotations

import sys
import types
from typing import NamedTuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
    rubin_megamoe as K,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import rubin_megamoe as M
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_megamoe_in_bucket import (  # noqa: E501
    rubin_megamoe_in_bucket,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="certified on SM107 only",
    ),
]

H, TOP_K, LOCAL, INTER = 64, 4, 2, 32
SF_COLS = H // 16


class _Regions(NamedTuple):
    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_weights: torch.Tensor
    combine_output: torch.Tensor
    shared_workspace: torch.Tensor
    peer_offsets: list


class _Engine:
    """The engine surface the launcher calls, recording the providers and the launches."""

    def __init__(self, direct=()):
        self.direct = set(direct)
        self.providers = []
        self.launches = []
        self.tuning = False
        self.in_op = False  # the fake op's own writes stand for the kernel's

    def module(self):
        m = types.ModuleType("cute_dsl_megamoe_custom_op")
        m.query_megamoe_kernel_properties = lambda **kw: (
            64,
            kw["max_tokens_per_rank"] in self.direct,
        )
        m.get_megamoe_symm_provider = self.provider
        m.megamoe_activation_sf_bytes_per_row = lambda hidden: hidden // 16
        m._sm107_use_in_kernel_fc2_reduce = lambda t, apply_topk: apply_topk and t == 32
        m.set_active_megamoe_profiling_scratch = lambda regions: None
        m.set_active_megamoe_profiling_scratch_factory = lambda factory: None
        m.get_megamoe_profiling_scratch = lambda **kw: None
        return m

    def provider(self, **kw):
        t, k = kw["max_tokens_per_rank"], kw["num_topk"]
        self.providers.append((t, kw["rank"], kw["world_size"], kw["combine_format"]))
        regions = _Regions(
            torch.full((t, H // 2), 0xAB, dtype=torch.uint8),
            torch.full((t, SF_COLS), 0xCD, dtype=torch.uint8),
            torch.full((t, k), 9.0),
            torch.zeros(t, 1, H, dtype=torch.bfloat16),
            torch.zeros(kw["shared_workspace_bytes"], dtype=torch.uint8),
            [0, 1, 2, 3],
        )
        return types.SimpleNamespace(get_regions=lambda: regions)

    def op(self, **kw):
        self.in_op = True
        try:
            self._op(**kw)
        finally:
            self.in_op = False

    def _op(self, **kw):
        n = kw["num_tokens"]
        self.launches.append(
            dict(
                bucket=kw["max_tokens_per_rank"],
                n=n,
                ids=kw["topk_idx"].clone(),
                weights=kw["topk_weights"].clone(),
                ids_tensor=kw["topk_idx"],
                weights_tensor=kw["topk_weights"],
                activation=kw["activation"],
                activation_sf=kw["activation_sf"],
                activation_rows=kw["activation"].view(torch.uint8)[:n].clone(),
                sf_rows=kw["activation_sf"].view(torch.uint8)[:n].clone(),
                fc1_sf_shape=tuple(kw["fc1_weight_sf"].shape),
                fc2_sf_shape=tuple(kw["fc2_weight_sf"].shape),
                situ=(kw["act_alpha"], kw["act_beta"]),
                apply_topk_in_fc1=kw["apply_topk_in_fc1"],
                combine_format=kw["combine_format"],
            )
        )
        # The routed rows: their activation bytes, so a caller can see which rows it got.
        kw["combine_output"][:n, 0, : H // 2] = (
            kw["activation"].view(torch.uint8)[:n].to(torch.bfloat16)
        )


class _Writes(TorchDispatchMode):
    """Every in-place tensor write outside the fake op, by aten name: on the GPU each is a launch."""

    def __init__(self, engine):
        super().__init__()
        self.engine, self.ops = engine, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if not self.engine.in_op and (name.endswith("_") or name == "copy_"):
            self.ops.append(name)
        return func(*args, **(kwargs or {}))


@pytest.fixture
def engine(monkeypatch):
    e = _Engine(direct=(96,))
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op", e.module()
    )
    tuner = types.SimpleNamespace(get=lambda: types.SimpleNamespace(is_tuning_mode=e.tuning))
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.autotuner",
        types.SimpleNamespace(AutoTuner=tuner),
    )
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_megamoe_nvfp4_blackwell", e.op, raising=False)
    K._PRIMED.clear()
    return e


def _ws(ep_size=4, rank=1, rungs=K.SM107_RUNGS, max_num_tokens=1024, chunk=16384):
    return K.build_workspace(
        object(),
        rank,
        ep_size,
        LOCAL,
        H,
        INTER,
        TOP_K,
        max_num_tokens,
        chunk,
        torch.device("cpu"),
        situ=(4.0, 25.0),
        rungs=rungs,
    )


def _experts():
    e = LOCAL
    return (
        torch.zeros(e, 2 * INTER, H // 2, dtype=torch.uint8),
        torch.zeros(e, 2 * INTER, H // 16, dtype=torch.uint8),
        torch.zeros(e, H, INTER // 2, dtype=torch.uint8),
        torch.zeros(e, H, INTER // 16, dtype=torch.uint8),
        torch.ones(e),
        torch.ones(e),
        torch.ones(e),
    )


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    x4 = torch.randint(0, 256, (n, H // 2), dtype=torch.uint8, generator=g)
    sf = torch.randint(0, 256, (n, SF_COLS), dtype=torch.uint8, generator=g)
    ids = torch.randint(0, 4 * LOCAL, (n, TOP_K), dtype=torch.int32, generator=g)
    weights = torch.rand(n, TOP_K, generator=g)
    return x4, sf, ids, weights


def test_ladder():
    assert K.ladder(8192) == (32, 64, 128, 192, 256, 384, 512, 1024, 4096, 8192)
    assert K.ladder(1024) == (32, 64, 128, 192, 256, 384, 512, 1024)
    assert K.ladder(0) == K.SM107_RUNGS
    assert K.ladder(1024, (*K.SM107_RUNGS, 96)) == (32, 64, 96, 128, 192, 256, 384, 512, 1024)


def test_buckets_built_in_ladder_order(engine):
    """One rendezvous per bucket, ascending, on every rank the same; ids start at -1."""
    ws = _ws(rungs=(*K.SM107_RUNGS, 96))
    want = [32, 64, 96, 128, 192, 256, 384, 512, 1024]
    assert [b.max_tokens for b in ws.buckets] == want
    assert [p[0] for p in engine.providers] == want
    assert {p[1:] for p in engine.providers} == {(1, 4, "bf16")}
    assert [b.direct_inputs for b in ws.buckets] == [t == 96 for t in want]
    assert all(bool((b.topk_idx == -1).all()) for b in ws.buckets)
    assert all(b.topk_idx.dtype == torch.int32 for b in ws.buckets)
    assert not any(b.in_kernel_fc2_reduce for b in ws.buckets)  # weights apply after FC2


def test_bucket_is_rank_identical(engine):
    """The bucket depends on the chunk's largest row count only, so every rank picks the same."""
    ws = _ws()
    for chunk_max in (1, 32, 33, 96, 128, 129, 1024):
        picked = {K.select_bucket(ws, chunk_max, False).max_tokens for _ in range(4)}
        assert len(picked) == 1 and min(picked) >= chunk_max
    assert K.select_bucket(ws, 96, False).max_tokens == 128
    assert K.select_bucket(ws, 0, False).max_tokens == 1024
    with pytest.raises(RuntimeError, match="exceed the top bucket"):
        K.select_bucket(ws, 1025, False)


def test_tail_after_a_shrink_on_every_layer(engine):
    """96 rows then 40 in the same bucket, layer after layer: every launch of 40 rows sees ids -1
    and weights 0 past its rows, whatever an earlier layer or step left there."""
    ws, ex = _ws(), _experts()
    big = _rows(96, 1)
    small = _rows(40, 2)
    for _layer in range(4):
        K.run(ws, *big, ex, 96, 96)
    for _layer in range(4):
        K.run(ws, *small, ex, 96, 40)
    for launch in engine.launches[4:]:
        assert launch["bucket"] == 128 and launch["n"] == 40
        assert torch.equal(launch["ids"][:40], small[2])
        assert bool((launch["ids"][40:] == -1).all()), "stale ids past the rows"
        assert torch.equal(launch["weights"][:40], small[3])
        assert not launch["weights"][40:].any(), "stale weights past the rows"
        assert torch.equal(launch["activation_rows"], small[0])


@pytest.mark.parametrize("order", [(40, 96, 40, 96), (96, 40, 96, 40)])
def test_replay_after_any_size(engine, order):
    """A call's writes depend only on its own rows: a graph captured at one row count replays
    the same staging after any other. A tail a replay of a larger size left behind is reset."""
    ws, ex = _ws(), _experts()
    rows = {40: _rows(40, 3), 96: _rows(96, 4)}
    seen = {}
    for n in order:
        bucket = K.select_bucket(ws, 96, False)
        bucket.topk_idx.fill_(7)  # what another graph's replay may have left
        bucket.topk_weights.fill_(3.0)
        K.run(ws, *rows[n], ex, 96, n)
        launch = engine.launches[-1]
        snap = (launch["ids"], launch["weights"], launch["activation_rows"])
        if n in seen:
            assert all(torch.equal(a, b) for a, b in zip(seen[n], snap))
        seen[n] = snap
        assert bool((launch["ids"][n:] == -1).all()) and not launch["weights"][n:].any()


def test_direct_inputs_in_a_full_bucket(engine):
    """A bucket of exactly the step's rows on the generation kernel reads the rows in place."""
    ws, ex = _ws(rungs=(*K.SM107_RUNGS, 96)), _experts()
    x4, sf, ids, weights = _rows(96, 5)
    bucket = K.select_bucket(ws, 96, False)
    assert bucket.max_tokens == 96 and bucket.direct_inputs
    out = K.run(ws, x4, sf, ids, weights, ex, 96, 96)
    launch = engine.launches[-1]
    assert launch["activation"].data_ptr() == x4.data_ptr()
    assert torch.equal(launch["ids"], ids) and torch.equal(launch["weights"], weights)
    assert bool((bucket.activation == 0xAB).all()), "the bucket's buffers were written"
    assert bool((bucket.topk_idx == -1).all())
    assert out.shape == (96, H) and out.dtype == torch.bfloat16
    # Fewer rows than the bucket: staged.
    K.run(ws, *_rows(80, 6), ex, 96, 80)
    assert engine.launches[-1]["activation"].data_ptr() == bucket.activation.data_ptr()


def test_a_rank_without_rows_still_launches(engine):
    ws, ex = _ws(), _experts()
    empty = (
        torch.empty(0, H // 2, dtype=torch.uint8),
        torch.empty(0, SF_COLS, dtype=torch.uint8),
        torch.empty(0, TOP_K, dtype=torch.int32),
        torch.empty(0, TOP_K),
    )
    out = K.run(ws, *empty, ex, 40, 0)
    assert out.shape == (0, H)
    launch = engine.launches[-1]
    assert launch["n"] == 0 and launch["bucket"] == 64
    assert bool((launch["ids"] == -1).all()) and not launch["weights"].any()
    # One rank alone has no peer to wait for: no launch.
    solo = _ws(ep_size=1, rank=0)
    count = len(engine.launches)
    assert K.run(solo, *empty, ex, 40, 0).shape == (0, H)
    assert len(engine.launches) == count


def test_launch_arguments(engine):
    """The engine op's arguments: flat per-expert scales, SiTU, weights after FC2, bf16 combine."""
    ws, ex = _ws(), _experts()
    K.run(ws, *_rows(8, 7), ex, 8, 8)
    launch = engine.launches[-1]
    assert launch["fc1_sf_shape"] == (LOCAL, 2 * INTER * H // 16)
    assert launch["fc2_sf_shape"] == (LOCAL, H * INTER // 16)
    assert launch["situ"] == (4.0, 25.0) and launch["apply_topk_in_fc1"] is False
    assert launch["combine_format"] == "bf16" and launch["bucket"] == 32


def test_priming_once_in_the_tuning_pass(engine):
    """Tuning launches every other bucket once with no rows, in ladder order, then never again."""
    ws, ex = _ws(), _experts()
    engine.tuning = True
    K.run(ws, *_rows(96, 8), ex, 96, 96)
    primes = [(launch["bucket"], launch["n"]) for launch in engine.launches[:-1]]
    assert primes == [(t, 0) for t in (32, 64, 192, 256, 384, 512, 1024)]
    assert (engine.launches[-1]["bucket"], engine.launches[-1]["n"]) == (128, 96)
    K.run(ws, *_rows(96, 9), ex, 96, 96)
    assert len(engine.launches) == len(primes) + 2
    engine.tuning = False


def test_entry_refuses_what_the_kernel_reads_wrong():
    ws = types.SimpleNamespace(
        hidden=H,
        top_k=TOP_K,
        num_local_experts=LOCAL,
        intermediate=INTER,
        buckets=(types.SimpleNamespace(max_tokens=128),),
    )
    x4, sf, ids, weights = _rows(8, 10)
    ex = list(_experts())
    M.rubin_megamoe.is_valid(x4, sf, ids, weights, *ex, ws, 8, 8)
    with pytest.raises(AssertionError):
        M.rubin_megamoe.is_valid(x4, sf, ids.long(), weights, *ex, ws, 8, 8)
    with pytest.raises(AssertionError, match="expanded"):
        bad = list(ex)
        bad[6] = torch.ones(1).expand(LOCAL)
        M.rubin_megamoe.is_valid(x4, sf, ids, weights, *bad, ws, 8, 8)
    with pytest.raises(AssertionError, match="top bucket"):
        M.rubin_megamoe.is_valid(x4, sf, ids, weights, *ex, ws, 129, 8)


# --- rubin_megamoe_in_bucket: the producers write the bucket, the launch is the kernel alone ---


def _fill_bucket(bucket, x4, sf, ids, weights, n):
    """What rubin_route_topk_out and rubin_megamoe_quantize write into a bucket, on the CPU."""
    bucket.activation[:n] = x4
    bucket.activation_sf[:n] = sf
    bucket.topk_idx[:n] = ids
    bucket.topk_weights[:n, :TOP_K] = weights
    bucket.topk_idx[n:] = -1
    bucket.topk_weights[n:, :TOP_K] = 0


def test_bucket_for_is_the_ladders_bucket(engine):
    """The producers write the bucket the launch reads: select_bucket's, the same on every rank."""
    ws = _ws()
    for chunk_max, n in ((1, 1), (96, 96), (96, 40), (129, 129), (1024, 7)):
        assert K.bucket_for(ws, chunk_max, n) is K.select_bucket(ws, chunk_max, False)
        assert M.megamoe_bucket(ws, chunk_max, n) is K.select_bucket(ws, chunk_max, False)
    with pytest.raises(RuntimeError, match="rows in a bucket of 128"):
        K.bucket_for(ws, 96, 129)


def test_in_bucket_launch_is_the_kernel_alone(engine):
    """96 rows in bucket 128: the ladder writes six times before its launch (the ids' tail, four
    copies, the weights' tail), the launch on a filled bucket writes nothing before it."""
    ws, ex = _ws(), _experts()
    rows = _rows(96, 21)
    with _Writes(engine) as writes:
        K.run(ws, *rows, ex, 96, 96)
    assert writes.ops == ["fill_", "copy_", "copy_", "copy_", "copy_", "zero_"], writes.ops
    bucket = M.megamoe_bucket(ws, 96, 96)
    _fill_bucket(bucket, *rows, 96)
    count = len(engine.launches)
    with _Writes(engine) as writes:
        out = rubin_megamoe_in_bucket(bucket, *ex, ws, 96)
    assert writes.ops == [] and len(engine.launches) == count + 1
    assert out.shape == (96, H) and out.dtype == torch.bfloat16
    # The direct path's full bucket also writes nothing; the in-bucket launch never stages.
    ws96 = _ws(rungs=(*K.SM107_RUNGS, 96))
    with _Writes(engine) as writes:
        K.run(ws96, *_rows(96, 22), ex, 96, 96)
    assert writes.ops == []


def test_in_bucket_reads_what_the_ladder_stages(engine):
    """The out_bound shape: the kernel reads the bucket's own buffers (strides and all), and the
    rows and the tail it sees are the ones the ladder stages for the same rows."""
    ws, ex = _ws(), _experts()
    rows = _rows(96, 23)
    K.run(ws, *rows, ex, 96, 96)
    ladder = engine.launches[-1]
    bucket = M.megamoe_bucket(ws, 96, 96)
    bucket.topk_idx.fill_(7)  # what another graph's replay may have left
    bucket.topk_weights.fill_(3.0)
    _fill_bucket(bucket, *rows, 96)
    rubin_megamoe_in_bucket(bucket, *ex, ws, 96)
    mine = engine.launches[-1]
    assert mine["bucket"] == ladder["bucket"] == 128 and mine["n"] == ladder["n"] == 96
    for key in ("activation", "activation_sf"):
        assert mine[key].data_ptr() == getattr(bucket, key).data_ptr(), key
        assert mine[key].shape == getattr(bucket, key).shape and mine[key].is_contiguous()
    assert mine["ids_tensor"].data_ptr() == bucket.topk_idx.data_ptr()
    assert mine["weights_tensor"].data_ptr() == bucket.topk_weights.data_ptr()
    assert tuple(mine["weights_tensor"].shape) == (128, TOP_K)
    for key in ("ids", "weights", "activation_rows", "sf_rows"):
        assert torch.equal(mine[key], ladder[key]), key
    assert bool((mine["ids"][96:] == -1).all()) and not mine["weights"][96:].any()


def test_in_bucket_primes_like_the_ladder(engine):
    """The tuning pass primes every other bucket once, in ladder order, whichever entry runs."""
    ws, ex = _ws(), _experts()
    engine.tuning = True
    bucket = M.megamoe_bucket(ws, 96, 96)
    _fill_bucket(bucket, *_rows(96, 24), 96)
    rubin_megamoe_in_bucket(bucket, *ex, ws, 96)
    primes = [(launch["bucket"], launch["n"]) for launch in engine.launches[:-1]]
    assert primes == [(t, 0) for t in (32, 64, 192, 256, 384, 512, 1024)]
    assert (engine.launches[-1]["bucket"], engine.launches[-1]["n"]) == (128, 96)
    K.run(ws, *_rows(96, 25), ex, 96, 96)
    assert len(engine.launches) == len(primes) + 2
    engine.tuning = False


def test_in_bucket_on_one_rank_without_rows(engine):
    """One rank alone has no peer to wait for: no rows, no launch (as rubin_megamoe)."""
    solo, ex = _ws(ep_size=1, rank=0), _experts()
    bucket = M.megamoe_bucket(solo, 40, 0)
    count = len(engine.launches)
    assert rubin_megamoe_in_bucket(bucket, *ex, solo, 0).shape == (0, H)
    assert len(engine.launches) == count


def test_in_bucket_entry_refuses_what_the_kernel_reads_wrong(engine):
    ws, ex = _ws(), _experts()
    bucket = M.megamoe_bucket(ws, 96, 96)
    _fill_bucket(bucket, *_rows(96, 26), 96)
    rubin_megamoe_in_bucket.is_valid(bucket, *ex, ws, 96)
    with pytest.raises(AssertionError, match="this workspace"):
        rubin_megamoe_in_bucket.is_valid(_ws().buckets[2], *ex, ws, 96)
    with pytest.raises(AssertionError):
        rubin_megamoe_in_bucket.is_valid(bucket, *ex, ws, 129)
    bucket.topk_idx[100] = 3  # a row past the rows that still routes somewhere
    with pytest.raises(AssertionError, match="-1"):
        rubin_megamoe_in_bucket.is_valid(bucket, *ex, ws, 96)
    bucket.topk_idx[100] = -1
    bucket.topk_weights[127, 0] = 0.5
    with pytest.raises(AssertionError, match="weights past the rows"):
        rubin_megamoe_in_bucket.is_valid(bucket, *ex, ws, 96)
    bucket.topk_weights[127, 0] = 0
    bad = list(ex)
    bad[6] = torch.ones(1).expand(LOCAL)
    with pytest.raises(AssertionError, match="expanded"):
        rubin_megamoe_in_bucket.is_valid(bucket, *bad, ws, 96)


def test_in_bucket_reference_is_rubin_megamoes():
    """The in-bucket entry's reference reads the rows the bucket holds through rubin_megamoe's."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
        SF_VEC,
    )

    # Rows of 128 (the reference unswizzles 128x4 scale tiles).
    hidden, inter, local, ranks, n, t = 256, 128, 2, 2, 3, 8
    g = torch.Generator().manual_seed(31)

    def u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def scales(*shape):  # E4M3 bytes 0x30-0x3f: finite, positive
        return torch.randint(0x30, 0x40, shape, dtype=torch.uint8, generator=g)

    experts = [
        (
            u8(local, 2 * inter, hidden // 2),
            scales(local, 2 * inter, hidden // SF_VEC),
            u8(local, hidden, inter // 2),
            scales(local, hidden, inter // SF_VEC),
            torch.full((local,), 0.03),
            torch.full((local,), 0.03),
            torch.ones(local),
        )
        for _ in range(ranks)
    ]
    every = [list(e) for e in zip(*experts)]
    x4, sf = u8(n, hidden // 2), scales(n, hidden // SF_VEC)
    ids = torch.stack([torch.randperm(local * ranks, generator=g)[:TOP_K] for _ in range(n)])
    weights = torch.rand(n, TOP_K, generator=g)
    rank = 1
    bucket = types.SimpleNamespace(
        activation=torch.cat([x4, u8(t - n, hidden // 2)]),
        activation_sf=torch.cat([sf, scales(t - n, hidden // SF_VEC)]),
        topk_idx=torch.cat(
            [ids.to(torch.int32), torch.full((t - n, TOP_K), -1, dtype=torch.int32)]
        ),
        topk_weights=torch.cat([weights, torch.zeros(t - n, TOP_K)]),
    )
    ws = types.SimpleNamespace(ep_rank=rank, top_k=TOP_K)
    got = rubin_megamoe_in_bucket.reference(bucket, *every, ws, n)
    want = M.rubin_megamoe.reference(
        {rank: x4}, {rank: sf}, {rank: ids.to(torch.int32)}, {rank: weights}, *every, ws, n, n
    )
    assert torch.equal(got, want)


# --- the entry's reference against the shared NVFP4 MoE reference ---


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """The expert loader's 128x4 block-scale swizzle (weights.py, ``w2_rows``)."""
    m, c = s.shape
    return s.reshape(m // 128, 4, 32, c // 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def _gate_up16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The expert loader's FC1 row order (weights.py, ``w2_rows``): 16 gate rows, then 16 up
    rows."""
    rows, cols = gate.shape
    return torch.stack((gate.reshape(-1, 16, cols), up.reshape(-1, 16, cols)), 1).reshape(
        2 * rows, cols
    )


def test_reference_against_the_shared_moe_reference():
    """Two ranks of two experts, laid out as the weight table stores them; every rank's rows
    against ``nvfp4_situ_moe_reference`` over all four experts as the checkpoint stores them."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
        _e2m1_rne,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import (
        nvfp4_situ_moe_reference,
    )

    hidden, inter, local, ranks, top_k = 256, 128, 2, 2, 3
    g = torch.Generator().manual_seed(11)

    def proj(n, k):
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
        steps = torch.randint(120, 132, (n, k // 16), generator=g).float()
        return codes, (steps / 126.0).to(torch.float8_e4m3fn), torch.tensor(0.03), torch.tensor(1.0)

    experts = [
        {"w1": proj(inter, hidden), "w3": proj(inter, hidden), "w2": proj(hidden, inter)}
        for _ in range(local * ranks)
    ]
    fc31_in = fc2_in = 1.0  # 1 / max input_scale
    per_rank = []
    for r in range(ranks):
        mine = experts[r * local : (r + 1) * local]
        per_rank.append(
            (
                torch.stack([_gate_up16(e["w1"][0], e["w3"][0]) for e in mine]),
                torch.stack(
                    [
                        _swizzle(
                            _gate_up16(e["w1"][1].view(torch.uint8), e["w3"][1].view(torch.uint8))
                        )
                        for e in mine
                    ]
                ),
                torch.stack([e["w2"][0] for e in mine]),
                torch.stack([_swizzle(e["w2"][1].view(torch.uint8)) for e in mine]),
                torch.stack([e["w1"][2] / fc31_in for e in mine]),
                torch.stack([e["w2"][2] / fc2_in for e in mine]),
                torch.stack([1.0 / e["w2"][3] for e in mine]),
            )
        )
    rows = 3
    xs, x4s, sfs, idss, wss = [], [], [], [], []
    table = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
    for r in range(ranks):
        x = torch.randn(rows, hidden, generator=g) * 0.2
        # fp4_quantize's codes and scales: what quantize_nvfp4 dequantizes (global scale 1).
        blocks = x.view(rows, hidden // 16, 16)
        sf = (blocks.abs().amax(-1, keepdim=True) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
        scale = torch.where(sf.float() == 0, torch.zeros(()), 1.0 / sf.float())
        values = _e2m1_rne(blocks * scale).view(rows, hidden)
        mag = (values.abs().unsqueeze(-1) - table).abs().argmin(-1)
        nib = (mag | ((values < 0).long() << 3)).to(torch.uint8).view(rows, hidden // 2, 2)
        x4s.append(nib[..., 0] | (nib[..., 1] << 4))
        sfs.append(sf.view(torch.uint8).view(rows, hidden // 16))
        xs.append(x)
        ids = torch.stack([torch.randperm(local * ranks, generator=g)[:top_k] for _ in range(rows)])
        idss.append(ids.to(torch.int32))
        wss.append(torch.rand(rows, top_k, generator=g))
    args = [list(t) for t in zip(*per_rank)]
    for r in range(ranks):
        ws = types.SimpleNamespace(ep_rank=r)
        got = M.rubin_megamoe.reference(x4s, sfs, idss, wss, *args, ws, rows, rows)
        want = nvfp4_situ_moe_reference(
            xs[r],
            idss[r],
            wss[r],
            experts,
            local_expert_offset=0,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
        # The same values and weights; the products accumulate in another order, and a borderline
        # activation may land on a neighbouring E2M1 step before FC2.
        cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0)
        assert cos > 0.9999, cos
        assert (got - want).abs().max() <= 0.05 * want.abs().max()


def test_the_engine_backends_variables_must_give_the_launch_settings():
    """``check_megamoe_lever_env``: the engine backend's variables, read as it reads them, must
    leave its launch settings at the flat target's constants."""
    M.check_megamoe_lever_env({})
    M.check_megamoe_lever_env(
        {
            "MEGAMOE_COMBINE_FORMAT": M.COMBINE_FORMAT,
            "MEGAMOE_TACTIC_AUTOTUNE": "0",
            "MEGAMOE_PRIME_LADDER": "1",
        }
    )
    for name, raw in (
        ("MEGAMOE_COMBINE_FORMAT", "32e4m3xe8m0"),
        ("MEGAMOE_TACTIC_AUTOTUNE", "1"),
        ("MEGAMOE_PRIME_LADDER", "0"),
    ):
        with pytest.raises(ValueError, match=name):
            M.check_megamoe_lever_env({name: raw})
