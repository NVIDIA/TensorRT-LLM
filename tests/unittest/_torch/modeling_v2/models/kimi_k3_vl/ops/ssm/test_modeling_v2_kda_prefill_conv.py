# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_prefill_conv op, on a real cache object.

Kimi K3's prefill short convolution on the conv pool of a real ``KimiK3HybridCacheManager`` (6 heads for TP16, 24
for TP4; K = V = 128, conv width 4): the per-layer bf16 conv view ``[slots, 3 H 128, 3]`` (slot rows strided by the
manager's coalescing) at the slots the manager assigned, scattered, and the fused ``[q | k | v | g]`` projection
rows as the source. The reference is the entry's fp32 conv; the outputs are gated at the kernel's documented band
(0.8% relative, 1e-4 absolute, the band of the kernel's own test) and the conv windows bit for bit (raw bf16 inputs).
"""

import zlib
from itertools import accumulate

import _kda_cells as kc
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="certified on SM100 and SM103")

LAYERS = 2
MAX_SEQS = max(len(c.spec["seq_lens"]) for c in kda_prefill_conv.CELLS)
HEADS = sorted({c.spec["heads"] for c in kda_prefill_conv.CELLS})
RTOL, ATOL = 0.008, 1e-4  # the kernel's fast-tanh SiLU against fp32 (its own test's band)


def _build_manager(heads: int, layers: int, max_batch_size: int):
    """``_kda_cells.build_manager`` without speculation, at ``heads`` heads per rank."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return KimiK3HybridCacheManager(
        mamba_d_state=kc.K,
        mamba_d_conv=kc.W,
        mamba_num_heads=heads,
        mamba_n_groups=heads,
        mamba_head_dim=kc.V,
        mamba_num_layers=layers,
        mamba_layer_mask=[True] * layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=64 * max_batch_size, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=None,
        layer_mask=[False] * layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=None,
        kda_draft_records=False,
    )


@pytest.fixture(scope="module")
def managers():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

    built = {}
    try:
        for i, heads in enumerate(HEADS):
            mgr = _build_manager(heads, LAYERS, MAX_SEQS)
            slots = kc.request_slots(mgr, MAX_SEQS, first_id=300 + 50 * i)
            built[heads] = (mgr, slots)
        yield built
    finally:
        for mgr, _ in built.values():
            mgr.shutdown()


def _layers(mgr) -> list:
    return [
        {"conv": mgr.get_conv_states(layer), "ssm": mgr.get_ssm_states(layer)}
        for layer in range(LAYERS)
    ]


def _fill(pools: list, seed: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for p in pools:
        p["conv"].copy_(torch.randn(p["conv"].shape, generator=gen, device="cuda").bfloat16())
        p["ssm"].copy_(torch.randn(p["ssm"].shape, generator=gen, device="cuda") * 0.05)


def _inputs(spec: dict, slots: torch.Tensor, gen: torch.Generator) -> dict:
    """One prefill batch as Kimi K3's layer hands it to the op: the fused projection rows, the module's bf16 conv
    taps, int32 starts / slots and bool initial-state flags."""
    lens = spec["seq_lens"]
    channels = 3 * spec["heads"] * 128
    return {
        "src": torch.randn(sum(lens), spec["src_cols"], generator=gen, device="cuda").bfloat16(),
        "weight": (torch.randn(channels, kc.W, generator=gen, device="cuda") * 0.3).bfloat16(),
        "starts": torch.tensor([0, *accumulate(lens)], dtype=torch.int32, device="cuda"),
        # The manager's slots, reversed: not monotone in the request order.
        "slots": slots[: len(lens)].flip(0).contiguous(),
        "has_init": torch.tensor(spec["has_init"], dtype=torch.bool, device="cuda"),
        "max_seqlen": max(lens),
    }


def _call(fn, inp: dict, conv: torch.Tensor, max_seqlen=None):
    return fn(
        inp["src"],
        inp["weight"],
        conv,
        inp["starts"],
        inp["slots"],
        inp["has_init"],
        inp["max_seqlen"] if max_seqlen is None else max_seqlen,
    )


def _check_outputs(got, want, tokens: int, heads: int) -> None:
    for a, b in zip(got, want):
        assert (
            a.shape == (1, tokens, heads, 128) and a.dtype == torch.bfloat16 and a.is_contiguous()
        )
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "cell", kda_prefill_conv.CELLS, ids=[c.why[:40] for c in kda_prefill_conv.CELLS]
)
def test_prefill_conv_on_manager_pools(managers, cell) -> None:
    """Every layer's call against fp32: q / k / v in the kernel's band, the used slots' conv windows bit for bit,
    the other slots of the layer, the layer's state pool, every other layer and the inputs untouched."""
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    mgr, all_slots = managers[spec["heads"]]
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        _fill(pools, seed)
        for layer, p in enumerate(pools):
            inp = _inputs(spec, all_slots, gen)
            ref_conv = kc.clone_pools({"conv": p["conv"]})["conv"]
            before = [kc.snapshot(q) for q in pools]
            src0, w0 = inp["src"].clone(), inp["weight"].clone()
            with validating(kda_prefill_conv):
                got = _call(kda_prefill_conv, inp, p["conv"])
            want = _call(kda_prefill_conv.reference, inp, ref_conv)
            torch.cuda.synchronize()
            _check_outputs(got, want, inp["src"].shape[0], spec["heads"])
            rows = inp["slots"].long()
            assert torch.equal(p["conv"][rows], ref_conv[rows]), layer
            others = kc.other_slots(p["conv"].shape[0], inp["slots"].tolist())
            assert torch.equal(p["conv"][others], before[layer]["conv"][others]), layer
            assert torch.equal(p["ssm"], before[layer]["ssm"]), layer
            for o, q in enumerate(pools):
                if o != layer:
                    assert all(torch.equal(q[n], before[o][n]) for n in ("conv", "ssm")), (layer, o)
            assert torch.equal(inp["src"], src0) and torch.equal(inp["weight"], w0)


@pytest.mark.parametrize("bound", ("one", "total_x2"))
def test_seqlen_bound_only_sizes_grid(managers, bound) -> None:
    """``max_seqlen`` understated (1) or overstated (twice the tokens): every token is still convolved."""
    cell = kda_prefill_conv.CELLS[0]
    spec = cell.spec
    mgr, all_slots = managers[spec["heads"]]
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(41)
    with torch.inference_mode():
        _fill(pools, 41)
        inp = _inputs(spec, all_slots, gen)
        conv = pools[0]["conv"]
        ref_conv = kc.clone_pools({"conv": conv})["conv"]
        max_seqlen = 1 if bound == "one" else 2 * inp["src"].shape[0]
        got = _call(kda_prefill_conv, inp, conv, max_seqlen)
        want = _call(kda_prefill_conv.reference, inp, ref_conv)
        torch.cuda.synchronize()
    _check_outputs(got, want, inp["src"].shape[0], spec["heads"])
    rows = inp["slots"].long()
    assert torch.equal(conv[rows], ref_conv[rows])


def test_graph_replay_matches_eager(managers) -> None:
    """One call captured in a CUDA graph (max_seqlen the token count) and replayed with new source rows, new
    sequence boundaries over the same token count and new initial-state flags written in place: the bits of the same
    calls run eagerly on a copy of the pool."""
    spec = kda_prefill_conv.CELLS[0].spec
    mgr, all_slots = managers[spec["heads"]]
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(43)
    lens = list(spec["seq_lens"])
    tokens = sum(lens)
    schedules = [lens, list(reversed(lens)), [tokens] + [0] * (len(lens) - 1)]
    with torch.inference_mode():
        _fill(pools, 43)
        conv = pools[0]["conv"]
        start = kc.clone_pools({"conv": conv})["conv"]
        eager_conv = kc.clone_pools({"conv": conv})["conv"]
        static = _inputs(spec, all_slots, gen)
        steps = []
        for i, cur in enumerate(schedules):
            step = _inputs(dict(spec, seq_lens=cur), all_slots, gen)
            step["has_init"] = torch.tensor(
                [(j + i) % 2 == 0 for j in range(len(lens))], dtype=torch.bool, device="cuda"
            )
            steps.append(step)
        want = []
        for step in steps:
            step = dict(step, weight=static["weight"])
            want.append([t.clone() for t in _call(kda_prefill_conv, step, eager_conv, tokens)])
        # Compile outside the capture, then put the pool back.
        _call(kda_prefill_conv, static, conv, tokens)
        conv.copy_(start)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = _call(kda_prefill_conv, static, conv, tokens)
        torch.cuda.current_stream().wait_stream(stream)
        conv.copy_(start)
        got = []
        for step in steps:
            for name in ("src", "starts", "has_init"):
                static[name].copy_(step[name])
            graph.replay()
            got.append([t.clone() for t in outs])
        torch.cuda.synchronize()
    for a, b in zip(got, want):
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(conv, eager_conv)


def test_rejects_pool_sharing_the_source_storage() -> None:
    """A conv pool in the source's allocation is refused before launch (the op's aliasing check)."""
    import tensorrt_llm  # noqa: F401

    tokens, channels = 16, kc.CONV_DIM
    with torch.inference_mode():
        backing = torch.randn(tokens * channels + 4 * channels * 3, device="cuda").bfloat16()
        src = backing[: tokens * channels].view(tokens, channels)
        conv = backing[tokens * channels :].view(4, channels, 3)
        before = backing.clone()
        weight = torch.randn(channels, kc.W, device="cuda").bfloat16()
        starts = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
        slots = torch.tensor([1], dtype=torch.int32, device="cuda")
        has_init = torch.tensor([True], device="cuda")
        with pytest.raises(ValueError, match="disjoint storage"):
            kda_prefill_conv(src, weight, conv, starts, slots, has_init, tokens)
        torch.cuda.synchronize()
    assert torch.equal(backing, before)


@pytest.mark.parametrize("misuse", ("duplicate_slot", "slot_range", "starts"))
def test_guard_rejects_silent_misuse(managers, misuse) -> None:
    """``is_valid`` refuses what the kernel would not check on device: a repeated slot, a slot past the pool, and
    starts that do not cover the source rows."""
    spec = kda_prefill_conv.CELLS[0].spec
    mgr, all_slots = managers[spec["heads"]]
    conv = _layers(mgr)[0]["conv"]
    gen = torch.Generator(device="cuda").manual_seed(47)
    with torch.inference_mode():
        inp = _inputs(spec, all_slots, gen)
        before = conv.clone()
        if misuse == "duplicate_slot":
            inp["slots"] = inp["slots"][:1].repeat(inp["slots"].numel())
            match = "unique"
        elif misuse == "slot_range":
            inp["slots"] = inp["slots"].clone()
            inp["slots"][0] = conv.shape[0]
            match = "out of range"
        else:
            inp["starts"] = inp["starts"].clone()
            inp["starts"][-1] -= 1
            match = "partition"
        with pytest.raises(AssertionError, match=match), validating(kda_prefill_conv):
            _call(kda_prefill_conv, inp, conv)
        torch.cuda.synchronize()
    assert torch.equal(conv, before)
