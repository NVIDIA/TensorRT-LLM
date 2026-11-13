# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ssm/kda_mtp_decode op, on a real cache object.

Kimi K3's KDA speculative verify from the replay caches (no draft records) at the TP16 rank slice (6 heads,
K = V = 128, conv width 4), on the pools of a real ``KimiK3HybridCacheManager`` built with MTP-style speculation of
``num_spec`` drafts, ``kda_replay_num_spec=num_spec`` and ``kda_draft_records=False``: every layer's bf16 conv
caches (dim-contiguous), its fp32 V-first state (strided by the manager's per-slot coalescing), its k / g / v /
beta caches (beta in head-padded rows), and the accepted-draft count all layers share
(``prev_num_accepted_tokens``), at the slots the manager assigned. Without the draft records the manager keeps the
replay caches and the count in a pool of their own (``replay_layer_group_id``), so the op reads the state by the state
slots and the rest by the replay slots (``get_replay_state_indices``). The cell with ``distinct_replay_slots``
pairs each request with another request's replay slot, so no request's replay slot equals its state slot whatever
numbers the manager assigned. Inputs as the model passes them: q / k / v and the output gate as strided views of one
fused bf16 projection, beta a strided view of the ``[f_a | b]`` rows.

Every round sets each request's accepted count (0 .. num_spec, different per request and per round) and runs the op
on the manager's pools and the entry's fp32 reference on a copy: outputs within 2e-2 and states within 1e-3 of
the reference's largest entry (the kernel's 3xTF32 recurrence and approximate exp / rcp against fp32), the conv
caches bit for bit (raw bf16 inputs), the bf16 k / v / beta caches within one bf16 step (1e-2) and the fp32 gate
cache within 1e-3.
"""

import importlib
import zlib

import pytest
import torch

__extra_import_path__ = ["../../..", "../ops/ssm"]
import _kda_cells as kc
from _validating import validating

kda_mtp_decode = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl."
    "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.ops.ssm.kda_mtp_decode"
).kda_mtp_decode

pytestmark = pytest.mark.skipif(not kc.sm100(), reason="certified on SM100 and SM103")

LAYERS = 2
MAX_REQUESTS = max(c.spec["requests"] for c in kda_mtp_decode.CELLS)
FA_COLS = 136  # the [f_a | b | pad] rows: f_a 128, b 6, padded to 16 bytes
TOL_CACHE_BF16 = 1e-2
TOL_CACHE_G = 1e-3
STATE = ("conv", "ssm")
REPLAY = ("cs_q", "cs_k", "cs_v", "k", "g", "v", "beta")
NAMES = (*STATE, *REPLAY)


def _build_manager(num_spec: int, layers: int, max_batch_size: int):
    """``_kda_cells.build_manager`` with the replay caches and without the draft records."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    return KimiK3HybridCacheManager(
        mamba_d_state=kc.K,
        mamba_d_conv=kc.W,
        mamba_num_heads=kc.H,
        mamba_n_groups=kc.H,
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
        spec_config=MTPDecodingConfig(max_draft_len=num_spec),
        layer_mask=[False] * layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_replay_num_spec=num_spec,
        kda_draft_records=False,
    )


@pytest.fixture(scope="module")
def managers():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

    built = {}
    try:
        for num_spec in sorted({c.spec["num_spec"] for c in kda_mtp_decode.CELLS}):
            mgr = _build_manager(num_spec, LAYERS, MAX_REQUESTS)
            built[num_spec] = mgr
            first_id = 700 + 100 * num_spec
            ids = list(range(first_id, first_id + MAX_REQUESTS))
            slots = kc.request_slots(mgr, MAX_REQUESTS, first_id=first_id)
            assert mgr.replay_layer_group_id is not None, "no replay pool of its own"
            replay = [int(r) for r in mgr.get_replay_state_indices(ids)]
            assert len(set(replay)) == MAX_REQUESTS, replay
            built[num_spec] = (
                mgr,
                slots,
                _cuda_slots(replay),
                _cuda_slots(_crossed(slots.tolist(), replay)),
            )
        yield built
    finally:
        for entry in built.values():
            (entry[0] if isinstance(entry, tuple) else entry).shutdown()


def _cuda_slots(values: list) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _crossed(state: list, replay: list) -> list:
    """The requests' replay slots rotated by the first offset at which no request keeps a replay slot equal to its
    state slot: still one unique replay slot per request, and different from the state slot in every row."""
    for shift in range(1, len(replay)):
        rotated = replay[shift:] + replay[:shift]
        if all(r != s for r, s in zip(rotated, state)):
            return rotated
    raise AssertionError(f"no rotation of the replay slots {replay} avoids the state slots {state}")


def _layer(mgr, layer: int) -> dict:
    cache = mgr.mamba_layer_cache(layer)
    return {
        "conv": cache.conv,
        "ssm": cache.temporal,
        "cs_q": cache.kda_conv_q,
        "cs_k": cache.kda_conv_k,
        "cs_v": cache.kda_conv_v,
        "k": cache.kda_k_cache,
        "g": cache.kda_g_cache,
        "v": cache.kda_v_cache,
        "beta": cache.kda_beta_cache,
        "pending": cache.prev_num_accepted_tokens,
    }


def _layers(mgr) -> list:
    return [_layer(mgr, layer) for layer in range(LAYERS)]


def _fill(p: dict, gen: torch.Generator) -> None:
    """Random contents a previous round could have left in every slot: raw bf16 conv inputs, k unit-norm per head,
    log decays in (lower_bound, 0), beta in (0, 1)."""

    def rnd(t, scale=1.0):
        return torch.randn(t.shape, generator=gen, device="cuda") * scale

    p["conv"].copy_(rnd(p["conv"]).bfloat16())
    p["ssm"].copy_(rnd(p["ssm"], 0.05))
    for name in ("cs_q", "cs_k", "cs_v"):
        p[name].copy_(rnd(p[name]).bfloat16())
    k = rnd(p["k"]).unflatten(-1, (kc.H, kc.K))
    p["k"].copy_((k * torch.rsqrt(k.square().sum(-1, keepdim=True))).flatten(-2).bfloat16())
    p["g"].copy_(kc.LOWER_BOUND * torch.sigmoid(rnd(p["g"])))
    p["v"].copy_(rnd(p["v"], 0.5).bfloat16())
    p["beta"].copy_(torch.sigmoid(rnd(p["beta"])).bfloat16())


def _clone(p: dict, pending: torch.Tensor) -> dict:
    """A standalone copy with the same strides, sharing ``pending`` (one count per slot for all layers)."""
    out = kc.clone_pools({n: p[n] for n in NAMES})
    out["pending"] = pending
    return out


def _weights(gen: torch.Generator) -> dict:
    """One layer's verify constants: fp32 conv taps [768, 4] holding bf16 values, the module's bf16 A_log / dt_bias,
    the fp32 output-norm weight."""

    def rnd(*s, scale=1.0):
        return torch.randn(*s, generator=gen, device="cuda") * scale

    taps = rnd(3, kc.HK, kc.W, scale=0.3).bfloat16().float()
    return {
        "w_q": taps[0].contiguous(),
        "w_k": taps[1].contiguous(),
        "w_v": taps[2].contiguous(),
        "a_log": rnd(kc.H, scale=0.5).bfloat16(),
        "dt_bias": rnd(kc.HK, scale=0.5).bfloat16(),
        "onorm_w": 1 + 0.1 * rnd(kc.V),
    }


def _inputs(num: int, steps: int, gen: torch.Generator) -> dict:
    rows = num * steps
    return {
        "qkvg": torch.randn(rows, 4 * kc.HK, generator=gen, device="cuda").bfloat16(),
        "fab": torch.randn(rows, FA_COLS, generator=gen, device="cuda").bfloat16(),
        "g": torch.randn(1, rows, kc.H, kc.K, generator=gen, device="cuda").bfloat16(),
    }


def _call(
    fn, wt: dict, p: dict, inp: dict, slots: torch.Tensor, num_spec: int, replay: torch.Tensor
) -> torch.Tensor:
    """The entry as the model calls it: section views of the fused projection, the fixed call form, the state slots
    ``slots`` and the replay slots ``replay``."""
    rows = inp["qkvg"].shape[0]
    x_q, x_k, x_v, onorm_g = (
        inp["qkvg"][:, i * kc.HK : (i + 1) * kc.HK].view(1, rows, kc.H, kc.K) for i in range(4)
    )
    beta = inp["fab"][:, kc.K : kc.K + kc.H].unsqueeze(0)
    return fn(
        x_q, x_k, x_v, wt["w_q"], wt["w_k"], wt["w_v"], p["cs_q"], p["cs_k"], p["cs_v"], inp["g"], beta,
        wt["a_log"], wt["dt_bias"], p["ssm"], p["k"], p["g"], p["v"], p["beta"], slots, None, num_spec,
        p["pending"], kc.LOWER_BOUND, kc.SCALE, True, onorm_g, wt["onorm_w"], kc.EPS, True, replay,
    )  # fmt: skip


def _accepted(num: int, num_spec: int, rnd: int) -> list:
    """Accepted drafts per request before round ``rnd``: 0 and num_spec first, then 1 .. num_spec - 1, shifted by
    one per round, so one request sees 0 and num_spec within two rounds and the requests differ."""
    order = [0, num_spec, *range(1, num_spec)]
    return [order[(n + rnd) % len(order)] for n in range(num)]


def _set_pending(pendings, replay: torch.Tensor, accepted: list) -> None:
    """The accepted counts, at the requests' replay slots."""
    for pending in pendings:
        pending[replay.long()] = torch.tensor(accepted, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize(
    "cell", kda_mtp_decode.CELLS, ids=[c.why[:40] for c in kda_mtp_decode.CELLS]
)
def test_mtp_decode_on_manager_pools(managers, cell) -> None:
    """``rounds`` rounds of every layer against the fp32 reference on a copy of the pools; the other slots of each
    layer, every other layer, the live conv pool and the accepted counts untouched."""
    num, num_spec, rounds = cell.spec["requests"], cell.spec["num_spec"], cell.spec["rounds"]
    steps = num_spec + 1
    seed = zlib.crc32(cell.why.encode()) % 2**31
    mgr, all_slots, all_replay, crossed_replay = managers[num_spec]
    slots = all_slots[:num].flip(0).contiguous()
    replay_source = crossed_replay if cell.spec.get("distinct_replay_slots") else all_replay
    replay = replay_source[:num].flip(0).contiguous()
    if cell.spec.get("distinct_replay_slots"):
        assert all(r != s for r, s in zip(replay.tolist(), slots.tolist())), (replay, slots)
    rows, replay_rows = slots.long(), replay.long()
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        for p in pools:
            _fill(p, gen)
        pending = pools[0]["pending"]
        pending.zero_()
        ref_pending = pending.clone()
        refs = [_clone(p, ref_pending) for p in pools]
        wts = [_weights(gen) for _ in range(LAYERS)]
        for rnd in range(rounds):
            accepted = _accepted(num, num_spec, rnd)
            _set_pending([pending, ref_pending], replay, accepted)
            for layer, (p, rp) in enumerate(zip(pools, refs)):
                where = (rnd, layer, accepted)
                before = [kc.snapshot({n: q[n] for n in (*NAMES, "pending")}) for q in pools]
                inp = _inputs(num, steps, gen)
                with validating(kda_mtp_decode):
                    out = _call(kda_mtp_decode, wts[layer], p, inp, slots, num_spec, replay)
                want = _call(kda_mtp_decode.reference, wts[layer], rp, inp, slots, num_spec, replay)
                torch.cuda.synchronize()
                assert out.shape == (1, num * steps, kc.H, kc.V) and out.dtype == torch.bfloat16
                assert torch.isfinite(out).all(), where
                assert kc.rel(out, want) <= kc.TOL_OUT, (where, kc.rel(out, want))
                assert kc.rel(p["ssm"][rows], rp["ssm"][rows]) <= kc.TOL_STATE, where
                for name in ("cs_q", "cs_k", "cs_v"):
                    assert torch.equal(p[name][replay_rows], rp[name][replay_rows]), (where, name)
                for name in ("k", "v", "beta"):
                    assert kc.rel(p[name][replay_rows], rp[name][replay_rows]) <= TOL_CACHE_BF16, (
                        where,
                        name,
                    )
                assert kc.rel(p["g"][replay_rows], rp["g"][replay_rows]) <= TOL_CACHE_G, where
                others = kc.other_slots(p["ssm"].shape[0], slots.tolist())
                assert kc.same_rows(p, before[layer], others, STATE), where
                other_replay = kc.other_slots(p["cs_q"].shape[0], replay.tolist())
                assert kc.same_rows(p, before[layer], other_replay, REPLAY), where
                assert torch.equal(p["conv"], before[layer]["conv"]), where
                assert torch.equal(pending, before[layer]["pending"]), where
                for o, q in enumerate(pools):
                    if o != layer:
                        assert all(torch.equal(q[n], before[o][n]) for n in NAMES), (where, o)


def test_rounds_graph_replay_matches_eager(managers) -> None:
    """Four requests, LAYERS layers x three rounds: one round of every layer captured once in a CUDA graph and
    replayed (rows rewritten in place, the accepted counts written between replays) gives the bits of the same
    rounds run eagerly on a copy of the pools."""
    num_spec, num, rounds = kc.NUM_SPEC, 4, 3
    steps = num_spec + 1
    mgr, all_slots, all_replay, _ = managers[num_spec]
    slots = all_slots[:num].flip(0).contiguous()
    replay = all_replay[:num].flip(0).contiguous()
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(91)
    with torch.inference_mode():
        for p in pools:
            _fill(p, gen)
        pending = pools[0]["pending"]
        pending.zero_()
        eager_pending = pending.clone()
        eager = [_clone(p, eager_pending) for p in pools]
        start = [_clone(p, pending.clone()) for p in pools]
        wts = [_weights(gen) for _ in range(LAYERS)]
        inputs = [[_inputs(num, steps, gen) for _ in range(LAYERS)] for _ in range(rounds)]
        want = []
        for rnd in range(rounds):
            _set_pending([eager_pending], replay, _accepted(num, num_spec, rnd))
            for layer in range(LAYERS):
                inp = inputs[rnd][layer]
                want.append(
                    _call(
                        kda_mtp_decode, wts[layer], eager[layer], inp, slots, num_spec, replay
                    ).clone()
                )
        static = [{n: t.clone() for n, t in inputs[0][layer].items()} for layer in range(LAYERS)]
        # Compile on the manager's own pools outside the capture, then put them back.
        for layer in range(LAYERS):
            _call(kda_mtp_decode, wts[layer], pools[layer], static[layer], slots, num_spec, replay)
        torch.cuda.synchronize()
        for p, s0 in zip(pools, start):
            for n in NAMES:
                p[n].copy_(s0[n])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            outs = [
                _call(
                    kda_mtp_decode, wts[layer], pools[layer], static[layer], slots, num_spec, replay
                )
                for layer in range(LAYERS)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        for p, s0 in zip(pools, start):
            for n in NAMES:
                p[n].copy_(s0[n])
        got = []
        for rnd in range(rounds):
            _set_pending([pending], replay, _accepted(num, num_spec, rnd))
            for layer in range(LAYERS):
                for n, t in static[layer].items():
                    t.copy_(inputs[rnd][layer][n])
            graph.replay()
            got.extend(o.clone() for o in outs)
        torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    for p, e in zip(pools, eager):
        assert all(torch.equal(p[n], e[n]) for n in NAMES)


def test_rejects_conv_cache_not_dim_contiguous(managers) -> None:
    """A slot-contiguous conv cache (the transpose of the manager's layout) is refused before anything is written."""
    mgr, all_slots, all_replay, _ = managers[kc.NUM_SPEC]
    slots, replay = all_slots[:2].clone(), all_replay[:2].clone()
    pools = _layers(mgr)
    gen = torch.Generator(device="cuda").manual_seed(93)
    with torch.inference_mode():
        _fill(pools[0], gen)
        before = kc.snapshot({n: pools[0][n] for n in NAMES})
        bad = dict(pools[0], cs_k=pools[0]["cs_k"].contiguous())
        inp = _inputs(2, kc.NT, gen)
        with pytest.raises(ValueError, match="dim-contiguous"):
            _call(kda_mtp_decode, _weights(gen), bad, inp, slots, kc.NUM_SPEC, replay)
        torch.cuda.synchronize()
    assert all(torch.equal(pools[0][n], before[n]) for n in NAMES)


@pytest.mark.parametrize("misuse", ("duplicate_slot", "rows", "call_form"))
def test_guard_rejects_silent_misuse(managers, misuse) -> None:
    """``is_valid`` refuses a repeated slot, token rows that are not N (1 + num_spec), and a call form other than the
    packed fused-norm one, before anything is written."""
    mgr, all_slots, all_replay, _ = managers[kc.NUM_SPEC]
    slots, replay = all_slots[:2].clone(), all_replay[:2].clone()
    p = _layer(mgr, 0)
    gen = torch.Generator(device="cuda").manual_seed(95)
    with torch.inference_mode():
        _fill(p, gen)
        before = kc.snapshot({n: p[n] for n in NAMES})
        wt = _weights(gen)
        inp = _inputs(2, kc.NT, gen)
        num_spec = kc.NUM_SPEC
        fn = kda_mtp_decode
        if misuse == "duplicate_slot":
            slots = slots[:1].repeat(2)
            match = "unique slot"
        elif misuse == "rows":
            inp = _inputs(2, kc.NT - 1, gen)
            match = "rows must be"
        else:
            match = "call form"

            def fn(*args):  # the legacy shifted-metadata layout
                args = list(args)
                args[19] = torch.arange(0, 3 * kc.NT, kc.NT, dtype=torch.int32, device="cuda")
                args[24] = False
                return kda_mtp_decode(*args)

        with pytest.raises(AssertionError, match=match), validating(kda_mtp_decode):
            _call(fn, wt, p, inp, slots, num_spec, replay)
        torch.cuda.synchronize()
    assert all(torch.equal(p[n], before[n]) for n in NAMES)
