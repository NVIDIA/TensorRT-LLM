# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the opt-in NVFP4 expert tactic pin (``expert_tactic_pin``).

Host: which AutoTuner cache entries the pin rewrites (the expert op, the 64-expert SwiGLU top-8
window, bucket 512) and how (tactic only, valid for every row count of the bucket); that it
waits for the tuning pass, re-checks only when the cache grows and warns once when it is idle;
that ``install`` wraps the expert forwards and refuses another device capability or a model
without the NVFP4 window.
GPU (sm_107): after a real tuning pass the op's 512..1023-row lookups return the pinned tactic
and every other bucket keeps its tuned entry; the pinned tactic's expert rows are bitwise those
of every 128-token tile at 512..768 rows.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.expert_tactic_pin import (  # noqa: E501
    pin as pin_path,
)

ON_SM107 = torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) == (10, 7)
needs_sm107 = pytest.mark.skipif(not ON_SM107, reason="the pin is measured on sm_107 only")

OP = pin_path.EXPERT_OP
PIN = (pin_path.DEFAULT_TILE_N, pin_path.DEFAULT_CONFIG)


def _shapes(rows: int, window: int = 64):
    """The cache-key profile of the op's 18 inputs at ``rows`` (``-1``: constrained dims)."""
    return (
        (0,),
        (0,),
        (rows, 2048),
        (-1,),
        (window, 8192, 2048),
        (window, 8192, 256),
        (0,),
        (0,),
        (0,),
        (0,),
        (window, 4096, 2048),
        (window, 4096, 256),
        (0,),
        (window,),
        (window,),
        (window,),
        (-1, 8),
        (-1, 8),
    )


def _key(rows: int, window: int = 64, top_k: int = 8, act: int = 0, op: str = OP):
    return (op, "FP4BlockScaleMoERunner", str((top_k, 4096, window, act)), _shapes(rows, window))


def _all_valid(top_k, hidden, intermediate, window, rows):
    return frozenset({PIN, (64, 31), (128, 22), (128, 23), (32, 7)})


def _cache():
    return {
        _key(256): (0, [128, 22], 1.0),
        _key(512): (0, [128, 23], 2.0),
        _key(1024): (0, [128, 20], 3.0),
        _key(512, window=68): (0, [128, 22], 4.0),
        _key(512, act=3): (0, [128, 22], 5.0),
        _key(512, top_k=10): (0, [128, 22], 6.0),
        _key(512, op="trtllm::fp8_block_scale_moe_runner"): (0, [128, 22], 7.0),
        ("trtllm::low_m_bf16_gemm", "Runner", "()", ((512, 8192),)): (0, 3, 8.0),
    }


# ------------------------------------------------------------------------------------------ host
def test_unique_id_parses_the_runner_id_and_nothing_else():
    assert pin_path.parse_unique_id("(8, 4096, 64, 0)") == (8, 4096, 64, 0)
    assert pin_path.parse_unique_id("(8, 4096, 64)") is None
    assert pin_path.parse_unique_id("(a, 4096, 64, 0)") is None


def test_only_the_window_bucket_entries_are_pinned():
    cache = _cache()
    assert pin_path.pinned_keys(cache) == [_key(512)]


def test_rewrite_sets_the_tactic_and_keeps_the_rest_of_the_entry():
    cache = _cache()
    before = dict(cache)
    rewritten, matched, invalid = pin_path.rewrite(cache, PIN, _all_valid)
    assert (rewritten, matched, invalid) == (1, 1, [])
    assert cache[_key(512)] == (0, list(PIN), 2.0)
    for key, value in before.items():
        if key != _key(512):
            assert cache[key] == value
    # Idempotent.
    assert pin_path.rewrite(cache, PIN, _all_valid)[:2] == (1, 1)
    assert cache[_key(512)] == (0, list(PIN), 2.0)


def test_a_tactic_invalid_at_any_row_of_the_bucket_is_not_pinned():
    seen = []

    def valid(top_k, hidden, intermediate, window, rows):
        seen.append((top_k, hidden, intermediate, window, rows))
        return frozenset() if rows == 1000 else frozenset({PIN})

    cache = _cache()
    rewritten, matched, invalid = pin_path.rewrite(cache, PIN, valid)
    assert (rewritten, matched, invalid) == (0, 1, [_key(512)])
    assert cache[_key(512)] == (0, [128, 23], 2.0)
    # Asked with the key's own shapes, rows 512 upward.
    assert seen[0] == (8, 4096, 4096, 64, 512)
    assert seen[-1][-1] == 1000


def test_valid_for_bucket_covers_every_row_count_of_the_bucket():
    rows = []
    assert pin_path.valid_for_bucket(
        PIN, 8, 4096, 4096, lambda *a: rows.append(a[-1]) or frozenset({PIN})
    )
    assert rows == list(range(512, 1024))


class _Tuner:
    def __init__(self, cache):
        self.is_tuning_mode = False
        self.profiling_cache = types.SimpleNamespace(cache=cache)


@pytest.fixture
def tuner(monkeypatch):
    from tensorrt_llm._torch import autotuner

    fake = _Tuner({})
    monkeypatch.setattr(autotuner.AutoTuner, "get", classmethod(lambda cls: fake))
    return fake


def test_ensure_waits_for_the_tuning_pass_and_rechecks_only_on_growth(tuner):
    pin = pin_path.ExpertTacticPin(PIN, valid=_all_valid)
    tuner.is_tuning_mode = True
    tuner.profiling_cache.cache.update(_cache())
    pin.ensure()
    assert tuner.profiling_cache.cache[_key(512)][1] == [128, 23]
    tuner.is_tuning_mode = False
    pin.ensure()
    assert tuner.profiling_cache.cache[_key(512)][1] == list(PIN)
    # Same size: no re-scan (an entry changed behind it stays as it is).
    tuner.profiling_cache.cache[_key(512)] = (0, [128, 22], 2.0)
    pin.ensure()
    assert tuner.profiling_cache.cache[_key(512)][1] == [128, 22]
    # Grown (a later tuning pass): re-scanned.
    tuner.profiling_cache.cache[_key(2048)] = (0, [128, 20], 9.0)
    pin.ensure()
    assert tuner.profiling_cache.cache[_key(512)][1] == list(PIN)


def test_ensure_warns_once_when_the_tuned_cache_has_no_entry_it_serves(tuner, monkeypatch):
    warnings = []
    monkeypatch.setattr(pin_path.logger, "warning", lambda msg: warnings.append(msg))
    pin = pin_path.ExpertTacticPin(PIN, valid=_all_valid)
    # No expert-op entry yet (before the tuning pass): nothing to say.
    tuner.profiling_cache.cache[_key(512, op="other::op")] = (0, [1, 1], 1.0)
    pin.ensure()
    assert warnings == []
    # The op was tuned, but only below the pinned bucket: the pin is idle, said once.
    tuner.profiling_cache.cache[_key(256)] = (0, [128, 22], 1.0)
    pin.ensure()
    tuner.profiling_cache.cache[_key(128)] = (0, [128, 22], 1.0)
    pin.ensure()
    assert len(warnings) == 1 and "changes no call" in warnings[0]
    # Once the bucket is tuned the entry is pinned.
    tuner.profiling_cache.cache[_key(512)] = (0, [128, 23], 2.0)
    pin.ensure()
    assert tuner.profiling_cache.cache[_key(512)][1] == list(PIN)
    assert len(warnings) == 1


def _model(n_layers: int):
    calls = []

    class Experts:
        def forward(self, *args, **kwargs):
            calls.append(("forward", args, kwargs))
            return "out"

    layers = [
        types.SimpleNamespace(feed_forward=types.SimpleNamespace(experts=Experts()))
        for _ in range(n_layers)
    ]
    return types.SimpleNamespace(layers=layers), calls


def test_install_wraps_the_listed_layers_and_checks_before_each_call(tuner):
    model, calls = _model(3)
    pin = pin_path.install(model, PIN, capability=(10, 7), layers=[1, 2])
    assert model._expert_tactic_pin is pin
    order = []
    pin.ensure = lambda: order.append("ensure")
    assert model.layers[1].feed_forward.experts.forward("x", "logits", do_finalize=False) == "out"
    assert order == ["ensure"]
    assert calls == [("forward", ("x", "logits"), {"do_finalize": False})]
    # Layer 0 is not wrapped.
    model.layers[0].feed_forward.experts.forward("y", "logits")
    assert order == ["ensure"]


def test_install_refuses_another_capability_and_a_model_without_the_window():
    model, _ = _model(2)
    with pytest.raises(ValueError, match="measured on"):
        pin_path.install(model, PIN, capability=(10, 3), layers=[0])
    with pytest.raises(ValueError, match="no layer"):
        pin_path.install(model, PIN, capability=(10, 7), layers=[])


# ------------------------------------------------------------------------------------------ GPU
HIDDEN, INTER, E, TOP_K, WIN = 4096, 4096, 256, 8, 64


def _gpu_inputs(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(0, 256, (rows, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g)
    sf = torch.randint(
        0x28, 0x41, (rows * HIDDEN // 16,), dtype=torch.uint8, device="cuda", generator=g
    )
    ids = torch.rand(rows, E, device="cuda", generator=g).topk(TOP_K, dim=1).indices
    w = torch.rand(rows, TOP_K, device="cuda", generator=g) + 0.05
    w = (w / w.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return x, sf.view(torch.float8_e4m3fn), ids.to(torch.int32).contiguous(), w.contiguous()


@pytest.fixture(scope="module")
def stacks():
    g = torch.Generator(device="cuda").manual_seed(7)
    w13 = torch.randint(
        0, 256, (WIN, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w13s = torch.randint(
        0x28, 0x38, (WIN, 2 * INTER, HIDDEN // 16), dtype=torch.uint8, device="cuda", generator=g
    )
    w2 = torch.randint(
        0, 256, (WIN, HIDDEN, INTER // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w2s = torch.randint(
        0x28, 0x38, (WIN, HIDDEN, INTER // 16), dtype=torch.uint8, device="cuda", generator=g
    )
    ones = torch.full((WIN,), 1e-3, device="cuda")
    return dict(
        w13=w13,
        w13s=w13s.view(torch.float8_e4m3fn),
        w2=w2,
        w2s=w2s.view(torch.float8_e4m3fn),
        c=ones,
        a13=ones.clone(),
        a2=ones.clone(),
    )


def _op(s, x, sf, ids, w):
    return torch.ops.trtllm.fp4_block_scale_moe_runner(
        None, None, x, sf, s["w13"], s["w13s"], None, None, None, None, s["w2"], s["w2s"], None,
        s["c"], s["a13"], s["a2"], E, TOP_K, None, None, INTER, 0, WIN, None, 5, False, 0, w, ids,
        None, 768, False,
    )  # fmt: skip


def _rows(outputs):
    rows, index = outputs[0], outputs[2].reshape(-1).long()
    valid = (index >= 0) & (index < rows.shape[0])
    out = torch.zeros(index.numel(), rows.shape[1], dtype=rows.dtype, device=rows.device)
    out[valid] = rows[index[valid]]
    return out.view(torch.int16), valid


@needs_sm107
def test_the_tuned_bucket_runs_the_pin_and_the_other_buckets_keep_their_tactics(stacks):
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune():
        _op(stacks, *_gpu_inputs(768, 1))
    before = {k: v[1] for k, v in tuner.profiling_cache.cache.items() if k[0] == OP}
    assert len(pin_path.pinned_keys(tuner.profiling_cache.cache)) == 1
    pin = pin_path.ExpertTacticPin(PIN)
    pin.ensure()
    after = {k: v[1] for k, v in tuner.profiling_cache.cache.items() if k[0] == OP}
    for key, tactic in after.items():
        if key[3][pin_path.HIDDEN_STATES][0] == pin_path.BUCKET:
            assert list(tactic) == list(PIN)
        else:
            assert tactic == before[key]


@needs_sm107
@pytest.mark.parametrize("rows", [512, 576, 640, 768])
def test_the_pinned_tactic_writes_the_rows_of_every_128_token_tile(stacks, rows):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner = FP4BlockScaleMoERunner(E, TOP_K, None, None, INTER, 0, WIN, None, 5, False, 0)
    x, sf, ids, w = _gpu_inputs(rows, 100 + rows)
    inputs = [None, None, x, sf, stacks["w13"], stacks["w13s"], None, None, None, None,
              stacks["w2"], stacks["w2s"], None, stacks["c"], stacks["a13"], stacks["a2"], w, ids]  # fmt: skip
    valid = pin_path._runner_valid(TOP_K, HIDDEN, INTER, WIN, rows)
    assert PIN in valid
    ref, ref_valid = _rows(runner(inputs, tactic=list(PIN)))
    others = [t for t in sorted(valid) if t[0] == 128]
    assert others
    for tactic in others:
        out, out_valid = _rows(runner(inputs, tactic=list(tactic)))
        assert torch.equal(out_valid, ref_valid), tactic
        assert torch.equal(out[ref_valid], ref[ref_valid]), tactic
