# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``dflash_window_attention`` entry.

The entry replaces FlashAttention-2's ``flash_attn_with_kvcache`` for the DFlash
drafter's non-causal sliding-window block attention. It computes the same
softmax attention with different split boundaries, so it is gated against its
fp32 ``reference`` within a derived band and, where flash-attn is installed,
against the built-in kernel's own error on the same inputs, both measured
against a float64 evaluation: a maximum error no larger than the built-in's
plus one ulp of the row scale, and a mean error within 10 % of the built-in's.

With ``k_new`` / ``v_new`` the op also appends the block's K/V to the arena;
that path must leave the whole arena bitwise as the drafter's ``index_put``
does and return bitwise the non-append result, eagerly and under CUDA-graph
replay. The written-out block decode (``draft.windowed_block_decode``) must be
bitwise the built-in ``dflash_forward`` running on this op.
"""

import json
import os
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    MAX_WINDOW_SPAN,
    WINDOW,
    attention_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    dflash_window_attention as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "dflash_window_attention requires a CUDA device"

#: Band against the fp32 reference, in ulp of each output row's largest
#: magnitude. Two roundings separate the kernel from the reference: the
#: probabilities are rounded to the operand dtype before the P.V product
#: (relative 2^-9 per term, which averages over the window rather than adding
#: up: the product's error is ~2^-9 of a row's typical magnitude) and the
#: output is rounded once (0.5 ulp). The reference itself rounds once. 3 ulp
#: per element and 1 ulp RMS bound both with margin.
ELEMENT_ULP, RMS_ULP = 3.0, 1.0

#: Serving arena columns per slot: max_seq_len (1M) plus one block.
SERVING_CAPACITY = 1_048_576 + 8

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")


def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _bits(t: torch.Tensor) -> torch.Tensor:
    """An integer view for bitwise comparison (NaN equals itself)."""
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _inputs(spec, seed):
    """q [B, M, Hq, D]; one layer's arena view [slots, capacity, Hkv, D]; L and slots per request.

    The arena has the drafter's [slots, layers, capacity, Hkv, D] layout, so the
    layer view handed to the op has the arena's outer strides (the whole arena
    is the view's ``_base``). Each request sits in its own slot (the last
    ones), with every column it can see filled: at least the last 4200 before
    ``L`` for long contexts, more for wider windows.

    With ``append`` the block's K/V are ``k_new`` / ``v_new`` -- strided views
    of one projection output, as the drafter slices them -- and the arena's
    block columns hold NaN, so a kernel that read them instead would show it.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dtype = spec.get("dtype", torch.bfloat16)
    batch, m, hq, hkv, d = spec["batch"], spec["m"], spec["hq"], spec["hkv"], spec["d"]
    capacity, slots, layers = spec["capacity"], spec["slots"], 3
    span = max(4200, spec["window"][0] + m + 64)
    k_arena = torch.zeros(slots, layers, capacity, hkv, d, dtype=dtype, device="cuda")
    v_arena = torch.zeros_like(k_arena)
    projection = torch.randn(batch * m, (hq + 2 * hkv) * d, generator=g, device="cuda").to(dtype)
    k_new = projection[:, hq * d : (hq + hkv) * d].reshape(-1, hkv, d)
    v_new = projection[:, (hq + hkv) * d :].reshape(-1, hkv, d)
    seqlens, slot_ids = [], []
    for b, ctx in enumerate(spec["ctx"]):
        slot = slots - 1 - b
        n = ctx + m
        lo = max(0, n - span)
        k_arena[slot, 1, lo:n] = torch.randn(n - lo, hkv, d, generator=g, device="cuda").to(dtype)
        v_arena[slot, 1, lo:n] = torch.randn(n - lo, hkv, d, generator=g, device="cuda").to(dtype)
        if spec.get("append"):
            k_arena[slot, 1, ctx:n] = float("nan")
            v_arena[slot, 1, ctx:n] = float("nan")
        seqlens.append(n)
        slot_ids.append(slot)
    q = (spec.get("q_scale", 2.0) * torch.randn(batch, m, hq, d, generator=g, device="cuda")).to(
        dtype
    )
    args = dict(
        q=q,
        k_cache=k_arena[:, 1],
        v_cache=v_arena[:, 1],
        cache_seqlens=torch.tensor(seqlens, dtype=torch.int32, device="cuda"),
        cache_batch_idx=torch.tensor(slot_ids, dtype=torch.int32, device="cuda"),
        window_size=tuple(spec["window"]),
    )
    if spec.get("append"):
        args.update(k_new=k_new, v_new=v_new)
    return args


def _arenas(args):
    """The whole [slots, layers, capacity, Hkv, D] K and V arenas behind the layer views."""
    return args["k_cache"]._base, args["v_cache"]._base


def _index_put(args, k_arena, v_arena, layer=1):
    """The drafter's built-in append: the block's K/V into [L - M, L) of each slot."""
    m = args["q"].shape[1]
    first = (args["cache_seqlens"] - m).long()
    slots = args["cache_batch_idx"].long().view(-1, 1).expand(-1, m).reshape(-1)
    cols = (first.view(-1, 1) + torch.arange(m, device="cuda")).reshape(-1)
    k_arena[slots, layer, cols] = args["k_new"]
    v_arena[slots, layer, cols] = args["v_new"]


def _flash_attn(args, num_splits=0):
    from flash_attn import flash_attn_with_kvcache

    return flash_attn_with_kvcache(
        q=args["q"],
        k_cache=args["k_cache"],
        v_cache=args["v_cache"],
        cache_seqlens=args["cache_seqlens"],
        cache_batch_idx=args["cache_batch_idx"],
        causal=False,
        window_size=args["window_size"],
        num_splits=num_splits,
    )


def _errors(out, exact):
    err = (out.float() - exact).abs()
    return err.max().item(), err.mean().item()


def _assert_no_less_accurate_than_flash_attention(out, args):
    """Budget: max error within 1 ulp of the row scale above FA2's, mean within 10 %."""
    exact = attention_reference(**args, softmax_scale=None, dtype=torch.float64).float()
    ours_max, ours_mean = _errors(out, exact)
    builtin_max, builtin_mean = _errors(_flash_attn(args), exact)
    row_ulp = 2.0 ** (torch.floor(torch.log2(exact.abs().max())).item() - 7)
    assert ours_max <= builtin_max + row_ulp, (ours_max, builtin_max, row_ulp)
    assert ours_mean <= 1.10 * builtin_mean + 1e-9, (ours_mean, builtin_mean)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:56] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _inputs(cell.spec, _seed(cell.why))
    pdl = cell.spec.get("pdl", False)
    entry = cell.spec.get("entry", False)
    append = "k_new" in args
    q_before = args["q"].clone()
    k_arena, v_arena = _arenas(args)
    expected_k, expected_v = k_arena.clone(), v_arena.clone()
    if append:
        _index_put(args, expected_k, expected_v)
    out = op(**args, enable_pdl=pdl, merge_entry_trigger=entry)
    assert out.shape == args["q"].shape and out.dtype == args["q"].dtype and out.is_contiguous()
    assert torch.equal(q_before, args["q"]), "q mutated"
    # The whole arena, bitwise: untouched without the append; with it, exactly
    # what the drafter's index_put leaves.
    assert _bitwise_equal(k_arena, expected_k) and _bitwise_equal(v_arena, expected_v), "arena"
    assert out.isfinite().all()

    ref = op.reference(**args, enable_pdl=pdl, merge_entry_trigger=entry)
    assert_within_ulp(out, ref, ELEMENT_ULP, RMS_ULP)

    if append:
        # The arena now holds the block: the same call without the append
        # (the drafter's index_put path) must give the same bits.
        plain = {k: v for k, v in args.items() if k not in ("k_new", "v_new")}
        assert torch.equal(out, op(**plain, enable_pdl=pdl)), "append path differs from index_put"
        args = plain

    if _has_flash_attn():
        _assert_no_less_accurate_than_flash_attention(out, args)


def test_pdl_launch_is_bitwise_the_plain_launch():
    spec = dict(
        batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[6_000], capacity=6_100, slots=2
    )
    args = _inputs(spec, 21)
    plain = op(**args, enable_pdl=False)
    assert torch.equal(op(**args, enable_pdl=True), plain)
    # pdl_merge: the merge's entry trigger changes the launch only.
    assert torch.equal(op(**args, enable_pdl=True, merge_entry_trigger=True), plain)


@pytest.mark.parametrize("pdl", [False, True])
def test_cuda_graph_replay_follows_new_lengths_and_inputs(pdl):
    """L and the slot are read on the device: one capture serves every step."""
    spec = dict(
        batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[5_000], capacity=9_000, slots=3
    )
    args = _inputs(spec, 31)
    op(**args, enable_pdl=pdl)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args, enable_pdl=pdl)
    fresh = _inputs(dict(spec, ctx=[8_100]), 32)
    for ctx, slot, scale in ((8_100, 2, 1.0), (300, 1, 0.5), (2_049, 0, 3.0)):
        args["q"].copy_(fresh["q"] * scale)
        args["k_cache"][slot].copy_(fresh["k_cache"][2])
        args["v_cache"][slot].copy_(fresh["v_cache"][2] * scale)
        args["cache_seqlens"].fill_(ctx + 8)
        args["cache_batch_idx"].fill_(slot)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op(**args, enable_pdl=pdl)), (ctx, slot)


@pytest.mark.parametrize("pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("m", [1, 8])
def test_cuda_graph_replay_of_the_append(m, pdl):
    """The serving path (append inside the op), two requests, captured once.

    Every replay brings new lengths (including an empty context and window
    edges), new slots and new inputs; output and whole arena must be bitwise
    what an eager ``index_put`` + op gives on the same state.
    """
    spec = dict(
        batch=2,
        m=m,
        hq=8,
        hkv=1,
        d=128,
        window=WINDOW,
        ctx=[5_000, 300],
        capacity=12_000,
        slots=4,
        append=True,
    )
    args = _inputs(spec, 101 + m)
    k_arena, v_arena = _arenas(args)
    op(**args, enable_pdl=pdl)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(**args, enable_pdl=pdl)
    trials = (
        ([8_100, 17], [3, 1]),
        ([300, 2_049], [0, 2]),
        ([2_079, 5_000], [1, 3]),
        ([0, 11_000], [2, 0]),
        ([2_047, 2_048], [3, 2]),
    )
    for t, (ctxs, slots) in enumerate(trials):
        fresh = _inputs(dict(spec, ctx=ctxs), 500 + 10 * m + t)
        fresh_k, fresh_v = _arenas(fresh)
        # _inputs puts request b in slot (slots - 1 - b); move them to this trial's slots.
        k_arena.zero_()
        v_arena.zero_()
        for b, slot in enumerate(slots):
            k_arena[slot].copy_(fresh_k[spec["slots"] - 1 - b])
            v_arena[slot].copy_(fresh_v[spec["slots"] - 1 - b])
        for key in ("q", "k_new", "v_new", "cache_seqlens"):
            args[key].copy_(fresh[key])
        args["cache_batch_idx"].copy_(torch.tensor(slots, dtype=torch.int32, device="cuda"))
        expected_k, expected_v = k_arena.clone(), v_arena.clone()
        graph.replay()
        torch.cuda.synchronize()
        _index_put(args, expected_k, expected_v)
        eager = dict(args, k_cache=expected_k[:, 1], v_cache=expected_v[:, 1])
        del eager["k_new"], eager["v_new"]
        expected = op(**eager, enable_pdl=False)
        assert _bitwise_equal(out, expected), (ctxs, slots)
        assert _bitwise_equal(k_arena, expected_k) and _bitwise_equal(v_arena, expected_v), t


def test_back_to_back_pdl_calls_in_one_graph():
    """Five appended layer calls chained by data, PDL on, in one graph.

    Consecutive calls reuse the allocator's partial-output block; each split
    kernel waits for its predecessor grid before it writes, so replays must be
    bitwise the eager non-PDL sequence.
    """
    m, layers, capacity = 8, 5, 12_000
    g = torch.Generator(device="cuda").manual_seed(3)
    k_arena = torch.zeros(2, layers, capacity, 1, 128, dtype=torch.bfloat16, device="cuda")
    v_arena = torch.zeros_like(k_arena)
    proj = torch.randn(layers, m, 10 * 128, generator=g, device="cuda").to(torch.bfloat16)
    seqlens = torch.tensor([9_000 + m], dtype=torch.int32, device="cuda")
    slot = torch.tensor([1], dtype=torch.int32, device="cuda")

    def run(pdl):
        outs, q = [], proj[0, :, :1024].reshape(1, m, 8, 128)
        for layer in range(layers):
            if layer:
                q = (outs[-1] * 0.5 + proj[layer, :, :1024].reshape(1, m, 8, 128)).to(q.dtype)
            k_new = proj[layer, :, 1024:1152].reshape(-1, 1, 128)
            v_new = proj[layer, :, 1152:].reshape(-1, 1, 128)
            outs.append(
                op(
                    q,
                    k_arena[:, layer],
                    v_arena[:, layer],
                    seqlens,
                    slot,
                    WINDOW,
                    None,
                    pdl,
                    k_new,
                    v_new,
                )
            )
        return outs

    def refill(seed, ctx):
        gg = torch.Generator(device="cuda").manual_seed(seed)
        lo = max(0, ctx - 4200)
        k_arena.zero_()
        v_arena.zero_()
        shape = (layers, ctx - lo, 1, 128)
        k_arena[1, :, lo:ctx] = torch.randn(shape, generator=gg, device="cuda").to(torch.bfloat16)
        v_arena[1, :, lo:ctx] = torch.randn(shape, generator=gg, device="cuda").to(torch.bfloat16)
        proj.copy_(torch.randn(proj.shape, generator=gg, device="cuda").to(torch.bfloat16))
        seqlens.fill_(ctx + m)

    refill(1, 9_000)
    run(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outs = run(True)
    for t, ctx in enumerate((9_000, 300, 2_049, 7, 2_100, 11_000)):
        refill(100 + t, ctx)
        k0, v0 = k_arena.clone(), v_arena.clone()
        graph.replay()
        torch.cuda.synchronize()
        got = [o.clone() for o in graph_outs]
        got_k, got_v = k_arena.clone(), v_arena.clone()
        k_arena.copy_(k0)
        v_arena.copy_(v0)
        expected = run(False)
        assert all(_bitwise_equal(a, b) for a, b in zip(got, expected)), ctx
        assert _bitwise_equal(got_k, k_arena) and _bitwise_equal(got_v, v_arena), ctx


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(window_size=(-1, -1)), "left window"),
        (dict(window_size=(2047, -2)), r"window_size\[1\]"),
        (dict(window_size=(MAX_WINDOW_SPAN - 7, 0)), "spans"),
        (dict(cache_seqlens="int64"), "int32"),
        (dict(dtype=torch.float32), "operands"),
        (dict(hq=12, hkv=8), "multiple of KV heads"),
        (dict(hq=32, hkv=1), "block rows"),
        (dict(d=96), "head width"),
    ],
    ids=[
        "unbounded-left",
        "bad-right",
        "too-wide",
        "int64-lengths",
        "fp32",
        "gqa-ratio",
        "tile",
        "width",
    ],
)
def test_is_valid_refuses_what_the_kernels_would_get_wrong(change, match):
    spec = dict(batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[100], capacity=200, slots=2)
    spec.update({k: v for k, v in change.items() if k in ("hq", "hkv", "d", "dtype")})
    args = _inputs(spec, 5)
    if "window_size" in change:
        args["window_size"] = change["window_size"]
    if change.get("cache_seqlens") == "int64":
        args["cache_seqlens"] = args["cache_seqlens"].long()
    with pytest.raises(ValueError, match=match):
        op.is_valid(**args)


def test_the_widest_window_taken_stays_within_the_merge():
    """A window at MAX_WINDOW_SPAN is accepted and computed (16-tile splits)."""
    left = MAX_WINDOW_SPAN - 8
    spec = dict(
        batch=1,
        m=8,
        hq=8,
        hkv=1,
        d=128,
        window=(left, left),
        ctx=[left + 100],
        capacity=left + 200,
        slots=1,
    )
    args = _inputs(spec, 9)
    op.is_valid(**args)
    out = op(**args, enable_pdl=True)
    assert_within_ulp(out, op.reference(**args), ELEMENT_ULP, RMS_ULP)


# --------------------------------------------------------------------------
# The drafter adapter
# --------------------------------------------------------------------------
class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return "built-in"


def _drafter_kwargs(**over):
    spec = dict(
        batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[3_000], capacity=3_100, slots=2
    )
    args = _inputs(spec, 41)
    kwargs = dict(
        q=args["q"],
        k_cache=args["k_cache"],
        v_cache=args["v_cache"],
        cache_seqlens=args["cache_seqlens"],
        cache_batch_idx=args["cache_batch_idx"],
        causal=False,
        window_size=WINDOW,
    )
    kwargs.update(over)
    return kwargs


@pytest.mark.parametrize(
    "over",
    [
        dict(causal=True),
        dict(window_size=(-1, -1)),
        dict(window_size=(MAX_WINDOW_SPAN, MAX_WINDOW_SPAN)),
        dict(num_splits=1),
        dict(cache_seqlens=None),
    ],
    ids=["causal", "unwindowed", "too-wide", "extra-keyword", "no-lengths"],
)
def test_adapter_sends_what_the_kernel_does_not_take_to_the_builtin(over):
    builtin = _Recorder()
    adapter = draft.WindowedBlockAttention(builtin, enable_pdl=False)
    kwargs = _drafter_kwargs(**over)
    assert adapter(**kwargs) == "built-in"
    (call,) = builtin.calls
    for key, value in kwargs.items():
        assert call[key] is value, key


def test_adapter_runs_the_windowed_call_on_the_kernel():
    builtin = _Recorder()
    adapter = draft.WindowedBlockAttention(builtin, enable_pdl=True)
    kwargs = _drafter_kwargs()
    out = adapter(**kwargs)
    assert not builtin.calls
    args = {k: v for k, v in kwargs.items() if k != "causal"}
    assert torch.equal(out, op(**args, enable_pdl=True))


# --------------------------------------------------------------------------
# The written-out block decode on a small synthetic DFlash 2 drafter
# --------------------------------------------------------------------------
_SYNTHETIC = dict(
    hidden=1024,
    heads=16,
    kv_heads=2,
    head_dim=128,
    intermediate=2048,
    layers=2,
    vocab=1024,
    block=8,
    window=256,
    captures=2,
)


def _synthetic_drafter(tmp_path):
    """A two-layer DFlash 2 drafter of the published layout, random weights, built-in loader.

    16 / 2 heads of width 128 (eight query heads per KV head, as one TP8 rank
    of the published drafter), a non-causal 256-token window, block 8, the
    DFlash 2 block convolutions and candidate selector.
    """
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    s = _SYNTHETIC
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "dflash_config": {
            "block_size": s["block"],
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": s["vocab"] - 1,
            "selector_rank": 32,
            "selector_top_k": 4,
            "target_layer_ids": list(range(s["captures"])),
        },
        "dtype": "bfloat16",
        "head_dim": s["head_dim"],
        "hidden_act": "silu",
        "hidden_size": s["hidden"],
        "intermediate_size": s["intermediate"],
        "is_causal": False,
        "layer_types": ["sliding_attention"] * s["layers"],
        "max_position_embeddings": 65_536,
        "max_window_layers": s["layers"],
        "num_attention_heads": s["heads"],
        "num_hidden_layers": s["layers"],
        "num_key_value_heads": s["kv_heads"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 500_000, "rope_type": "default"},
        "rope_theta": 500_000,
        "sliding_window": s["window"],
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": s["vocab"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    pretrained = AutoConfig.from_pretrained(str(tmp_path))
    with torch.device("cuda"):
        drafter = DFlashForCausalLM(
            ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM")
        )

    g = torch.Generator(device="cuda").manual_seed(0)

    def linear(out_features, in_features):
        w = torch.randn(out_features, in_features, generator=g, device="cuda")
        return (w / in_features**0.5).to(torch.bfloat16)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16)

    h, hd, inter = s["hidden"], s["head_dim"], s["intermediate"]
    conv_rows = 2 * 2 * h // 16
    weights = {
        "fc.weight": linear(h, h * s["captures"]),
        "hidden_norm.weight": norm(h),
        "norm.weight": norm(h),
        "candidate_selector.hidden_projection.weight": linear(32, h),
        "candidate_selector.predecessor_codebook": linear(s["vocab"], 32),
        "candidate_selector.successor_codebook": linear(s["vocab"], 32),
    }
    for i in range(s["layers"]):
        p = f"layers.{i}."
        weights.update(
            {
                p + "self_attn.q_proj.weight": linear(s["heads"] * hd, h),
                p + "self_attn.k_proj.weight": linear(s["kv_heads"] * hd, h),
                p + "self_attn.v_proj.weight": linear(s["kv_heads"] * hd, h),
                p + "self_attn.o_proj.weight": linear(h, s["heads"] * hd),
                p + "self_attn.q_norm.weight": norm(hd),
                p + "self_attn.k_norm.weight": norm(hd),
                p + "input_layernorm.weight": norm(h),
                p + "post_attention_layernorm.weight": norm(h),
                p + "mlp.gate_proj.weight": linear(inter, h),
                p + "mlp.up_proj.weight": linear(inter, h),
                p + "mlp.down_proj.weight": linear(h, inter),
            }
        )
        for conv in ("attention_conv", "mlp_conv"):
            base = 0.5 * torch.randn(2, 2, h, generator=g, device="cuda")
            weights[p + conv + ".base_kernel"] = base.to(torch.bfloat16)
            weights[p + conv + ".kernel_projection.weight"] = linear(conv_rows, h) * 0.1
    drafter.load_weights(weights)
    return drafter


def _fill_context(drafter, k_arena, v_arena, slot, ctx, seed):
    """The drafter's context K/V for positions [0, ctx) of one slot, from random target features."""
    s = _SYNTHETIC
    g = torch.Generator(device="cuda").manual_seed(seed)
    captured = torch.randn(ctx, s["hidden"] * s["captures"], generator=g, device="cuda")
    k, v = drafter.precompute_context_kv(
        drafter.project_target_hidden(captured.to(torch.bfloat16)),
        torch.arange(ctx, device="cuda"),
    )
    k_arena[slot, :, :ctx] = k.permute(1, 0, 2, 3)
    v_arena[slot, :, :ctx] = v.permute(1, 0, 2, 3)


@torch.inference_mode()
def test_synthetic_drafter_block_decode_is_bitwise_the_builtin_forward_on_the_op(tmp_path):
    """``windowed_block_decode`` == built-in ``dflash_forward`` with the windowed attention.

    Two requests in distinct slots, contexts inside and past the window; the
    block columns start as NaN (the block decode must overwrite them, the built-in's
    index_put does). Output and whole arena bitwise, eagerly and from one
    CUDA-graph capture replayed with new contexts, slots and inputs.
    """
    s = _SYNTHETIC
    drafter = _synthetic_drafter(tmp_path)
    assert drafter.dflash_attention_backend == "VANILLA"
    window = (s["window"] - 1, s["window"] - 1)
    assert all(drafter._resolve_block_attention(i) == (False, window) for i in range(s["layers"]))
    assert draft.block_decode_unsupported_reason(drafter) is None

    slots, capacity, block = 3, 1_400, s["block"]
    shape = (slots, s["layers"], capacity, s["kv_heads"], s["head_dim"])
    k_arena = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    v_arena = torch.zeros_like(k_arena)
    noise = torch.zeros(2, block, s["hidden"], dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros(2, block, dtype=torch.long, device="cuda")
    num_ctx = torch.zeros(2, dtype=torch.long, device="cuda")
    slot_idx = torch.zeros(2, dtype=torch.long, device="cuda")

    def set_step(ctxs, slot_ids, seed):
        k_arena.zero_()
        v_arena.zero_()
        for b, (ctx, slot) in enumerate(zip(ctxs, slot_ids)):
            _fill_context(drafter, k_arena, v_arena, slot, ctx, seed + b)
            k_arena[slot, :, ctx : ctx + block] = float("nan")
            v_arena[slot, :, ctx : ctx + block] = float("nan")
        g = torch.Generator(device="cuda").manual_seed(seed)
        noise.copy_(torch.randn(noise.shape, generator=g, device="cuda").to(torch.bfloat16))
        ctx_t = torch.tensor(ctxs, device="cuda")
        positions.copy_(ctx_t.view(-1, 1) + torch.arange(block, device="cuda"))
        num_ctx.copy_(ctx_t)
        slot_idx.copy_(torch.tensor(slot_ids, device="cuda"))

    def decode(forward):
        return forward(
            noise_embedding=noise,
            query_positions=positions,
            num_ctx_per_req=num_ctx,
            ctx_k_cache=k_arena,
            ctx_v_cache=v_arena,
            ctx_cache_batch_idx=slot_idx,
        )

    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    def builtin_forward(**kwargs):
        return DFlashForCausalLM.dflash_forward(drafter, **kwargs)

    def expected(ctxs, slot_ids, seed):
        set_step(ctxs, slot_ids, seed)
        out = decode(builtin_forward).clone()
        return out, k_arena.clone(), v_arena.clone()

    assert draft.install_windowed_block_attention(drafter, enable_pdl=True)
    assert draft.install_block_decode(drafter, enable_pdl=True) is None
    assert drafter.dflash_forward.__func__ is draft.windowed_block_decode

    steps = (([700, 100], [2, 0]), ([1_300, 255], [0, 1]), ([5, 900], [1, 2]))
    for t, (ctxs, slot_ids) in enumerate(steps):
        out_ref, k_ref, v_ref = expected(ctxs, slot_ids, 10 + t)
        assert out_ref.isfinite().all()
        set_step(ctxs, slot_ids, 10 + t)
        out = decode(drafter.dflash_forward)
        assert _bitwise_equal(out, out_ref), (ctxs, (out.float() - out_ref.float()).abs().max())
        assert _bitwise_equal(k_arena, k_ref) and _bitwise_equal(v_arena, v_ref), ctxs

    set_step(*steps[0], 10)
    decode(drafter.dflash_forward)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    set_step(*steps[0], 10)
    with torch.cuda.graph(graph):
        graph_out = decode(drafter.dflash_forward)
    for t, (ctxs, slot_ids) in enumerate(steps + (([1_000, 3], [2, 1]),)):
        out_ref, k_ref, v_ref = expected(ctxs, slot_ids, 20 + t)
        set_step(ctxs, slot_ids, 20 + t)
        graph.replay()
        torch.cuda.synchronize()
        assert _bitwise_equal(graph_out, out_ref), ctxs
        assert _bitwise_equal(k_arena, k_ref) and _bitwise_equal(v_arena, v_ref), ctxs


# --------------------------------------------------------------------------
# The published drafter, end to end
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not _DRAFTER_DIR or not _has_flash_attn(),
    reason="needs MUSESPARK_REFERENCE_DRAFTER and flash_attn",
)
@torch.inference_mode()
def test_published_drafter_block_decode_in_the_serving_arena():
    """The five-layer block decode of the published drafter (TP1: 64 / 8 heads)
    in an arena of the serving capacity (1M + 8 columns), before and after the
    windowed-attention install, at a 3,000- and a 100,770-token context.

    Gate: every layer's attention call meets the budget against the built-in
    kernel on the same inputs (float64 evaluation; max within 1 ulp of the row
    scale above FlashAttention's, mean within 10 %). The decode's distance from
    the built-in is a numerics change, not noise: in this arena FlashAttention
    keeps the whole window in one split, so the built-in is deterministic and
    the windowed attention differs from it by accumulation order (bounded here
    at 2 % RMS on the block hidden states as a gross-error check). Then the
    written-out block decode must be bitwise the windowed-attention decode,
    arena included.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda"):
        drafter = DFlashForCausalLM(
            ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM")
        )
    drafter.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert drafter.dflash_attention_backend == "VANILLA"
    windows = [drafter._resolve_block_attention(i) for i in range(pretrained.num_hidden_layers)]
    assert all(w == (False, WINDOW) for w in windows), windows

    block, num_capture = 8, len(drafter.target_layer_ids)
    drafter._build_fused_kv_buffers()
    shape = (
        1,
        drafter._num_attn_layers,
        SERVING_CAPACITY,
        drafter._num_kv_heads,
        drafter._head_dim,
    )
    pool_k = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    pool_v = torch.zeros_like(pool_k)

    def filled(ctx_len):
        # Only the window's worth of context is ever read; fill the last 2,200.
        return max(0, ctx_len - 2_200), ctx_len + block

    def fill(ctx_len, seed):
        lo, _ = filled(ctx_len)
        g = torch.Generator(device="cuda").manual_seed(seed)
        captured = torch.randn(
            ctx_len - lo, pretrained.hidden_size * num_capture, generator=g, device="cuda"
        ).to(torch.bfloat16)
        k, v = drafter.precompute_context_kv(
            drafter.project_target_hidden(captured), torch.arange(lo, ctx_len, device="cuda")
        )
        pool_k.zero_()
        pool_v.zero_()
        pool_k[0, :, lo:ctx_len] = k.permute(1, 0, 2, 3)
        pool_v[0, :, lo:ctx_len] = v.permute(1, 0, 2, 3)

    def decode(ctx_len, seed):
        noise = torch.randn(
            1,
            block,
            pretrained.hidden_size,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            device="cuda",
        ).to(torch.bfloat16)
        return drafter.dflash_forward(
            noise_embedding=noise,
            query_positions=torch.arange(ctx_len, ctx_len + block, device="cuda").unsqueeze(0),
            num_ctx_per_req=torch.tensor([ctx_len], device="cuda"),
            ctx_k_cache=pool_k,
            ctx_v_cache=pool_v,
            ctx_cache_batch_idx=torch.tensor([0], device="cuda"),
        ).float()

    def rel_rms(a, b):
        return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()

    cases = [(ctx, seed) for ctx in (3_000, 100_770) for seed in (1, 2)]
    builtin = []
    for ctx, seed in cases:
        fill(ctx, 100 + ctx)
        builtin.append(decode(ctx, seed))

    assert draft.install_windowed_block_attention(drafter, enable_pdl=True)
    adapter = drafter._dflash_flash_attention
    assert isinstance(adapter, draft.WindowedBlockAttention)
    calls = []

    def checked(**kwargs):
        out = adapter(**kwargs)
        args = {k: v for k, v in kwargs.items() if k != "causal"}
        _assert_no_less_accurate_than_flash_attention(out, args)
        calls.append(kwargs["window_size"])
        return out

    drafter._dflash_flash_attention = checked
    ours, arena = [], []
    for ctx, seed in cases:
        fill(ctx, 100 + ctx)
        ours.append(decode(ctx, seed))
        lo, hi = filled(ctx)
        # Outside [lo, hi) the arena is zero; keep only the written columns.
        arena.append((pool_k[0, :, lo:hi].clone(), pool_v[0, :, lo:hi].clone()))
    assert len(calls) == len(cases) * pretrained.num_hidden_layers
    drafter._dflash_flash_attention = adapter
    distance = rel_rms(torch.cat(ours), torch.cat(builtin))
    assert distance <= 0.02, distance

    # The written-out block decode with the append fused is bitwise the
    # windowed-attention decode, and leaves the arena exactly as the built-in index_put does.
    assert draft.install_block_decode(drafter, enable_pdl=True) is None
    assert draft.install_block_decode(drafter, enable_pdl=True) is None  # idempotent
    assert drafter.dflash_forward.__func__ is draft.windowed_block_decode
    for (ctx, seed), expected, (k_expected, v_expected) in zip(cases, ours, arena):
        fill(ctx, 100 + ctx)
        pool_k[0, :, ctx : ctx + block].fill_(float("nan"))  # the append must overwrite these
        pool_v[0, :, ctx : ctx + block].fill_(float("nan"))
        got = decode(ctx, seed)
        assert torch.equal(got, expected), (ctx, seed, (got - expected).abs().max().item())
        lo, hi = filled(ctx)
        for pool, written in ((pool_k, k_expected), (pool_v, v_expected)):
            assert _bitwise_equal(pool[0, :, lo:hi], written), ctx
            assert torch.count_nonzero(pool) == torch.count_nonzero(written), "outside the block"


# --------------------------------------------------------------------------
# Microbenchmark (MODELING_V2_TIMING=1): one drafter layer's block attention
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("MODELING_V2_TIMING") or not _has_flash_attn(),
    reason="MODELING_V2_TIMING unset or no flash_attn",
)
@pytest.mark.parametrize("hq, hkv", ((8, 1), (16, 2)), ids=("tep8", "tep4"))
def test_microbenchmark_block_attention(hq, hkv):
    """Per drafter layer, B=1, block 8, a 100,770-token context in the 1M arena: the built-in
    (``index_put`` append + ``flash_attn_with_kvcache``) vs the op with the append fused, each
    replayed 20 times in one CUDA graph. Printed, never gated."""
    spec = dict(
        batch=1, m=8, hq=hq, hkv=hkv, d=128, window=WINDOW, ctx=[100_770],
        capacity=SERVING_CAPACITY, slots=2, append=True,
    )  # fmt: skip
    args = _inputs(spec, 1)
    k_arena, v_arena = _arenas(args)
    plain = {k: v for k, v in args.items() if k not in ("k_new", "v_new")}

    def builtin():
        _index_put(args, k_arena, v_arena)
        _flash_attn(plain)

    def fused():
        op(**args, enable_pdl=True)

    times = {}
    for name, fn in (("builtin index_put+flash_attn", builtin), ("dflash_window_attention", fused)):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                fn()
        graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / 20)
        times[name] = sorted(samples)[2]
    line = ", ".join(f"{name} {us:.1f} us" for name, us in times.items())
    print(f"\nTIMING drafter block attention {hq} q / {hkv} kv heads: {line}")
