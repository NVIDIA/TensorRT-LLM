# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The musespark_open TEP target over each layout of the DFlash drafter's context.

* ``compose`` plans every fast path over the dense arena and every one but
  ``DENSE_ARENA_PATHS`` and their dependents over a paged context, whatever the
  attention backend that keeps it (no GPU).
* ``dflash_window_attention`` and ``dflash_window_attention_wide``, dense and
  paged, against their reference; on the same values a paged call is bitwise the
  dense call, the appended block included.
* ``windowed_block_decode`` over a TRTLLM drafter's pages, on either window op, is
  bitwise the VANILLA drafter's over the dense arena, and agrees with the
  built-in TRTLLM forward (SM100 family, trtllm-gen).
"""

import importlib.util
import os
import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import ModelingV2Mode
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    WINDOW,
    dflash_window_attention,
    unsupported_reason,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_window_attention_wide import (  # noqa: E501
    dflash_window_attention_wide,
    wide_unsupported_reason,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

_KEYS = sorted(modeling.TARGET_PATHS)
_SEMANTICS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    *([".."] * 4),
    "speculative",
    "hw_agnostic",
    "test_dflash2_semantics.py",
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------
# compose: the plan per drafter context layout
# --------------------------------------------------------------------------
def _model_config(backend, tp):
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    spec = DFlashDecodingConfig(
        max_draft_len=7, speculative_model="/drafter", attention_backend=backend
    )
    mapping = types.SimpleNamespace(tp_size=tp, gpus_per_node=4, tp_group=list(range(tp)))
    return types.SimpleNamespace(
        mapping=mapping,
        spec_config=spec,
        quant_config=types.SimpleNamespace(quant_algo=QuantAlgo.NVFP4),
    )


def _plan(backend, monkeypatch, spec_worker=None, key=(103, 4)):
    """A composition's plan with no precondition read and no TP collective."""
    monkeypatch.setattr(dispatch, "missing_preconditions", lambda *args: [])
    monkeypatch.setattr(dispatch, "agreed_union", lambda mapping, local: list(dict.fromkeys(local)))
    sm, tp = key
    plan = dispatch.compose(
        None,
        _model_config(backend, tp),
        ModelingV2Mode.REQUIRE,
        modeling.TARGET_PATHS,
        divmod(sm, 10),
        spec_worker,
    )
    return set(modeling.TARGET_PATHS[key].served), set(plan.paths)


def _left_out(served, *names):
    return set(dispatch.dependents(names)) & served


@pytest.mark.parametrize("key", _KEYS)
def test_the_full_dense_arena_keeps_every_full_arena_fast_path(key, monkeypatch):
    served, planned = _plan("VANILLA", monkeypatch, key=key)
    assert served - planned == _left_out(served, *dispatch.WINDOWED_CONTEXT_ARENA_PATHS)


@pytest.mark.parametrize("key", _KEYS)
@pytest.mark.parametrize("backend", ["TRTLLM", "FA4"])
def test_a_paged_context_drops_only_the_dense_arena_paths(backend, key, monkeypatch):
    served, planned = _plan(backend, monkeypatch, key=key)
    dropped = dispatch.DENSE_ARENA_PATHS + dispatch.WINDOWED_CONTEXT_ARENA_PATHS
    assert served - planned == _left_out(served, *dropped)
    assert "context_kv_fork" not in planned
    kept = {
        "drafter_block_decode",
        "wide_drafter_window_attention",
        "drafter_hops",
        "drafter_pdl_chain",
        "verify_fold",
    }
    assert kept & served <= planned


def test_the_window_rule_bounds_the_dense_arena_only(monkeypatch):
    windowed = types.SimpleNamespace(windowed_ctx_arena=True)
    served, dense = _plan("VANILLA", monkeypatch, windowed)
    # sm103 tep4 serves the window compaction, so a windowed arena keeps every path.
    assert dense == served
    assert _plan("TRTLLM", monkeypatch, windowed)[1] == _plan("TRTLLM", monkeypatch)[1]


def test_the_worker_backend_settles_an_open_auto(monkeypatch):
    worker = types.SimpleNamespace(_dflash_attention_backend="TRTLLM")
    assert _plan("AUTO", monkeypatch, worker)[1] == _plan("TRTLLM", monkeypatch)[1]


# --------------------------------------------------------------------------
# The window ops, dense and paged
# --------------------------------------------------------------------------
def _case(**over):
    """A call: block 8, a TP8 rank's heads, the served window, one slot per request plus one."""
    spec = dict(batch=1, m=8, hq=8, hkv=1, d=128, window=WINDOW, ctx=[5_000], capacity=8_200)
    spec.update(over)
    spec.setdefault("slots", spec["batch"] + 1)
    return spec


def _wide(batch, ctx=None, **over):
    """A wide call: a TP4 rank's heads, by default every slot holding a full window."""
    return _case(batch=batch, ctx=ctx or [2_056] * batch, hq=16, hkv=2, capacity=2_080, **over)


#: The TP4 serving step: a ~100k context, the block appended, PDL launches.
_TP4_STEP = dict(hq=16, hkv=2, ctx=[100_770], capacity=100_800, pdl=True, append=True)
#: Several requests of mixed lengths.
_MIXED = dict(batch=3, ctx=[9_000, 17, 2_100], capacity=9_024)

#: The single-request op: (what the call covers, spec, tokens per page).
_SINGLE = [
    ("tp4 serving step", _case(**_TP4_STEP), 32),
    ("tp4, the merge released at entry", _case(**_TP4_STEP, entry=True), 32),
    ("mixed lengths", _case(**_MIXED), 16),
    ("mixed lengths", _case(**_MIXED), 64),
    ("tp4, a context shorter than the window", _case(hq=16, hkv=2, ctx=[500], capacity=544), 32),
    (
        "narrow window, one split, a block across a page boundary",
        _case(batch=2, window=(31, 3), ctx=[396, 20], capacity=416, append=True),
        16,
    ),
]

#: The multi-request op: (what the call covers, spec, tokens per page).
_WIDE = [
    ("8 requests, full windows, appended, PDL", _wide(8, append=True, pdl=True), 32),
    (
        "16 requests, mixed lengths",
        _wide(16, [2_056, 1, 17, 500, 2_000, 33, 64, 65] * 2, append=True),
        16,
    ),
    ("8 requests, the block read from the cache", _wide(8), 64),
    ("40 requests, appended", _wide(40, append=True), 32),
    (
        "6 requests, narrow window, one split written directly",
        _wide(6, [396, 20, 1, 40, 64, 100], window=(31, 3), append=True),
        16,
    ),
]


def _paged(k_arena, v_arena, page, gen, table=None):
    """The arena's slots on scattered pages: a pool [pages, K/V, Hkv, page, D] and its table."""
    slots, capacity, hkv, d = k_arena.shape
    per_slot = -(-capacity // page)
    num_pages = slots * per_slot + 3
    if table is None:
        table = torch.randperm(num_pages, device="cuda", generator=gen)[: slots * per_slot]
        table = table.view(slots, per_slot).int().contiguous()
    pool = torch.randn(num_pages, 2, hkv, page, d, device="cuda", generator=gen).to(k_arena.dtype)
    pos = torch.arange(capacity, device="cuda")
    for half, arena in enumerate((k_arena, v_arena)):
        pool[:, half].transpose(1, 2)[table[:, pos // page].long(), pos % page] = arena
    return table, pool


def _check_paged_is_dense(op, why, spec, page):
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{why}/{page}".encode()) % 2**31)
    batch, m, hq, hkv, d = (spec[key] for key in ("batch", "m", "hq", "hkv", "d"))
    shape = (spec["slots"], spec["capacity"], hkv, d)
    k_arena = torch.randn(shape, device="cuda", generator=gen).bfloat16()
    v_arena = torch.randn(shape, device="cuda", generator=gen).bfloat16()
    q = (2 * torch.randn(batch, m, hq, d, device="cuda", generator=gen)).bfloat16()
    seqlens = torch.tensor(spec["ctx"], dtype=torch.int32, device="cuda") + m
    slots = (spec["slots"] - 1 - torch.arange(batch, device="cuda")).int()
    kwargs = {}
    if spec.get("append"):
        kwargs["k_new"] = torch.randn(batch * m, hkv, d, device="cuda", generator=gen).bfloat16()
        kwargs["v_new"] = torch.randn(batch * m, hkv, d, device="cuda", generator=gen).bfloat16()
    if spec.get("entry"):
        kwargs["merge_entry_trigger"] = True
    table, pool = _paged(k_arena, v_arena, page, gen)
    call = (seqlens, slots, spec["window"], None, spec.get("pdl", False))
    with validating(op):
        out = op(q, k_arena, v_arena, *call, **kwargs)
    ref = op.reference(q, k_arena, v_arena, *call, **kwargs)
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=1.0)
    k_pages, v_pages = pool[:, 0].transpose(1, 2), pool[:, 1].transpose(1, 2)
    with validating(op):
        paged = op(q, k_pages, v_pages, *call, page_table=table, **kwargs)
    assert torch.equal(paged, out)
    if "k_new" in kwargs:
        for b, ctx in enumerate(spec["ctx"]):
            cols = torch.arange(ctx, ctx + m, device="cuda")
            rows = table[slots[b].long(), cols // page].long(), cols % page
            assert torch.equal(k_pages[rows], k_arena[slots[b].long(), ctx : ctx + m])
            assert torch.equal(v_pages[rows], v_arena[slots[b].long(), ctx : ctx + m])


@needs_cuda
@pytest.mark.parametrize(("why", "spec", "page"), _SINGLE, ids=[f"{w}-{p}" for w, _, p in _SINGLE])
def test_dflash_window_attention_over_pages_is_the_dense_call(why, spec, page):
    _check_paged_is_dense(dflash_window_attention, why, spec, page)


@needs_cuda
@pytest.mark.parametrize(("why", "spec", "page"), _WIDE, ids=[f"{w}-{p}" for w, _, p in _WIDE])
def test_dflash_window_attention_wide_over_pages_is_the_dense_call(why, spec, page):
    _check_paged_is_dense(dflash_window_attention_wide, why, spec, page)


@needs_cuda
@pytest.mark.parametrize("reason_of", [unsupported_reason, wide_unsupported_reason])
def test_a_page_stride_off_the_row_grid_is_refused(reason_of):
    base = torch.zeros(10 * 4224, dtype=torch.bfloat16, device="cuda")
    pages = base.as_strided((9, 16, 2, 128), (4224, 256, 128, 1))
    q = torch.zeros(1, 8, 4, 128, dtype=torch.bfloat16, device="cuda")
    lens = torch.full((1,), 9, dtype=torch.int32, device="cuda")
    rows = torch.zeros(1, dtype=torch.int32, device="cuda")
    table = torch.zeros(1, 2, dtype=torch.int32, device="cuda")
    reason = reason_of(q, pages, pages, lens, rows, (15, 15), None, None, table)
    assert reason is not None and "whole number of rows" in reason


# --------------------------------------------------------------------------
# windowed_block_decode over a TRTLLM drafter's pages
# --------------------------------------------------------------------------
def _semantics():
    """The tiny DFlash 2 drafter of the speculative tests (config, weights)."""
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", _SEMANTICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trtllm_gen_unavailable():
    if not torch.cuda.is_available():
        return "needs CUDA"
    from tensorrt_llm._torch.speculative.dflash_attention import (
        dflash_trtllm_gen_unavailability_reason,
    )

    return dflash_trtllm_gen_unavailability_reason()


def _drafter(sem, backend):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    model_config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(model_config, dflash_attention_backend=backend).to("cuda")
    drafter.load_weights(dict(sem._tiny_weights(True)))
    drafter._build_fused_kv_buffers()
    return drafter


def _install(drafter, block_conv, wide):
    """``drafter_block_decode`` (with or without the fused convolution), then the wide op."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
        block_decode,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_drafter_window_attention import (  # noqa: E501
        attention as wide_attention,
    )

    block_decode.install(drafter, block_conv, True)
    assert drafter.dflash_forward.__func__ is block_decode.windowed_block_decode
    if wide:
        wide_attention.install(drafter)


def _context(sem, drafter, ctx_lens, page_size, seed=3):
    """Each request's context K/V on scattered pages, and the same values in a dense arena."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = sem.TINY["hidden_size"]
    layers, nkv, hd = drafter._num_attn_layers, drafter._num_kv_heads, drafter._head_dim
    capacity = (max(ctx_lens) + sem.BLOCK) // page_size * page_size + page_size
    k_arena = torch.zeros(
        len(ctx_lens), layers, capacity, nkv, hd, device="cuda", dtype=torch.bfloat16
    )
    v_arena = torch.zeros_like(k_arena)
    for b, ctx in enumerate(ctx_lens):
        feats = torch.randn(ctx, hidden * sem.NUM_CAPTURE, device="cuda", generator=gen) * 0.5
        projected = drafter.project_target_hidden(feats.bfloat16())
        k, v = drafter.precompute_context_kv(projected, torch.arange(ctx, device="cuda"))
        k_arena[b, :, :ctx], v_arena[b, :, :ctx] = k.transpose(0, 1), v.transpose(0, 1)
    # One table for every layer's pool, as the worker's.
    table, pools = None, []
    for layer in range(layers):
        table, pool = _paged(k_arena[:, layer], v_arena[:, layer], page_size, gen, table)
        pools.append(pool)
    noise = (
        torch.randn(len(ctx_lens), sem.BLOCK, hidden, device="cuda", generator=gen) * 0.5
    ).bfloat16()
    return table, pools, k_arena, v_arena, noise


def _forward(fn, sem, ctx_lens, noise, **context):
    lens = torch.tensor(ctx_lens, device="cuda")
    return fn(
        noise_embedding=noise,
        query_positions=lens[:, None] + torch.arange(sem.BLOCK, device="cuda"),
        num_ctx_per_req=lens,
        ctx_cache_batch_idx=torch.arange(len(ctx_lens), device="cuda"),
        **context,
    )


def _appended(pools, table, ctx_lens, block, page_size):
    """Each layer's block K/V as appended to the pages, [B, L, block, Hkv, D] for K and V."""
    out = []
    for half in (0, 1):
        per_request = []
        for b, ctx in enumerate(ctx_lens):
            cols = torch.arange(ctx, ctx + block, device="cuda")
            rows = table[b, cols // page_size].long(), cols % page_size
            per_request.append(torch.stack([p[:, half].transpose(1, 2)[rows] for p in pools]))
        out.append(torch.stack(per_request))
    return out


#: Windows that start mid-page, a one-token context, contexts past the window; the
#: six requests' 48 rows go to the wide op where it is installed.
_CTX_LENS = {False: [24, 1, 40, 15], True: [24, 1, 40, 15, 33, 7]}


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("page_size", [16, 32])
def test_the_block_decode_over_pages_is_bitwise_the_dense_arena_one(page_size, wide):
    if (reason := _trtllm_gen_unavailable()) is not None:
        pytest.skip(reason)
    sem = _semantics()
    ctx_lens = _CTX_LENS[wide]
    paged, dense = _drafter(sem, "TRTLLM"), _drafter(sem, "VANILLA")
    table, pools, k_arena, v_arena, noise = _context(sem, paged, ctx_lens, page_size)
    _install(paged, True, wide)
    _install(dense, True, wide)
    got = _forward(
        paged.dflash_forward,
        sem,
        ctx_lens,
        noise,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_kv_cache=pools,
        ctx_page_table=table,
    )
    want = _forward(
        dense.dflash_forward, sem, ctx_lens, noise, ctx_k_cache=k_arena, ctx_v_cache=v_arena
    )
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
    k_pages, v_pages = _appended(pools, table, ctx_lens, sem.BLOCK, page_size)
    for b, ctx in enumerate(ctx_lens):
        assert torch.equal(k_pages[b], k_arena[b, :, ctx : ctx + sem.BLOCK])
        assert torch.equal(v_pages[b], v_arena[b, :, ctx : ctx + sem.BLOCK])


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("block_conv", [False, True])
def test_the_block_decode_over_pages_agrees_with_the_builtin_trtllm_forward(
    page_size, block_conv, wide
):
    if (reason := _trtllm_gen_unavailable()) is not None:
        pytest.skip(reason)
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
        block_decode,
    )
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    sem = _semantics()
    ctx_lens = _CTX_LENS[wide]
    drafter = _drafter(sem, "TRTLLM")
    table, pools, _, _, noise = _context(sem, drafter, ctx_lens, page_size)
    if block_conv:
        block_decode.install(drafter, True, False)
    builtin_pools = [pool.clone() for pool in pools]
    want = _forward(
        lambda **kw: DFlashForCausalLM.dflash_forward(drafter, **kw),
        sem,
        ctx_lens,
        noise,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_kv_cache=builtin_pools,
        ctx_page_table=table,
    )
    _install(drafter, block_conv, wide)
    got = _forward(
        drafter.dflash_forward,
        sem,
        ctx_lens,
        noise,
        ctx_k_cache=None,
        ctx_v_cache=None,
        ctx_kv_cache=pools,
        ctx_page_table=table,
    )
    # The built-in attention is TRTLLM-Gen's causal decode plus a block fixup: bf16-close.
    err = (got.float() - want.float()).abs().max().item()
    assert err <= 0.02 * max(want.float().abs().max().item(), 1.0), err
    # Layer 0 appends the same block K/V; a later layer's follow the attention above it.
    assert torch.equal(pools[0], builtin_pools[0])
    for got_pool, want_pool in zip(pools[1:], builtin_pools[1:]):
        err = (got_pool.float() - want_pool.float()).abs().max().item()
        assert err <= 0.02 * max(want_pool.float().abs().max().item(), 1.0), err
