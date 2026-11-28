# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 rank's decode forms (`TEP_FAST_DECODE_ENV`).

* **Layout** (`_tep_fast_decode`, CPU): on by default at tep4 only (attention TP4 without
  attention DP, four 128-expert windows); `0` turns it off; set at another layout it warns.
* **Hyper-connection sites** (`DecodeTarget._hc_site` / `_hc_moe_site`, CPU, recorders): with
  `hc_decode_pairs` a decode site of one row takes the GEMV pair, of 2..`_HC_DECODE_ROWS_MAX`
  rows the tcgen05 rows pair (the MoE site with its NVFP4 epilogue), larger ones the
  three-launch site; without it every site keeps the three launches.
* **Sparse GQA at the tep4 shard** (GPU): `qsa_sparse_gqa_splitk_cute` over 6 q heads on one
  e4m3 KV head, gated by the trunk's projection row, within the band every sparse GQA entry
  holds against its fp32 reference, eagerly and replayed from a CUDA graph. The shard's first
  call, which names the target's reach (`max_rows`), compiles every plan up to 16 rows.

The route front at tep4 is covered by `test_route_front.py` and the op's tep4 cells.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.attention import (
    qsa_sparse_gqa_splitk_cute__decode as splitk_mod,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402 — needs the path declared above


def _core(world=4, attn_tp=4, dp=1, local=128, one_rank=False):
    return SimpleNamespace(
        one_rank=one_rank,
        dp_size=dp,
        attn_tp_size=attn_tp,
        mapping=SimpleNamespace(world_size=world),
        local_experts=local,
        num_experts=512,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "value, core, expected",
    [
        (None, _core(), True),  # tep4, the default
        ("0", _core(), False),
        ("1", _core(), True),
        (None, _core(local=512), False),  # tp4: every expert at a quarter of its intermediate
        (None, _core(attn_tp=1, dp=4), False),  # dep4
        (None, _core(world=2, attn_tp=2, local=256), False),  # tep2
        (None, _core(world=1, attn_tp=1, local=512, one_rank=True), False),
    ],
)
def test_layout(monkeypatch, value, core, expected):
    if value is None:
        monkeypatch.delenv(M.TEP_FAST_DECODE_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_FAST_DECODE_ENV, value)
    assert M._tep_fast_decode(core) is expected


@pytest.mark.cpu_only
def test_set_at_another_layout_warns(monkeypatch):
    warned = []
    monkeypatch.setenv(M.TEP_FAST_DECODE_ENV, "1")
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert not M._tep_fast_decode(_core(attn_tp=1, dp=4))
    assert len(warned) == 1 and M.TEP_FAST_DECODE_ENV in warned[0]


def _sites(monkeypatch, pairs: bool):
    calls = []

    def rec(name, out):
        def f(*args, **kw):
            calls.append(name)
            return out

        return f

    hc = torch.zeros(1)
    monkeypatch.setattr(M, "qwen4_exp_hc_decode_rows_first", rec("rows", (hc, hc, hc, hc)))
    monkeypatch.setattr(M, "qwen4_exp_hc_decode_rows_mix", rec("rows_mix", (hc, "xq", "xsf")))
    monkeypatch.setattr(M, "qwen4_exp_hc_decode_first_cute", rec("gemv", (hc, hc, hc)))
    monkeypatch.setattr(M, "qwen4_exp_hc_decode_mix_cute", rec("gemv_mix", (hc, hc)))
    monkeypatch.setattr(
        M, "qwen4_exp_hc_decode_mix_fp4_cute", rec("gemv_mix_fp4", (hc, hc, "xq", "xsf"))
    )
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", rec("inject_norm", (hc, hc)))
    monkeypatch.setattr(M, "_hc_mixer", rec("mixer", (hc, hc)))
    monkeypatch.setattr(M, "_hc_moe_mixer", rec("moe_mixer", (hc, hc, None)))
    target = M.DecodeTarget.__new__(M.DecodeTarget)
    target.core = SimpleNamespace(
        hc_decode_pairs=pairs, hc_lowrank=24, hc_count=4, eps=1e-6, dp_size=1
    )
    return target, calls


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "pairs, rows, expected",
    [
        (True, 1, ["gemv", "gemv_mix"]),
        (True, 4, ["rows", "rows_mix"]),
        (True, M._HC_DECODE_ROWS_MAX, ["rows", "rows_mix"]),
        (True, M._HC_DECODE_ROWS_MAX + 1, ["inject_norm", "mixer"]),
        (False, 4, ["inject_norm", "mixer"]),
    ],
)
def test_hc_site_forms(monkeypatch, pairs, rows, expected):
    target, calls = _sites(monkeypatch, pairs)
    target._hc_site(torch.zeros(rows, 8), None, None, None, None, None, True)
    assert calls == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "pairs, rows, expected, fp4",
    [
        (True, 1, ["gemv", "gemv_mix_fp4"], ("xq", "xsf")),
        (True, 16, ["rows", "rows_mix"], ("xq", "xsf")),
        (False, 16, ["inject_norm", "moe_mixer"], None),
    ],
)
def test_hc_moe_site_forms(monkeypatch, pairs, rows, expected, fp4):
    """The MoE site returns the runner's NVFP4 activation from the pair's own epilogue."""
    target, calls = _sites(monkeypatch, pairs)
    L = SimpleNamespace(
        mhc_norm=None, mhc_first=None, mhc_up=None, moe_gscale=torch.ones(1), mhc_rows=None
    )
    out = target._hc_moe_site(torch.zeros(rows, 8), None, None, L)
    assert calls == expected and out[3] == fp4


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the CuTe split-K decode kernel serves SM10x",
)

#: The tep4 shard: 6 q heads on one KV head of 256, 32-token pages, 2048 + 4 - 1 selected tokens.
_HQ, _HKV, _D, _TPB, _WIDTH = 6, 1, 256, 32, 2051


def _tep4_call(rows: int, seed: int, device):
    """`rows` verify rows, four a request, deep in each request's sequence: an e4m3 pool, the
    trunk's gate (columns of its fused projection row) and a selection of every visible token
    up to the width, front-packed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    requests = rows // 4
    pages_per_request = 96
    pages = requests * pages_per_request + 7
    shape = (pages, _HKV, _TPB, _D)
    k = (torch.randn(shape, generator=gen, device=device) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(shape, generator=gen, device=device) * 0.5).to(torch.float8_e4m3fn)
    perm = torch.randperm(pages, generator=gen, device=device)[: requests * pages_per_request]
    table = perm.view(requests, pages_per_request).to(torch.int32).contiguous()
    req = torch.arange(rows, device=device).div(4, rounding_mode="floor").to(torch.int32)
    last = 3000 - 37 * torch.arange(requests, device=device)
    pos = (last.repeat_interleave(4) - 3 + torch.arange(rows, device=device) % 4).long()
    col = torch.arange(_WIDTH, device=device)[None, :]
    start = (pos + 1 - _WIDTH).clamp_min(0)[:, None]
    sel = torch.where(col + start <= pos[:, None], col + start, -1).to(torch.int32)
    row_heads = _HQ + 2 * _HKV + _HQ + 3
    row = torch.randn(rows, row_heads * _D, generator=gen, device=device).to(torch.bfloat16)
    q = row[:, : _HQ * _D].view(rows, _HQ, _D)
    gate_lo = (_HQ + 2 * _HKV) * _D
    return dict(
        q=q,
        k_cache=k,
        v_cache=v,
        block_table=table,
        selected_tokens=sel,
        request_indices=req,
        tokens_per_block=_TPB,
        softmax_scale=_D**-0.5,
        output_gate=row[:, gate_lo : gate_lo + _HQ * _D],
    )


@_GPU
@pytest.mark.parametrize("rows", [4, 8, 12, 16])
def test_splitk_serves_the_tep4_shard(rows):
    op = M.qsa_sparse_gqa_splitk_cute
    device = torch.device("cuda")
    args = _tep4_call(rows, 1000 + rows, device)
    core = SimpleNamespace(tep_fast=True)
    assert not op.serves(args["q"], args["k_cache"], args["selected_tokens"])
    assert M._splitk_serves(core, args["q"], args["k_cache"], args["selected_tokens"])
    out = op(**args)
    assert out.shape == (rows, _HQ, _D) and not bool(out.isnan().any())
    assert_within_ulp(out, op.reference(**args), element_ulp=4.0, rms_ulp=2.0)

    # Captured once, the operands refilled in place, replayed: the eager call's bits.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = op(**args)
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _tep4_call(rows, 2000 + rows, device)
    for name in ("k_cache", "v_cache", "q", "output_gate"):
        args[name].copy_(fresh[name])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(**args))
    assert_within_ulp(captured, op.reference(**args), element_ulp=4.0, rms_ulp=2.0)


def _splitk_compiled():
    """The split-K op's compiled `(splits, tiles per split)` split passes and merge widths."""
    keys = splitk_mod._COMPILED
    splits = {k[-3:-1] for k in keys if k[0] != "merge"}
    merges = {k[2] for k in keys if k[0] == "merge"}
    return splits, merges


@_GPU
def test_splitk_first_call_compiles_every_plan_up_to_the_callers_reach(monkeypatch):
    """The tep4 shard is not in the op's `MAX_ROWS`: a first call of 4 rows compiles the 4-row
    plans only, unless the caller names its reach (`max_rows`, the target's 16). With it, that
    call compiles the split pass and merge of every plan 1..16 rows reach, and no later call of
    up to 16 rows, gated or not, compiles another."""
    device = torch.device("cuda")
    op = M.qsa_sparse_gqa_splitk_cute
    reach = M._TEP_SPLITK_MAX_ROWS[(_HQ, _HKV, torch.float8_e4m3fn)]
    args = _tep4_call(reach, 5000, device)
    assert (_HQ, _HKV, torch.float8_e4m3fn) not in splitk_mod.MAX_ROWS
    elem = 1  # e4m3
    sliced = ("q", "selected_tokens", "request_indices", "output_gate")

    def call(rows, gated=True, **kwargs):
        a = {n: (v[:rows] if n in sliced else v) for n, v in args.items()}
        if not gated:
            del a["output_gate"]
        return op(**a, **kwargs)

    def plans(rows):
        return set(splitk_mod.reachable_plans(rows, _HKV, _WIDTH, elem))

    assert plans(4) < plans(reach), "the reach adds plans past the first call's rows"
    for named in (None, reach):
        monkeypatch.setattr(splitk_mod, "_COMPILED", {})
        monkeypatch.setattr(splitk_mod, "_PLANNED", set())
        call(4, max_rows=named)
        want = plans(4 if named is None else reach)
        splits, merges = _splitk_compiled()
        assert splits == want, (named, sorted(splits), sorted(want))
        assert merges == {n for n, _ in want if n > 1}, (named, sorted(merges))
    compiled = set(splitk_mod._COMPILED)
    for rows in range(1, reach + 1):
        for gated in (True, False):
            call(rows, gated)
            assert set(splitk_mod._COMPILED) == compiled, f"{rows} rows, gated={gated} compiled"


@_GPU
def test_draft_gate_view_through_the_dispatch():
    """The draft layer's gate is the second half of each head's 512-wide q projection
    (`_mtp_qkv`: `[rows, 6, 256]` at a head stride of 512). Through `_sparse_gqa_decode` at the
    tep4 shard every head is gated by its own half: the result is the reference's with that view,
    and not what the gate read as contiguous heads would give."""
    rows = 4
    op = M.qsa_sparse_gqa_splitk_cute
    device = torch.device("cuda")
    args = _tep4_call(rows, 3000, device)
    gen = torch.Generator(device=device).manual_seed(3001)
    qg = torch.randn(rows, _HQ, 2 * _D, generator=gen, device=device)
    qg[:, :, _D:] += 2.0 * torch.arange(_HQ, device=device)[None, :, None] - 5.0  # per head
    qg = qg.to(torch.bfloat16)
    gate = qg[:, :, _D:]
    assert gate.stride() == (2 * _HQ * _D, 2 * _D, 1)
    core = SimpleNamespace(tep_fast=True, softmax_scale=args["softmax_scale"])
    kv = SimpleNamespace(k=args["k_cache"], v=args["v_cache"], k_scale=None)
    out = M._sparse_gqa_decode(
        core, kv, args["q"], args["block_table"], args["selected_tokens"],
        args["request_indices"], args["tokens_per_block"], gate, 4,
    )  # fmt: skip
    assert out is not None and out.shape == (rows, _HQ, _D)
    assert_within_ulp(
        out, op.reference(**dict(args, output_gate=gate)), element_ulp=4.0, rms_ulp=2.0
    )
    flat = torch.as_strided(gate, (rows, _HQ, _D), (2 * _HQ * _D, _D, 1))
    assert not torch.equal(out, op.reference(**dict(args, output_gate=flat)))
