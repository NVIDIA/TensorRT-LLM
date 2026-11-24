# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CuteDSL MoE entries of the context path: rubin_moe_sort, rubin_moe_output_memset (and
rubin_moe_zero_local_rows, the DEP16 context's zero of the received rows), rubin_cutedsl_fc1
and rubin_cutedsl_fc2_finalize.

Off the GPU: each reference against an independent formulation, the four references composed
into the layer (rubin_fc12's reference), the guards, the construction check of the CuteDSL
variables, and each entry's call against the engine CuteDSL backend's call of the same op.
On SM107: every entry's cells against its reference, and the four composed against the layer.
"""

import inspect
import zlib
from types import SimpleNamespace

import _rubin_nvfp4_moe_inputs as moe_inputs
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_moe_common as common,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc1 import (  # noqa: E501
    rubin_cutedsl_fc1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc2_finalize import (  # noqa: E501
    check_cutedsl_lever_env,
    rubin_cutedsl_fc2_finalize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_fc12 import (
    nvfp4_situ_moe_reference,
    situ,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_output_memset import (  # noqa: E501
    rubin_moe_output_memset,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_sort import (
    rubin_moe_sort,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_zero_local_rows import (  # noqa: E501
    rubin_moe_zero_local_rows,
)
from tensorrt_llm._torch.utils import ActivationType

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401 -- path above

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")
cpu = pytest.mark.cpu_only

SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)
#: As rubin_fc12's cells: BF16 ulps of each row's largest output and of the relative RMS, the
#: catalog fp4_block_scale_moe_runner's band for the same NVFP4 two-GEMM chain.
ELEMENT_ULP, RMS_ULP = 16.0, 2.0


def _route(rows, num_experts, top_k, seed, device="cpu"):
    """``top_k`` distinct experts per row and FP32 weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(rows, num_experts, generator=g, device=device)
    ids = torch.topk(logits, top_k, dim=-1).indices.to(torch.int32)
    weights = torch.rand(rows, top_k, generator=g, device=device)
    return ids, weights / weights.sum(-1, keepdim=True)


# References (CPU)


@cpu
@pytest.mark.parametrize("tile", [128, 256, 512])
@pytest.mark.parametrize("rows, offset, local", [(1, 0, 16), (40, 8, 16), (300, 16, 8)])
def test_sort_reference_is_the_token_ordered_sort(tile, rows, offset, local):
    """Stable in the expanded index is token order: the tables equal the routing tail's sort
    (``token_ordered_sort``), the unwritten entries -1 in both."""
    ids, weights = _route(rows, 32, 4, rows + offset)
    ref = rubin_moe_sort.reference(ids, weights, 32, 4, offset, local, tile)
    want = token_ordered_sort4(ids, offset, local, tile)
    for got, exp in zip(ref, want):
        assert torch.equal(got, exp)


def token_ordered_sort4(ids, offset, local, tile):
    """``token_ordered_sort`` for any top-k (the module's helper is cut for the router's 16)."""
    m, k = ids.shape
    expanded = m * k
    filled = min(local, expanded)
    max_tiles = filled + (expanded - filled) // tile
    flat = ids.reshape(-1).long() - offset
    keep = (flat >= 0) & (flat < local)
    experts, rows = flat[keep], torch.arange(expanded)[keep]
    order = torch.argsort(experts * expanded + rows)
    experts, rows = experts[order], rows[order]
    counts = torch.bincount(experts, minlength=local)
    tiles = (counts + tile - 1) // tile
    base = (torch.cumsum(tiles, 0) - tiles) * tile
    permuted = (
        base[experts] + torch.arange(experts.numel()) - (torch.cumsum(counts, 0) - counts)[experts]
    )
    e2p = torch.full((expanded,), -1, dtype=torch.int32)
    e2p[rows] = permuted.int()
    p2e = torch.full((max_tiles * tile,), -1, dtype=torch.int32)
    p2e[permuted] = rows.int()
    n = int(tiles.sum())
    owner = torch.repeat_interleave(torch.arange(local), tiles)
    nth = torch.arange(n) - (torch.cumsum(tiles, 0) - tiles)[owner]
    t2e = torch.full((max_tiles,), -1, dtype=torch.int32)
    t2l = torch.full((max_tiles,), -1, dtype=torch.int32)
    t2e[:n] = owner.int()
    t2l[:n] = torch.minimum(base[owner] + (nth + 1) * tile, base[owner] + counts[owner]).int()
    return (
        t2e,
        t2l,
        e2p.view(m, k),
        p2e,
        torch.tensor([n * tile], dtype=torch.int32),
        torch.tensor([n], dtype=torch.int32),
    )


@cpu
def test_sort_reference_shapes_follow_the_engine_sizes():
    """``max_tiles`` and the permuted rows are ``moe_sort``'s allocation (the family's
    ``sort_buffer_sizes`` at the router's top 16)."""
    ids, weights = _route(96, 896, 16, 1)
    for tile in (128, 256):
        out = rubin_moe_sort.reference(ids, weights, 896, 16, 112, 112, tile)
        tiles, permuted = common.sort_buffer_sizes(96, 112, tile)
        assert out[0].numel() == tiles and out[3].numel() == permuted


@cpu
def test_memset_reference_full_and_sparse():
    """No all-to-all, or EP no larger than top-k: every row. EP above top-k: only the tokens
    with a local pair."""
    ids, weights = _route(20, 64, 2, 3)
    tables = rubin_moe_sort.reference(ids, weights, 64, 2, 0, 8, 128)
    t2e, t2l, e2p, p2e, _, n = tables
    out = torch.ones(20, 16, dtype=torch.bfloat16)
    for ep, a2a in ((8, False), (2, True), (16, False)):
        assert not rubin_moe_output_memset.reference(out, t2l, e2p, p2e, n, 128, 2, ep, a2a).any()
    got = rubin_moe_output_memset.reference(out, t2l, e2p, p2e, n, 128, 2, 8, True)
    local = ((ids >= 0) & (ids < 8)).any(1)
    assert torch.equal(got.eq(0).all(1), local) and got[~local].eq(1).all()


@cpu
@pytest.mark.parametrize("tile, offset", [(128, 0), (256, 4), (512, 2)])
def test_composed_references_are_the_layer(tile, offset):
    """sort -> memset -> FC1 -> FC2 finalize, each by its reference, is rubin_fc12's reference
    over the window (the token loop of test_rubin_fc12.py stands behind that one)."""
    hidden, inter, local, total, top_k, rows = 64, 32, 4, 10, 3, 9
    experts = moe_inputs.experts(local, hidden, inter, tile + offset, device="cpu")
    ids, weights = _route(rows, total, top_k, tile)
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(rows, hidden, generator=g) * 0.2).to(torch.bfloat16)
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort.reference(
        ids, weights, total, top_k, offset, local, tile
    )
    out = rubin_moe_output_memset.reference(
        torch.full((rows, hidden), 3.0, dtype=torch.bfloat16),
        t2l,
        e2p,
        p2e,
        n,
        tile,
        top_k,
        1,
        False,
    )
    act = rubin_cutedsl_fc1.reference(
        x, experts, t2e, t2l, p2e, n, top_k=top_k, tile_size=tile, **SITU
    )
    got = rubin_cutedsl_fc2_finalize.reference(
        act, experts, t2e, t2l, p2e, n, weights, out, top_k=top_k, tile_size=tile
    )
    want = nvfp4_situ_moe_reference(x, ids, weights, experts, local_expert_offset=offset, **SITU)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@cpu
def test_fc1_reference_rows_are_per_pair():
    """Each written permuted row holds its own (token, expert) pair's activation; rows the sort
    leaves unwritten stay 0."""
    hidden, inter, local, top_k, rows, tile = 32, 16, 3, 2, 5, 128
    experts = moe_inputs.experts(local, hidden, inter, 11, device="cpu")
    ids, weights = _route(rows, local, top_k, 12)
    x = torch.randn(rows, hidden, generator=torch.Generator().manual_seed(13)).to(torch.bfloat16)
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort.reference(ids, weights, local, top_k, 0, local, tile)
    act = rubin_cutedsl_fc1.reference(
        x, experts, t2e, t2l, p2e, n, top_k=top_k, tile_size=tile, **SITU
    )
    written = p2e >= 0
    assert act[~written].eq(0).all()
    g1 = 1.0 / max(float(ex[w][3]) for ex in experts for w in ("w1", "w3"))
    g2 = 1.0 / max(float(ex["w2"][3]) for ex in experts)
    for p in written.nonzero().flatten().tolist():
        t = int(p2e[p]) // top_k
        e = int(t2e[p // tile])
        xq = common.quantize_nvfp4(x[t : t + 1].float(), g1)
        w = {
            n_: common.dequantize_nvfp4(experts[e][n_][0], experts[e][n_][1]) * experts[e][n_][2]
            for n_ in ("w1", "w3")
        }
        want = common.quantize_nvfp4(situ(xq @ w["w1"].t(), xq @ w["w3"].t(), 4.0, 25.0), g2)
        torch.testing.assert_close(act[p : p + 1], want, rtol=1e-6, atol=1e-6)


# The received-rows zero of the DEP16 context (rubin_moe_zero_local_rows)


def _received(counts, slots, total, top_k, offset, local, seed, device="cpu", one_local=False):
    """``(ids, weights)`` of an all-to-all's receive slots, ``[len(counts) * slots, top_k]``.

    Source ``s`` sends ``counts[s]`` tokens, packed at the front of its ``slots``: ``top_k``
    distinct experts each (global ids), at least one of the window ``[offset, offset + local)``
    and exactly one with ``one_local``. Every id of the slots past them is -1, as the dispatch's
    sanitize leaves them, with weight 0.
    """
    g = torch.Generator().manual_seed(seed)
    n, rows = len(counts), sum(counts)
    logits = torch.randn(rows, total, generator=g)
    if one_local:
        logits[:, offset : offset + local] = float("-inf")
    ids = torch.topk(logits, top_k, dim=-1).indices
    need = ~((ids >= offset) & (ids < offset + local)).any(1)
    pick = offset + torch.randint(0, local, (rows,), generator=g)
    col = torch.randint(0, top_k, (rows,), generator=g)
    ids[need, col[need]] = pick[need]
    weights = torch.rand(rows, top_k, generator=g)
    weights = weights / weights.sum(-1, keepdim=True)
    slot_ids = torch.full((n * slots, top_k), -1, dtype=torch.int32)
    slot_w = torch.zeros(n * slots, top_k)
    at = 0
    for s, c in enumerate(counts):
        slot_ids[s * slots : s * slots + c] = ids[at : at + c].to(torch.int32)
        slot_w[s * slots : s * slots + c] = weights[at : at + c]
        at += c
    return slot_ids.to(device), slot_w.to(device)


def _received_rows(counts, slots, device="cpu"):
    """The bool mask of the slots a source filled: the front ``counts[s]`` of each block."""
    pos = torch.arange(len(counts) * slots, device=device)
    return pos % slots < torch.tensor(counts, device=device).repeat_interleave(slots)


@cpu
def test_zero_local_rows_reference_zeroes_the_received_rows():
    """Behind the all-to-all the rows with a pair of the window are the received slots: the
    reference zeroes exactly those (+0 in every byte, as the memset) and keeps the rest, while the
    memset at EP 16 = top 16 zeroes every slot."""
    counts, slots, offset, local = (5, 0, 8, 3, 1, 8, 2, 7), 8, 16, 8
    ids, weights = _received(counts, slots, 64, 16, offset, local, 21)
    out = torch.full((ids.shape[0], 32), -3.0, dtype=torch.bfloat16)
    got = rubin_moe_zero_local_rows.reference(out, ids, offset, local)
    live = _received_rows(counts, slots)
    assert torch.equal(got[live].view(torch.int16), torch.zeros_like(got[live]).view(torch.int16))
    assert torch.equal(got[~live], out[~live])
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort.reference(ids, weights, 64, 16, offset, local, 128)
    full = rubin_moe_output_memset.reference(out, t2l, e2p, p2e, n, 128, 16, 16, True)
    assert not full.any() and torch.equal(full[live], got[live])
    # The rows the sort gives this window's pairs are the received rows.
    assert torch.equal((e2p >= 0).any(1), live)


@cpu
def test_composed_references_with_the_received_rows_zero_are_the_layer():
    """sort -> received-rows zero -> FC1 -> FC2 finalize is rubin_fc12's reference on every
    received row; the slots past each source's count keep what they held (NaN here) and are the
    rows the combine never reads."""
    hidden, inter, local, total, top_k, tile, offset = 64, 32, 4, 16, 3, 128, 4
    counts, slots = (3, 0, 4, 1), 4
    experts = moe_inputs.experts(local, hidden, inter, 31, device="cpu")
    ids, weights = _received(counts, slots, total, top_k, offset, local, 32)
    g = torch.Generator().manual_seed(33)
    x = (torch.randn(ids.shape[0], hidden, generator=g) * 0.2).to(torch.bfloat16)
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort.reference(
        ids, weights, total, top_k, offset, local, tile
    )
    out = rubin_moe_zero_local_rows.reference(
        torch.full((ids.shape[0], hidden), float("nan"), dtype=torch.bfloat16), ids, offset, local
    )
    act = rubin_cutedsl_fc1.reference(
        x, experts, t2e, t2l, p2e, n, top_k=top_k, tile_size=tile, **SITU
    )
    got = rubin_cutedsl_fc2_finalize.reference(
        act, experts, t2e, t2l, p2e, n, weights, out, top_k=top_k, tile_size=tile
    )
    want = nvfp4_situ_moe_reference(x, ids, weights, experts, local_expert_offset=offset, **SITU)
    live = _received_rows(counts, slots)
    torch.testing.assert_close(got[live], want[live], rtol=1e-6, atol=1e-6)
    assert got[~live].isnan().all()


@cpu
def test_zero_local_rows_logs_its_line_after_each_launch(monkeypatch):
    """Each call launches, then hands ``LOG_LINE`` to ``logger.info_once`` under one key, so a
    process logs the line once: the witness a DEP16 context rank's log carries that its MoE
    zeroes only the received rows."""
    module = inspect.getmodule(type(rubin_moe_zero_local_rows))
    out = torch.zeros(4, 16, dtype=torch.bfloat16)
    ids = torch.full((4, 16), -1, dtype=torch.int32)
    events = []
    monkeypatch.setattr(
        module,
        "zero_local_rows",
        lambda o, i, lo, n: events.append(("launch", o is out, i is ids, lo, n)),
    )
    monkeypatch.setattr(
        module,
        "logger",
        SimpleNamespace(info_once=lambda *msg, key: events.append(("log", msg, key))),
    )
    for _ in range(2):
        rubin_moe_zero_local_rows(out, ids, 728, 56)
    launch, line = ("launch", True, True, 728, 56), ("log", (module.LOG_LINE,), module.LOG_KEY)
    assert events == [launch, line, launch, line]
    assert module.LOG_LINE.startswith("Kimi K3.1 Rubin MoE zero: received rows only")


@cpu
def test_zero_local_rows_guards():
    """The guard refuses the windows that zero the wrong rows; the launcher refuses, before any
    launch, the layouts it would read wrongly, and launches nothing for no rows."""
    ids, _ = _received((2, 1), 2, 32, 4, 8, 8, 41)
    out = torch.zeros(ids.shape[0], 16, dtype=torch.bfloat16)
    rubin_moe_zero_local_rows.is_valid(out, ids, 8, 8)
    for args, match in (((8, 0), "empty window"), ((-1, 8), "below 0")):
        with pytest.raises(AssertionError, match=match):
            rubin_moe_zero_local_rows.is_valid(out, ids, *args)
    for o, i in (
        (torch.zeros(16, ids.shape[0], dtype=torch.bfloat16).t(), ids),
        (out, ids.long()),
        (out, ids[:-1]),
        (out, ids.t().contiguous().t()),
    ):
        with pytest.raises(ValueError, match="zero_local_rows"):
            rubin_moe_zero_local_rows(o, i, 8, 8)
    empty = torch.zeros(0, 16, dtype=torch.bfloat16)
    assert rubin_moe_zero_local_rows(empty, ids[:0], 8, 8) is None


_INTERPRETED_ZERO = """
import importlib.util
import sys
import torch
spec = importlib.util.spec_from_file_location("_c8_zero_rows_interpreted", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
{received}
for counts, slots, top_k, offset, local, hidden, dtype in (
    ((5, 0, 3, 8), 8, 16, 8, 4, 3584, torch.bfloat16),
    ((2, 7, 1), 7, 4, 0, 3, 40, torch.bfloat16),
    ((1,) * 16, 1, 16, 840, 56, 3584, torch.bfloat16),
    ((3, 3), 3, 16, 56, 56, 72, torch.float32),
):
    ids, _ = _received(counts, slots, 896 if top_k == 16 else 32, top_k, offset, local, hidden)
    g = torch.Generator().manual_seed(7)
    width = hidden * (2 if dtype == torch.float32 else 1)
    bits = torch.randint(-32768, 32767, (ids.shape[0], width), generator=g).to(torch.int16)
    out = bits.view(dtype).clone()
    want = out.clone()
    live = ((ids >= offset) & (ids < offset + local)).any(1)
    want[live] = 0
    module.zero_local_rows(out, ids, offset, local)
    assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (counts, hidden, dtype)
    print("ok", counts, hidden, dtype)
print("ok all")
"""


@cpu
def test_zero_local_rows_kernel_interpreted():
    """The Triton kernel in Triton's interpreter, in a fresh process (TRITON_INTERPRET must be set
    before triton is first imported): the received rows +0 in every byte and every other row's
    bits untouched (random bits, NaN patterns among them), at K3.1's row and top-16, at a row
    length that is not a whole number of store tiles, and a row count that is not one of
    programs."""
    import inspect
    import os
    import subprocess
    import sys

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_moe_zero_rows,
    )

    code = _INTERPRETED_ZERO.format(received=inspect.getsource(_received))
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-c", code, rubin_moe_zero_rows.__file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0 and "ok all" in proc.stdout, proc.stdout + proc.stderr


# The calls are the engine backend's (CPU, the ops recorded)


class _Recorder:
    def __init__(self):
        self.calls = []

    def op(self, name, result=None):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return result

        return call


@cpu
def test_calls_are_the_engine_cutedsl_backends(monkeypatch):
    """With the four ops recorded, the four entries issue the calls the engine CuteDSL backend's
    ``run_moe_nvfp4_impl`` issues for a SiTU layer on SM107 (fused finalize, no aux stream), keyword
    for keyword and in the same order."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.grouped_gemm import nvfp4 as engine_mod
    from tensorrt_llm._torch.moe.fused_moe.cutedsl.kernel_inputs import NvFp4WeightView

    rows, local, hidden, inter, top_k, tile = 6, 4, 64, 32, 2, 128
    ids, weights = _route(rows, 8, top_k, 5)
    tables = rubin_moe_sort.reference(ids, weights, 8, top_k, 4, local, tile)
    fc1_out = (torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8))
    rec = _Recorder()
    ops = {
        "moe_sort": rec.op("moe_sort", tables),
        "moe_output_memset_inplace": rec.op("memset"),
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_rubin": rec.op("fc1", fc1_out),
        "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_rubin": rec.op("fc2"),
    }
    fake_trtllm = type("trtllm", (), {k: staticmethod(v) for k, v in ops.items()})
    monkeypatch.setattr(torch.ops, "trtllm", fake_trtllm, raising=False)
    monkeypatch.setattr(engine_mod, "get_sm_version", lambda: 107)
    monkeypatch.setattr(engine_mod, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)

    def u8(*s):
        return torch.zeros(*s, dtype=torch.uint8)

    x, x_sf = u8(rows, hidden // 2), u8(rows, hidden // 16)
    view = NvFp4WeightView(
        w3_w1_weight=u8(local, 2 * inter, hidden // 2),
        fc1_weight_scale=u8(local, 2 * inter, hidden // 16),
        fc1_global_scale=torch.ones(local),
        w2_weight=u8(local, hidden, inter // 2),
        fc2_weight_scale=u8(local, hidden, inter // 16),
        fc2_global_scale=torch.ones(local),
        expert_size_per_partition=local,
        slot_start=4,
    )
    out = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    fc2_in = torch.ones(1)
    backend = object.__new__(engine_mod.TrtllmCutedslGroupedGemmNvfp4Impl)
    backend.__dict__.update(
        num_slots=8,
        use_fused_finalize=True,
        activation_type=ActivationType.SiTu,
        act_alpha=SITU["situ_beta"],
        act_beta=SITU["situ_linear_beta"],
        act_clamp=float("inf"),
        fc2_input_scale=fc2_in,
        mapping=type("M", (), {"moe_ep_size": 2})(),
        _pending_x_ready_event=None,
    )
    monkeypatch.setattr(backend, "_has_moe_output_memset_aux_stream", lambda: False, raising=False)
    backend.run_moe_nvfp4_impl(
        x, ids, weights, x_sf, out, view, enable_alltoall=True, tile_size=tile
    )
    engine_calls, rec.calls = rec.calls, []

    sort = rubin_moe_sort(ids, weights, 8, top_k, 4, local, tile)
    t2e, t2l, e2p, p2e, _, n = sort
    rubin_moe_output_memset(out, t2l, e2p, p2e, n, tile, top_k, 2, True)
    h, h_sf = rubin_cutedsl_fc1(
        x,
        view.w3_w1_weight,
        x_sf,
        view.fc1_weight_scale,
        view.fc1_global_scale,
        t2e,
        t2l,
        p2e,
        n,
        fc2_in,
        8,
        top_k,
        local,
        4,
        tile,
        int(ActivationType.SiTu),
        **SITU,
    )
    rubin_cutedsl_fc2_finalize(
        h,
        view.w2_weight,
        h_sf,
        view.fc2_weight_scale,
        view.fc2_global_scale,
        out,
        t2e,
        t2l,
        p2e,
        n,
        weights,
        8,
        top_k,
        local,
        4,
        tile,
    )
    ours = rec.calls
    # The engine runs the memset after FC1 without an aux stream; with one (the targets' memset
    # stream) it runs beside FC1. The entries' calls are the same set; the targets order them.
    key = {"moe_sort": 0, "memset": 1, "fc1": 2, "fc2": 3}
    assert sorted(c[0] for c in engine_calls) == sorted(c[0] for c in ours)
    for (name, a1, k1), (name2, a2, k2) in zip(
        sorted(engine_calls, key=lambda c: key[c[0]]), sorted(ours, key=lambda c: key[c[0]])
    ):
        assert name == name2 and a1 == a2 == (), name
        # An argument the engine leaves out takes the op's schema default, which the entry
        # passes explicitly (the FC1 epilogue's -1.0 for an absent cap or clamp, and no pinned
        # tactic).
        defaults = {
            "swiglu_limit_scalar": -1.0,
            "situ_beta": -1.0,
            "situ_linear_beta": -1.0,
            "precomputed_tactic": None,
        }
        assert set(k1) <= set(k2) and set(k2) - set(k1) <= set(defaults), (name, set(k1) ^ set(k2))
        for k in k2:
            v1, v2 = k1.get(k, defaults.get(k)), k2[k]
            if isinstance(v1, torch.Tensor):
                assert (v1.data_ptr(), v1.dtype, v1.shape) == (v2.data_ptr(), v2.dtype, v2.shape), (
                    name,
                    k,
                )
            else:
                assert v1 == v2, (name, k, v1, v2)


# Guards and construction (CPU)


@cpu
def test_guards_admit_a_valid_call_and_refuse_what_answers_wrongly():
    ids, weights = _route(8, 16, 2, 9)
    sort_args = (ids, weights, 16, 2, 0, 8, 128)
    rubin_moe_sort.is_valid(*sort_args)
    for bad, match in (
        ((ids.t().contiguous().t(),) + sort_args[1:], "contiguous"),
        ((ids.long(),) + sort_args[1:], "int32"),
        (sort_args[:6] + (0,), "positive"),
    ):
        with pytest.raises(AssertionError, match=match):
            rubin_moe_sort.is_valid(*bad)
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort.reference(*sort_args)
    out = torch.zeros(8, 64, dtype=torch.bfloat16)
    rubin_moe_output_memset.is_valid(out, t2l, e2p, p2e, n, 128, 2, 8, False)
    with pytest.raises(AssertionError, match="dense"):
        rubin_moe_output_memset.is_valid(
            torch.zeros(64, 8, dtype=torch.bfloat16).t(), t2l, e2p, p2e, n, 128, 2, 8, False
        )
    u8 = torch.uint8
    fc1 = dict(
        input=torch.zeros(8, 32, dtype=u8),
        weight=torch.zeros(8, 64, 32, dtype=u8),
        input_scale=torch.zeros(8, 4, dtype=u8),
        weight_scale=torch.zeros(8, 64, 4, dtype=u8),
        alpha=torch.ones(8),
        tile_idx_to_group_idx=t2e,
        tile_idx_to_mn_limit=t2l,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=n,
        global_sf=torch.ones(1),
        num_experts=16,
        top_k=2,
        num_local_experts=8,
        local_expert_offset=0,
        tile_size=128,
        activation_type=int(ActivationType.SiTu),
        **SITU,
    )
    rubin_cutedsl_fc1.is_valid(**fc1)
    for bad, match in (
        (dict(tile_size=256), "not cut for tile 256"),
        (dict(precomputed_tactic=common.FC1_TILE256_TACTIC), "cut for tile 256, not 128"),
        (dict(alpha=torch.ones(7)), "one alpha"),
        (dict(input_scale=torch.zeros(8, 2, dtype=u8)), "linear e4m3"),
    ):
        with pytest.raises(AssertionError, match=match):
            rubin_cutedsl_fc1.is_valid(**{**fc1, **bad})
    fc2 = dict(
        input=torch.zeros(p2e.numel(), 16, dtype=u8),
        weight=torch.zeros(8, 64, 16, dtype=u8),
        input_scale=torch.zeros(p2e.numel() * 2, dtype=u8),
        weight_scale=torch.zeros(8, 64, 2, dtype=u8),
        alpha=torch.ones(8),
        output=out,
        tile_idx_to_group_idx=t2e,
        tile_idx_to_mn_limit=t2l,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=n,
        token_final_scales=weights,
        num_experts=16,
        top_k=2,
        num_local_experts=8,
        local_expert_offset=0,
        tile_size=128,
    )
    rubin_cutedsl_fc2_finalize.is_valid(**fc2)
    for bad, match in (
        (dict(output=out.float()), "BF16"),
        (dict(token_final_scales=weights.to(torch.bfloat16)), "FP32"),
        (dict(top_k=3), "routing of"),
        (dict(tile_size=512), "not cut for tile 512"),
        (dict(precomputed_tactic=common.FC2_TILE256_TACTIC), "cut for tile 256, not 128"),
    ):
        with pytest.raises(AssertionError, match=match):
            rubin_cutedsl_fc2_finalize.is_valid(**{**fc2, **bad})


@cpu
def test_cutedsl_lever_env():
    """The FC2 N-tile pin unset, the input quantization in order (unset or 0)."""
    check_cutedsl_lever_env({})
    check_cutedsl_lever_env({"TLLM_MOE_ASYNC_INPUT_QUANT": "0"})
    for env in (
        {"TRTLLM_CUTEDSL_FC2_N_TILE_SIZE": "128"},
        {"TRTLLM_CUTEDSL_FC2_N_TILE_SIZE": "0"},
        {"TLLM_MOE_ASYNC_INPUT_QUANT": "1"},
    ):
        with pytest.raises(ValueError, match=next(iter(env))):
            check_cutedsl_lever_env(env)


@cpu
def test_pinned_tile256_tactics_pass_the_runners_filter():
    """The tactics a context server pins at tile 256 are the runners' own kind at K3.1's widths
    (FC1 N = 2 x 3072 gate / up rows, FC2 N = hidden 3584): one 256-row routing tile per 2-CTA
    cluster, whole N tiles."""
    for tactic, n in ((common.FC1_TILE256_TACTIC, 2 * 3072), (common.FC2_TILE256_TACTIC, 3584)):
        common.check_cutedsl_tactic(tactic, 256, n)
        with pytest.raises(AssertionError, match="cut for tile 256, not 128"):
            common.check_cutedsl_tactic(tactic, 128, n)
    common.check_cutedsl_tactic(None, 128, 64)


@cpu
@pytest.mark.parametrize(
    "tactic, tile, n, match",
    [
        ("((256, 256, 256), (256, 256, 128), (1, 1), False)", 256, 3584, "CTA group"),
        ("((128, 256, 256), (128, 256, 128), (2, 1), False)", 128, 3584, "CTA group"),
        ("((128, 256, 256), (128, 128, 128), (1, 1), False)", 128, 3584, "instruction N"),
        ("((256, 256, 256), (256, 256, 128), (2, 2), False)", 256, 3840, "partial N tile"),
        ("((128, 256, 256), (128, 256, 128), (1, 1), False)", 128, 3000, "partial N tile"),
        ("((128, 128, 256), (128, 128, 128), (1, 2), False, 'tma')", 128, 6144, "multicast"),
    ],
)
def test_check_cutedsl_tactic_refuses_what_the_runners_filter_out(tactic, tile, n, match):
    with pytest.raises(AssertionError, match=match):
        common.check_cutedsl_tactic(tactic, tile, n)


@cpu
def test_signatures_carry_the_engine_ops_defaults():
    """The FC1 epilogue arguments default as the engine op's (-1 disables a cap or the clamp)."""
    params = inspect.signature(type(rubin_cutedsl_fc1).__call__).parameters
    for name in ("situ_beta", "situ_linear_beta", "swiglu_limit_scalar"):
        assert params[name].default == -1.0


# Cells (SM107)


def _ulp_check(out, ref):
    assert_within_ulp(out, ref.to(out.dtype), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _run_cell(spec, seed):
    """The four entries in the targets' order over one rank's window of the cell; returns
    ``(output, own, layer, sort tables, sort reference)``. FC1 and FC2 run the cell's pinned
    tactics (``fc1_tactic`` / ``fc2_tactic``), or the AutoTuner's picks when it names none.

    ``own`` is FC1's reference finalized by FC2's, over the inputs of the two calls: the sort
    tables the kernels ran on and the memset rows FC2 adds into. ``layer`` is rubin_fc12's
    reference of the window, which test_composed_references_are_the_layer ties to ``own``."""
    rows, local, inter, tile, top_k, hidden = (
        spec["tokens"],
        spec["local"],
        spec["inter"],
        spec["tile_size"],
        spec["top_k"],
        spec["hidden"],
    )
    offset = spec.get("local_expert_offset", 0)
    total = max(896, offset + local)
    experts = moe_inputs.experts(local, hidden, inter, seed)
    layer = moe_inputs.layer(experts, hidden)
    ids, weights = _route(rows, total, top_k, seed, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(x, layer["g1"], 16, False, False)
    sf = sf.view(rows, hidden // 16)
    sort = rubin_moe_sort(ids, weights, total, top_k, offset, local, tile)
    t2e, t2l, e2p, p2e, _, n = sort
    out = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    rubin_moe_output_memset(out, t2l, e2p, p2e, n, tile, top_k, 8, False)
    zeroed = out.clone()  # FC2 finalizes into ``out`` in place; its reference starts here too
    h, h_sf = rubin_cutedsl_fc1(
        x4,
        layer["fc1_w"],
        sf,
        layer["fc1_s"],
        layer["fc1_alpha"],
        t2e,
        t2l,
        p2e,
        n,
        layer["g2"],
        total,
        top_k,
        local,
        offset,
        tile,
        int(ActivationType.SiTu),
        **SITU,
        precomputed_tactic=spec.get("fc1_tactic"),
    )
    rubin_cutedsl_fc2_finalize(
        h,
        layer["fc2_w"],
        h_sf,
        layer["fc2_s"],
        layer["fc2_alpha"],
        out,
        t2e,
        t2l,
        p2e,
        n,
        weights,
        total,
        top_k,
        local,
        offset,
        tile,
        precomputed_tactic=spec.get("fc2_tactic"),
    )
    torch.cuda.synchronize()
    # The entries' references take the BF16 input and the experts as stored (their documented
    # exception); FC1's rows are what FC2's reference reads, as (h, h_sf) are what FC2 reads.
    act = rubin_cutedsl_fc1.reference(
        x, experts, t2e, t2l, p2e, n, top_k=top_k, tile_size=tile, **SITU
    )
    own = rubin_cutedsl_fc2_finalize.reference(
        act, experts, t2e, t2l, p2e, n, weights, zeroed, top_k=top_k, tile_size=tile
    )
    ref = nvfp4_situ_moe_reference(x, ids, weights, experts, local_expert_offset=offset, **SITU)
    sort_ref = rubin_moe_sort.reference(ids, weights, total, top_k, offset, local, tile)
    return out, own, ref, sort, sort_ref


def _same_sort(got, want, tile):
    """Tile arrays and counts equal; within each expert's run the same set of pairs."""
    t2e, t2l, e2p, p2e, padded, n = got
    k = int(n[0])
    assert int(n[0]) == int(want[5][0]) and int(padded[0]) == int(want[4][0])
    assert torch.equal(t2e[:k].cpu(), want[0][:k].cpu()) and torch.equal(
        t2l[:k].cpu(), want[1][:k].cpu()
    )
    live = torch.arange(k * tile, device=p2e.device)
    live = live[live < t2l.long()[live // tile]]
    for tile_i in range(k):
        mine = live[live // tile == tile_i]
        assert sorted(p2e[mine].tolist()) == sorted(want[3].to(p2e.device)[mine].tolist())
    assert torch.equal((e2p >= 0).cpu(), (want[2] >= 0).cpu())


@gpu
@pytest.mark.parametrize(
    "cell", rubin_moe_sort.CELLS, ids=[c.why[:40] for c in rubin_moe_sort.CELLS]
)
def test_sort_cells(cell, certifying):  # noqa: F811
    certifying(rubin_moe_sort)
    spec = cell.spec
    ids, weights = _route(
        spec["tokens"], spec["num_experts"], spec["top_k"], zlib.crc32(cell.why.encode()), "cuda"
    )
    args = (
        ids,
        weights,
        spec["num_experts"],
        spec["top_k"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        spec["tile_tokens_dim"],
    )
    _same_sort(rubin_moe_sort(*args), rubin_moe_sort.reference(*args), spec["tile_tokens_dim"])


@gpu
@pytest.mark.parametrize(
    "cell", rubin_moe_output_memset.CELLS, ids=[c.why[:40] for c in rubin_moe_output_memset.CELLS]
)
def test_memset_cells(cell, certifying):  # noqa: F811
    certifying(rubin_moe_output_memset)
    spec = cell.spec
    ids, weights = _route(spec["tokens"], 896, spec["top_k"], zlib.crc32(cell.why.encode()), "cuda")
    t2e, t2l, e2p, p2e, _, n = rubin_moe_sort(
        ids, weights, 896, spec["top_k"], 0, spec["local_num_experts"], spec["tile_tokens_dim"]
    )
    out = torch.ones(spec["tokens"], spec["hidden"], dtype=torch.bfloat16, device="cuda")
    args = (
        t2l,
        e2p,
        p2e,
        n,
        spec["tile_tokens_dim"],
        spec["top_k"],
        spec["ep_size"],
        spec["enable_alltoall"],
    )
    ref = rubin_moe_output_memset.reference(out, *args)
    rubin_moe_output_memset(out, *args)
    assert torch.equal(out, ref)


@gpu
@pytest.mark.parametrize(
    "cell", rubin_cutedsl_fc1.CELLS, ids=[c.why[:40] for c in rubin_cutedsl_fc1.CELLS]
)
def test_fc1_cells(cell, certifying):  # noqa: F811
    """FC1 is checked through FC2: its output only exists in the swizzled NVFP4 form
    rubin_cutedsl_fc2_finalize reads, so the chain is held to FC1's reference finalized by
    FC2's, and to rubin_fc12's reference of the layer."""
    certifying(rubin_cutedsl_fc1)
    out, own, ref, _, _ = _run_cell(cell.spec, zlib.crc32(cell.why.encode()))
    _ulp_check(out, own)
    _ulp_check(out, ref)


@gpu
@pytest.mark.parametrize(
    "cell",
    rubin_cutedsl_fc2_finalize.CELLS,
    ids=[c.why[:40] for c in rubin_cutedsl_fc2_finalize.CELLS],
)
def test_fc2_finalize_cells(cell, certifying):  # noqa: F811
    certifying(rubin_cutedsl_fc2_finalize)
    out, own, ref, sort, sort_ref = _run_cell(cell.spec, zlib.crc32(cell.why.encode()) + 1)
    _same_sort(sort, sort_ref, cell.spec["tile_size"])
    _ulp_check(out, own)
    _ulp_check(out, ref)


@gpu
@pytest.mark.parametrize(
    "cell",
    rubin_moe_zero_local_rows.CELLS,
    ids=[c.why[:40] for c in rubin_moe_zero_local_rows.CELLS],
)
def test_zero_local_rows_cells(cell, certifying):  # noqa: F811
    """Bit for bit: the received rows +0, every other row's random bits kept."""
    certifying(rubin_moe_zero_local_rows)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode())
    counts = spec["counts"]
    ids, _ = _received(
        counts,
        spec["slots"],
        spec["num_experts"],
        spec["top_k"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
        seed,
        "cuda",
    )
    g = torch.Generator(device="cuda").manual_seed(seed)
    bits = torch.randint(
        -32768, 32767, (ids.shape[0], spec["hidden"]), generator=g, device="cuda"
    ).to(torch.int16)
    out = bits.view(torch.bfloat16).clone()
    args = (ids, spec["local_expert_offset"], spec["local_num_experts"])
    ref = rubin_moe_zero_local_rows.reference(out, *args)
    rubin_moe_zero_local_rows(out, *args)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    live = _received_rows(counts, spec["slots"], "cuda")
    assert not out[live].view(torch.int16).any()


@gpu
def test_received_rows_zero_matches_the_memset_through_the_layer():
    """The DEP16 context's experts with the received-rows zero against the memset: sort, zero,
    FC1 and FC2 finalize give the same bits on every received row. One pair of the window per
    token, so the finalize's adds into a row cannot change order between the two runs. Uneven
    sources, one of them empty: the slots past each count keep their NaN under the received-rows
    zero and are zero under the memset."""
    counts = (64, 0, 37, 5, 64, 18, 50, 1, 29, 64, 44, 12, 60, 7, 33, 25)
    slots, local, offset, total, top_k, tile = 64, 56, 56, 896, 16, 128
    hidden, inter = 3584, 3072
    experts = moe_inputs.experts(local, hidden, inter, 51)
    layer = moe_inputs.layer(experts, hidden)
    ids, weights = _received(counts, slots, total, top_k, offset, local, 52, "cuda", one_local=True)
    rows = ids.shape[0]
    g = torch.Generator(device="cuda").manual_seed(53)
    x = (torch.randn(rows, hidden, generator=g, device="cuda") * 0.15).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(x, layer["g1"], 16, False, False)
    sf = sf.view(rows, hidden // 16)
    outs = []
    for zero in ("memset", "received"):
        sort = rubin_moe_sort(ids, weights, total, top_k, offset, local, tile)
        t2e, t2l, e2p, p2e, _, n = sort
        out = torch.full((rows, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
        if zero == "memset":
            rubin_moe_output_memset(out, t2l, e2p, p2e, n, tile, top_k, 16, True)
        else:
            rubin_moe_zero_local_rows(out, ids, offset, local)
        h, h_sf = rubin_cutedsl_fc1(
            x4,
            layer["fc1_w"],
            sf,
            layer["fc1_s"],
            layer["fc1_alpha"],
            t2e,
            t2l,
            p2e,
            n,
            layer["g2"],
            total,
            top_k,
            local,
            offset,
            tile,
            int(ActivationType.SiTu),
            **SITU,
        )
        rubin_cutedsl_fc2_finalize(
            h,
            layer["fc2_w"],
            h_sf,
            layer["fc2_s"],
            layer["fc2_alpha"],
            out,
            t2e,
            t2l,
            p2e,
            n,
            weights,
            total,
            top_k,
            local,
            offset,
            tile,
        )
        outs.append(out)
    torch.cuda.synchronize()
    live = _received_rows(counts, slots, "cuda")
    assert (~live).any(), "the cell must have slots past the counts"
    assert torch.equal(outs[0][live].view(torch.int16), outs[1][live].view(torch.int16))
    assert outs[1][~live].isnan().all() and not outs[0][~live].any()
