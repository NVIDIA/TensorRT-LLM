# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_qkv_a_rmsnorm_q8 entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_qkv_a_rmsnorm as plain,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_qkv_a_rmsnorm_q8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

op = entry.cute_dsl_qkv_a_rmsnorm_q8
_Q, _KV, _ROW = 2048, 512, 2624
_EPS = 1e-5


def _build(rows: int, seed: int):
    """A `[rows, 2624]` projection output and its q / kv views, as the target slices it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    parent = torch.randn(rows, _ROW, generator=g, device="cuda").to(torch.bfloat16)
    q_w = (1.0 + 0.25 * torch.randn(_Q, generator=g, device="cuda")).to(torch.bfloat16)
    kv_w = (1.0 + 0.25 * torch.randn(_KV, generator=g, device="cuda")).to(torch.bfloat16)
    q_raw, ckv_raw, _ = torch.split(parent, [_Q, _KV, _ROW - _Q - _KV], -1)
    return parent, q_raw, q_w, ckv_raw, kv_w


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device(), _EPS)


@pytest.fixture(scope="module")
def compiled_plain():
    return plain.compile(torch.cuda.current_device(), _EPS)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, compiled_plain, certifying) -> None:  # noqa: F811
    certifying(op)
    rows, r8 = cell.spec["rows"], cell.spec["r8"]
    store_from = cell.spec.get("q_store_from", 0)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    parent, q_raw, q_w, ckv_raw, kv_w = _build(rows, seed)
    twin = parent.clone()
    q8, sf = entry.operand(r8, q_raw.device)
    ref_q, ref_kv, _, _ = op.reference(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, compiled)

    with validating(op):
        q_a = op(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, compiled, q_store_from=store_from)
    torch.cuda.synchronize()

    assert q_a.is_contiguous() and q_a.shape == (rows, _Q)
    assert_within_ulp(q_a[store_from:], ref_q[store_from:], element_ulp=2.0, rms_ulp=0.5)
    assert_within_ulp(parent[:, _Q : _Q + _KV], ref_kv, element_ulp=2.0, rms_ulp=0.5)
    # The bf16 rows are the plain entry's bytes, the kv half too.
    tq, tkv, _ = torch.split(twin, [_Q, _KV, _ROW - _Q - _KV], -1)
    q_plain, _ = plain.cute_dsl_qkv_a_rmsnorm(tq, q_w, tkv, kv_w, _EPS, compiled_plain)
    torch.cuda.synchronize()
    assert torch.equal(q_a[store_from:], q_plain[store_from:]), "bf16 q rows differ"
    assert torch.equal(parent[:, _Q : _Q + _KV], twin[:, _Q : _Q + _KV]), "kv rows differ"
    # The MXF8 operand is the activation quant's bytes of those bf16 rows.
    want_q8, want_sf = entry._quant.reference(q_plain[:r8].contiguous())
    assert torch.equal(q8, want_q8), "q8 differs from the activation quant of q_a"
    # Scales of the real rows (the padded rows' entries are not part of the contract).
    got = entry._quant.dequantize(q8, sf, r8, _Q)
    want = entry._quant.dequantize(want_q8, want_sf, r8, _Q)
    assert torch.equal(got, want), "block scales differ from the activation quant's"


def test_is_valid_refuses_a_partial_decode_operand(compiled) -> None:
    """The decode kernel quantizes every row: an operand shorter than the rows is a misuse."""
    _, q_raw, q_w, ckv_raw, kv_w = _build(8, seed=3)
    q8, sf = entry.operand(4, q_raw.device)
    with pytest.raises(AssertionError, match="every row"):
        with validating(op):
            op(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, compiled)


#: Context row counts of the ILP build: a first warp-row count past 512, mid-size and served
#: steps, the token budget and twice it.
_ILP_ROWS = (513, 4128, 7340, 8192, 16448)


@pytest.fixture(scope="module")
def compiled_plain_ctx():
    """The q_store_from variant without the ILP build: the one it replaces."""
    return entry.compile(torch.cuda.current_device(), _EPS, ilp=False)


def _ctx_call(handle, rows: int, seed: int):
    parent, q_raw, q_w, ckv_raw, kv_w = _build(rows, seed)
    q8, sf = entry.operand(rows - 16, q_raw.device)
    store_from = rows - 16 - 64
    q_a = op(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, handle, q_store_from=store_from)
    return q_a[store_from:], parent[:, _Q : _Q + _KV], q8, sf


@pytest.mark.parametrize("rows", _ILP_ROWS)
def test_ilp_build_is_the_plain_build_bitwise(rows, compiled, compiled_plain_ctx) -> None:
    assert compiled.ilp and not compiled_plain_ctx.ilp
    got = _ctx_call(compiled, rows, seed=rows)
    want = _ctx_call(compiled_plain_ctx, rows, seed=rows)
    torch.cuda.synchronize()
    for g, w, name in zip(got, want, ("q rows", "kv rows", "q8", "scales")):
        assert torch.equal(g, w), f"{name}: the ILP build differs at {rows} rows"


def test_ilp_build_graph_replay_is_eager(compiled) -> None:
    rows = 7340
    eager = _ctx_call(compiled, rows, seed=7)
    parent, q_raw, q_w, ckv_raw, kv_w = _build(rows, seed=7)
    pristine = parent.clone()
    q8, sf = entry.operand(rows - 16, q_raw.device)
    store_from = rows - 16 - 64
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, compiled, q_store_from=store_from)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        q_a = op(q_raw, q_w, ckv_raw, kv_w, _EPS, q8, sf, compiled, q_store_from=store_from)
    parent.copy_(pristine)  # the kv half is normalized in place: restore it, then replay
    graph.replay()
    torch.cuda.synchronize()
    got = (q_a[store_from:], parent[:, _Q : _Q + _KV], q8, sf)
    for g, w, name in zip(got, eager, ("q rows", "kv rows", "q8", "scales")):
        assert torch.equal(g, w), f"{name}: graph replay differs from eager"
