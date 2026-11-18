# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_mha_prefill entry.

Each cell is a packed context step: per request its query rows and its K/V rows (a
cached prefix first). The output buffer starts as NaN, so a row the kernel never
writes fails; a second launch must give the same bytes.

The `kv_strided` cells are the served handles: K and V are the two halves of one
`[T_k, 8192]` buffer (the one-launch K / V expansion's rows). Beyond the reference, each
must write the packed handle's bytes for the same K / V made contiguous.
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_prefill as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_mha_prefill
HEADS, DIM = entry.NUM_HEADS, entry.HEAD_DIM


def _cu(lengths) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32).cuda()


def _pow2(amax: torch.Tensor) -> torch.Tensor:
    """The smallest power of two `s` with `amax / s <= 448`."""
    return torch.exp2(torch.ceil(torch.log2((amax.float() / 448.0).clamp_min(2.0**-126))))


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    new, cached = spec["new"], spec["cached"]
    kv = [c + n for c, n in zip(cached, new)]
    t, tk = sum(new), sum(kv)
    q = torch.randn(t, HEADS, DIM, generator=g, device="cuda")
    k = torch.randn(tk, HEADS, DIM, generator=g, device="cuda")
    v = torch.randn(tk, HEADS, DIM, generator=g, device="cuda")
    head_scales = None
    if spec.get("fp8"):
        # As the target quantizes them: power-of-two scales, per tensor for q, per head
        # for k and v, so every value fits e4m3. A sharper q than the bf16 cells' makes
        # the softmax peaked enough for the e4m3 P to matter.
        q, k = q * spec.get("q_scale", 3.0), k * 0.7
        s_q = _pow2(q.abs().amax())
        s_k, s_v = _pow2(k.abs().amax(dim=(0, 2))), _pow2(v.abs().amax(dim=(0, 2)))
        q = (q / s_q).to(torch.float8_e4m3fn)
        k = (k / s_k.view(1, -1, 1)).to(torch.float8_e4m3fn)
        v = (v / s_v.view(1, -1, 1)).to(torch.float8_e4m3fn)
        head_scales = entry.fp8_head_scales(float(s_q), s_k, s_v).cuda()
    else:
        q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    if spec.get("kv_strided"):
        k, v = _halves(k, v)
    return dict(
        q=q,
        k=k,
        v=v,
        out=torch.full((t, HEADS, DIM), float("nan"), dtype=torch.bfloat16, device="cuda"),
        cu_seqlens=_cu(new),
        max_seqlen=max(new),
        cu_seqlens_k=_cu(kv) if any(cached) else None,
        head_scales=head_scales,
    )


def _halves(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K and V as the two halves of one `[T_k, 2 * 16 * 256]` buffer, as the one-launch
    expansion writes them: token-row stride 8192 elements, V at element 4096 of each row."""
    tk = k.shape[0]
    kv = torch.empty(tk, 2 * HEADS * DIM, dtype=k.dtype, device=k.device)
    kv[:, : HEADS * DIM] = k.reshape(tk, -1)
    kv[:, HEADS * DIM :] = v.reshape(tk, -1)
    k_s, v_s = kv[:, : HEADS * DIM].view(tk, HEADS, DIM), kv[:, HEADS * DIM :].view(tk, HEADS, DIM)
    assert k_s.stride(0) == v_s.stride(0) == 2 * HEADS * DIM
    return k_s, v_s


_HANDLES: dict = {}


def _compiled(fp8: bool, q8: bool = False, kv_strided: bool = False):
    """The handles the target builds at load, one per variant."""
    key = (fp8, q8, kv_strided)
    if key not in _HANDLES:
        _HANDLES[key] = entry.compile(
            torch.cuda.current_device(), fp8=fp8, q8=q8, kv_strided=kv_strided
        )
    return _HANDLES[key]


@pytest.fixture(scope="module")
def compiled():
    return _compiled(False)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Against fp32 attention: relative RMS below 1e-2 (bf16) or 3e-2 (fp8). The bf16
    kernel accumulates QK and PV in fp32 but rounds P to bf16 (relative error up to
    2^-9 per weight, applied to K/V rows of O(1)) and then O to bf16; over these random
    operands that measures 1.6e-3 .. 2.3e-3 relative RMS, so 1e-2 leaves 4x margin. The
    fp8 kernel's reference is the fp32 attention over the dequantized e4m3 operands; its
    P is e4m3 (2^-4 relative per weight), which measures 1.3e-2 .. 1.6e-2, so 3e-2 leaves
    2x margin. A wrong mask, request boundary or head scale misses by orders. Every row
    is written and a repeated launch gives the same bytes."""
    if cell.spec.get("q8"):
        _check_q8_cell(cell)
        return
    compiled = _compiled(bool(cell.spec.get("fp8")), kv_strided=bool(cell.spec.get("kv_strided")))
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    ref = op.reference(**args, compiled=compiled).float()
    assert ref.abs().sum() > 0, "a degenerate reference"
    with validating(op):
        got = op(**args, compiled=compiled)
    assert got is args["out"]
    again = op(**{**args, "out": torch.full_like(args["out"], float("nan"))}, compiled=compiled)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(got).all()), "a row was not written"
    assert torch.equal(got, again), "a repeated launch differs"
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    band = 3e-2 if cell.spec.get("fp8") else 1e-2
    assert rel < band, f"relative RMS {rel:.3e} vs the fp32 reference"


def _check_q8_cell(cell) -> None:
    """The MXF8-O handle writes O as o_proj's operand: exactly the activation quant of the
    bf16 O the fp8 handle writes for the same operands (the same tile math; the epilogue
    rounds O to bf16 and quantizes it with the activation quant's rule). The rows past the
    query rows (a mixed step's generation rows, quantized later by their own launch) are
    not touched, and `out` is not written."""
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    strided = bool(cell.spec.get("kv_strided"))
    t = int(args["q"].shape[0])
    t_o = t + int(cell.spec.get("tail", 0))
    plain = op(**args, compiled=_compiled(True, kv_strided=strided))
    q8 = torch.full((t_o, HEADS * DIM), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(entry.sf_bytes(t_o), dtype=torch.uint8, device="cuda")
    out8 = torch.full_like(args["out"], float("nan"))
    with validating(op):
        op(**{**args, "out": out8}, compiled=_compiled(True, True, strided), q8_out=(q8, sf))
    torch.cuda.synchronize()
    want_q8, want_sf = entry._kernel._mxf8.reference(plain.reshape(t, HEADS * DIM))
    assert torch.equal(q8[:t], want_q8), "the MXF8 O is not the activation quant of bf16 O"
    deq = entry._kernel._mxf8.dequantize
    assert torch.equal(
        deq(q8[:t].contiguous(), sf, t, HEADS * DIM), deq(want_q8, want_sf, t, HEADS * DIM)
    ), "block scales differ"
    assert bool((q8[t:] == 0x5A).all()), "rows past the query rows were written"
    assert bool(torch.isnan(out8).all()), "the MXF8-O handle wrote the bf16 output"


_KV_STRIDED = [c for c in op.CELLS if c.spec.get("kv_strided")]


@pytest.mark.parametrize("cell", _KV_STRIDED, ids=[c.why[:44] for c in _KV_STRIDED])
def test_kv_strided_is_the_packed_handle_bitwise(cell) -> None:
    """The `kv_strided` handle on the halves == the packed handle on the same K / V made
    contiguous: the MXF8 O operand and its scales (q8, the served handle) or the bf16 O."""
    q8 = bool(cell.spec.get("q8"))
    args = _build(cell.spec, zlib.crc32(f"strided {cell.why}".encode()) % 2**31)
    packed = {**args, "k": args["k"].contiguous(), "v": args["v"].contiguous()}
    assert not args["k"].is_contiguous() and args["v"].stride(0) == 2 * HEADS * DIM
    outs = []
    for a, strided in ((packed, False), (args, True)):
        handle = _compiled(True, q8, strided)
        if q8:
            outs.append(_q8_run(a, handle))
        else:
            out = op(**{**a, "out": torch.full_like(a["out"], float("nan"))}, compiled=handle)
            outs.append((out.view(torch.uint8),))
    torch.cuda.synchronize()
    for got, want in zip(outs[1], outs[0]):
        assert torch.equal(got, want), "the kv_strided handle differs from the packed handle"
    if q8:
        assert not bool((outs[0][0] == 0x5A).all(-1).any()), "an O row was not written"
    else:
        assert bool(torch.isfinite(outs[0][0].view(torch.bfloat16)).all()), "O not written"


def test_packed_handle_refuses_strided_kv() -> None:
    """The packed handle ignores tensor strides: it must not accept the halves."""
    args = _build(dict(new=[513], cached=[0], fp8=True, kv_strided=True), seed=6)
    with pytest.raises(AssertionError):
        with validating(op):
            op(**args, compiled=_compiled(True))


def test_kv_strided_graph_replay_is_eager() -> None:
    """The served MXF8-O `kv_strided` handle captured in a CUDA graph and replayed == eager."""
    spec = dict(new=[987] * 7 + [431], cached=[0] * 8, fp8=True, q8=True, kv_strided=True)
    args = _build(spec, seed=7)
    handle = _compiled(True, True, True)
    eager = _q8_run(args, handle)
    t = int(args["q"].shape[0])
    q8 = torch.full((t, HEADS * DIM), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(entry.sf_bytes(t), dtype=torch.uint8, device="cuda")
    out = torch.full_like(args["out"], float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(**{**args, "out": out}, compiled=handle, q8_out=(q8, sf))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(**{**args, "out": out}, compiled=handle, q8_out=(q8, sf))
    q8.fill_(0x5A)
    sf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q8, eager[0]) and torch.equal(sf, eager[1]), (
        "graph replay differs from eager"
    )


def test_is_valid_refuses_a_max_seqlen_short_of_the_longest_request(compiled) -> None:
    """The grid is sized by max_seqlen: a request's rows past it are never computed."""
    args = _build(dict(new=[300, 40], cached=[0, 0]), seed=1)
    args["max_seqlen"] = 256
    with pytest.raises(AssertionError, match="max_seqlen"):
        with validating(op):
            op(**args, compiled=compiled)


def test_is_valid_refuses_fewer_kv_rows_than_query_rows(compiled) -> None:
    """K/V rows are the cached prefix and then the query rows: never fewer."""
    args = _build(dict(new=[64, 64], cached=[8, 0]), seed=2)
    args["cu_seqlens_k"] = _cu([72, 0])
    args["k"], args["v"] = args["k"][:72], args["v"][:72]
    with pytest.raises(AssertionError, match="fewer K/V rows"):
        with validating(op):
            op(**args, compiled=compiled)


def test_fp8_handle_refuses_a_call_without_head_scales() -> None:
    """The fp8 kernel reads its dequantization factors on every call."""
    args = _build(dict(new=[64], cached=[0], fp8=True), seed=3)
    args["head_scales"] = None
    with pytest.raises(AssertionError, match="head_scales"):
        with validating(op):
            op(**args, compiled=_compiled(True))


_NO_OFFLOAD: dict = {}


def _q8_handle_without_offload():
    """The MXF8-O handle without the offloaded epilogue: the build it replaces."""
    if "h" not in _NO_OFFLOAD:
        _NO_OFFLOAD["h"] = entry.compile(
            torch.cuda.current_device(), fp8=True, q8=True, epi_offload=False
        )
    return _NO_OFFLOAD["h"]


#: The q8 cells' shapes, and a peaked softmax (a much sharper q: a few keys take nearly all of
#: each row's weight, so the e4m3 P rounding and the row sums are at their extremes).
_OFFLOAD_SPECS = [c.spec for c in op.CELLS if c.spec.get("q8") and not c.spec.get("kv_strided")] + [
    dict(new=[987] * 8, cached=[0] * 8, fp8=True, q8=True, q_scale=24.0),
    dict(new=[300, 40, 1], cached=[17, 64, 300], fp8=True, q8=True, q_scale=24.0),
]


def _q8_run(args, handle):
    t = int(args["q"].shape[0])
    q8 = torch.full((t, HEADS * DIM), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(entry.sf_bytes(t), dtype=torch.uint8, device="cuda")
    op(**{**args, "out": torch.full_like(args["out"], float("nan"))}, compiled=handle,
       q8_out=(q8, sf))  # fmt: skip
    return q8, sf


@pytest.mark.parametrize("spec", _OFFLOAD_SPECS, ids=range(len(_OFFLOAD_SPECS)))
def test_offloaded_epilogue_is_the_plain_q8_build_bitwise(spec) -> None:
    args = _build(spec, seed=31 + sum(spec["new"]))
    got = _q8_run(args, _compiled(True, True))
    want = _q8_run(args, _q8_handle_without_offload())
    torch.cuda.synchronize()
    assert torch.equal(got[0], want[0]), "MXF8 O bytes differ from the build without the offload"
    assert torch.equal(got[1], want[1]), "block scales differ from the build without the offload"


def test_offloaded_epilogue_graph_replay_is_eager() -> None:
    spec = dict(new=[987] * 8, cached=[0] * 8, fp8=True, q8=True)
    args = _build(spec, seed=5)
    handle = _compiled(True, True)
    eager = _q8_run(args, handle)
    t = int(args["q"].shape[0])
    q8 = torch.full((t, HEADS * DIM), 0x5A, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(entry.sf_bytes(t), dtype=torch.uint8, device="cuda")
    out = torch.full_like(args["out"], float("nan"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(**{**args, "out": out}, compiled=handle, q8_out=(q8, sf))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op(**{**args, "out": out}, compiled=handle, q8_out=(q8, sf))
    q8.fill_(0x5A)
    sf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q8, eager[0]) and torch.equal(sf, eager[1]), (
        "graph replay differs from eager"
    )
