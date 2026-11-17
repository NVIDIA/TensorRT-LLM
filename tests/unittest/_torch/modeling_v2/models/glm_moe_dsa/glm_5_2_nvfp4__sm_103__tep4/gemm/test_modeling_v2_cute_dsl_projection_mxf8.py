# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection_mxf8 entry.

The `kv_strided` cells are the fp8 context attention's served K / V expansion: `kv_b_kv8` in
one launch. Beyond the reference, its `[T_k, 8192]` output must be the two launches `kv_b_k8`
+ `kv_b_v8` side by side, bit for bit, on the operands the target issues (the A operands
written by `cute_dsl_mha_kv_gather_mxf8`, the expansion weights as `_build_context_mha_fp8`
folds them, the row slices and per-128-row scale offsets of `_expand_mxf8`).
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_projection_mxf8 runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_kv_gather_mxf8 as gather_entry,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection_mxf8 as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

op = entry.cute_dsl_projection_mxf8

#: One runner per entry, built once per session against one layer's weight.
_RUNNERS: dict = {}


def _shape(kernel: str) -> tuple[int, int]:
    abi = entry._kernel.ABIS[entry.ENTRIES[kernel]]
    return abi.out_features, abi.in_features


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    """A row-major bf16 `[N, K]` weight scaled so each output is O(1); zero where a split
    entry's tiles do not read it (`n_split_k`, the entry's contract)."""
    n, k = _shape(kernel)
    w = (torch.randn(n, k, generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)
    split = entry._kernel.ABIS[entry.ENTRIES[kernel]].n_split_k
    if split is not None:
        w[split[0] :, split[1] :] = 0
    return w


def _runner(kernel: str) -> entry.Mxf8ProjectionRunner:
    if kernel not in _RUNNERS:
        g = torch.Generator(device="cuda").manual_seed(1)
        _RUNNERS[kernel] = entry.build(kernel, entry.quantize_weight(_weight(kernel, g)))
    return _RUNNERS[kernel]


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Against the same MXF8 operands multiplied in fp32: the products of two e4m3 values
    and power-of-two scales are exact in fp32, so the two sides differ only by the order
    of K fp32 additions before one rounding to the output. bf16 out: 4 ulp element-wise
    and 2 ulp relative RMS, as the bf16 projections. e4m3 out: an element differs only
    where the two sums straddle an e4m3 rounding boundary, by one step (3 mantissa bits),
    at no more than 1% of the elements. A wrong scale factor (a whole 32-element block off
    by a power of two) or a wrong tile misses by many steps."""
    if out.dtype == torch.float8_e4m3fn:
        got, want = out.float(), ref.float()
        assert bool(torch.isfinite(got).all()), "an element was not written"
        step = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-6))) - 3)
        diff = (got - want).abs()
        assert bool((diff <= step).all()), f"max error {float((diff / step).max()):.1f} steps"
        frac = float((diff > 0).float().mean())
        assert frac <= 0.01, f"{frac:.2%} of the elements differ from the reference"
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _e4m3_bytes(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Finite e4m3 bytes of O(1) values: a latent-like pre-quantized A operand."""
    vals = torch.randn(rows, k, generator=g, device="cuda") * 4.0
    return vals.to(torch.float8_e4m3fn).view(torch.uint8)


def _block_scales(rows: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Blocked e8m0 scales with codes around 2^0 (2^-4 .. 2^3): every block's own scale."""
    n = entry._kernel.unit_sf_bytes(rows, k)
    return torch.randint(123, 131, (n,), generator=g, device="cuda", dtype=torch.int32).to(
        torch.uint8
    )


def _call(x, weight, out, runner, pre=None, sfa=None) -> None:
    """The op as the target calls it: a pre-quantized operand in row slices of the runner's
    capacity (a multiple of 128 rows), each slice's scales from its first 128-row block."""
    if not runner.prequant or x.shape[0] <= runner.launch.max_rows:
        assert op(x, weight, out, runner, pre=pre, sfa=sfa) is out, (
            "the op writes `out` and returns it"
        )
        return
    step = runner.launch.max_rows
    blk = entry._kernel.unit_sf_bytes(128, runner.k)
    for r0 in range(0, x.shape[0], step):
        r1 = min(x.shape[0], r0 + step)
        part = None if sfa is None else sfa[r0 // 128 * blk :]
        op(x[r0:r1], weight, out[r0:r1], runner, sfa=part)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    n, k = _shape(kernel)
    runner = _runner(kernel)
    assert (runner.n, runner.k) == (n, k)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    # Another layer's weight: the runner was built against a different one.
    weight = entry.quantize_weight(_weight(kernel, g))
    pre = sfa = None
    if runner.prequant:
        x = _e4m3_bytes(rows, k, g)
        if cell.spec.get("sfa"):
            sfa = _block_scales(rows, k, g)
    else:
        x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
        if cell.spec.get("x_row_stride"):
            # The tail of a wider row, as q_b reads q_a in the q_a | kv_a projection row.
            wide = torch.zeros(rows, cell.spec["x_row_stride"], dtype=torch.bfloat16, device="cuda")
            wide[:, :k] = x
            x = wide[:, :k]
        if cell.spec.get("pre"):
            # The producer's operand: the activation quant's exact bytes.
            pre = tuple(t.contiguous() for t in entry._kernel._quant.reference(x))
    # The dense down's call: the SwiGLU wrote only the operand, which is passed as `x` too.
    x_call = pre[0] if cell.spec.get("x_is_pre") else x
    # NaN, so an element the kernel never writes fails the comparison.
    out = torch.full((rows, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = out.to(runner.out_dtype)
    with validating(op):
        _call(x_call, weight, out, runner, pre=pre, sfa=sfa)
    torch.cuda.synchronize()
    _compare(out, op.reference(x_call, weight, out, runner, pre=pre, sfa=sfa))
    if pre is not None:
        # Same bytes in, same launch: the producer's operand must give the quant path's output.
        plain = torch.full_like(out, float("nan"))
        op(x, weight, plain, runner)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), plain.view(torch.uint8)), (
            "a producer-quantized operand must give the activation quant's output bitwise"
        )
    # FP8-level against the bf16 product: the MXF8 rounding of both operands only.
    if out.dtype == torch.bfloat16 and not runner.prequant:
        dense = (x.float() @ _dequantized(weight, n, k).t()).float()
        rel = ((out.float() - dense).norm() / dense.norm()).item()
        assert rel < 5e-2, f"relative RMS {rel:.3e} vs the product with x unquantized"


def _dequantized(weight, n: int, k: int) -> torch.Tensor:
    return entry._kernel.dequantize_weight(weight)


def test_weight_quantization_is_the_activation_rule() -> None:
    """The load-time weight quantization and the per-step activation quantization are the
    same kernel: equal bytes and scale factors for the same matrix."""
    g = torch.Generator(device="cuda").manual_seed(7)
    w = _weight("o_proj", g)
    wq = entry.quantize_weight(w)
    q, sf = entry._kernel._quant.reference(w)
    assert torch.equal(wq.q, q)
    assert torch.equal(wq.sf[: sf.numel()], sf)


def test_is_valid_refuses_an_unaligned_row_stride() -> None:
    """The quant kernel loads 8 bf16 per thread: a row stride off a multiple of 8 would
    read across rows."""
    runner = _runner("qkva")
    n, k = _shape("qkva")
    wide = torch.zeros(65, k + 4, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(65, n, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="row stride"):
        with validating(op):
            op(wide[:, :k], entry.quantize_weight(torch.zeros(n, k, dtype=torch.bfloat16,
                                                              device="cuda")), out, runner)  # fmt: skip


# ---------------------------------------------------------------------------
# kv_strided: `kv_b_kv8` is `kv_b_k8` | `kv_b_v8`, bitwise.
# ---------------------------------------------------------------------------

H, NOPE, ROPE, V, LORA = 16, 192, 64, 256, 512
LAT = LORA + ROPE
QK = NOPE + ROPE
K_ROWS, V_ROWS = H * QK, H * V  # 4096, 4096

_KV: dict = {}


def _kv_weights(g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The folded K / V expansion weights (bf16) as `_build_context_mha_fp8` makes them: per
    head `[W_UK_h | 0 ; 0 | I_64]` and `W_UV_h`, a power-of-two per-head scale folded in."""
    w_uk = torch.randn(H, NOPE, LORA, generator=g, device="cuda") * LORA**-0.5
    w_uv = torch.randn(H, V, LORA, generator=g, device="cuda") * LORA**-0.5
    w_k = torch.zeros(H, QK, LAT, device="cuda")
    w_k[:, :NOPE, :LORA] = w_uk
    w_k[:, NOPE:, LORA:] = torch.eye(ROPE, device="cuda")
    s_k = torch.exp2(torch.randint(-3, 3, (H, 1, 1), generator=g, device="cuda").float())
    s_v = torch.exp2(torch.randint(-3, 3, (H, 1, 1), generator=g, device="cuda").float())
    w_k = (w_k / s_k).to(torch.bfloat16).view(K_ROWS, LAT)
    w_v = (w_uv / s_v).to(torch.bfloat16).view(V_ROWS, LORA)
    return w_k, w_v


def _kv_state() -> dict:
    """One layer's expansion weights in both forms, and the gather's launcher."""
    if not _KV:
        g = torch.Generator(device="cuda").manual_seed(zlib.crc32(b"kv weights") % 2**31)
        w_k, w_v = _kv_weights(g)
        merged = torch.zeros(K_ROWS + V_ROWS, LAT, dtype=torch.bfloat16, device="cuda")
        merged[:K_ROWS] = w_k
        merged[K_ROWS:, :LORA] = w_v
        q = entry.quantize_weight
        _KV["wk8"], _KV["wv8"], _KV["wkv8"] = q(w_k), q(w_v), q(merged)
        _KV["gather"] = gather_entry.compile(torch.cuda.current_device())
    return _KV


def _a8(tk: int, seed: int):
    """The gather's A operands `(a_k, a_v, sfa_k, sfa_v)` for `tk` new latent rows, with block
    magnitudes from tiny to O(10) (every e8m0 code path)."""
    st = _kv_state()
    g = torch.Generator(device="cuda").manual_seed(seed)
    lat = torch.randn(tk, LAT, generator=g, device="cuda")
    block_exp = torch.randint(-12, 4, (tk, LAT // 32), generator=g, device="cuda")
    lat = (lat * torch.exp2(block_exp.float()).repeat_interleave(32, dim=1)).to(torch.bfloat16)
    a8 = tuple(t.fill_(0xA5) for t in gather_entry.buffers(tk, "cuda"))
    d = torch.device("cuda")
    gather_entry.cute_dsl_mha_kv_gather_mxf8(
        torch.zeros(1, LAT, dtype=torch.uint8, device=d),
        torch.zeros(1, dtype=torch.int64, device=d),
        0,
        torch.zeros(1, dtype=torch.int32, device=d),
        torch.zeros(1, 1, dtype=torch.int32, device=d),
        lat,
        a8,
        0,
        1,
        1,
        False,
        compiled=st["gather"],
    )
    return a8


def _two_launch(a8, k: torch.Tensor, v: torch.Tensor) -> None:
    """The two-launch `_expand_mxf8` body: K then V per row slice."""
    st = _kv_state()
    a_k, a_v, sfa_k, sfa_v = a8
    rk, rv = _runner("kv_b_k8"), _runner("kv_b_v8")
    step = rk.launch.max_rows
    blk_k, blk_v = gather_entry.sf_bytes_k(128), gather_entry.sf_bytes_v(128)
    for r0 in range(0, k.shape[0], step):
        r1 = min(k.shape[0], r0 + step)
        b = r0 // 128
        op(a_k[r0:r1], st["wk8"], k[r0:r1], rk, sfa=sfa_k[b * blk_k :])
        op(a_v[r0:r1], st["wv8"], v[r0:r1], rv, sfa=sfa_v[b * blk_v :])


def _one_launch(a8, kv: torch.Tensor) -> None:
    """The served `_expand_mxf8` body: one launch per row slice on `a_k`."""
    st = _kv_state()
    a_k, _, sfa_k, _ = a8
    rkv = _runner("kv_b_kv8")
    step = rkv.launch.max_rows
    blk_k = gather_entry.sf_bytes_k(128)
    for r0 in range(0, kv.shape[0], step):
        r1 = min(kv.shape[0], r0 + step)
        b = r0 // 128
        op(a_k[r0:r1], st["wkv8"], kv[r0:r1], rkv, sfa=sfa_k[b * blk_k :])


def _nan_e4m3(rows: int, cols: int) -> torch.Tensor:
    """e4m3 NaN bytes, so an element the kernel never writes fails the byte comparison."""
    return torch.full((rows, cols), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )


_KV_STRIDED = [c for c in op.CELLS if c.spec.get("kv_strided")]


def test_merged_weight_bytes_are_the_two_weights() -> None:
    """Rows quantize independently: the stacked weight's K rows are `kv_b_k8`'s bytes, its V
    rows' first 512 columns `kv_b_v8`'s, the rest zero (`kv_b_kv8`'s contract)."""
    st = _kv_state()
    wkv, wk, wv = st["wkv8"], st["wk8"], st["wv8"]
    assert torch.equal(wkv.q[:K_ROWS], wk.q)
    assert torch.equal(wkv.q[K_ROWS:, :LORA], wv.q)
    assert not bool(wkv.q[K_ROWS:, LORA:].any())


@pytest.mark.parametrize("cell", _KV_STRIDED, ids=[c.why[:44] for c in _KV_STRIDED])
def test_kv_strided_one_launch_is_the_two_launches_bitwise(cell) -> None:
    """K | V of the one launch == the two launches' K and V, byte for byte (tile choice,
    row slices and scale offsets as the target issues them)."""
    tk = cell.spec["rows"]
    a8 = _a8(tk, zlib.crc32(f"a8 {tk}".encode()) % 2**31)
    k, v = _nan_e4m3(tk, K_ROWS), _nan_e4m3(tk, V_ROWS)
    kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
    _two_launch(a8, k, v)
    _one_launch(a8, kv)
    torch.cuda.synchronize()
    # the gather's a_v is a_k's first 512 columns: the V tiles' operand is the same bytes
    assert torch.equal(a8[0][:, :LORA], a8[1])
    kb, vb, kvb = k.view(torch.uint8), v.view(torch.uint8), kv.view(torch.uint8)
    assert not bool((kb == 0x7F).all(-1).any()), "a K row of the two launches was not written"
    assert torch.equal(kvb[:, :K_ROWS], kb), "K half differs from kv_b_k8"
    assert torch.equal(kvb[:, K_ROWS:], vb), "V half differs from kv_b_v8"
    # and both are the MXF8 product (not two equal wrong answers)
    st = _kv_state()
    rows = min(tk, 1024)
    ref = op.reference(a8[0][:rows], st["wkv8"], kv[:rows], _runner("kv_b_kv8"), sfa=a8[2])
    got, want = kv[:rows].float(), ref.float()
    step = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-6))) - 3)
    assert bool(((got - want).abs() <= step).all()), "one-launch output off the MXF8 product"


def test_kv_b_kv8_refuses_nonzero_v_rows_past_column_512() -> None:
    """`kv_b_kv8`'s V rows must be zero past column 512 (its V tiles never read them)."""
    st = _kv_state()
    bad = st["wkv8"].q.clone()
    bad[K_ROWS + 5, LORA + 3] = 0x38
    w = entry.QuantizedWeight(bad, st["wkv8"].sf)
    x = torch.zeros(128, LAT, dtype=torch.uint8, device="cuda")
    out = torch.empty(128, K_ROWS + V_ROWS, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(AssertionError):
        op.is_valid(x, w, out, _runner("kv_b_kv8"))


def test_kv_b_kv8_graph_replay_equals_eager() -> None:
    """The one launch captured in a CUDA graph, replayed on new operands copied into its
    buffers, == the eager calls on those operands (7340 rows)."""
    tk = 7340
    a8 = _a8(tk, zlib.crc32(b"graph a") % 2**31)
    kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
    _one_launch(a8, kv)  # warm (module loads outside the capture)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _one_launch(a8, kv)
    for rep in range(3):
        for dst, src in zip(a8, _a8(tk, zlib.crc32(f"graph a {rep}".encode()) % 2**31)):
            dst.copy_(src)
        kv.view(torch.uint8).fill_(0x7F)
        graph.replay()
        torch.cuda.synchronize()
        got = kv.view(torch.uint8).clone()
        kv.view(torch.uint8).fill_(0x7F)
        _one_launch(a8, kv)
        torch.cuda.synchronize()
        assert torch.equal(got, kv.view(torch.uint8)), f"graph replay {rep} differs from eager"
