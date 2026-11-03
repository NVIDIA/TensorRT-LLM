# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8 context attention's K / V expansion in one launch.

`cute_dsl_projection_mxf8` `kv_b_kv8` (one GEMM, N = 8192, the V tiles on K < 512) against
the served two launches `kv_b_k8` + `kv_b_v8`, and the `kv_strided` MHA handles reading K / V
as the two halves of its `[T_k, 8192]` rows against the packed handles reading the two launches'
outputs. Everything is compared bit for bit: the merged path must be the two-launch path's bytes.

The operands are the target's: the A operands written by `cute_dsl_mha_kv_gather_mxf8` from
roped bf16 latent rows with block magnitudes from tiny to O(10) (every e8m0 code path), the
expansion weights built like `_build_context_mha_fp8` builds them (per head `[W_UK_h | 0 ; 0 |
I_64]` and `W_UV_h`, a power-of-two per-head scale folded in), the context row counts the
per-layer call meets (513 .. 16448, i.e. one and two runner slices of 16384 rows; with a cached
prefix the K/V rows exceed the query rows), and the call sequence `_expand_mxf8` issues (row
slices of `max_rows`, the scale bytes offset per 128-row block).
"""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the K / V expansion runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_kv_gather_mxf8 as gather_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention import (  # noqa: E402, E501
    cute_dsl_mha_prefill as mha_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm import (  # noqa: E402, E501
    cute_dsl_projection_mxf8 as proj_entry,
)

proj = proj_entry.cute_dsl_projection_mxf8
gather = gather_entry.cute_dsl_mha_kv_gather_mxf8
mha = mha_entry.cute_dsl_mha_prefill

H, NOPE, ROPE, V, LORA = 16, 192, 64, 256, 512
LAT = LORA + ROPE
QK = NOPE + ROPE
K_ROWS, V_ROWS = H * QK, H * V  # 4096, 4096

#: (query rows per request, cached rows per request): T_k = sum of both.
CASES = {
    513: ([513], [0]),
    4128: ([1032] * 4, [0] * 4),
    7340: ([987] * 7 + [431], [0] * 8),
    8192: ([1024] * 8, [0] * 8),
    # 8192 query rows over 16448 K/V rows: two runner slices (16384 + 64)
    16448: ([1024] * 8, [1032] * 8),
}

_STATE: dict = {}


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _weights(g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The folded K / V expansion weights (bf16) as `_build_context_mha_fp8` makes them."""
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


def _merged_weight(w_k: torch.Tensor, w_v: torch.Tensor) -> torch.Tensor:
    w = torch.zeros(K_ROWS + V_ROWS, LAT, dtype=torch.bfloat16, device="cuda")
    w[:K_ROWS] = w_k
    w[K_ROWS:, :LORA] = w_v
    return w


def _state() -> dict:
    """Runners (built once against zero weights, as the target builds them), one layer's
    weights in both forms, and the MHA handles."""
    if not _STATE:
        dev = torch.cuda.current_device()
        z = dict(dtype=torch.bfloat16, device="cuda")
        q = proj_entry.quantize_weight
        _STATE["rk"] = proj_entry.build("kv_b_k8", q(torch.zeros(K_ROWS, LAT, **z)))
        _STATE["rv"] = proj_entry.build("kv_b_v8", q(torch.zeros(V_ROWS, LORA, **z)))
        _STATE["rkv"] = proj_entry.build("kv_b_kv8", q(torch.zeros(K_ROWS + V_ROWS, LAT, **z)))
        _STATE["gather"] = gather_entry.compile(dev)
        g = torch.Generator(device="cuda").manual_seed(_seed("weights"))
        w_k, w_v = _weights(g)
        _STATE["wk8"], _STATE["wv8"] = q(w_k), q(w_v)
        _STATE["wkv8"] = q(_merged_weight(w_k, w_v))
        for name, kw in (
            ("q8", dict(fp8=True, q8=True)),
            ("q8s", dict(fp8=True, q8=True, kv_strided=True)),
            ("fp8", dict(fp8=True)),
            ("fp8s", dict(fp8=True, kv_strided=True)),
        ):
            _STATE[name] = mha_entry.compile(dev, **kw)
    return _STATE


def _a8(tk: int, seed: int):
    """The gather's A operands for `tk` new latent rows (no cached prefix: rows in order)."""
    st = _state()
    g = torch.Generator(device="cuda").manual_seed(seed)
    lat = torch.randn(tk, LAT, generator=g, device="cuda")
    block_exp = torch.randint(-12, 4, (tk, LAT // 32), generator=g, device="cuda")
    lat = (lat * torch.exp2(block_exp.float()).repeat_interleave(32, dim=1)).to(torch.bfloat16)
    a8 = tuple(t.fill_(0xA5) for t in gather_entry.buffers(tk, "cuda"))
    d = torch.device("cuda")
    gather(
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
    """The served `_expand_mxf8` body: K then V per row slice."""
    st = _state()
    a_k, a_v, sfa_k, sfa_v = a8
    rk, rv = st["rk"], st["rv"]
    step = rk.launch.max_rows
    blk_k, blk_v = gather_entry.sf_bytes_k(128), gather_entry.sf_bytes_v(128)
    for r0 in range(0, k.shape[0], step):
        r1 = min(k.shape[0], r0 + step)
        b = r0 // 128
        proj(a_k[r0:r1], st["wk8"], k[r0:r1], rk, sfa=sfa_k[b * blk_k :])
        proj(a_v[r0:r1], st["wv8"], v[r0:r1], rv, sfa=sfa_v[b * blk_v :])


def _one_launch(a8, kv: torch.Tensor) -> None:
    """The merged `_expand_mxf8` body: one launch per row slice on `a_k`."""
    st = _state()
    a_k, _, sfa_k, _ = a8
    rkv = st["rkv"]
    step = rkv.launch.max_rows
    blk_k = gather_entry.sf_bytes_k(128)
    for r0 in range(0, kv.shape[0], step):
        r1 = min(kv.shape[0], r0 + step)
        b = r0 // 128
        proj(a_k[r0:r1], st["wkv8"], kv[r0:r1], rkv, sfa=sfa_k[b * blk_k :])


def _nan_e4m3(rows: int, cols: int) -> torch.Tensor:
    """e4m3 NaN bytes, so an element the kernel never writes fails the byte comparison."""
    return torch.full((rows, cols), 0x7F, dtype=torch.uint8, device="cuda").view(
        torch.float8_e4m3fn
    )


def test_merged_weight_bytes_are_the_two_weights() -> None:
    """Rows quantize independently: the stacked weight's K rows are `kv_b_k8`'s bytes, its V
    rows' first 512 columns `kv_b_v8`'s, the rest zero (`kv_b_kv8`'s contract)."""
    st = _state()
    wkv, wk, wv = st["wkv8"], st["wk8"], st["wv8"]
    assert torch.equal(wkv.q[:K_ROWS], wk.q)
    assert torch.equal(wkv.q[K_ROWS:, :LORA], wv.q)
    assert not bool(wkv.q[K_ROWS:, LORA:].any())


@pytest.mark.parametrize("tk", sorted(CASES))
def test_one_launch_is_the_two_launches_bitwise(tk: int) -> None:
    """K | V of the one launch == the two launches' K and V, byte for byte (tile choice,
    row slices and scale offsets as the target issues them)."""
    a8 = _a8(tk, _seed(f"a8 {tk}"))
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
    # and both are the MXF8 product (sanity: not two equal wrong answers)
    st = _state()
    ref = proj.reference(a8[0][:1024], st["wkv8"], kv[:1024], st["rkv"], sfa=a8[2])
    got, want = kv[:1024].float(), ref.float()
    step = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-6))) - 3)
    assert bool(((got - want).abs() <= step).all()), "one-launch output off the MXF8 product"


def _mha_args(tk: int, seed: int):
    news, cached = CASES[tk]
    assert sum(news) + sum(cached) == tk
    g = torch.Generator(device="cuda").manual_seed(seed)
    tq = sum(news)
    q = (torch.randn(tq, H, QK, generator=g, device="cuda") * 2.0).to(torch.float8_e4m3fn)
    cu, cu_k = [0], [0]
    for n, c in zip(news, cached):
        cu.append(cu[-1] + n)
        cu_k.append(cu_k[-1] + n + c)
    cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
    cu_k = torch.tensor(cu_k, dtype=torch.int32, device="cuda") if sum(cached) else None
    s_k = torch.exp2(torch.randint(-3, 3, (H,), generator=g, device="cuda").float())
    s_v = torch.exp2(torch.randint(-3, 3, (H,), generator=g, device="cuda").float())
    hs = mha_entry.fp8_head_scales(2.0**-3, s_k.cpu(), s_v.cpu()).cuda()
    return q, cu, cu_k, max(news), hs


@pytest.mark.parametrize("tk", sorted(CASES))
@pytest.mark.parametrize("handle", ["q8", "fp8"])
def test_strided_mha_reads_the_halves_bitwise(tk: int, handle: str) -> None:
    """The `kv_strided` MHA on the one launch's `[T_k, 8192]` halves == the packed MHA on the
    two launches' K and V: the MXF8 O operand (q8, the served handle) or the bf16 O."""
    st = _state()
    a8 = _a8(tk, _seed(f"a8 {tk}"))
    k, v = _nan_e4m3(tk, K_ROWS), _nan_e4m3(tk, V_ROWS)
    kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
    _two_launch(a8, k, v)
    _one_launch(a8, kv)
    q, cu, cu_k, max_len, hs = _mha_args(tk, _seed(f"mha {tk}"))
    tq = q.shape[0]
    k_s, v_s = kv[:, :K_ROWS].view(tk, H, QK), kv[:, K_ROWS:].view(tk, H, V)
    assert k_s.data_ptr() == kv.data_ptr() and v_s.stride(0) == K_ROWS + V_ROWS
    outs = []
    for name, kk, vv in ((handle, k.view(tk, H, QK), v.view(tk, H, V)), (handle + "s", k_s, v_s)):
        out = torch.full((tq, H, V), float("nan"), dtype=torch.bfloat16, device="cuda")
        q8_out = None
        if handle == "q8":
            q8_out = (
                torch.full((tq, H * V), 0x7F, dtype=torch.uint8, device="cuda"),
                torch.full((mha_entry.sf_bytes(tq),), 0xA5, dtype=torch.uint8, device="cuda"),
            )
        mha.is_valid(q, kk, vv, out, cu, max_len, st[name], cu_k, hs, q8_out)
        mha(q, kk, vv, out, cu, max_len, st[name], cu_seqlens_k=cu_k, head_scales=hs,
            q8_out=q8_out)  # fmt: skip
        outs.append(q8_out if handle == "q8" else (out.view(torch.uint8),))
    torch.cuda.synchronize()
    for a, b in zip(*outs):
        assert torch.equal(a, b), f"{handle}: strided MHA output differs at T_k {tk}"
    if handle == "q8":
        assert not bool((outs[0][0] == 0x7F).all(-1).any()), "an O row was not written"
    else:
        assert bool(torch.isfinite(outs[0][0].view(torch.bfloat16)).all()), "O not written"


def test_packed_handle_refuses_strided_k() -> None:
    """The packed handle ignores tensor strides: it must not accept the halves."""
    st = _state()
    tk = 513
    kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
    q, cu, cu_k, max_len, hs = _mha_args(tk, 1)
    out = torch.empty(q.shape[0], H, V, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        st["fp8"].check(
            q, kv[:, :K_ROWS].view(tk, H, QK), kv[:, K_ROWS:].view(tk, H, V), out, cu, cu_k, hs
        )


def test_split_weight_contract_is_checked() -> None:
    """`kv_b_kv8`'s V rows must be zero past column 512 (its V tiles never read them)."""
    st = _state()
    bad = st["wkv8"].q.clone()
    bad[K_ROWS + 5, LORA + 3] = 0x38
    w = proj_entry.QuantizedWeight(bad, st["wkv8"].sf)
    x = torch.zeros(128, LAT, dtype=torch.uint8, device="cuda")
    out = torch.empty(128, K_ROWS + V_ROWS, dtype=torch.float8_e4m3fn, device="cuda")
    with pytest.raises(AssertionError):
        proj.is_valid(x, w, out, st["rkv"])


def test_graph_replay_equals_eager() -> None:
    """One launch + strided MXF8-O MHA captured in a CUDA graph, replayed on new operands
    copied into its buffers, == the eager calls on those operands (7340 rows)."""
    st = _state()
    tk = 7340
    a8 = _a8(tk, _seed("graph a"))
    q, cu, cu_k, max_len, hs = _mha_args(tk, _seed("graph q"))
    tq = q.shape[0]
    kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
    q8 = torch.empty(tq, H * V, dtype=torch.uint8, device="cuda")
    sf = torch.empty(mha_entry.sf_bytes(tq), dtype=torch.uint8, device="cuda")
    out = torch.empty(tq, H, V, dtype=torch.bfloat16, device="cuda")

    def step():
        _one_launch(a8, kv)
        mha(q, kv[:, :K_ROWS].view(tk, H, QK), kv[:, K_ROWS:].view(tk, H, V), out, cu,
            max_len, st["q8s"], cu_seqlens_k=cu_k, head_scales=hs, q8_out=(q8, sf))  # fmt: skip

    step()  # warm (module loads outside the capture)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for rep in range(3):
        new_a8 = _a8(tk, _seed(f"graph a {rep}"))
        new_q = _mha_args(tk, _seed(f"graph q {rep}"))[0]
        for dst, src in zip(a8, new_a8):
            dst.copy_(src)
        q.copy_(new_q)
        kv.view(torch.uint8).fill_(0x7F)
        graph.replay()
        torch.cuda.synchronize()
        got = (kv.view(torch.uint8).clone(), q8.clone(), sf.clone())
        kv.view(torch.uint8).fill_(0x7F)
        step()
        torch.cuda.synchronize()
        want = (kv.view(torch.uint8), q8, sf)
        for a, b in zip(got, want):
            assert torch.equal(a, b), f"graph replay {rep} differs from eager"


def test_timing_report() -> None:
    """Per-launch device time, two launches vs one, at the served context shapes (CUDA events,
    interleaved, medians of 30; printed, not asserted). Run with -s to see it."""
    for tk in (4128, 7340, 8192):
        a8 = _a8(tk, _seed(f"time {tk}"))
        k, v = _nan_e4m3(tk, K_ROWS), _nan_e4m3(tk, V_ROWS)
        kv = _nan_e4m3(tk, K_ROWS + V_ROWS)
        flush = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        times = {"two": [], "one": []}
        for _ in range(30):
            for name, fn in (
                ("two", lambda: _two_launch(a8, k, v)),
                ("one", lambda: _one_launch(a8, kv)),
            ):
                flush.zero_()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                times[name].append(e0.elapsed_time(e1) * 1e3)
        med = {n: sorted(t)[len(t) // 2] for n, t in times.items()}
        print(
            f"KVB_MERGE_TIME T_k={tk} two_launch_us={med['two']:.2f} one_launch_us={med['one']:.2f} "
            f"delta_us={med['one'] - med['two']:+.2f}",
            flush=True,
        )
