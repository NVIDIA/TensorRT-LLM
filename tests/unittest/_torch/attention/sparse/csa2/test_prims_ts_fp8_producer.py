# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the packed producer and compare its E4M3 operands and native attention."""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="CSA2 PrimTS requires SM100, SM103 or SM107",
)

_Kind = Literal["swa", "swa_fp8", "main", "main_fp8"]
_Probe = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]
_POOL_LAYOUTS = [
    pytest.param("swa", 528, id="swa-compact"),
    pytest.param("swa", 609, id="swa-odd-stride"),
    pytest.param("swa_fp8", 512, id="swa-fp8-compact"),
    pytest.param("swa_fp8", 544, id="swa-fp8-padded-stride"),
    pytest.param("swa_fp8", 513, id="swa-fp8-odd-stride"),
    pytest.param("main", 288, id="main-compact"),
    pytest.param("main", 320, id="main-padded-stride"),
    pytest.param("main", 321, id="main-odd-stride"),
    pytest.param("main_fp8", 512, id="main-fp8-compact"),
    pytest.param("main_fp8", 544, id="main-fp8-padded-stride"),
    pytest.param("main_fp8", 513, id="main-fp8-odd-stride"),
]


@pytest.fixture(scope="module")
def _producer_probes() -> dict[_Kind, _Probe]:
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_producer import (
        _PackedSmemKvResource,
    )

    class _DecodeProbe:
        def __init__(self, kind: _Kind) -> None:
            self.is_main = kind in ("main", "main_fp8")
            self.swa_cache_format = "swa_fp8" if kind == "swa_fp8" else "swa"
            self.main_cache_format = "main_fp8" if kind == "main_fp8" else "main"

        @cute.kernel
        def kernel(
            self,
            pool: cute.Tensor,
            indices: cute.Tensor,
            output: cute.Tensor,
        ) -> None:
            lane = cute.arch.thread_idx()[0]
            selected = cute.arch.block_idx()[0]
            row = cutlass.Int32(indices[selected])
            dimension = cutlass.Int32(lane * 16)
            # Only the chosen source is accessed; no task graph or SMEM is needed
            # to invoke the same decoder that fills the native operand ring.
            resource = _PackedSmemKvResource(
                swa=pool,
                main=pool,
                swa_cache_format=self.swa_cache_format,
                main_cache_format=self.main_cache_format,
            )
            values = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for word in cutlass.range_constexpr(4):
                values[word] = cutlass.Int32(0)
            if cutlass.const_expr(self.is_main):
                resource._decode_main_chunk(row, dimension, values)
            else:
                resource._decode_swa_chunk(row, dimension, values)
            for word in cutlass.range_constexpr(4):
                output[selected, lane * 4 + word] = values[word]

        @cute.jit
        def __call__(
            self,
            pool: cute.Tensor,
            indices: cute.Tensor,
            output: cute.Tensor,
            stream: object,
        ) -> None:
            self.kernel(pool, indices, output).launch(
                grid=(indices.shape[0], 1, 1), block=(32, 1, 1), stream=stream
            )

    selected_rows = cute.sym_int()
    indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (selected_rows,), stride_order=(0,), assumed_align=4
    )
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (selected_rows, 128), stride_order=(1, 0), assumed_align=16
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = {}
    for kind, width in (("swa", 528), ("swa_fp8", 512), ("main", 288), ("main_fp8", 512)):
        pool = cute.runtime.make_fake_tensor(
            cutlass.Uint8,
            (cute.sym_int(), width),
            stride=(cute.sym_int64(divisibility=1), 1),
            assumed_align=16,
        )
        compiled[kind] = cute.compile[cute.FrontendNext](
            _DecodeProbe(kind),
            pool,
            indices,
            output,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )
    return compiled


def _pool(kind: _Kind, stride: int, rows: int = 5) -> torch.Tensor:
    layouts = {
        "swa": (528, 512, 16),
        "swa_fp8": (512, 512, 0),
        "main": (288, 256, 32),
        "main_fp8": (512, 512, 0),
    }
    width, payload_bytes, groups = layouts[kind]
    poison = 255 if kind in ("swa_fp8", "main_fp8") else 0xA5
    owner = torch.full((rows, stride), poison, device="cuda", dtype=torch.uint8)
    pool = owner[:, :width]
    row = torch.arange(rows, device=pool.device, dtype=torch.int32)[:, None]
    column = torch.arange(payload_bytes, device=pool.device, dtype=torch.int32)[None, :]
    payload = (column * 29 + row * 17) % 256
    if kind in ("swa", "swa_fp8", "main_fp8"):
        # All finite E4M3 encodings, including signed zeros and subnormals.
        payload = torch.where((payload & 127) == 127, payload - 1, payload)
        scales = [0, 1, 120, 124, 126, 127, 128, 130]
    else:
        # Nibbles cover every E2M1 encoding in both low/high positions.
        scales = [0, 128, 1, 129, 16, 144, 51, 179, 56, 184, 95, 223, 126, 254]
    pool[:, :payload_bytes] = payload.to(torch.uint8)
    if groups:
        scale_codes = torch.tensor(scales, device=pool.device, dtype=torch.uint8)
        group = torch.arange(groups, device=pool.device)[None, :]
        pool[:, payload_bytes:] = scale_codes[(group + row * 5) % len(scales)]
    return pool


def _decode_e4m3(encoded: torch.Tensor) -> torch.Tensor:
    """Decode the format arithmetically, without production conversion helpers."""
    bits = encoded.to(torch.int32)
    magnitude = bits & 127
    exponent, fraction = magnitude >> 3, magnitude & 7
    normal = (1.0 + fraction.float() / 8.0) * torch.exp2(exponent.float() - 7.0)
    value = torch.where(exponent == 0, fraction.float() * 2.0**-9, normal)
    value = torch.where(magnitude == 127, float("nan"), value)
    return torch.where((bits & 128) != 0, -value, value)


def _expected_bytes(pool: torch.Tensor, kind: _Kind) -> torch.Tensor:
    """Independently decode at unit tensor scale in FP32, then round/saturate to E4M3."""
    rows = pool.shape[0]
    if kind in ("swa_fp8", "main_fp8"):
        payload = _decode_e4m3(pool[:, :512]).reshape(rows, 16, 32)
        scales = torch.ones((rows, 16), device=pool.device)
    elif kind == "swa":
        payload = _decode_e4m3(pool[:, :512]).reshape(rows, 16, 32)
        scale_codes = pool[:, 512:].to(torch.int32)
        scales = torch.ldexp(torch.ones_like(scale_codes, dtype=torch.float32), scale_codes - 127)
        scales = torch.where(scale_codes == 255, float("nan"), scales)
    else:
        packed = pool[:, :256].to(torch.int32)
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=pool.device)
        payload = levels[codes & 7]
        payload = torch.where((codes & 8) != 0, -payload, payload).reshape(rows, 32, 16)
        scales = _decode_e4m3(pool[:, 256:])
    decoded = (payload * scales[..., None]).reshape(rows, 512)
    return decoded.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _probe_output(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    owner = torch.full(
        (indices.numel() + 2, 128), 0x25252525, device=indices.device, dtype=torch.int32
    )
    return owner, owner[1:-1]


def _assert_probe(
    output: torch.Tensor,
    owner: torch.Tensor,
    pool: torch.Tensor,
    kind: _Kind,
    indices: torch.Tensor,
) -> None:
    expected = torch.zeros((indices.numel(), 512), device=pool.device, dtype=torch.uint8)
    valid = (indices >= 0) & (indices < pool.shape[0])
    if pool.shape[0]:
        expected[valid] = _expected_bytes(pool, kind)[indices[valid].long()]
    torch.testing.assert_close(output.view(torch.uint8), expected, atol=0, rtol=0)
    for guard in (owner[0], owner[-1]):
        torch.testing.assert_close(guard, torch.full_like(guard, 0x25252525), atol=0, rtol=0)


@pytest.mark.parametrize(("kind", "stride"), _POOL_LAYOUTS)
@torch.inference_mode()
def test_prims_ts_fp8_producer_bytes(
    _producer_probes: dict[_Kind, _Probe], kind: _Kind, stride: int
) -> None:
    pool = _pool(kind, stride)
    indices = torch.tensor(
        [0, 1, 4, 0, 2, -1, -2147483648, 5, 6, 2147483647], device=pool.device, dtype=torch.int32
    )
    owner, output = _probe_output(indices)
    _producer_probes[kind](pool, indices, output)
    _assert_probe(output, owner, pool, kind, indices)
    assert pool.stride(0) == stride
    assert torch.count_nonzero(output[0]) > 0
    torch.testing.assert_close(output[0], output[3], atol=0, rtol=0)


@pytest.mark.parametrize("scale_byte", [0, 1, 254])
@torch.inference_mode()
def test_prims_ts_fp8_producer_ue8m0_extremes(
    _producer_probes: dict[_Kind, _Probe], scale_byte: int
) -> None:
    pool = _pool("swa", 609)
    pool[:, 512:].fill_(scale_byte)
    indices = torch.tensor([0, 1, 4], device=pool.device, dtype=torch.int32)
    owner, output = _probe_output(indices)
    _producer_probes["swa"](pool, indices, output)
    _assert_probe(output, owner, pool, "swa", indices)
    # At unit tensor scale, byte zero (2**-127) and one flush every payload to
    # zero, and byte 254's FP32 product overflows but saturates to +-448.
    decoded = output.view(torch.uint8).view(torch.float8_e4m3fn).float()
    assert torch.isfinite(decoded).all()
    if scale_byte == 254:
        assert decoded.abs().max() == 448
    else:
        assert torch.count_nonzero(decoded) == 0


@torch.inference_mode()
def test_prims_ts_fp8_producer_empty_main(_producer_probes: dict[_Kind, _Probe]) -> None:
    pool = _pool("main", 321, rows=1)[:0]
    indices = torch.tensor(
        [0, -1, 1, 2147483647, -2147483648], device=pool.device, dtype=torch.int32
    )
    owner, output = _probe_output(indices)
    _producer_probes["main"](pool, indices, output)
    _assert_probe(output, owner, pool, "main", indices)


@pytest.mark.parametrize("stride", [512, 513])
@torch.inference_mode()
def test_prims_ts_fp8_producer_empty_ordinary_swa(
    _producer_probes: dict[_Kind, _Probe], stride: int
) -> None:
    pool = _pool("swa_fp8", stride, rows=1)[:0]
    indices = torch.tensor(
        [0, -1, 1, 2147483647, -2147483648], device=pool.device, dtype=torch.int32
    )
    owner, output = _probe_output(indices)
    _producer_probes["swa_fp8"](pool, indices, output)
    _assert_probe(output, owner, pool, "swa_fp8", indices)


@pytest.mark.parametrize("kind", ["swa", "main"])
@torch.inference_mode()
def test_prims_ts_fp8_producer_nan_scale(
    _producer_probes: dict[_Kind, _Probe], kind: _Kind
) -> None:
    pool = _pool(kind, 609 if kind == "swa" else 321)
    pool[:, 512 if kind == "swa" else 256 :].fill_(255 if kind == "swa" else 127)
    indices = torch.tensor([0, -1, pool.shape[0]], device=pool.device, dtype=torch.int32)
    owner, output = _probe_output(indices)
    _producer_probes[kind](pool, indices, output)
    # E4M3 has two NaN encodings; the NaN sign is not part of this contract.
    assert ((output[0].view(torch.uint8) & 127) == 127).all()
    assert torch.count_nonzero(output[1:]) == 0
    for guard in (owner[0], owner[-1]):
        torch.testing.assert_close(guard, torch.full_like(guard, 0x25252525), atol=0, rtol=0)


@pytest.mark.parametrize("kind", ["swa", "swa_fp8", "main", "main_fp8"])
@torch.inference_mode()
def test_prims_ts_fp8_producer_graph_live_rows(
    _producer_probes: dict[_Kind, _Probe], kind: _Kind
) -> None:
    stride = {"swa": 609, "swa_fp8": 513, "main": 321, "main_fp8": 513}[kind]
    pool = _pool(kind, stride)
    indices = torch.tensor([0, 1, 4, -1], device=pool.device, dtype=torch.int32)
    owner, output = _probe_output(indices)
    probe = _producer_probes[kind]
    probe(pool, indices, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probe(pool, indices, output)
    before = output.clone()
    # The same executable must reread live indices and pool bytes on replay.
    addresses = tuple(tensor.data_ptr() for tensor in (pool, indices, output))
    indices[1] = 0
    # Flip payload signs only; flipped scale bytes could encode NaN groups.
    pool[:, : 256 if kind == "main" else 512].bitwise_xor_(128)
    if kind in ("swa_fp8", "main_fp8"):
        indices[2] = pool.shape[0]
    graph.replay()
    _assert_probe(output, owner, pool, kind, indices)
    assert not torch.equal(output, before)
    assert tuple(tensor.data_ptr() for tensor in (pool, indices, output)) == addresses


@pytest.mark.parametrize(
    ("query_band", "heads", "swa_stride", "main_stride", "swa_cache_format"),
    [
        pytest.param("split", 64, 609, 321, "swa", id="split-64-609-321"),
        pytest.param("split", 65, 609, 321, "swa", id="split-65-609-321"),
        pytest.param("split", 127, 609, 321, "swa", id="split-127-609-321"),
        pytest.param("unsplit", 3, 528, 320, "swa", id="unsplit-3-528-320"),
        ("split", 65, 513, 321, "swa_fp8"),
        ("unsplit", 3, 512, 320, "swa_fp8"),
        ("vsplit2", 64, 513, 321, "swa_fp8"),
    ],
)
@torch.inference_mode()
def test_prims_ts_fp8_attention_matches_native(
    query_band: Literal["split", "unsplit", "vsplit2"],
    heads: int,
    swa_stride: int,
    main_stride: int,
    swa_cache_format: Literal["swa", "swa_fp8"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode as native
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import (
        PackedSparseMLA,
        _resolve_launch,
    )

    torch.manual_seed(9341)
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # A few rows split V four ways, sm_count // 8 rows two ways and a full wave not at all.
    rows = {"split": 4, "vsplit2": sm_count // 8, "unsplit": sm_count + 1}[query_band]
    swa_width, main_width = 137, 149
    launch = _resolve_launch(device, rows, heads, swa_width, main_width, swa_cache_format)
    workspace = torch.empty(launch.workspace_bytes, device=device, dtype=torch.uint8)
    packed = PackedSparseMLA(workspace)
    packed.plan(
        device,
        rows,
        heads,
        max_topk=swa_width,
        max_extra_topk=main_width,
        has_sinks=True,
        swa_cache_format=swa_cache_format,
    )
    state = packed._state
    assert state is not None
    assert state.buffers["q8"].dtype == torch.float8_e4m3fn
    assert (
        state.buffers["q8"].untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    )
    assert (state.split_kv > 1) == (query_band != "unsplit")
    assert launch.profile.head_dim_ctas == {"split": 4, "vsplit2": 2, "unsplit": 1}[query_band]
    assert state.spec.kernel._make_config().sparse_reuse_kv

    # The native wrapper's heuristic otherwise chooses a different schedule.
    # Use exactly the packed profile, including split count and native KV8/Q2
    # defaults (no explicit reuse_kv_stages override), without changing native code.
    monkeypatch.setattr(native, "select_sparse_mla_profile", lambda **_: launch.profile)
    baseline = native.BatchSparseMLADecodePagedTSWrapper()
    baseline.plan(
        device,
        rows,
        heads,
        max_topk=swa_width,
        max_extra_topk=main_width,
        q_dtype=torch.float8_e4m3fn,
        use_attention_sinks=True,
        return_lse=True,
    )
    assert baseline._state["requested_splits"] == state.split_kv

    q = (torch.randn((rows, 1, heads, 512), device=device) * 0.125).bfloat16()
    swa, main = _pool(swa_cache_format, swa_stride, 13), _pool("main", main_stride, 11)
    if swa_cache_format == "swa":
        swa[:, 512:] = (120 + torch.arange(16, device=device) % 5).to(torch.uint8)
    main_scales = torch.tensor(
        [16, 160, 48, 184, 64, 200, 80, 208], device=device, dtype=torch.uint8
    )
    main[:, 256:] = main_scales[torch.arange(32, device=device) % main_scales.numel()]
    query_row = torch.arange(rows, device=device, dtype=torch.int32)[:, None]
    si = (torch.arange(swa_width, device=device, dtype=torch.int32)[None, :] + query_row * 3) % 13
    mi = (torch.arange(main_width, device=device, dtype=torch.int32)[None, :] + query_row * 5) % 11
    si[:, 0], si[:, 1], si[:, 2], si[:, 3], si[:, 4] = 0, -1, 0, 13, 2147483647
    mi[:, 0], mi[:, 1], mi[:, 2], mi[:, 3] = 0, 0, -1, 11
    sl = torch.full((rows,), swa_width, device=device, dtype=torch.int32)
    ml = torch.full((rows,), main_width, device=device, dtype=torch.int32)
    sl[1::4], sl[2::4], sl[3::4] = 17, 0, 0
    ml[1::4], ml[2::4], ml[3::4] = 0, main_width - 11, 0
    softmax_scale, output_scale = 512**-0.5, 0.75
    sinks = torch.linspace(-1, 1, heads, device=device)
    output_owner = torch.full((rows + 2, 1, heads, 512), 7.5, device=device, dtype=torch.bfloat16)
    output = output_owner[1:-1]
    output.fill_(torch.nan)
    expected = torch.empty_like(q)
    expected_lse = torch.empty((rows, 1, heads), device=device)

    def run_packed() -> None:
        packed.run(
            q,
            swa,
            si,
            sl,
            extra_kv_cache=main,
            extra_sparse_indices=mi,
            extra_sparse_lens=ml,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            sinks=sinks,
            out=output,
        )

    def assert_native() -> None:
        # Q8 is Q rounded RN/satfinite at unit scale.
        expected_q8 = q.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        torch.testing.assert_close(
            state.buffers["q8"].view(torch.uint8), expected_q8, atol=0, rtol=0
        )
        # Dense E4M3 pools are an independent test oracle, never a production
        # staging path. Native gather has no physical-pool bound, so use -1 for
        # the same entries masked by the actual packed producer's bounds check.
        native_swa = (
            _expected_bytes(swa, swa_cache_format).view(torch.float8_e4m3fn).view(-1, 1, 512)
        )
        native_main = _expected_bytes(main, "main").view(torch.float8_e4m3fn).view(-1, 1, 512)
        native_si = torch.where((si >= 0) & (si < swa.shape[0]), si, -1)
        native_mi = torch.where((mi >= 0) & (mi < main.shape[0]), mi, -1)
        baseline.run(
            state.buffers["q8"],
            native_swa,
            native_si,
            sl,
            extra_kv_cache=native_main,
            extra_indices=native_mi,
            extra_lengths=ml,
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            attention_sinks=sinks,
            out=expected,
            lse=expected_lse,
            validate=False,
        )
        # Both paths use identical FP8 Q/K/V, schedules and reduction splits;
        # only the packed producer differs. The external interface stays BF16.
        assert q.dtype == output.dtype == expected.dtype == torch.bfloat16
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(state.buffers["lse"], expected_lse, atol=0.002, rtol=0.002)
        assert torch.count_nonzero(output[3::4]) == 0
        for guard in (output_owner[0], output_owner[-1]):
            torch.testing.assert_close(guard, torch.full_like(guard, 7.5), atol=0, rtol=0)

    run_packed()
    assert_native()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        run_packed()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_packed()
    # Replay rereads Q, both pools, both index lists and the sinks; the host
    # scales stay as captured. The largest MAIN and UE8M0 SWA scales saturate
    # products beyond 448 in both the producer and the oracle.
    if swa_cache_format == "swa_fp8":
        swa.bitwise_xor_(128)
        main[:, 256:].fill_(126)
    else:
        swa[:, 512:].fill_(129)
    q.mul_(0.5)
    si[:, 0] = 1
    mi[:, 0] = 2
    sinks.add_(0.25)
    graph.replay()
    assert_native()


def _main_bank_case(
    page_dtype: torch.dtype, stride: int
) -> tuple[torch.Tensor, object, torch.Tensor]:
    """A MAIN pool, a three-request bank plan over it and each bank row's physical row."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2PrimsTSMainBank

    pool = _pool("main", stride, rows=37)
    page_size, base, counts = 8, 6, [29, 0, 26]
    # Request 0 ends on an unbacked page and crosses the pool end on page 4;
    # request 1 owns no rows; request 2 maps a page beyond the pool.
    page_rows = [[3, 0, 4, -1], [0, 0, 0, 0], [1, 9, -1, 2]]
    pages = torch.tensor(page_rows, device=pool.device, dtype=page_dtype)
    starts = [base, base + 29, base + 29, base + 55]
    plan = CSA2PrimsTSMainBank(
        offsets=torch.tensor(starts, device=pool.device, dtype=torch.int64),
        base=base,
        rows=sum(counts),
        pages=pages,
        page_size=page_size,
        num_requests=len(counts),
    )
    physical = []
    for request, count in enumerate(counts):
        for logical in range(count):
            page = page_rows[request][logical // page_size]
            row = page * page_size + logical % page_size
            physical.append(row if page >= 0 and row < pool.shape[0] else -1)
    return pool, plan, torch.tensor(physical, device=pool.device, dtype=torch.int32)


@pytest.mark.parametrize("stride", [288, 320, 321])
@pytest.mark.parametrize("page_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prims_ts_main_bank_matches_producer(
    _producer_probes: dict[_Kind, _Probe], stride: int, page_dtype: torch.dtype
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import decode_main_bank

    pool, plan, physical = _main_bank_case(page_dtype, stride)
    owner = torch.full((plan.rows + 2, 512), 0x5A, device=pool.device, dtype=torch.uint8)
    bank = owner[1:-1]
    decode_main_bank(pool, plan, bank)
    # The producer's own MAIN decode of each bank row's physical record; unbacked
    # pages and rows beyond the pool decode to zero rows.
    probe_owner, probe = _probe_output(physical)
    _producer_probes["main"](pool, physical, probe)
    _assert_probe(probe, probe_owner, pool, "main", physical)
    torch.testing.assert_close(bank, probe.view(torch.uint8), atol=0, rtol=0)
    assert torch.count_nonzero(bank[physical >= 0]) > 0
    assert torch.count_nonzero(bank[physical < 0]) == 0
    for guard in (owner[0], owner[-1]):
        torch.testing.assert_close(guard, torch.full_like(guard, 0x5A), atol=0, rtol=0)


@pytest.mark.parametrize("page_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_prims_ts_main_bank_decodes_marked_rows(page_dtype: torch.dtype) -> None:
    """With marks, exactly the marked rows decode as a full decode would, and their marks clear."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        decode_main_bank,
        main_bank_marks,
    )

    pool, plan, physical = _main_bank_case(page_dtype, 321)
    full = torch.empty((plan.rows, 512), device=pool.device, dtype=torch.uint8)
    decode_main_bank(pool, plan, full)
    arenas = {}
    marks = main_bank_marks(arenas, pool.device, plan.rows)
    assert marks.shape == (plan.rows,) and torch.count_nonzero(marks) == 0
    # Both requests that own rows, request 0's unbacked last row and the bank's
    # last row; any nonzero byte marks a row.
    chosen = [0, 3, 28, 29, 40, plan.rows - 1]
    assert physical[28] < 0
    marks[chosen] = torch.tensor([1, 7, 1, 1, 255, 1], device=pool.device, dtype=torch.uint8)
    selected = torch.zeros(plan.rows, device=pool.device, dtype=torch.bool)
    selected[chosen] = True
    bank = torch.full_like(full, 0x5A)
    decode_main_bank(pool, plan, bank, marks)
    torch.testing.assert_close(bank[selected], full[selected], atol=0, rtol=0)
    assert bool((bank[~selected] == 0x5A).all())
    assert torch.count_nonzero(marks) == 0
    # Cleared marks decode nothing more.
    bank.fill_(0x5A)
    decode_main_bank(pool, plan, bank, marks)
    assert bool((bank == 0x5A).all())
    # Retained marks grow zero-filled, and must cover exactly the bank's rows.
    grown = main_bank_marks(arenas, pool.device, 300)
    assert grown.shape == (300,) and torch.count_nonzero(grown) == 0
    with pytest.raises(ValueError, match="marks"):
        decode_main_bank(pool, plan, bank, marks[:-1])


@torch.inference_mode()
def test_prims_ts_main_bank_nan_scale() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import decode_main_bank

    pool, plan, physical = _main_bank_case(torch.int32, 321)
    pool[:, 256:].fill_(127)
    bank = torch.empty((plan.rows, 512), device=pool.device, dtype=torch.uint8)
    decode_main_bank(pool, plan, bank)
    # E4M3 has two NaN encodings; the NaN sign is not part of this contract.
    assert ((bank[physical >= 0] & 127) == 127).all()
    assert torch.count_nonzero(bank[physical < 0]) == 0


@pytest.mark.parametrize("query_band", ["split", "unsplit"])
@torch.inference_mode()
def test_prims_ts_main_bank_attention_matches_packed(
    query_band: Literal["split", "unsplit"],
) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import decode_main_bank
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import (
        PackedSparseMLA,
        _resolve_launch,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2PrimsTSMainBank

    torch.manual_seed(9343)
    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    rows = {"split": 4, "unsplit": sm_count + 1}[query_band]
    heads, swa_width, main_width = 64, 137, 149
    plans = {}
    for main_format in ("main", "main_fp8"):
        launch = _resolve_launch(
            device, rows, heads, swa_width, main_width, "swa_fp8", main_cache_format=main_format
        )
        packed = PackedSparseMLA(
            torch.empty(launch.workspace_bytes, device=device, dtype=torch.uint8)
        )
        packed.plan(
            device,
            rows,
            heads,
            max_topk=swa_width,
            max_extra_topk=main_width,
            has_sinks=True,
            swa_cache_format="swa_fp8",
            main_cache_format=main_format,
        )
        plans[main_format] = packed
    assert (plans["main"]._state.split_kv > 1) == (query_band == "split")
    q = (torch.randn((rows, 1, heads, 512), device=device) * 0.125).bfloat16()
    swa, main = _pool("swa_fp8", 513, 13), _pool("main", 321, 11)
    main_scales = torch.tensor(
        [16, 160, 48, 184, 64, 200, 80, 208], device=device, dtype=torch.uint8
    )
    main[:, 256:] = main_scales[torch.arange(32, device=device) % main_scales.numel()]
    # One request whose pages map every pool row to the same bank row.
    bank_plan = CSA2PrimsTSMainBank(
        offsets=torch.tensor([0, main.shape[0]], device=device, dtype=torch.int64),
        base=0,
        rows=main.shape[0],
        pages=torch.arange(main.shape[0], device=device, dtype=torch.int32)[None, :],
        page_size=1,
        num_requests=1,
    )
    bank = torch.empty((bank_plan.rows, 512), device=device, dtype=torch.uint8)
    decode_main_bank(main, bank_plan, bank)
    query_row = torch.arange(rows, device=device, dtype=torch.int32)[:, None]
    si = (torch.arange(swa_width, device=device, dtype=torch.int32)[None, :] + query_row * 3) % 13
    mi = (torch.arange(main_width, device=device, dtype=torch.int32)[None, :] + query_row * 5) % 11
    si[:, 0], si[:, 1], si[:, 2], si[:, 3], si[:, 4] = 0, -1, 0, 13, 2147483647
    mi[:, 0], mi[:, 1], mi[:, 2], mi[:, 3] = 0, 0, -1, 11
    sl = torch.full((rows,), swa_width, device=device, dtype=torch.int32)
    ml = torch.full((rows,), main_width, device=device, dtype=torch.int32)
    sl[1::4], sl[2::4], sl[3::4] = 17, 0, 0
    ml[1::4], ml[2::4], ml[3::4] = 0, main_width - 11, 0
    sinks = torch.linspace(-1, 1, heads, device=device)
    outputs, lses = {}, {}
    for main_format, source in (("main", main), ("main_fp8", bank)):
        outputs[main_format] = torch.full_like(q, torch.nan)
        plans[main_format].run(
            q,
            swa,
            si,
            sl,
            extra_kv_cache=source,
            extra_sparse_indices=mi,
            extra_sparse_lens=ml,
            softmax_scale=512**-0.5,
            output_scale=0.75,
            sinks=sinks,
            out=outputs[main_format],
        )
        lses[main_format] = plans[main_format]._state.buffers["lse"].clone()
    # Identical operands, schedule and reduction: the bank changes only how they arrive.
    assert torch.isfinite(outputs["main"]).all()
    torch.testing.assert_close(outputs["main_fp8"], outputs["main"], atol=0, rtol=0)
    torch.testing.assert_close(lses["main_fp8"], lses["main"], atol=0, rtol=0)
    # Q already rounded RN/satfinite to unit-scale E4M3 is read as is.
    q8 = q.float().clamp(-448, 448).to(torch.float8_e4m3fn)
    prequantized = torch.full_like(q, torch.nan)
    plans["main_fp8"].run(
        q8,
        swa,
        si,
        sl,
        extra_kv_cache=bank,
        extra_sparse_indices=mi,
        extra_sparse_lens=ml,
        softmax_scale=512**-0.5,
        output_scale=0.75,
        sinks=sinks,
        out=prequantized,
    )
    torch.testing.assert_close(prequantized, outputs["main"], atol=0, rtol=0)


@pytest.mark.parametrize("band", ["vsplit", "split", "one_wave", "persistent", "waves"])
@torch.inference_mode()
def test_prims_ts_packed_schedule_matches_plain_keep_ab(
    band: Literal["vsplit", "split", "one_wave", "persistent", "waves"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected schedule computes exactly what plain nonpersistent Keep-AB does.

    Beyond one resident wave the static persistent scheduler runs single-split tiles;
    every grid also caches sparse coordinates once and pairs correction rows.
    """
    import dataclasses

    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    heads, swa_width, main_width = 64, 128, 149
    capacity = 128 + (main_width + 127) // 128 * 128
    # One CTA per SM: grids beyond one wave persist, with one split and one V owner.
    for rows, scheduler in ((sm_count, "nonpersistent"), (sm_count + 1, "static")):
        profile = packed._select_packed_profile(rows, heads, capacity, sm_count)
        assert profile.scheduler == scheduler, rows
        assert profile.uniform_offset_cache and profile.paired_correction
    # Half a wave still splits KV but leaves no room to also split V.
    rows = {
        "vsplit": 4,
        "split": sm_count // 2,
        "one_wave": sm_count,
        "persistent": sm_count + 1,
        "waves": 2 * sm_count + 3,
    }[band]
    selected = packed._select_packed_profile(rows, heads, capacity, sm_count)
    assert (selected.split_kv > 1) == (band in ("vsplit", "split"))
    assert (selected.head_dim_ctas > 1) == (band == "vsplit")
    assert (selected.scheduler == "static") == (band in ("persistent", "waves"))
    torch.manual_seed(9347)
    q = (torch.randn((rows, 1, heads, 512), device=device) * 0.125).bfloat16()
    swa, main = _pool("swa_fp8", 513, 13), _pool("main", 321, 11)
    query_row = torch.arange(rows, device=device, dtype=torch.int32)[:, None]
    si = (torch.arange(swa_width, device=device, dtype=torch.int32)[None, :] + query_row * 3) % 13
    mi = (torch.arange(main_width, device=device, dtype=torch.int32)[None, :] + query_row * 5) % 11
    si[:, 1], mi[:, 2], mi[:, 3] = -1, -1, 11
    sl = torch.full((rows,), swa_width, device=device, dtype=torch.int32)
    ml = torch.full((rows,), main_width, device=device, dtype=torch.int32)
    sl[1::4], ml[2::4], sl[3::4], ml[3::4] = 17, main_width - 11, 0, 0
    sinks = torch.linspace(-1, 1, heads, device=device)
    plain = dataclasses.replace(
        selected, scheduler="nonpersistent", uniform_offset_cache=False, paired_correction=False
    )
    choose = packed._select_packed_profile
    outputs, lses = {}, {}
    for name, profile in (("selected", selected), ("plain", plain)):
        monkeypatch.setattr(packed, "_select_packed_profile", lambda *_, _p=profile: _p)
        launch = packed._resolve_launch(device, rows, heads, swa_width, main_width, "swa_fp8")
        assert launch.profile == profile
        wrapper = packed.PackedSparseMLA(
            torch.empty(launch.workspace_bytes, device=device, dtype=torch.uint8)
        )
        wrapper.plan(
            device,
            rows,
            heads,
            max_topk=swa_width,
            max_extra_topk=main_width,
            swa_cache_format="swa_fp8",
        )
        outputs[name] = torch.full_like(q, torch.nan)
        wrapper.run(
            q,
            swa,
            si,
            sl,
            extra_kv_cache=main,
            extra_sparse_indices=mi,
            extra_sparse_lens=ml,
            softmax_scale=512**-0.5,
            sinks=sinks,
            out=outputs[name],
        )
        lses[name] = wrapper._state.buffers["lse"].clone()
    monkeypatch.setattr(packed, "_select_packed_profile", choose)
    assert torch.isfinite(outputs["selected"]).all()
    torch.testing.assert_close(outputs["selected"], outputs["plain"], atol=0, rtol=0)
    torch.testing.assert_close(lses["selected"], lses["plain"], atol=0, rtol=0)


def _valid_prefix_lists(
    lengths: torch.Tensor, capacity: int, pool_rows: int, step: int
) -> torch.Tensor:
    """Compact lists: in-pool rows, duplicates included, before each live length; -1 after."""
    query_row = torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.int32)[:, None]
    column = torch.arange(capacity, device=lengths.device, dtype=torch.int32)[None, :]
    indices = (column * 7 + query_row * step) % pool_rows
    return torch.where(column < lengths[:, None], indices, -1).contiguous()


def _gather_case(rows: int, heads: int, has_main: bool, has_sinks: bool, swa, main) -> dict:
    device = swa.device
    swa_width, main_width = 128, 149
    sl = torch.full((rows,), swa_width, device=device, dtype=torch.int32)
    sl[1::4] = 17
    ml = torch.full((rows,), main_width, device=device, dtype=torch.int32)
    ml[2::4] = main_width - 11
    if has_sinks:
        # Rows that attend only to the sinks.
        sl[3::4] = 0
        ml[3::4] = 0
    torch.manual_seed(5521)
    case = dict(
        q=(torch.randn((rows, 1, heads, 512), device=device) * 0.125).bfloat16(),
        swa_cache=swa,
        swa_indices=_valid_prefix_lists(sl, swa_width, swa.shape[0], 3),
        swa_lengths=sl,
        sinks=torch.linspace(-1, 1, heads, device=device) if has_sinks else None,
    )
    if has_main:
        case.update(
            extra_kv_cache=main,
            extra_sparse_indices=_valid_prefix_lists(ml, main_width, main.shape[0], 5),
            extra_sparse_lens=ml,
        )
    return case


def _run_gather_case(packed, case: dict, *, valid_prefix: bool) -> tuple:
    rows, _, heads, _ = case["q"].shape
    has_main = "extra_kv_cache" in case
    max_extra_topk = case["extra_sparse_indices"].shape[1] if has_main else 0
    device = case["q"].device
    size = packed.get_packed_sparse_mla_workspace_size(
        device, rows, heads, max_topk=128, max_extra_topk=max_extra_topk
    )
    wrapper = packed.PackedSparseMLA(torch.empty(size, device=device, dtype=torch.uint8))
    wrapper.plan(
        device,
        rows,
        heads,
        max_topk=128,
        max_extra_topk=max_extra_topk,
        has_sinks=case["sinks"] is not None,
        swa_cache_format="swa_fp8",
        main_cache_format="main_fp8" if has_main else "main",
        valid_prefix=valid_prefix,
    )
    out = torch.full_like(case["q"], torch.nan)
    inputs = {key: value for key, value in case.items() if key != "q"}
    wrapper.run(case["q"], **inputs, softmax_scale=512**-0.5, out=out, validate=True)
    return wrapper, out, wrapper._state.buffers["lse"].clone()


def _force_gather(packed, monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """Run the gather kernel on the producer's own schedule, with overrides, for every grid."""
    monkeypatch.setattr(
        packed,
        "_select_gather_profile",
        lambda rows, heads, capacity, sm: replace(
            packed._select_packed_profile(rows, heads, capacity, sm), **overrides
        ),
    )


_WS_GATHER = dict(
    ws_mma=True,
    ws_store_cols=64,
    ws_tma_store=True,
    page_pipeline_stages=2,
    ws_tail_staged_stripes=5,
)
# Plans with MAIN rows (capacity above the 256-row minimum) also stage Q in the
# K/V ring and retain ten D128 K/V stages.
_WS_MULTI_TILE = dict(ws_q_in_kv=True, reuse_kv_stages=10)


@pytest.mark.parametrize("band", ["vsplit", "split", "one_wave", "persistent"])
@pytest.mark.parametrize("has_main", [True, False])
@pytest.mark.parametrize("has_sinks", [True, False])
@torch.inference_mode()
def test_prims_ts_gather_matches_packed_producer(
    band: Literal["vsplit", "split", "one_wave", "persistent"],
    has_main: bool,
    has_sinks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TMA gather4 over plain E4M3 rows computes exactly what the packed producer does."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    rows = {"vsplit": 4, "split": 32, "one_wave": sm_count, "persistent": sm_count + 1}[band]
    _force_gather(packed, monkeypatch)
    # Dense rows of every finite E4M3 encoding, signed zeros and subnormals included.
    case = _gather_case(
        rows, 64, has_main, has_sinks, _pool("swa_fp8", 512, 300), _pool("main_fp8", 512, 200)
    )
    gathered, out, lse = _run_gather_case(packed, case, valid_prefix=True)
    produced, expected, expected_lse = _run_gather_case(packed, case, valid_prefix=False)
    assert gathered._state.gather is not None and produced._state.gather is None
    assert (gathered._state.gather.split_kv > 1) == (band in ("vsplit", "split"))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)


@pytest.mark.parametrize("band", ["one_wave", "persistent", "multi_tile"])
@pytest.mark.parametrize("has_main", [True, False])
@pytest.mark.parametrize("has_sinks", [True, False])
@torch.inference_mode()
def test_prims_ts_gather_ws_matches_packed_producer(
    band: Literal["one_wave", "persistent", "multi_tile"],
    has_main: bool,
    has_sinks: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 64-head WS gather schedule (TMEM Q, TMA output) computes exactly what the producer does."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    rows = {"one_wave": sm_count, "persistent": sm_count + 1, "multi_tile": 4 * sm_count + 3}[band]
    _force_gather(packed, monkeypatch, **_WS_GATHER, **(_WS_MULTI_TILE if has_main else {}))
    case = _gather_case(
        rows, 64, has_main, has_sinks, _pool("swa_fp8", 512, 300), _pool("main_fp8", 512, 200)
    )
    gathered, out, lse = _run_gather_case(packed, case, valid_prefix=True)
    produced, expected, expected_lse = _run_gather_case(packed, case, valid_prefix=False)
    assert gathered._state.gather is not None and produced._state.gather is None
    assert gathered._state.gather.split_kv == 1
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)


@pytest.mark.parametrize(("swa_stride", "main_stride"), [(1024, 512), (512, 513)])
@torch.inference_mode()
def test_prims_ts_gather_plan_reads_padded_pools_with_producer(
    swa_stride: int, main_stride: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows that are not dense 512-byte records take the producer kernel they need."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    _force_gather(packed, monkeypatch)
    case = _gather_case(
        8, 64, True, True, _pool("swa_fp8", swa_stride, 300), _pool("main_fp8", main_stride, 200)
    )
    gathered, out, _ = _run_gather_case(packed, case, valid_prefix=True)
    _, expected, _ = _run_gather_case(packed, case, valid_prefix=False)
    aligned = swa_stride % 16 == 0 and main_stride % 16 == 0
    assert gathered._state.compiled is None and set(gathered._packed) == {aligned}
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_prims_ts_gather_plan_validates_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    _force_gather(packed, monkeypatch)
    case = _gather_case(4, 64, True, True, _pool("swa_fp8", 512, 300), _pool("main_fp8", 512, 200))
    for key in ("swa_indices", "extra_sparse_indices"):
        holed = dict(case)
        holed[key] = case[key].clone()
        holed[key][0, 1] = -1
        with pytest.raises(ValueError, match="valid_prefix"):
            _run_gather_case(packed, holed, valid_prefix=True)
        # Without the promise, the producer masks the hole.
        _run_gather_case(packed, holed, valid_prefix=False)


def test_prims_ts_gather_profile_selection() -> None:
    """Plain E4M3 rows are gathered by TMA: 64-head single-split, single-V plans on every
    grid, other plans on persistent grids of over four KV tiles per CTA."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    assert packed.gathers_plain_rows("swa_fp8", "main_fp8", 512)
    assert packed.gathers_plain_rows("swa_fp8", "main", 0)
    assert not packed.gathers_plain_rows("swa_fp8", "main", 512)
    assert not packed.gathers_plain_rows("swa", "main_fp8", 512)
    for sm_count in (148, 152):
        for capacity in (640, 256):
            # Sixty-four heads: grids too small for one split and one V owner per tile
            # keep the producer; every larger one takes the WS gather schedule.
            for rows, gathered in (
                (4, False),
                (sm_count // 2, False),
                (sm_count // 2 + 1, True),
                (140, True),
                (sm_count + 1, True),
                (8064, True),
            ):
                profile = packed._select_gather_profile(rows, 64, capacity, sm_count)
                producer = packed._select_packed_profile(rows, 64, capacity, sm_count)
                assert packed._quantizes_output(producer) == gathered, (rows, capacity, sm_count)
                multi_tile = _WS_MULTI_TILE if capacity > 256 else {}
                expected = replace(producer, **_WS_GATHER, **multi_tile) if gathered else None
                assert profile == expected, (rows, capacity, sm_count)
        narrow_bands = {
            # Five KV tiles: every persistent grid gathers.
            640: ((140, False), (sm_count, False), (sm_count + 1, True)),
            # Two KV tiles: persistent grids gather from over four tiles per CTA.
            256: ((sm_count + 1, False), (2 * sm_count, False), (2 * sm_count + 1, True)),
        }
        # Sixteen heads keep the producer's plain schedule.
        for capacity, cases in narrow_bands.items():
            for rows, gathered in cases:
                profile = packed._select_gather_profile(rows, 16, capacity, sm_count)
                narrow = packed._select_packed_profile(rows, 16, capacity, sm_count)
                assert profile == (narrow if gathered else None), (rows, capacity, sm_count)
            assert packed._select_packed_profile(8064, 16, capacity, sm_count).scheduler == "static"


def _rope_table(device: torch.device, positions: int) -> torch.Tensor:
    """FP32 [P, 64] GPT-J table: 32 cosines, then 32 sines, of independent angles."""
    angles = torch.rand((positions, 32), device=device, dtype=torch.float32) * 6.2832
    return torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()


def _run_mxfp8_case(
    packed, case: dict, groups: int, *, valid_prefix: bool
) -> tuple[torch.Tensor, ...]:
    """The plan's BF16 output, and its MXFP8 output written into poisoned buffers."""
    _, bf16, _ = _run_gather_case(packed, case, valid_prefix=valid_prefix)
    rows, _, heads, _ = case["q"].shape
    has_main = "extra_kv_cache" in case
    max_extra_topk = case["extra_sparse_indices"].shape[1] if has_main else 0
    device = case["q"].device
    size = packed.get_packed_sparse_mla_workspace_size(
        device, rows, heads, max_topk=128, max_extra_topk=max_extra_topk
    )
    wrapper = packed.PackedSparseMLA(torch.empty(size, device=device, dtype=torch.uint8))
    wrapper.plan(
        device,
        rows,
        heads,
        max_topk=128,
        max_extra_topk=max_extra_topk,
        has_sinks=case["sinks"] is not None,
        swa_cache_format="swa_fp8",
        main_cache_format="main_fp8" if has_main else "main",
        valid_prefix=valid_prefix,
        output_groups=groups,
    )
    depth = heads * 512 // groups
    data = torch.full((groups, rows, depth), 0x7F, device=device, dtype=torch.uint8)
    scale_rows = (rows + 127) // 128 * 128
    scales = torch.full(
        (groups * scale_rows * depth // 32,), 0xAB, device=device, dtype=torch.uint8
    )
    positions = torch.randint(0, 1000, (rows,), device=device, dtype=torch.int32)
    cos_sin = _rope_table(device, 1000)
    inputs = {key: value for key, value in case.items() if key != "q"}
    returned = wrapper.run(
        case["q"],
        **inputs,
        softmax_scale=512**-0.5,
        out=None,
        validate=True,
        output_quant=(data.view(torch.float8_e4m3fn), scales, positions, cos_sin),
    )
    assert returned.data_ptr() == data.data_ptr()
    return wrapper, bf16, data, scales, positions, cos_sin


@pytest.mark.parametrize("engine", ["gather", "producer", "ws_gather"])
@pytest.mark.parametrize("band", ["one_wave", "persistent", "aligned"])
@pytest.mark.parametrize(("heads", "groups"), [(64, 8), (16, 2)])
@torch.inference_mode()
def test_prims_ts_mxfp8_output_matches_inverse_rope_mxfp8(
    engine: Literal["gather", "producer", "ws_gather"],
    band: Literal["one_wave", "persistent", "aligned"],
    heads: int,
    groups: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused MXFP8 output equals inverse_rope_mxfp8 of the BF16 output, byte for byte.

    ``ws_gather`` stages the quantized output of the 64-head WS schedule for TMA stores;
    the other engines quantize per lane and store directly.
    """
    if engine == "ws_gather" and heads != 64:
        pytest.skip("the WS TMA output needs whole 64-head query tiles")
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import inverse_rope_mxfp8

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    # One resident wave, a persistent grid with padded scale rows, and a 128-row multiple.
    rows = {"one_wave": 140, "persistent": 2 * sm_count + 37, "aligned": 384}[band]
    _force_gather(
        packed,
        monkeypatch,
        **({**_WS_GATHER, **_WS_MULTI_TILE} if engine == "ws_gather" else {}),
    )
    case = _gather_case(
        rows, heads, True, True, _pool("swa_fp8", 512, 300), _pool("main_fp8", 512, 200)
    )
    wrapper, bf16, data, scales, positions, cos_sin = _run_mxfp8_case(
        packed, case, groups, valid_prefix=engine != "producer"
    )
    assert (wrapper._state.gather is not None) == (engine != "producer")
    assert torch.isfinite(bf16).all()
    expected, expected_scales = inverse_rope_mxfp8(
        bf16.view(rows, heads * 512), positions, cos_sin, groups
    )
    assert torch.equal(data, expected.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@torch.inference_mode()
def test_prims_ts_mxfp8_output_plan_contract() -> None:
    """Only single-split, single-V-CTA plans store MXFP8, and their run takes no BF16 output.

    CSA2 offers the MXFP8 output only to plans of the 64-head WS gather schedule.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_packed as packed

    device = torch.device("cuda", 0)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    assert not packed.fuses_mxfp8_output(device, 4, 64, 128, 149)
    assert packed.fuses_mxfp8_output(device, 140, 64, 128, 149)
    assert packed.fuses_mxfp8_output(device, 2 * sm_count + 37, 64, 128, 512)
    assert not packed.fuses_mxfp8_output(device, 8 * sm_count + 37, 16, 128, 512)
    size = packed.get_packed_sparse_mla_workspace_size(device, 4, 64, max_extra_topk=149)
    wrapper = packed.PackedSparseMLA(torch.empty(size, device=device, dtype=torch.uint8))
    with pytest.raises(ValueError, match="single-split"):
        wrapper.plan(device, 4, 64, max_extra_topk=149, output_groups=8)
    case = _gather_case(4, 64, True, True, _pool("swa_fp8", 512, 300), _pool("main_fp8", 512, 200))
    planned, out, _ = _run_gather_case(packed, case, valid_prefix=False)
    inputs = {key: value for key, value in case.items() if key != "q"}
    data = torch.empty((8, 4, 4096), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(8 * 128 * 128, device=device, dtype=torch.uint8)
    quant = (data, scales, torch.zeros(4, device=device, dtype=torch.int32), _rope_table(device, 4))
    with pytest.raises(ValueError, match="output_quant"):
        planned.run(case["q"], **inputs, softmax_scale=512**-0.5, out=out, output_quant=quant)


@pytest.mark.parametrize("stride", [288, 321])
@torch.inference_mode()
def test_prims_ts_main_records_match_main_bank(stride: int) -> None:
    """A selection bank row decodes its record exactly as the request bank decodes that row."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        decode_main_bank,
        decode_main_records,
        main_records_storage,
    )

    pool, plan, physical = _main_bank_case(torch.int32, stride)
    full = torch.empty((plan.rows, 512), device=pool.device, dtype=torch.uint8)
    decode_main_bank(pool, plan, full)
    # Bank rows of records past the pool or negative keep their bytes: nothing reads them.
    extra = torch.tensor([pool.shape[0], -5], device=pool.device, dtype=torch.int32)
    records = torch.cat([physical, extra]).view(-1, 1).contiguous()
    arenas = {}
    bank = main_records_storage(arenas, pool.device, records.numel())
    bank.fill_(0x5A)
    decode_main_records(pool, records, bank, arenas)
    valid = records.view(-1) >= 0
    valid[-2:] = False
    torch.testing.assert_close(
        bank[: plan.rows][valid[: plan.rows]], full[valid[: plan.rows]], atol=0, rtol=0
    )
    assert bool((bank[~valid] == 0x5A).all())
    # The storage is allocated once at its reserved capacity and never moves.
    assert main_records_storage(arenas, pool.device, 7).data_ptr() == bank.data_ptr()
    with pytest.raises(ValueError, match="selection bank"):
        decode_main_records(pool, records, bank[:-1], arenas)


def _dedup_reference(records: torch.Tensor, pool_rows: int) -> torch.Tensor:
    """Each selection's owner: the smallest flat index selecting its pool row, else itself."""
    flat = records.view(-1).tolist()
    first, owners = {}, []
    for index, record in enumerate(flat):
        if 0 <= record < pool_rows:
            owners.append(first.setdefault(record, index))
        else:
            owners.append(index)
    return torch.tensor(owners, dtype=torch.int32).view(records.shape)


@torch.inference_mode()
def test_prims_ts_main_records_dedup() -> None:
    """Deduplicated selection banks decode each selected row once, at its first selection.

    Every selection reads its owner's bank row, which holds exactly what the plain selection
    bank decodes for it. Owner slots carry an epoch, so later calls, eager or replayed from a
    CUDA Graph, with other records never see an earlier call's owners.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        decode_main_records,
        dedup_main_records,
        main_records_storage,
    )

    torch.manual_seed(4111)
    device = torch.device("cuda", torch.cuda.current_device())
    main = _pool("main", 288, 2000)
    rows, width = 24, 512
    arenas, plain_arenas = {}, {}
    bank = main_records_storage(arenas, device, rows * width)
    plain = main_records_storage(plain_arenas, device, rows * width)
    bank_rows = torch.empty((rows, width), device=device, dtype=torch.int32)

    def check(records: torch.Tensor) -> None:
        decode_main_records(main, records, plain, plain_arenas)
        owners = _dedup_reference(records.cpu(), main.shape[0]).to(device)
        torch.testing.assert_close(bank_rows, owners, atol=0, rtol=0)
        valid = ((records >= 0) & (records < main.shape[0])).view(-1)
        read = bank[bank_rows.view(-1).long()]
        torch.testing.assert_close(read[valid], plain[valid], atol=0, rtol=0)

    def case(seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        # Four query rows per request draw from one shared set, as draft tokens do.
        shared = torch.randint(
            0, main.shape[0], (rows // 4, 700), device=device, generator=generator
        )
        pick = torch.randint(0, 700, (rows, width), device=device, generator=generator)
        records = shared[torch.arange(rows, device=device)[:, None] // 4, pick].to(torch.int32)
        records[:, -7:] = -1
        records[3, :5] = main.shape[0] + 9
        return records.contiguous()

    for seed in (1, 2):
        records = case(seed)
        dedup_main_records(main, records, bank_rows, bank, arenas)
        check(records)
    # A captured call claims with the device epoch each replay advances.
    records = case(3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dedup_main_records(main, records, bank_rows, bank, arenas)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dedup_main_records(main, records, bank_rows, bank, arenas)
    for seed in (4, 5):
        records.copy_(case(seed))
        graph.replay()
        torch.cuda.synchronize()
        check(records)
    with pytest.raises(ValueError, match="like records"):
        dedup_main_records(main, records, bank_rows[:-1], bank, arenas)


@pytest.mark.parametrize("mapped", [False, True], ids=["physical", "paged"])
@torch.inference_mode()
def test_prims_ts_compaction_claims_dedup_owners(
    mapped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction claims the owner slots the deduplicated selection bank decodes from.

    Claiming each kept MAIN row while compacting leaves the decode exactly the owners its
    own claim would: every selection reads its first selection's bank row, in eager calls,
    in CUDA Graph replays with other selections, and with the compaction on another stream.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import _prims_ts_bank
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        decode_main_records,
        dedup_main_records,
        main_record_owners,
        main_records_storage,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_metadata import (
        prepare_prims_ts_metadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import _PrimsTSMetadata

    def no_claim_kernel():
        raise AssertionError("the compaction claims the owner slots")

    monkeypatch.setattr(_prims_ts_bank, "_claim_kernel", no_claim_kernel)
    torch.manual_seed(4127)
    device = torch.device("cuda", torch.cuda.current_device())
    main = _pool("main", 288, 2000)
    rows, width, page = 24, 512, 64
    selection = torch.empty((rows, width), device=device, dtype=torch.int32)
    compact = _PrimsTSMetadata(
        torch.empty((rows, 128), device=device, dtype=torch.int32),
        torch.empty((rows,), device=device, dtype=torch.int32),
        torch.empty((rows, width), device=device, dtype=torch.int32),
        torch.empty((rows,), device=device, dtype=torch.int32),
    )
    pages = (main.shape[0] + page - 1) // page
    inputs = CSA2BackendForwardArgs(
        swa_pool=_pool("swa_fp8", 512, 300),
        swa_indices=torch.arange(128, device=device, dtype=torch.int32).repeat(rows, 1),
        main_pool=main,
        topk_indices=selection,
        # Every request's page table maps logical MAIN row i to pool row i.
        main_mapping=(
            torch.arange(pages, device=device, dtype=torch.int32).repeat(rows // 4, 1),
            page,
            main.shape[0],
            torch.arange(rows, device=device, dtype=torch.int32) // 4,
            torch.full((rows,), main.shape[0], device=device, dtype=torch.int32),
        )
        if mapped
        else None,
    )
    arenas, plain_arenas = {}, {}
    bank = main_records_storage(arenas, device, rows * width)
    plain = main_records_storage(plain_arenas, device, rows * width)
    bank_rows = torch.empty((rows, width), device=device, dtype=torch.int32)
    claims = main_record_owners(arenas, device, main.shape[0])

    def case(seed: int) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(seed)
        # Four query rows per request draw from one shared set, as draft tokens do.
        shared = torch.randint(
            0, main.shape[0], (rows // 4, 700), device=device, generator=generator
        )
        pick = torch.randint(0, 700, (rows, width), device=device, generator=generator)
        records = shared[torch.arange(rows, device=device)[:, None] // 4, pick].to(torch.int32)
        # Invalid selections between valid ones, which the compaction drops.
        records[:, 5::31] = -1
        records[3, 7::13] = main.shape[0] + 9
        return records

    def check(source: torch.Tensor) -> None:
        kept = [[v for v in row if 0 <= v < main.shape[0]] for row in source.tolist()]
        expected = torch.tensor([row + [-1] * (width - len(row)) for row in kept], device=device)
        torch.testing.assert_close(compact.extra_indices, expected.int(), atol=0, rtol=0)
        assert compact.extra_lengths.tolist() == [len(row) for row in kept]
        records = compact.extra_indices
        decode_main_records(main, records, plain, plain_arenas)
        owners = _dedup_reference(records.cpu(), main.shape[0]).to(device)
        torch.testing.assert_close(bank_rows, owners, atol=0, rtol=0)
        valid = (records >= 0).view(-1)
        read = bank[bank_rows.view(-1).long()]
        torch.testing.assert_close(read[valid], plain[valid], atol=0, rtol=0)

    def forward() -> None:
        prepare_prims_ts_metadata(inputs, compact, claims=claims)
        dedup_main_records(main, compact.extra_indices, bank_rows, bank, arenas, claimed=True)

    for seed in (1, 2):
        selection.copy_(case(seed))
        forward()
        check(selection)
    # Compaction on a side stream that waits only for the selections.
    selection.copy_(case(3))
    ready, done = torch.cuda.Event(), torch.cuda.Event()
    ready.record()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        side.wait_event(ready)
        prepare_prims_ts_metadata(inputs, compact, claims=claims)
        done.record()
    done.wait()
    dedup_main_records(main, compact.extra_indices, bank_rows, bank, arenas, claimed=True)
    check(selection)
    # A captured call claims with the device epoch each replay advances.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for seed in (4, 5):
        selection.copy_(case(seed))
        graph.replay()
        torch.cuda.synchronize()
        check(selection)
    padded = _PrimsTSMetadata(
        compact.indices,
        compact.lengths,
        torch.empty((rows, width + 8), device=device, dtype=torch.int32)[:, :width],
        compact.extra_lengths,
    )
    with pytest.raises(ValueError, match="MAIN claims"):
        prepare_prims_ts_metadata(inputs, padded, claims=claims)


@torch.inference_mode()
def test_prims_ts_selection_bank_storage_capacity() -> None:
    """The selection bank is sized once for its largest planned tile and never moves under it.

    A request for more rows than the retained storage holds replaces it and keeps the old
    allocation alive for the CUDA Graphs that captured it; the reservation covers the bank of
    the largest generation tile.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        main_bank_reserved_rows,
        main_records_rows_limit,
        main_records_storage,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.prims_ts import (
        workspace_reservation_bytes,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    arenas = {}
    small = main_records_storage(arenas, device, 7)
    assert small.shape == (7, 512)
    # Never smaller than the reserved context bank rows, whatever the requested capacity.
    retained = main_records_storage(arenas, device, 1).untyped_storage()
    assert retained.nbytes() == main_bank_reserved_rows() * 512
    # A larger tile: one allocation at the planned capacity, retaining the old storage.
    rows, capacity = 768 * 512, 2048 * 512
    large = main_records_storage(arenas, device, rows, capacity=capacity)
    assert large.shape == (rows, 512) and large.data_ptr() != small.data_ptr()
    assert large.untyped_storage().nbytes() == capacity * 512
    assert any(
        old.data_ptr() == small.data_ptr()
        for value in arenas.values()
        if isinstance(value, list)
        for old in value
    )
    # Smaller tiles then reuse the large storage in place.
    assert main_records_storage(arenas, device, 192 * 512).data_ptr() == large.data_ptr()
    with pytest.raises(ValueError, match="selection bank"):
        main_records_storage(arenas, device, main_records_rows_limit() + 1)
    base = workspace_reservation_bytes(device, 64, 640, 768)
    # The selection bank, plus its epoch and an INT64 owner slot per MAIN pool row.
    assert workspace_reservation_bytes(device, 64, 640, 768, generation_queries=768) == (
        base + 768 * 512 * 512 + 8
    )
    assert workspace_reservation_bytes(
        device, 64, 640, 768, generation_queries=768, main_pool_rows=1000
    ) == (base + 768 * 512 * 512 + 8 * 1001)
    assert workspace_reservation_bytes(device, 64, 128, 768, generation_queries=768) == (
        workspace_reservation_bytes(device, 64, 128, 768)
    )


@torch.inference_mode()
def test_prims_ts_main_bank_storage_grows_past_reservation() -> None:
    """A context bank of more rows than the reservation covers grows on demand, up to its limit."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        main_bank_marks,
        main_bank_reserved_rows,
        main_bank_rows_limit,
        main_bank_storage,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    reserved = main_bank_reserved_rows()
    assert reserved < main_bank_rows_limit()
    arenas = {}
    bank = main_bank_storage(arenas, device, reserved + 1)
    assert bank.shape == (reserved + 1, 512)
    # The next power of two of the rows.
    assert bank.untyped_storage().nbytes() == 2 * reserved * 512
    marks = main_bank_marks(arenas, device, reserved + 1)
    assert marks.shape == (reserved + 1,) and torch.count_nonzero(marks) == 0
    for retained in (main_bank_storage, main_bank_marks):
        with pytest.raises(ValueError, match="MAIN bank rows"):
            retained(arenas, device, main_bank_rows_limit() + 1)


@pytest.mark.parametrize("rows", [4, 192, 768])
@torch.inference_mode()
def test_prims_ts_selection_bank_attention_matches_packed(rows: int) -> None:
    """Gathering one decoded bank row per compacted selection reads what in-kernel decoding reads.

    With valid-prefix selections every row within a query's MAIN length is a pool record, so
    the bank row at ``r * width + c`` holds the operand the packed producer decodes for it.
    """
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_bank import (
        decode_main_records,
        dedup_main_records,
        main_records_storage,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2._prims_ts_packed import (
        PackedSparseMLA,
        _resolve_launch,
    )

    torch.manual_seed(5527)
    device = torch.device("cuda", torch.cuda.current_device())
    heads, swa_width, main_width = 64, 128, 512
    plans = {}
    for main_format in ("main", "main_fp8"):
        launch = _resolve_launch(
            device, rows, heads, swa_width, main_width, "swa_fp8", main_cache_format=main_format
        )
        packed = PackedSparseMLA(
            torch.empty(launch.workspace_bytes, device=device, dtype=torch.uint8)
        )
        packed.plan(
            device,
            rows,
            heads,
            max_topk=swa_width,
            max_extra_topk=main_width,
            has_sinks=True,
            swa_cache_format="swa_fp8",
            main_cache_format=main_format,
            valid_prefix=True,
        )
        plans[main_format] = packed
    q = (torch.randn((rows, 1, heads, 512), device=device) * 0.125).bfloat16()
    swa, main = _pool("swa_fp8", 512, 300), _pool("main", 288, 4096)
    main_scales = torch.tensor(
        [16, 160, 48, 184, 64, 200, 80, 208], device=device, dtype=torch.uint8
    )
    main[:, 256:] = main_scales[torch.arange(32, device=device) % main_scales.numel()]
    query_row = torch.arange(rows, device=device, dtype=torch.int32)[:, None]
    si = (torch.arange(swa_width, device=device, dtype=torch.int32)[None, :] + query_row * 3) % 300
    sl = torch.full((rows,), swa_width, device=device, dtype=torch.int32)
    sl[1::4] = 17
    # Rows of a request draw from a shared set of 900 records, as its draft tokens do.
    group = torch.arange(rows, device=device)[:, None] // 6
    shared = torch.randint(0, main.shape[0], ((rows + 5) // 6, 900), device=device)
    pick = torch.randint(0, 900, (rows, main_width), device=device)
    mi = shared[group, pick].to(torch.int32)
    ml = torch.full((rows,), main_width, device=device, dtype=torch.int32)
    ml[1::3], ml[2::7] = 77, 0
    # Selections past a query's length are not read; poison them in the record table.
    mi[torch.arange(main_width, device=device)[None, :] >= ml[:, None]] = -1
    arenas, dedup_arenas = {}, {}
    bank = main_records_storage(arenas, device, rows * main_width)
    decode_main_records(main, mi, bank, arenas)
    flat = torch.arange(rows * main_width, device=device, dtype=torch.int32).view(rows, main_width)
    # The deduplicated bank decodes each selected row once; selections read their owner's row.
    dedup_bank = main_records_storage(dedup_arenas, device, rows * main_width)
    owners = torch.empty_like(mi)
    dedup_main_records(main, mi, owners, dedup_bank, dedup_arenas)
    sinks = torch.linspace(-1, 1, heads, device=device)
    outputs, lses = {}, {}
    for name, main_format, source, indices in (
        ("main", "main", main, mi),
        ("main_fp8", "main_fp8", bank, flat),
        ("dedup", "main_fp8", dedup_bank, owners),
    ):
        outputs[name] = torch.full_like(q, torch.nan)
        plans[main_format].run(
            q,
            swa,
            si,
            sl,
            extra_kv_cache=source,
            extra_sparse_indices=indices,
            extra_sparse_lens=ml,
            softmax_scale=512**-0.5,
            output_scale=0.75,
            sinks=sinks,
            out=outputs[name],
        )
        lses[name] = plans[main_format]._state.buffers["lse"].clone()
    assert torch.isfinite(outputs["main"]).all()
    for name in ("main_fp8", "dedup"):
        torch.testing.assert_close(outputs[name], outputs["main"], atol=0, rtol=0)
        torch.testing.assert_close(lses[name], lses["main"], atol=0, rtol=0)
