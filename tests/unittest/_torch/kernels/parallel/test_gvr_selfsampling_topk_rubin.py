# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubin GVR dispatch, allocation ownership and graph-replay regression tests."""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from math import gcd, lcm
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from cuda.bindings import driver as cuda

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling_host as gvr
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_locality as locality
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_locality_policy as policy
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_rubin_dispatch as dispatch

_rubin_gpu = pytest.mark.usefixtures("rubin_gpu")

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def rubin_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin kernel tests require an SM107 GPU")
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cutlass.utils.rubin_helpers")


def _checked(result: tuple) -> object:
    assert result[0] == cuda.CUresult.CUDA_SUCCESS, result[0]
    return result[1] if len(result) == 2 else result[1:]


def _cleanup_mappings(device: int, mapped: list[tuple[int, int]], base: int, size: int) -> None:
    error = None
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        error = exc
    for pointer, span in mapped:
        try:
            _checked(cuda.cuMemUnmap(pointer, span))
        except AssertionError as exc:
            error = error or exc
    try:
        _checked(cuda.cuMemAddressFree(base, size))
    except AssertionError as exc:
        error = error or exc
    if error is not None:
        raise error


@pytest.fixture
def domain_executor() -> Iterator[locality.LocalityDomainExecutor]:
    attribute = getattr(cuda.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT", None)
    location = getattr(cuda.CUmemLocationType, "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN", None)
    if attribute is None or location is None or not hasattr(cuda, "cuDevSmResourceSplit"):
        pytest.skip("CUDA bindings do not provide locality-domain APIs")
    result = cuda.cuDeviceGetAttribute(attribute, torch.cuda.current_device())
    if result[0] == cuda.CUresult.CUDA_ERROR_NOT_SUPPORTED:
        pytest.skip("Locality domains are not supported by the driver")
    if int(_checked(result)) < 2:
        pytest.skip("Cross-domain row ownership requires at least two domains")
    with locality.LocalityDomainExecutor(torch.cuda.current_device(), backfill=False) as executor:
        yield executor


class _ByteView:
    def __init__(self, pointer: int, size: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


@contextmanager
def _localized_rows(
    rows: int, n: int, dtype: torch.dtype, executor: locality.LocalityDomainExecutor
) -> Iterator[tuple[torch.Tensor, tuple[int, ...] | None]]:
    """Map contiguous row extents to queried domains without copying their allocation."""
    device = executor.device
    weights = [d.sm_count for d in executor.domains]
    total = sum(weights)
    counts = [rows * w // total for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (rows * weights[i] % total, -i), reverse=True)
    for i in order[: rows - sum(counts)]:
        counts[i] += 1
    properties = []
    alignment = 1
    allocation_alignment = 1
    for domain, count in enumerate(counts):
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        prop.location.localized.deviceId = device
        prop.location.localized.localityDomainId = domain
        properties.append(prop)
        unit = int(
            _checked(
                cuda.cuMemGetAllocationGranularity(
                    prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
                )
            )
        )
        allocation_alignment = lcm(allocation_alignment, unit)
        if count:
            alignment = lcm(alignment, unit // gcd(unit, count))
    element_size = torch.empty((), dtype=dtype).element_size()
    stride_bytes = (n * element_size + alignment - 1) // alignment * alignment
    size = rows * stride_bytes
    base = int(_checked(cuda.cuMemAddressReserve(size, allocation_alignment, 0, 0)))
    mapped = []
    try:
        offset = 0
        for count, prop in zip(counts, properties):
            if not count:
                continue
            span = count * stride_bytes
            handle = _checked(cuda.cuMemCreate(span, prop, 0))
            try:
                _checked(cuda.cuMemMap(base + offset, span, 0, handle, 0))
                mapped.append((base + offset, span))
            finally:
                _checked(cuda.cuMemRelease(handle))
            access = cuda.CUmemAccessDesc()
            access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _checked(cuda.cuMemSetAccess(base + offset, span, [access], 1))
            offset += span
        owner = _ByteView(base, size)
        storage = torch.as_tensor(owner, device=f"cuda:{device}")
        assert storage.data_ptr() == base
        x = storage.view(dtype).reshape(rows, stride_bytes // element_size)
        try:
            result = cuda.cuPointerGetAttribute(
                cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, base
            )
        except (TypeError, RuntimeError):
            expected = None
        else:
            assert int(_checked(result)) == next(i for i, count in enumerate(counts) if count)
            expected = tuple(counts)
        assert locality._allocation_rows(x, device) == expected
        yield x, expected
    finally:
        error = sys.exc_info()[1]
        try:
            _cleanup_mappings(device, mapped, base, size)
        except (RuntimeError, AssertionError) as cleanup_error:
            if error is not None:
                raise error from cleanup_error
            raise


def _fill(x: torch.Tensor, n: int) -> None:
    generator = torch.Generator(device=x.device).manual_seed(n + x.shape[0])
    x.uniform_(-30, 30, generator=generator)
    x[:, n:] = float("inf")


def _check_exact(x: torch.Tensor, out: torch.Tensor, lengths: list[int]) -> None:
    k = out.shape[1]
    for row, n in enumerate(lengths):
        idx = out[row].long()
        if n <= k:
            assert torch.equal(idx[:n], torch.arange(n, device=x.device))
            assert bool((idx[n:] == -1).all())
        else:
            assert bool(((idx >= 0) & (idx < n)).all())
            ordered = idx.sort().values
            assert not bool((ordered[1:] == ordered[:-1]).any())
            got = x[row].gather(0, idx).sort().values
            want = torch.topk(x[row, :n], k).values.sort().values
            assert torch.equal(got, want), f"row {row}: incorrect top-k value multiset"


def _check_replay(
    call: Callable[[], None],
    x: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    n: int,
    cr: int = 1,
    *,
    replay_lengths: list[int] | None = None,
    next_n: int = 1,
) -> None:
    def row_lengths(request_lengths):
        return [
            max(0, (length - next_n + token + 1) // cr)
            for length in request_lengths
            for token in range(next_n)
        ]

    original = x.clone()
    call()
    torch.cuda.synchronize()
    _check_exact(x, out, row_lengths([n * cr] * len(lengths)))
    assert torch.equal(x, original)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            call()
        # Replay once with the prepared envelope, then with changed scores and lengths.
        for step in range(2):
            request_lengths = [n] * len(lengths)
            if step == 1:
                request_lengths = (
                    replay_lengths
                    if replay_lengths is not None
                    else [
                        (0, out.shape[1] - 3, out.shape[1] + 3, n - 11)[r % 4]
                        for r in range(len(lengths))
                    ]
                )
            totals = [v * cr for v in request_lengths]
            lengths.copy_(torch.tensor(totals, dtype=torch.int32, device=x.device))
            valid = row_lengths(totals)
            x.copy_(original if step == 0 else -original)
            for row, count in enumerate(valid):
                x[row, count:] = float("inf")
            before = x.clone()
            out.fill_(-777)
            graph.replay()
            torch.cuda.synchronize()
            _check_exact(x, out, valid)
            assert torch.equal(x, before)
    finally:
        graph.reset()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("k,rows,n", [(512, 4, 262127), (1024, 16, 65539), (2048, 1, 131075)])
@_rubin_gpu
def test_rubin_gvr_graph_exactness(dtype: torch.dtype, k: int, rows: int, n: int) -> None:
    """Native dtype dispatch must cover odd tails and changing device KV lengths."""
    cr = 1 if k == 2048 else 4
    x = torch.empty((rows, (n + 7) // 8 * 8), dtype=dtype, device="cuda")
    if dtype == torch.float32 and k == 512:
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        plan = gvr._route_fp32_varlen(rows, n, x.shape[1], k, sms, 107)
        assert plan["kernel"] == "main" and plan["rt"]["R"] > 1
        assert plan["_rubin_runtime"] == {"amin_num": 4}
    _fill(x, n)
    lengths = torch.full((rows,), n * cr, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    _check_replay(
        lambda: gvr.run_varlen(x, lengths, out, max_seq_len=n * cr, compress_ratio=cr),
        x,
        lengths,
        out,
        n,
        cr,
    )


@pytest.mark.parametrize("next_n", [1, 3, 5])
@_rubin_gpu
def test_rubin_gvr_prepared_next_n_graph(next_n: int) -> None:
    # Keep the logits shape fixed while changing how rows are grouped into requests.
    rows, n, k = 15, 2048, 512
    x = torch.empty((rows, n), dtype=torch.float32, device="cuda")
    _fill(x, n)
    lengths = torch.full((rows // next_n,), n, dtype=torch.int32, device=x.device)
    out = torch.empty((rows, k), dtype=torch.int32, device=x.device)
    prepared = locality.PreparedRowTopK(x, lengths, out, next_n=next_n, max_seq_len=n)
    try:
        _check_replay(prepared, x, lengths, out, n, next_n=next_n)
    finally:
        prepared.close()


@pytest.mark.parametrize("dtype,next_n,cr", [(torch.float16, 2, 1), (torch.bfloat16, 5, 4)])
@_rubin_gpu
def test_rubin_gvr_prepared_mtp_graph(dtype, next_n, cr):
    batch, n, k = 4, 8192, 512
    x = torch.empty((batch * next_n, n), dtype=dtype, device="cuda")
    _fill(x, n)
    lengths = torch.full((batch,), n * cr, dtype=torch.int32, device=x.device)
    out = torch.empty((len(x), k), dtype=torch.int32, device=x.device)
    prepared = locality.PreparedRowTopK(
        x, lengths, out, next_n=next_n, max_seq_len=n * cr, compress_ratio=cr
    )
    try:
        _check_replay(prepared, x, lengths, out, n, cr, next_n=next_n)
    finally:
        prepared.close()


@pytest.mark.parametrize("k,rows,n", [(512, 1, 8195), (2048, 16, 8207)])
@_rubin_gpu
def test_rubin_gvr_bf16_pcw_graph(
    k: int, rows: int, n: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PCW must handle 16-byte row alignment with both histogram bin counts."""
    cr = 1 if k == 2048 else 4
    x = torch.empty((rows, (n + 7) // 8 * 8), dtype=torch.bfloat16, device="cuda")
    assert x.shape[1] % 8 == 0 and x.shape[1] % 64 != 0
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tuning = dispatch.select_lowp_tuning(rows, n, x.shape[1], k, sms, 107)
    assert tuning is not None and tuning.pcw_aligned
    device = gvr._device()
    compile_spy = Mock(wraps=device.get_compiled__reg)
    monkeypatch.setattr(device, "get_compiled__reg", compile_spy)
    monkeypatch.setattr(gvr, "_VARLEN_CACHE_BF16", {})
    _fill(x, n)
    lengths = torch.full((rows,), n * cr, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    _check_replay(
        lambda: gvr.run_varlen(x, lengths, out, max_seq_len=n * cr, compress_ratio=cr),
        x,
        lengths,
        out,
        n,
        cr,
    )
    assert compile_spy.call_count == 1
    options = compile_spy.call_args.kwargs
    assert options["pcw"] and options["dtype"] == "bf16"
    assert options["pcw_bins"] == (512 if k == 2048 else 1024)


@_rubin_gpu
def test_rubin_gvr_pcw_minimum_row_alignment() -> None:
    """The direct PCW kernel accepts 8B rows; the public GVR API requires 16B."""
    rows, n, npad, k = 2, 8195, 8204, 512
    assert npad % gvr.PCW_VECTOR_ELEMENTS == 0 and npad % gvr.PCW_API_ROW_ELEMENTS != 0
    x = torch.empty((rows, npad), dtype=torch.bfloat16, device="cuda")
    _fill(x, n)
    lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    kernel = gvr._device().get_compiled__reg(
        (512, 4, 1, 1, True, False, False, 1024),
        dtype="bf16",
        pcw=True,
        pcw_bins=1024,
        varlen=True,
        static_k=k,
        static_npad=npad,
    )
    _check_replay(
        lambda: kernel(x, out, lengths, out, n, 0, 0, 0),
        x,
        lengths,
        out,
        n,
        replay_lengths=[k - 3, n - 11],
    )


@_rubin_gpu
def test_rubin_gvr_handbuilt_reg_graph() -> None:
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows, n, k = max(1, 150 * sms // 208), 6000, 1024
    tuning = dispatch.select_tuning(rows, n, k, sms, 107)
    assert tuning is not None and tuning.family == "reg"
    plan = gvr._route_fp32_varlen(rows, n, n, k, sms, 107)
    assert plan["tpl"] == (512, 3, 2, 2, True, False, False, 1024)
    assert plan["smem"] == 24576 and plan["_rubin_options"] == {"static_npad": n}
    x = torch.empty((rows, n), dtype=torch.float32, device="cuda")
    _fill(x, n)
    lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    _check_replay(lambda: gvr.run_varlen(x, lengths, out, max_seq_len=n), x, lengths, out, n)


@_rubin_gpu
def test_rubin_gvr_output_alignment_graph() -> None:
    rows, n, k = 209, 5120, 512
    x = torch.empty((rows, n), dtype=torch.float16, device="cuda")
    _fill(x, n)
    lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
    storage = torch.empty(rows * k + 1, dtype=torch.int32, device="cuda")
    shifted = storage[1:].view(rows, k)
    assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="Output base"):
        locality.PreparedRowTopK(x, lengths, shifted, max_seq_len=n, co_local=False)
    out = storage[:-1].view(rows, k)
    prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n, co_local=False)
    try:
        _check_replay(prepared, x, lengths, out, n)
    finally:
        prepared.close()


@pytest.mark.parametrize(
    "dtype,k,next_n",
    [(dtype, k, 1) for dtype in _DTYPES for k in (512, 1024, 2048)] + [(torch.float32, 1024, 3)],
)
@_rubin_gpu
def test_rubin_gvr_locality_graph(
    dtype: torch.dtype, k: int, next_n: int, domain_executor: locality.LocalityDomainExecutor
) -> None:
    """Full-batch planning and per-domain workspaces survive graph replay."""
    rows, n = max(512, (512 * domain_executor.physical_sm_count + 207) // 208), 131075
    if next_n > 1:
        weights = [domain.sm_count for domain in domain_executor.domains]
        multiple = next_n * sum(weights) // gcd(*weights)
        rows = (rows + multiple - 1) // multiple * multiple
    cr = 1 if k == 2048 else 4
    with _localized_rows(rows, n, dtype, domain_executor) as (x, counts):
        _fill(x, n)
        lengths = torch.full((rows // next_n,), n * cr, dtype=torch.int32, device="cuda")
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(
            x,
            lengths,
            out,
            max_seq_len=n * cr,
            next_n=next_n,
            compress_ratio=cr,
            co_local=True,
            row_counts=counts,
            executor=domain_executor,
        )
        try:
            if counts is None:
                assert prepared.mode == "tuned" and prepared.executor is None
                _check_replay(prepared, x, lengths, out, n, cr, next_n=next_n)
                return
            assert prepared.locality_allocation and prepared.mode == "local_preserve"
            assert len(prepared.parts) == sum(c > 0 for c in counts)
            plan = gvr._varlen_launcher(
                rows,
                x.shape[1],
                k,
                n,
                next_n,
                cr,
                domain_executor.physical_sm_count,
                107,
                dtype=dtype,
                device_index=x.get_device(),
            )
            assert all(bound.launcher is plan for _, bound, _ in prepared.parts)
            _check_replay(prepared, x, lengths, out, n, cr, next_n=next_n)
        finally:
            prepared.close()
        assert not domain_executor.closed  # Caller-owned resources outlive a prepared call.
        prepared.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared()


@_rubin_gpu
def test_rubin_gvr_ordinary_allocation_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An eligible shape alone must neither localize memory nor create green contexts."""
    rows, n, k = 512, 131075, 512
    x = torch.empty((rows, (n + 7) // 8 * 8), device="cuda")
    _fill(x, n)
    lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")

    def forbidden(*args, **kwargs) -> None:
        pytest.fail("ordinary allocation must not create locality resources")

    monkeypatch.setattr(locality, "LocalityDomainExecutor", forbidden)
    prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n, co_local=True)
    try:
        assert not prepared.locality_allocation and prepared.mode == "tuned"
        assert prepared.executor is None
        _check_replay(prepared, x, lengths, out, n)
    finally:
        prepared.close()
    monkeypatch.setattr(locality, "_allocation_rows", forbidden)
    # Explicit opt-out and the ordinary public entry must not query allocation ownership.
    prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n, co_local=False)
    try:
        prepared()
        gvr.run_varlen(x, lengths, out, max_seq_len=n)
        torch.cuda.synchronize()
    finally:
        prepared.close()


@pytest.mark.parametrize("dtype", _DTYPES)
@_rubin_gpu
def test_rubin_gvr_automatic_backfill(
    domain_executor: locality.LocalityDomainExecutor, dtype: torch.dtype
) -> None:
    rows = (512 * domain_executor.physical_sm_count + 207) // 208
    n, k = 131075, 512
    with _localized_rows(rows, n, dtype, domain_executor) as (x, counts):
        _fill(x, n)
        lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n, co_local=True)
        owned = prepared.executor
        try:
            if counts is None:
                assert prepared.mode == "tuned" and owned is None
                _check_replay(prepared, x, lengths, out, n)
                return
            assert prepared.mode == "local_preserve" and prepared.owns_executor
            assert owned is not None
            assert owned.backfill_requested
            assert sum(d.sm_count for d in owned.domains) <= owned.physical_sm_count
            if owned.backfill_applied:
                assert sum(d.sm_count for d in owned.domains) > sum(
                    d.sm_count for d in domain_executor.domains
                )
            _check_replay(prepared, x, lengths, out, n)
        finally:
            prepared.close()
        assert owned.closed


@_rubin_gpu
def test_rubin_gvr_complete_row_ownership(domain_executor: locality.LocalityDomainExecutor) -> None:
    weights = [d.sm_count for d in domain_executor.domains]
    rows, n = sum(weights) // gcd(*weights), 4099
    with _localized_rows(rows, n, torch.float32, domain_executor) as (x, counts):
        assert locality._allocation_rows(x.reshape(1, -1), x.get_device()) is None
        bad_counts = (rows,) + (0,) * (len(weights) - 1)
        out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(
            x, lengths, out, max_seq_len=n, co_local=True, row_counts=bad_counts
        )
        try:
            assert not prepared.locality_allocation and prepared.executor is None
        finally:
            prepared.close()


@pytest.mark.parametrize("sms,sm", [(132, 90), (148, 100), (148, 103), (208, 120)])
def test_rubin_gvr_preserves_other_architectures(sms: int, sm: int) -> None:
    for rows, n, k in ((4, 262127, 512), (512, 32768, 1024), (16, 131075, 2048)):
        npad = (n + 7) // 8 * 8
        assert dispatch.select_tuning(rows, n, k, sms, sm) is None
        assert dispatch.select_lowp_tuning(rows, n, npad, k, sms, sm) is None
        assert gvr._route_fp32_varlen(rows, n, npad, k, sms, sm) == gvr.route(
            rows, n, npad, k, sms, sm
        )


def test_rubin_gvr_explicit_routing_constraints() -> None:
    assert dispatch.select_tuning(4, 262127, 512, 208, 107) is not None
    assert dispatch.select_lowp_tuning(4, 262127, 262128, 512, 208, 107) is not None
    for kwargs in ({"det": True}, {"r_cap": 1}, {"r_cap": 2}):
        args = (4, 262127, 262128, 512, 208, 107)
        assert gvr._route_fp32_varlen(*args, **kwargs) == gvr.route(*args, **kwargs)


@pytest.mark.parametrize(
    "counts,alignment,remainder",
    [
        ((92, 92), (4, 4), 24),
        ((48, 24, 24), (8, 4, 8), 32),
        ((24, 16, 8), (4, 4, 4), 7),
    ],
)
def test_rubin_gvr_backfill_resource_invariants(
    counts: tuple[int, ...], alignment: tuple[int, ...], remainder: int
) -> None:
    actual = locality._backfill_counts(counts, alignment, remainder)
    assert len(actual) == len(counts)
    assert sum(counts) <= sum(actual) <= sum(counts) + remainder
    assert all(a >= c and a % align == 0 for a, c, align in zip(actual, counts, alignment))
    assert all(a * counts[0] == actual[0] * c for a, c in zip(actual, counts))
    if remainder < sum(alignment):
        assert actual == counts
    else:
        assert sum(actual) > sum(counts)


def test_rubin_gvr_policy_boundaries() -> None:
    for sms, rows in ((104, 256), (208, 512), (212, 522), (416, 1024)):
        assert policy.choose_locality(torch.float32, 512, 131072, rows, sms, 107) == (
            "local_preserve",
            4,
        )
    assert policy.choose_locality(torch.float32, 512, 131072, 512, 208, 100) is None


@_rubin_gpu
def test_rubin_gvr_pcw_cache_separates_histogram_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different PCW bin counts must not share a compiled callable."""
    device = gvr._device()
    compiled = []

    def record_compile(*args, **kwargs) -> object:
        result = object()
        compiled.append(result)
        return result

    monkeypatch.setattr(device, "_COMPILE_CACHE__reg", {})
    monkeypatch.setattr(device.cute, "compile", record_compile)
    tpl = (256, 4, 2, 1, True, False, False, 1024)
    options = dict(dtype="bf16", pcw=True, varlen=True, static_k=2048, static_npad=8200)
    first = device.get_compiled__reg(tpl, pcw_bins=512, **options)
    second = device.get_compiled__reg(tpl, pcw_bins=1024, **options)
    assert first is not second
    assert device.get_compiled__reg(tpl, pcw_bins=512, **options) is first
    assert len(compiled) == 2


@pytest.fixture
def recorded_launcher(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Exercise the real host router while recording, rather than compiling, device code."""
    compile_spy = Mock(return_value=object())
    device = SimpleNamespace(
        STATIC_BYTES=0,
        **{
            name: compile_spy
            for name in (
                "get_compiled",
                "get_compiled__reg",
                "get_compiled__regclus",
                "get_compiled__clus",
                "get_compiled_complement",
            )
        },
    )
    monkeypatch.setattr(gvr, "_dev_mod", device)
    for name in ("_VARLEN_CACHE", "_VARLEN_CACHE_BF16", "_VARLEN_CACHE_FP16"):
        monkeypatch.setattr(gvr, name, {})

    def launch(dtype, k, rows, n, *, tuned=True, npad=None, sms=208):
        compile_spy.reset_mock()
        result = gvr._varlen_launcher(
            rows,
            npad or (n + 7) // 8 * 8,
            k,
            n,
            1,
            1,
            sms,
            107,
            dtype=dtype,
            device_index=0,
            use_rubin_tuning=tuned,
        )
        assert compile_spy.call_count == 1
        return result, compile_spy.call_args

    return launch


@pytest.mark.parametrize(
    "dtype,k,rows,n,family,tpl,cluster,options",
    [
        (torch.float16, 512, 4, 262127, "reg", (1024, 4, 1), 16, {"code_radix": True, "clus": 16}),
        (torch.float16, 1024, 16, 65539, "reg", (512, 8, 1), 4, {"code_radix": True, "clus": 4}),
        (torch.float16, 2048, 1, 131075, "reg", (512, 8, 1), 8, {"code_radix": True, "clus": 8}),
        (torch.float16, 512, 209, 5120, "reg", (256, 8, 4), 1, {"cr2": True, "warp_rank": True}),
        (torch.float16, 2048, 16, 8207, "reg", (512, 4, 1), 1, {"cr2": False, "warp_rank": True}),
        (torch.bfloat16, 512, 4, 262127, "reg_clus", (1024, 2, 16), 16, {"nbh": 512}),
        (torch.bfloat16, 1024, 16, 65539, "reg_clus", (1024, 2, 4), 4, {"nbh": 1024}),
        (torch.bfloat16, 2048, 1, 131075, "reg_clus", (1024, 2, 8), 8, {"nbh": 512}),
        (torch.bfloat16, 512, 1, 8195, "reg", (512, 4, 1), 8, {"pcw": True, "pcw_bins": 1024}),
        (torch.bfloat16, 2048, 16, 8207, "reg", (512, 4, 1), 1, {"pcw": True, "pcw_bins": 512}),
    ],
)
def test_rubin_gvr_lowp_routing_options(
    recorded_launcher: Callable, dtype, k, rows, n, family, tpl, cluster, options
) -> None:
    launcher, call = recorded_launcher(dtype, k, rows, n)
    assert launcher[0] == family and launcher.cluster_size == cluster
    assert call.args[0][:3] == tpl
    assert {key: call.kwargs[key] for key in options} == options
    key = next(iter(gvr._varlen_cache_for(dtype)))
    assert key[7] == gvr._pack_device_profile(208, 107)


@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 9, 16, 64])
def test_rubin_gvr_pcw_cluster_metadata(
    recorded_launcher: Callable, tuned: bool, rows: int
) -> None:
    launcher, call = recorded_launcher(torch.bfloat16, 512, rows, 8195, tuned=tuned, npad=8256)
    assert call.kwargs["pcw"]
    parts = call.kwargs["parts"]
    expected_cluster = parts > 1 and (parts < 8 or parts * rows <= 64)
    assert call.kwargs["pcw_cluster"] == expected_cluster
    assert launcher.cluster_size == (parts if expected_cluster else 1)


def test_rubin_gvr_tuning_cache_and_wave(recorded_launcher: Callable) -> None:
    tuned, tuned_call = recorded_launcher(torch.bfloat16, 512, 1, 8195, npad=8256)
    original, original_call = recorded_launcher(
        torch.bfloat16, 512, 1, 8195, tuned=False, npad=8256
    )
    assert tuned is not original
    assert tuned_call.kwargs["warp_rank"] and not original_call.kwargs["warp_rank"]
    keys = list(gvr._VARLEN_CACHE_BF16)
    assert len(keys) == 2 and keys[1] == keys[0] + ("original_route",)
    assert keys[0][7] == gvr._pack_device_profile(208, 107)
    _, inside = recorded_launcher(torch.float16, 2048, 148, 8195)
    _, outside = recorded_launcher(torch.float16, 2048, 149, 8195)
    assert inside.kwargs["binproof"] and not outside.kwargs.get("binproof", False)


@pytest.mark.parametrize("dtype,n", [(torch.float16, 131075), (torch.bfloat16, 65539)])
def test_rubin_gvr_lowp_decode_wave(recorded_launcher, monkeypatch, dtype, n):
    route = Mock(wraps=gvr._route_16bit)
    stream = Mock(wraps=gvr._route_streaming_16bit)
    monkeypatch.setattr(gvr, "_route_16bit", route)
    monkeypatch.setattr(gvr, "_route_streaming_16bit", stream)
    tall, compile_call = recorded_launcher(dtype, 1024, 1024, n, sms=212)
    assert route.call_args.args[-2:] == (212, 107)
    assert stream.call_args.args[-2:] == (212, 107)
    assert tall[0] == "main" and compile_call.args[0][0] == 256 and tall[3][0] == 4096
    short, compile_call = recorded_launcher(dtype, 1024, 64, n, sms=212)
    assert short[0] == "clus" and compile_call.kwargs["wave_sms"] == gvr.wave_sms(212, 107) == 148


def test_rubin_gvr_fp32_launch_staging(recorded_launcher):
    for tuned, size in [(False, 28672), (True, 32768)]:
        launcher, compiled = recorded_launcher(torch.float32, 1024, 64, 16384, tuned=tuned)
        assert launcher[0] == "reg" and bool(compiled.kwargs.get("stage_output")) == tuned
        assert launcher[2][-1] == size


@pytest.mark.parametrize(
    "k,n,rows,region,family,block",
    [
        (2048, 131087, 256, "main", "main", 512),
        (2048, 163775, 256, "main", "main", 512),
        (2048, 16399, 4, "reg", "reg", 1024),
        (2048, 24575, 8, "reg", "reg", 1024),
        (2048, 24576, 8, None, "reg_clus", 1024),
        (1024, 32771, 74, "main", "reg_clus", 1024),
        (1024, 32771, 75, "main", "main", 1024),
        (1024, 32771, 130, "main", "main", 1024),
        (1024, 32771, 131, "main", "main", 512),
    ],
)
def test_rubin_gvr_fp32_region_on_212_sms(recorded_launcher, k, n, rows, region, family, block):
    tuning = dispatch.select_tuning(rows, n, k, 212, 107)
    assert (tuning.family if tuning else None) == region
    launcher, compiled = recorded_launcher(torch.float32, k, rows, n, sms=212)
    assert launcher[0] == family and compiled.args[0][0] == block
    if family == "reg":
        assert compiled.args[0][7] == (2048 if k == 1024 else 1024)


@pytest.mark.parametrize("sms", [208, 212, 216])
def test_rubin_fp32_long_row_wave_boundaries(recorded_launcher, sms):
    for rows, block in [(sms, 1024), (sms + 1, 512), (2 * sms, 512)]:
        launcher, compiled = recorded_launcher(torch.float32, 2048, rows, 131087, sms=sms)
        assert launcher[0] == "main" and compiled.args[0][0] == block
        assert launcher[3][0] == 4096
    tuning = dispatch.select_tuning(2 * sms + 1, 131087, 2048, sms, 107)
    assert tuning is not None and tuning.family == "blackwell_options"


def test_rubin_gvr_pcw_alignment_policy() -> None:
    assert gvr.PCW_VECTOR_ELEMENTS == 4 and gvr.PCW_API_ROW_ELEMENTS == 8
    for n in (2048, 2049):
        for npad in (2052, 2056, 2112):
            tuning = dispatch.select_lowp_tuning(1, n, npad, 512, 208, 107)
            assert tuning.pcw_aligned == (npad == 2112 or (n > 2048 and npad == 2056))


def test_rubin_gvr_main_runtime_override(recorded_launcher: Callable) -> None:
    launcher, _ = recorded_launcher(torch.float32, 512, 4, 262127)
    assert launcher[0] == "main"
    assert launcher[2][5] == 16  # Runtime R.
    assert launcher[3][2] == 1024  # amin_num * K / 2, not just the intermediate override.


def _test_executor() -> locality.LocalityDomainExecutor:
    executor = locality.LocalityDomainExecutor.__new__(locality.LocalityDomainExecutor)
    executor.device = 0
    executor.closed = False
    executor.physical_sm_count = 208
    executor.coscheduled_sm_count = 4
    executor.backfill_applied = False
    executor.domains = [
        locality._Domain(i, 104, f"context{i}", f"stream{i}", Mock()) for i in range(2)
    ]
    return executor


@pytest.mark.parametrize("failure", ["synchronize", "stream", "context"])
def test_rubin_gvr_executor_cleanup_retry(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    executor = _test_executor()
    success = (cuda.CUresult.CUDA_SUCCESS,)
    failure_status = (cuda.CUresult.CUDA_ERROR_UNKNOWN,)
    sync = Mock(
        side_effect=[failure_status, success] if failure == "synchronize" else None,
        return_value=success,
    )
    stream = Mock(
        side_effect=[failure_status, success] if failure == "stream" else None, return_value=success
    )
    context = Mock(
        side_effect=[failure_status, success] if failure == "context" else None,
        return_value=success,
    )
    monkeypatch.setattr(cuda, "cuStreamSynchronize", sync)
    monkeypatch.setattr(cuda, "cuStreamDestroy", stream)
    monkeypatch.setattr(cuda, "cuGreenCtxDestroy", context)
    with pytest.raises(locality._CudaError):
        executor.close()
    assert executor.closed
    assert sync.call_count == stream.call_count == 2
    assert context.call_count == (1 if failure == "stream" else 2)
    if failure == "context":
        assert len(executor.domains) == 1
        assert executor.domains[0].driver_stream is None
    for operation in (sync, stream, context):
        operation.side_effect = None
    executor.close()
    assert not executor.domains
    assert stream.call_count == (3 if failure == "stream" else 2)
    assert sync.call_count == (3 if failure == "stream" else 2)
    assert context.call_count == (3 if failure == "context" else 2)
    calls = (sync.call_count, stream.call_count, context.call_count)
    executor.close()
    assert calls == (sync.call_count, stream.call_count, context.call_count)


@pytest.mark.parametrize("owned", [False, True])
def test_rubin_gvr_prepared_cleanup_after_sync_error(
    monkeypatch: pytest.MonkeyPatch, owned: bool
) -> None:
    executor = _test_executor()
    close = Mock(side_effect=executor.domains.clear)
    monkeypatch.setattr(executor, "close", close)
    prepared = locality.PreparedRowTopK.__new__(locality.PreparedRowTopK)
    prepared.device, prepared.executor, prepared.owns_executor = 0, executor, owned
    prepared.closed, prepared.parts = False, [object()]
    error = RuntimeError("synchronization failed")
    sync = Mock(side_effect=[error, None])
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    with pytest.raises(RuntimeError) as raised:
        prepared.close()
    assert raised.value is error
    assert close.call_count == int(owned)
    assert prepared.closed and len(prepared.parts) == 1
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    with pytest.raises(RuntimeError, match="closed"):
        prepared()
    prepared.close()
    assert not prepared.parts and sync.call_count == 2
    assert close.call_count == int(owned)


@pytest.mark.parametrize(
    "status",
    [
        cuda.CUresult.CUDA_SUCCESS,
        cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,
        cuda.CUresult.CUDA_ERROR_INVALID_VALUE,
        cuda.CUresult.CUDA_ERROR_NOT_SUPPORTED,
        cuda.CUresult.CUDA_ERROR_UNKNOWN,
    ],
)
def test_rubin_gvr_backfill_split(monkeypatch: pytest.MonkeyPatch, status, mock_domain_api) -> None:
    executor = _test_executor()

    def resource(count):
        return SimpleNamespace(sm=SimpleNamespace(smCount=count, smCoscheduledAlignment=4))

    plain, filled = [resource(92), resource(92)], [resource(104), resource(104)]
    remainder, rest = resource(24), resource(0)
    params = [SimpleNamespace(smCount=0, flags=2) for _ in plain]
    monkeypatch.setattr(
        cuda,
        "cuDeviceGetDevResource",
        Mock(return_value=(cuda.CUresult.CUDA_SUCCESS, resource(208))),
    )
    split = Mock(
        return_value=(status, filled, rest) if status == cuda.CUresult.CUDA_SUCCESS else (status,)
    )
    monkeypatch.setattr(cuda, "cuDevSmResourceSplit", split)
    if status == cuda.CUresult.CUDA_ERROR_UNKNOWN:
        with pytest.raises(locality._CudaError):
            executor._try_backfill(plain, remainder, params)
    else:
        actual, remaining = executor._try_backfill(plain, remainder, params)
        applied = status == cuda.CUresult.CUDA_SUCCESS
        assert actual is (filled if applied else plain)
        assert remaining is (rest if applied else remainder)
        assert executor.backfill_applied == applied
    assert [p.smCount for p in params] == [104, 104]
    assert all(
        p.flags & int(cuda.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_BACKFILL)
        for p in params
    )
    assert split.call_count == 1


class _HostTensor(torch.Tensor):
    """CPU storage for prepared-API validation tests; all CUDA operations are mocked."""

    @property
    def is_cuda(self) -> bool:
        return True

    def get_device(self) -> int:
        return 0


@pytest.fixture
def host_preparation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(major=10, minor=7, multi_processor_count=208),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: Mock())
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(locality, "_allocation_rows", lambda *_: (4, 4))
    monkeypatch.setattr(locality, "choose_locality", lambda *_: ("local_preserve", 4))
    bound = Mock(
        side_effect=lambda *args, **kwargs: SimpleNamespace(cluster_size=1, launcher=("reg",))
    )
    monkeypatch.setattr(locality, "_Bound", bound)
    x = torch.empty((8, 32), dtype=torch.float16).as_subclass(_HostTensor)
    lengths = torch.full((8,), 32, dtype=torch.int32).as_subclass(_HostTensor)
    output = torch.empty((8, 512), dtype=torch.int32).as_subclass(_HostTensor)
    return x, lengths, output, bound


@pytest.mark.parametrize(
    "status,operation",
    [
        (cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION, "split SM resources"),
        (cuda.CUresult.CUDA_ERROR_UNKNOWN, "split SM resources"),
        (cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION, "green context"),
    ],
)
def test_rubin_gvr_implicit_split_fallback(
    host_preparation, monkeypatch, status, operation
) -> None:
    x, lengths, output, bound = host_preparation
    error = locality._CudaError(status, operation)
    monkeypatch.setattr(locality, "LocalityDomainExecutor", Mock(side_effect=error))
    if (
        operation == "split SM resources"
        and status == cuda.CUresult.CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION
    ):
        prepared = locality.PreparedRowTopK(x, lengths, output, max_seq_len=32, co_local=True)
        assert prepared.mode == "tuned" and prepared.executor is None
        assert not prepared.owns_executor and bound.call_count == 1
        prepared.close()
    else:
        with pytest.raises(locality._CudaError) as raised:
            locality.PreparedRowTopK(x, lengths, output, max_seq_len=32, co_local=True)
        assert raised.value is error and bound.call_count == 0


def test_rubin_gvr_bound_compile_failure_closes_owned_executor(host_preparation, monkeypatch):
    from cutlass.base_dsl.common import DSLRuntimeError

    x, lengths, output, bound = host_preparation
    executor = _test_executor()
    success = (cuda.CUresult.CUDA_SUCCESS,)
    monkeypatch.setattr(cuda, "cuStreamSynchronize", Mock(return_value=success))
    destroy_stream = Mock(return_value=success)
    destroy_context = Mock(side_effect=[(cuda.CUresult.CUDA_ERROR_UNKNOWN,), success, success])
    monkeypatch.setattr(cuda, "cuStreamDestroy", destroy_stream)
    monkeypatch.setattr(cuda, "cuGreenCtxDestroy", destroy_context)
    monkeypatch.setattr(locality, "LocalityDomainExecutor", Mock(return_value=executor))
    failure = DSLRuntimeError("bound compilation failed")
    bound.side_effect = failure
    with pytest.raises(DSLRuntimeError) as caught:
        locality.PreparedRowTopK(x, lengths, output, max_seq_len=32, co_local=True)
    assert caught.value is failure
    retry = caught.value.__cause__
    assert isinstance(retry, locality.LocalityCleanupError) and retry.executor is executor
    assert [d.ordinal for d in executor.domains] == [1]
    retry.executor.close()
    assert not executor.domains and destroy_stream.call_count == 2
    assert destroy_context.call_args_list == [call("context1"), call("context0"), call("context1")]


def test_rubin_gvr_preparation_alignment(host_preparation) -> None:
    x, lengths, output, bound = host_preparation
    shifted = (
        torch.empty(output.numel() + 1, dtype=torch.int32)[1:]
        .view_as(output)
        .as_subclass(_HostTensor)
    )
    assert shifted.is_contiguous() and shifted.data_ptr() % 16
    with pytest.raises(ValueError, match="Output base"):
        locality.PreparedRowTopK(x, lengths, shifted, max_seq_len=32, co_local=False)
    assert bound.call_count == 0
    prepared = locality.PreparedRowTopK(x, lengths, output, max_seq_len=32, co_local=False)
    assert prepared.mode == "tuned"
    prepared.close()


@pytest.mark.parametrize(
    "options", [{}, {"co_local": None}, {"co_local": False}, {"co_local": True}]
)
def test_rubin_gvr_probe_semantics(host_preparation, monkeypatch, options) -> None:
    x, lengths, output, _ = host_preparation
    probe = Mock(return_value=None)
    create = Mock(side_effect=AssertionError("must not create resources"))
    monkeypatch.setattr(locality, "_allocation_rows", probe)
    monkeypatch.setattr(locality, "LocalityDomainExecutor", create)
    prepared = locality.PreparedRowTopK(x, lengths, output, max_seq_len=32, **options)
    assert prepared.mode == "tuned" and not prepared.locality_allocation
    assert probe.call_count == int(options.get("co_local") is True)
    create.assert_not_called()
    prepared.close()


@pytest.mark.parametrize(
    "query", ["first", "row_start", "last_byte", "interior", "domain_count", "granularity"]
)
@pytest.mark.parametrize("exception", [TypeError, RuntimeError])
def test_rubin_gvr_optional_driver_query(monkeypatch, query, exception):
    x = torch.empty((2, 32), dtype=torch.float32)
    start = x.data_ptr()
    stride = x.stride(0) * x.element_size()
    success = cuda.CUresult.CUDA_SUCCESS
    calls = 0

    def ordinal(attr, ptr):
        nonlocal calls
        calls += 1
        if (
            (query == "first" and calls == 1)
            or (query == "row_start" and ptr == start + stride)
            or (query == "last_byte" and ptr == start + stride - 1)
            or (query == "interior" and ptr == start + 16)
        ):
            raise exception("unsupported pointer query")
        return success, int(ptr >= start + stride)

    def attribute(kind, value):
        if kind == query:
            raise exception("unsupported device query")
        return success, value

    driver = SimpleNamespace(
        CUresult=cuda.CUresult,
        CUpointer_attribute=SimpleNamespace(CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL=1),
        CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT=2),
        CUmemLocationType=SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN=3),
        CUmemAllocationType=cuda.CUmemAllocationType,
        CUmemAllocationGranularity_flags=cuda.CUmemAllocationGranularity_flags,
        CUmemAllocationProp=lambda: SimpleNamespace(
            location=SimpleNamespace(localized=SimpleNamespace())
        ),
        cuPointerGetAttribute=ordinal,
        cuDeviceGetAttribute=lambda *_: attribute("domain_count", 2),
        cuMemGetAllocationGranularity=lambda *_: attribute("granularity", 16),
    )
    monkeypatch.setattr(locality, "cuda", driver)
    assert locality._allocation_rows(x, 0) is None


@pytest.mark.parametrize("invalid", ["closed", "device", "count"])
def test_rubin_gvr_borrowed_executor_validation(host_preparation, invalid: str) -> None:
    x, lengths, output, _ = host_preparation
    executor = _test_executor()
    if invalid == "closed":
        executor.closed = True
    elif invalid == "device":
        executor.device = 1
    else:
        executor.domains.pop()
    executor.close = Mock()
    with pytest.raises(ValueError):
        locality.PreparedRowTopK(
            x, lengths, output, max_seq_len=32, executor=executor, co_local=True
        )
    executor.close.assert_not_called()


@pytest.mark.parametrize("budgets,counts", [((156, 52), (6, 2)), ((104, 52, 52), (4, 2, 2))])
def test_rubin_gvr_preserves_full_batch_plan(
    host_preparation, monkeypatch, budgets, counts
) -> None:
    x, lengths, output, bound = host_preparation
    executor = _test_executor()
    executor.domains = [
        locality._Domain(i, count, Mock(), Mock(), Mock()) for i, count in enumerate(budgets)
    ]
    monkeypatch.setattr(locality, "_allocation_rows", lambda *_: counts)
    lengths = lengths[:4]
    prepared = locality.PreparedRowTopK(
        x, lengths, output, next_n=2, max_seq_len=32, executor=executor, co_local=True
    )
    assert prepared.mode == "local_preserve"
    assert [call.args[0].shape[0] for call in bound.call_args_list] == list(counts)
    assert all(call.args[5:] == (8, 208, 107, 2) for call in bound.call_args_list)
    offset = 0
    for binding, count in zip(bound.call_args_list, counts):
        assert binding.args[1].data_ptr() == lengths[offset // 2 :].data_ptr()
        assert len(binding.args[1]) == count // 2
        offset += count
    prepared.close()
    assert not executor.closed


def test_rubin_gvr_mtp_split_request(host_preparation, monkeypatch):
    x, lengths, output, bound = host_preparation
    monkeypatch.setattr(locality, "_allocation_rows", lambda *_: (3, 5))
    make_executor = Mock()
    monkeypatch.setattr(locality, "LocalityDomainExecutor", make_executor)
    lengths = lengths[:4]
    prepared = locality.PreparedRowTopK(x, lengths, output, next_n=2, max_seq_len=32, co_local=True)
    assert prepared.locality_allocation and prepared.mode == "tuned"
    make_executor.assert_not_called()
    assert bound.call_count == 1 and bound.call_args.args[1] is lengths
    assert bound.call_args.args[-1] == 2
    prepared.close()


@pytest.mark.parametrize("next_n,length_count", [(0, 8), (3, 2), (2, 8)])
def test_rubin_gvr_invalid_request_shape(host_preparation, next_n, length_count):
    x, lengths, output, bound = host_preparation
    with pytest.raises(ValueError, match="next_n|one length per request"):
        locality.PreparedRowTopK(x, lengths[:length_count], output, next_n=next_n, max_seq_len=32)
    bound.assert_not_called()


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("event_failure", [False, True])
def test_rubin_gvr_failed_submission_join(
    monkeypatch, capturing: bool, event_failure: bool
) -> None:
    parent, first_stream, second_stream = Mock(), Mock(), Mock()
    first_done, second_done = Mock(), Mock()
    if event_failure:
        first_done.record.side_effect = RuntimeError("event failed")
    order = Mock()
    order.attach_mock(second_done.record, "record")
    order.attach_mock(parent.wait_event, "wait")
    failure = KeyboardInterrupt("submission interrupted")
    prepared = locality.PreparedRowTopK.__new__(locality.PreparedRowTopK)
    prepared.device, prepared.executor, prepared.closed, prepared.mode = (
        0,
        None,
        False,
        "local_preserve",
    )
    prepared.start = Mock()
    prepared.parts = [
        (SimpleNamespace(stream=first_stream), Mock(), first_done),
        (SimpleNamespace(stream=second_stream), Mock(side_effect=failure), second_done),
    ]
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: parent)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    with pytest.raises(
        RuntimeError if event_failure else KeyboardInterrupt,
        match="event failed" if event_failure else "submission interrupted",
    ):
        prepared()
    assert prepared.closed
    if capturing:
        parent.wait_event.assert_not_called()
        first_stream.synchronize.assert_not_called()
        second_stream.synchronize.assert_not_called()
    elif event_failure:
        first_stream.synchronize.assert_called_once()
    else:
        assert order.mock_calls == [
            call.wait(first_done),
            call.record(second_stream),
            call.wait(second_done),
        ]
        first_stream.synchronize.assert_not_called()
        second_stream.synchronize.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        prepared()


@pytest.mark.parametrize("sync_failure", [False, True])
def test_rubin_gvr_vmm_cleanup_all_mappings(monkeypatch, sync_failure: bool) -> None:
    sync_error = RuntimeError("synchronize failed")
    monkeypatch.setattr(
        torch.cuda, "synchronize", Mock(side_effect=sync_error if sync_failure else None)
    )
    unmap = Mock(side_effect=[(cuda.CUresult.CUDA_ERROR_UNKNOWN,), (cuda.CUresult.CUDA_SUCCESS,)])
    free = Mock(return_value=(cuda.CUresult.CUDA_SUCCESS,))
    monkeypatch.setattr(cuda, "cuMemUnmap", unmap)
    monkeypatch.setattr(cuda, "cuMemAddressFree", free)
    with pytest.raises(RuntimeError if sync_failure else AssertionError) as raised:
        _cleanup_mappings(0, [(1024, 128), (1152, 128)], 1024, 256)
    if sync_failure:
        assert raised.value is sync_error
    assert unmap.call_count == 2
    free.assert_called_once_with(1024, 256)


@pytest.mark.parametrize("owned", [False, True])
def test_rubin_gvr_initialization_preserves_error(monkeypatch, owned: bool) -> None:
    failure = ValueError("preparation failed")
    cleanup = RuntimeError("cleanup failed")
    executor = SimpleNamespace(domains=[object()], close=Mock(side_effect=[cleanup, None]))

    def initialize(prepared, *args, **kwargs):
        prepared.owns_executor, prepared.executor = owned, executor
        raise failure

    monkeypatch.setattr(locality.PreparedRowTopK, "_initialize", initialize)
    with pytest.raises(ValueError) as raised:
        locality.PreparedRowTopK(None, None, None, max_seq_len=32)
    assert raised.value is failure
    assert executor.close.call_count == int(owned)
    if owned:
        retry = raised.value.__cause__
        assert isinstance(retry, locality.LocalityCleanupError) and retry.__cause__ is cleanup
        retry.executor.close()
        assert executor.close.call_count == 2
    else:
        assert raised.value.__cause__ is None


@pytest.fixture
def mock_domain_api(monkeypatch):
    monkeypatch.setattr(
        cuda, "CUdevice_attribute", SimpleNamespace(CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT=1)
    )
    monkeypatch.setattr(cuda, "CU_DEV_SM_RESOURCE_GROUP_PARAMS", SimpleNamespace, raising=False)
    monkeypatch.setattr(
        cuda,
        "CUdevSmResourceGroup_flags",
        SimpleNamespace(
            CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID=2, CU_DEV_SM_RESOURCE_GROUP_BACKFILL=4
        ),
        raising=False,
    )
    monkeypatch.setattr(cuda, "cuDevSmResourceSplit", Mock(), raising=False)


@pytest.mark.parametrize("destroy_failure", [False, True])
def test_rubin_gvr_executor_partial_creation_cleanup(
    monkeypatch, mock_domain_api, destroy_failure
) -> None:
    success = cuda.CUresult.CUDA_SUCCESS
    resource = SimpleNamespace(sm=SimpleNamespace(smCount=104))
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "init", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(major=10, minor=7, multi_processor_count=208),
    )
    monkeypatch.setattr(cuda, "cuDeviceGetAttribute", lambda *_: (success, 2))
    monkeypatch.setattr(cuda, "cuDeviceGetDevResource", lambda *_: (success, resource))
    monkeypatch.setattr(
        cuda,
        "cuDevSmResourceSplit",
        lambda *_: (success, [resource, resource], SimpleNamespace(sm=SimpleNamespace(smCount=0))),
    )
    monkeypatch.setattr(cuda, "cuDevResourceGenerateDesc", lambda *_: (success, 1))
    monkeypatch.setattr(cuda, "cuGreenCtxCreate", Mock(side_effect=[(success, 10), (success, 11)]))
    monkeypatch.setattr(
        cuda,
        "cuGreenCtxStreamCreate",
        Mock(side_effect=[(success, 20), (cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY,)]),
    )
    monkeypatch.setattr(torch.cuda, "ExternalStream", Mock())
    monkeypatch.setattr(
        cuda,
        "cuStreamGetDevResource",
        lambda *_: (success, SimpleNamespace(sm=SimpleNamespace(smCount=104, localityDomainId=0))),
    )
    monkeypatch.setattr(cuda, "cuStreamSynchronize", lambda *_: (cuda.CUresult.CUDA_ERROR_UNKNOWN,))
    stream_destroy = Mock(
        side_effect=[(cuda.CUresult.CUDA_ERROR_UNKNOWN,), (success,)] if destroy_failure else None,
        return_value=(success,),
    )
    context_destroy = Mock(return_value=(success,))
    monkeypatch.setattr(cuda, "cuStreamDestroy", stream_destroy)
    monkeypatch.setattr(cuda, "cuGreenCtxDestroy", context_destroy)
    with pytest.raises(locality._CudaError) as raised:
        locality.LocalityDomainExecutor()
    assert raised.value.status == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY
    cause = raised.value.__cause__
    stream_destroy.assert_called_once_with(20)
    if destroy_failure:
        assert isinstance(cause, locality.LocalityCleanupError)
        assert [c.args[0] for c in context_destroy.call_args_list] == [11]
        monkeypatch.setattr(cuda, "cuStreamSynchronize", lambda *_: (success,))
        cause.executor.close()
        assert not cause.executor.domains and stream_destroy.call_count == 2
        cause = cause.__cause__
    assert cause.status == cuda.CUresult.CUDA_ERROR_UNKNOWN
    assert [call.args[0] for call in context_destroy.call_args_list] == [11, 10]


@pytest.mark.parametrize("invalid", ["counts", "alignment", "total"])
def test_rubin_gvr_rejects_incompatible_backfill(
    monkeypatch, invalid: str, mock_domain_api
) -> None:
    executor = _test_executor()

    def resource(count, alignment=4):
        return SimpleNamespace(sm=SimpleNamespace(smCount=count, smCoscheduledAlignment=alignment))

    plain, remainder = [resource(92), resource(92)], resource(24)
    filled = [
        resource(100 if invalid == "counts" else 104, 2 if invalid == "alignment" else 4),
        resource(104),
    ]
    rest = resource(4 if invalid == "total" else 0)
    monkeypatch.setattr(
        cuda, "cuDeviceGetDevResource", lambda *_: (cuda.CUresult.CUDA_SUCCESS, resource(208))
    )
    monkeypatch.setattr(
        cuda, "cuDevSmResourceSplit", lambda *_: (cuda.CUresult.CUDA_SUCCESS, filled, rest)
    )
    actual, remaining = executor._try_backfill(
        plain, remainder, [SimpleNamespace(smCount=0, flags=2) for _ in plain]
    )
    assert actual is plain and remaining is remainder and not executor.backfill_applied
