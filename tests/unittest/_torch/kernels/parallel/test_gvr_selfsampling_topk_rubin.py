# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rubin GVR dispatch, allocation ownership and graph-replay regression tests."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import gcd, lcm

import pytest
import torch
from cuda.bindings import driver as cuda

pytest.importorskip("cutlass.cute")
pytest.importorskip("cutlass.utils.rubin_helpers")

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling_host as gvr
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_locality as locality
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_locality_policy as policy
from tensorrt_llm._torch.kernels.rubin.top_k import gvr_topk_rubin_dispatch as dispatch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin GVR tests require an SM107 GPU",
)

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _checked(result: tuple) -> object:
    assert result[0] == cuda.CUresult.CUDA_SUCCESS, result[0]
    return result[1] if len(result) == 2 else result[1:]


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
    with locality.LocalityDomainExecutor(torch.cuda.current_device()) as executor:
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
) -> Iterator[tuple[torch.Tensor, tuple[int, ...]]]:
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
            finally:
                _checked(cuda.cuMemRelease(handle))
            mapped.append((base + offset, span))
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
        assert locality._allocation_rows(x, device) == tuple(counts)
        yield x, tuple(counts)
    finally:
        torch.cuda.synchronize(device)
        for pointer, span in mapped:
            _checked(cuda.cuMemUnmap(pointer, span))
        _checked(cuda.cuMemAddressFree(base, size))


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
) -> None:
    original = x.clone()
    call()
    torch.cuda.synchronize()
    _check_exact(x, out, [n] * len(x))
    assert torch.equal(x, original)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            call()
        # Replay once with the prepared envelope, then with changed scores and lengths.
        for step in range(2):
            valid = (
                [n] * len(x)
                if step == 0
                else [(0, out.shape[1] - 3, out.shape[1] + 3, n - 11)[r % 4] for r in range(len(x))]
            )
            lengths.copy_(torch.tensor(valid, dtype=torch.int32, device=x.device) * cr)
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
def test_rubin_gvr_graph_exactness(dtype: torch.dtype, k: int, rows: int, n: int) -> None:
    """Native dtype dispatch must cover odd tails and changing device KV lengths."""
    cr = 1 if k == 2048 else 4
    x = torch.empty((rows, (n + 7) // 8 * 8), dtype=dtype, device="cuda")
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


@pytest.mark.parametrize("k,rows,n", [(512, 1, 8195), (2048, 16, 8207)])
def test_rubin_gvr_bf16_pcw_graph(k: int, rows: int, n: int) -> None:
    """PCW must handle 16-byte row alignment with both histogram bin counts."""
    cr = 1 if k == 2048 else 4
    x = torch.empty((rows, (n + 7) // 8 * 8), dtype=torch.bfloat16, device="cuda")
    assert x.shape[1] % 8 == 0 and x.shape[1] % 64 != 0
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tuning = dispatch.select_lowp_tuning(rows, n, x.shape[1], k, sms, 107)
    assert tuning is not None and tuning.pcw_aligned
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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_rubin_gvr_locality_graph(
    dtype: torch.dtype, k: int, domain_executor: locality.LocalityDomainExecutor
) -> None:
    """Full-batch planning and per-domain workspaces survive graph replay."""
    rows, n = max(512, (512 * domain_executor.physical_sm_count + 207) // 208), 131075
    cr = 1 if k == 2048 else 4
    with _localized_rows(rows, n, dtype, domain_executor) as (x, counts):
        _fill(x, n)
        lengths = torch.full((rows,), n * cr, dtype=torch.int32, device="cuda")
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(
            x,
            lengths,
            out,
            max_seq_len=n * cr,
            compress_ratio=cr,
            row_counts=counts,
            executor=domain_executor,
        )
        try:
            assert prepared.locality_allocation and prepared.mode == "local_preserve"
            assert len(prepared.parts) == sum(c > 0 for c in counts)
            plan = gvr._varlen_launcher(
                rows,
                x.shape[1],
                k,
                n,
                1,
                cr,
                domain_executor.physical_sm_count,
                107,
                dtype=dtype,
                device_index=x.get_device(),
            )
            assert all(bound.launcher is plan for _, bound, _ in prepared.parts)
            _check_replay(prepared, x, lengths, out, n, cr)
        finally:
            prepared.close()
        assert not domain_executor.closed  # Caller-owned resources outlive a prepared call.
        prepared.close()
        with pytest.raises(RuntimeError, match="closed"):
            prepared()


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


def test_rubin_gvr_automatic_backfill(domain_executor: locality.LocalityDomainExecutor) -> None:
    rows = 1024 * domain_executor.physical_sm_count // 208
    n, k = 131075, 512
    with _localized_rows(rows, n, torch.float32, domain_executor) as (x, _):
        _fill(x, n)
        lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n)
        owned = prepared.executor
        try:
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


def test_rubin_gvr_complete_row_ownership(domain_executor: locality.LocalityDomainExecutor) -> None:
    weights = [d.sm_count for d in domain_executor.domains]
    rows, n = sum(weights) // gcd(*weights), 4099
    with _localized_rows(rows, n, torch.float32, domain_executor) as (x, counts):
        assert locality._allocation_rows(x.reshape(1, -1), x.get_device()) is None
        bad_counts = (counts[0] + 1, *counts[1:-1], counts[-1] - 1)
        out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        lengths = torch.full((rows,), n, dtype=torch.int32, device="cuda")
        prepared = locality.PreparedRowTopK(x, lengths, out, max_seq_len=n, row_counts=bad_counts)
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
    assert not policy.choose_backfill(torch.float32, 512, 131071, 1024, 208, 107)
    assert policy.choose_backfill(torch.float32, 512, 131072, 1024, 208, 107)
    assert not policy.choose_backfill(torch.float32, 512, 262143, 1025, 208, 107)
    assert policy.choose_backfill(torch.float32, 512, 262144, 1025, 208, 107)
    assert not policy.choose_backfill(torch.float32, 512, 131072, 4096, 208, 107)
    assert policy.choose_backfill(torch.float32, 512, 131072, 4097, 208, 107)
    for dtype in (torch.float16, torch.bfloat16):
        assert not policy.choose_backfill(dtype, 512, 1 << 20, 8192, 208, 107)
    for sms, rows in ((104, 256), (208, 512), (416, 1024)):
        assert policy.choose_locality(torch.float32, 512, 131072, rows, sms, 107) == (
            "local_preserve",
            4,
        )
    assert policy.choose_locality(torch.float32, 512, 131072, 512, 208, 100) is None


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
