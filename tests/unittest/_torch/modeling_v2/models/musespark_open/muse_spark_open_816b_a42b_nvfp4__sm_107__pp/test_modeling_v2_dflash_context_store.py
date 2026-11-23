# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 PP target's one-launch drafter-context store, on the host only (CPU torch).

``kernels/dflash_context/store.py`` is loaded from its file (it imports nothing of the
package at load). The tests pin:

* the step's RoPE positions: per request ``arange(first, first + rows)`` as int32, every
  position repeated per drafter layer, as ``_fused_rope_inplace`` builds them;
* the step's host buffer: the store table and the positions, read back through the views
  the device copy is split into;
* the torch reference of the store against flashinfer's ``append_paged_kv_cache``
  addressing written out token by token (``kv_indptr[row] + position // page`` into the
  flattened table, offset ``position % page``), with permuted pages, block-table rows other
  than the batch order, first positions off a page boundary and rows crossing pages;
* FP8 draft pages: the store's conversion (``to_cache_dtype``) is the shared draft-KV cast
  (``speculative/dflash_kv_storage.py``) on every BF16 bit pattern, the reference store into
  FP8 pages holds the converted BF16 store, the K/V are computed in the weights' dtype, and
  ``refusal`` takes BF16 or FP8 pages with BF16 weights, one dtype for every layer.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_STORE = (
    _ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    / "muse_spark_open_816b_a42b_nvfp4__sm_107__pp/kernels/dflash_context/store.py"
)


def _load_store():
    name = "_mso_pp_dflash_store_under_test"
    spec = importlib.util.spec_from_file_location(name, _STORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STORE = _load_store()


def _runs(*fields):
    return [STORE.Run(offset=o, rows=r, table_row=t, first=f) for o, r, t, f in fields]


@pytest.mark.parametrize("layers", [1, 2, 5])
def test_rope_positions_are_the_built_in_repeated_int32_positions(layers):
    runs = _runs((0, 1, 0, 0), (1, 31, 3, 5), (32, 33, 1, 4093), (65, 32, 2, 32))
    expected = torch.cat(
        [
            torch.arange(run.first, run.first + run.rows, dtype=torch.long)
            .to(torch.int32)
            .repeat_interleave(layers)
            for run in runs
        ]
    )
    got = STORE.rope_positions(runs, layers)
    assert got.dtype == torch.int32
    assert torch.equal(got, expected)
    assert STORE.rope_positions([], layers).numel() == 0


@pytest.mark.parametrize("total_odd", [False, True])
def test_host_table_round_trips_through_the_device_views(total_odd):
    layers = 5
    runs = _runs((0, 3, 2, 7), (3, 2 if total_odd else 4, 0, 64))
    addresses = [(0x7F0000001000, 0x7F0000802000), (0x7F0000003000, 0x7F0000904000)]
    buffer = STORE.host_table(runs, addresses, layers, pin=False)
    total = sum(run.rows for run in runs)
    meta, positions = STORE.split_table(buffer, len(runs), total * layers)
    assert meta.tolist() == [
        0x7F0000001000,
        0x7F0000802000,
        3,
        2,
        7,
        0,
        0x7F0000003000,
        0x7F0000904000,
        runs[1].rows,
        0,
        64,
        0,
    ]
    assert torch.equal(positions, STORE.rope_positions(runs, layers))
    assert buffer.numel() == len(runs) * STORE.META_FIELDS + (total * layers + 1) // 2


def _append_by_token(runs, ks, kv_flats, table, caches):
    """flashinfer's append addressing, one token at a time: ``kv_indptr[b] = b * width``."""
    indices = table.flatten().tolist()
    width = table.shape[1]
    page_size = caches[0].shape[3]
    for run, k, kv_flat in zip(runs, ks, kv_flats):
        rows, layers, heads, head_dim = k.shape
        v = kv_flat.view(rows, layers, 2, heads, head_dim)[:, :, 1]
        for t in range(rows):
            position = run.first + t
            page = indices[run.table_row * width + position // page_size]
            entry = position % page_size
            for layer, cache in enumerate(caches):
                cache[page, 0, :, entry, :] = k[t, layer]
                cache[page, 1, :, entry, :] = v[t, layer]


def _random_bits(*shape, generator):
    bits = torch.randint(-(1 << 15), 1 << 15, shape, dtype=torch.int32, generator=generator)
    return bits.to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize("layout", ["arena", "page_major"])
def test_reference_store_is_the_append_addressing(layout):
    generator = torch.Generator().manual_seed(3)
    layers, heads, head_dim, page_size, rows_per_table = 3, 2, 8, 4, 6
    table_rows = 4
    pages = table_rows * rows_per_table
    if layout == "arena":
        storage = _random_bits(layers, pages, 2, heads, page_size, head_dim, generator=generator)
        caches = [storage[layer] for layer in range(layers)]
    else:
        storage = _random_bits(pages, layers, 2, heads, page_size, head_dim, generator=generator)
        caches = [storage[:, layer] for layer in range(layers)]
    table = torch.randperm(pages, generator=generator).to(torch.int32).view(table_rows, -1)
    runs = _runs((0, 1, 3, 0), (1, 9, 0, 3), (10, 4, 2, 8), (14, 7, 1, 13))
    ks = [_random_bits(r.rows, layers, heads, head_dim, generator=generator) for r in runs]
    kv_flats = [
        _random_bits(r.rows, layers * 2 * heads * head_dim, generator=generator) for r in runs
    ]
    expected = storage.clone()
    if layout == "arena":
        expected_caches = [expected[layer] for layer in range(layers)]
    else:
        expected_caches = [expected[:, layer] for layer in range(layers)]
    _append_by_token(runs, ks, kv_flats, table, expected_caches)
    STORE.reference_store(runs, ks, kv_flats, table, caches)
    assert torch.equal(storage.view(torch.int16), expected.view(torch.int16))


# --------------------------------------------------------------------------
# FP8 draft pages
# --------------------------------------------------------------------------
_STORAGE_PATH = _ROOT / "tensorrt_llm/_torch/speculative/dflash_kv_storage.py"


def _load_storage():
    name = "_mso_pp_dflash_kv_storage_under_test"
    spec = importlib.util.spec_from_file_location(name, _STORAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STORAGE = _load_storage()
FP8 = torch.float8_e4m3fn


def test_store_conversion_is_the_shared_draft_kv_cast():
    bits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    ours = STORE.to_cache_dtype(x, FP8)
    shared = STORAGE.to_kv_storage(x, FP8)
    nan = x.float().isnan()
    assert torch.equal(ours.view(torch.uint8)[~nan], shared.view(torch.uint8)[~nan])
    assert bool(ours[nan].float().isnan().all())
    assert STORE.E4M3_MAX == STORAGE.E4M3_MAX
    saturated = STORE.to_cache_dtype(
        torch.tensor([1.0e4, -math.inf, 464.0], dtype=torch.bfloat16), FP8
    )
    assert saturated.float().tolist() == [448.0, -448.0, 448.0]
    same = torch.zeros(2, dtype=torch.bfloat16)
    assert STORE.to_cache_dtype(same, torch.bfloat16) is same


def test_compute_dtype_is_the_weights_for_fp8_pages():
    drafter = SimpleNamespace(_fused_kv_weight=torch.zeros(1, dtype=torch.bfloat16))
    assert STORE.compute_dtype(drafter, FP8) == torch.bfloat16
    # Other pages are computed in their own dtype; the weights are not read.
    assert STORE.compute_dtype(SimpleNamespace(), torch.bfloat16) == torch.bfloat16
    assert STORE.CACHE_DTYPES == (torch.bfloat16, FP8)


def test_compute_dtype_builds_the_fused_kv_weights_for_fp8_pages():
    """A step that skipped ``refusal`` (store_override, stream capture) can be the first to ask."""
    built = []

    def build():
        built.append(True)
        drafter._fused_kv_weight = torch.zeros(1, dtype=torch.bfloat16)

    drafter = SimpleNamespace(_fused_kv_weight=None, _build_fused_kv_buffers=build)
    assert STORE.compute_dtype(drafter, FP8) == torch.bfloat16
    assert built == [True]
    # Built once: a second call reads the weights.
    assert STORE.compute_dtype(drafter, FP8) == torch.bfloat16
    assert built == [True]
    # BF16 pages never build them.
    unbuilt = SimpleNamespace(_fused_kv_weight=None, _build_fused_kv_buffers=build)
    assert STORE.compute_dtype(unbuilt, torch.bfloat16) == torch.bfloat16
    assert built == [True]


@pytest.mark.parametrize("layout", ["arena", "page_major"])
def test_reference_store_into_fp8_pages_holds_the_converted_bf16_store(layout):
    generator = torch.Generator().manual_seed(5)
    layers, heads, head_dim, page_size, rows_per_table = 3, 2, 8, 4, 6
    table_rows = 4
    pages = table_rows * rows_per_table
    shape = (
        (layers, pages, 2, heads, page_size, head_dim)
        if layout == "arena"
        else (pages, layers, 2, heads, page_size, head_dim)
    )
    bf16 = _random_bits(*shape, generator=generator)
    fp8 = STORE.to_cache_dtype(bf16, FP8)

    def caches(storage):
        if layout == "arena":
            return [storage[layer] for layer in range(layers)]
        return [storage[:, layer] for layer in range(layers)]

    table = torch.randperm(pages, generator=generator).to(torch.int32).view(table_rows, -1)
    runs = _runs((0, 1, 3, 0), (1, 9, 0, 3), (10, 4, 2, 8), (14, 7, 1, 13))
    ks = [_random_bits(r.rows, layers, heads, head_dim, generator=generator) for r in runs]
    kv_flats = [
        _random_bits(r.rows, layers * 2 * heads * head_dim, generator=generator) for r in runs
    ]
    STORE.reference_store(runs, ks, kv_flats, table, caches(bf16))
    STORE.reference_store(runs, ks, kv_flats, table, caches(fp8))
    expected = STORE.to_cache_dtype(bf16, FP8)
    nan = expected.float().isnan()
    assert torch.equal(fp8.view(torch.uint8)[~nan], expected.view(torch.uint8)[~nan])
    assert bool(fp8[nan].float().isnan().all())


class _OnDevice:
    """A host tensor that reports itself as a CUDA tensor (``refusal`` reads ``is_cuda`` only)."""

    is_cuda = True

    def __init__(self, tensor):
        self._tensor = tensor

    def __getattr__(self, name):
        return getattr(self._tensor, name)


class _Drafter:
    """The drafter attributes ``refusal`` reads: the built-in class, its fused K/V tensors."""

    _input_ln_eps = None
    _num_attn_layers = 3
    _num_kv_heads = 2
    _head_dim = 8

    def __init__(self, base):
        self.fc = object()
        self.hidden_norm = object()
        self._fused_kv_weight = _OnDevice(torch.zeros(1, dtype=torch.bfloat16))
        self._fused_kv_bias = None
        self._k_norm_stacked = torch.zeros(1, dtype=torch.bfloat16)
        self.__class__ = type("_DrafterOfBase", (base, _Drafter), {})

    def _build_fused_kv_buffers(self):
        pass


@pytest.fixture
def built_in_dflash(monkeypatch):
    """The ``modeling_dflash`` names ``refusal`` imports, without the package."""

    class DFlashForCausalLM:
        def precompute_context_kv(self, *args):
            raise AssertionError("not called")

        def _fused_rope_inplace(self, *args):
            raise AssertionError("not called")

    modeling_dflash = types.ModuleType("tensorrt_llm._torch.models.modeling_dflash")
    modeling_dflash.DFlashForCausalLM = DFlashForCausalLM
    modeling_dflash._flashinfer_rope = object()
    names = ("tensorrt_llm", "tensorrt_llm._torch", "tensorrt_llm._torch.models")
    for name in names:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["tensorrt_llm._torch.models"].modeling_dflash = modeling_dflash
    monkeypatch.setitem(sys.modules, modeling_dflash.__name__, modeling_dflash)
    return DFlashForCausalLM


def _worker(page_dtypes):
    caches = [
        _OnDevice(torch.zeros(4, 2, 2, 4, 8, dtype=torch.bfloat16).to(dtype))
        for dtype in page_dtypes
    ]
    return SimpleNamespace(
        _ctx_paged=True,
        _ctx_kv_buf=caches,
        _ctx_block_tables=torch.zeros(2, 3, dtype=torch.int32),
        _ctx_page_table=None,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, FP8])
def test_refusal_takes_bf16_or_fp8_pages(built_in_dflash, dtype):
    assert STORE.refusal(_worker([dtype] * 3), _Drafter(built_in_dflash)) is None


@pytest.mark.parametrize(
    "page_dtypes,match",
    [
        ([FP8, torch.bfloat16, torch.bfloat16], "draft layer's pages"),
        ([torch.bfloat16, FP8, FP8], "draft layer's pages"),
        ([torch.float16] * 3, "not one of"),
    ],
    ids=["fp8_first", "bf16_first", "fp16"],
)
def test_refusal_names_page_dtypes_it_does_not_take(built_in_dflash, page_dtypes, match):
    reason = STORE.refusal(_worker(page_dtypes), _Drafter(built_in_dflash))
    assert reason is not None and match in reason
    assert str(page_dtypes[1] if match != "not one of" else page_dtypes[0]) in reason
