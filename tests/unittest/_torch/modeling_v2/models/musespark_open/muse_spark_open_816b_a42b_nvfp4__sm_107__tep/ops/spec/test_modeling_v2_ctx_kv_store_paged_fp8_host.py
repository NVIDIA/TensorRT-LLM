# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The page dtypes ``ctx_kv_store_paged`` takes, and its reference into FP8 pages, on the host.

``ops/spec/ctx_kv_store_paged.py`` is loaded from its file with Triton and the op catalog
stubbed and ``dflash_kv_storage`` loaded by path (the kernel is not run). The tests pin which
K/V / page dtype pairs the op takes: pages of the K/V dtype, and FP8 E4M3 pages for BF16 K/V
(the FP8 draft KV pool), all layers one dtype; and that its reference stores into FP8 pages
exactly what ``to_kv_storage`` makes of the BF16 values (saturated at +-448, NaN kept).
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_SPEC = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/spec"
)
_STORAGE = _ROOT / "tensorrt_llm/_torch/speculative/dflash_kv_storage.py"
_PACKAGE = "_musespark_open_tep_ctx_kv_store_paged_fp8_under_test"
_CATALOG = "tensorrt_llm._torch._experimental.modeling_v2.catalog._op"
_STORAGE_NAME = "tensorrt_llm._torch.speculative.dflash_kv_storage"
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
FP16 = torch.float16
PAGE = 32
LAYERS = 5
HEADS = 2
HEAD_DIM = 128


def _triton_stub() -> tuple[types.ModuleType, types.ModuleType]:
    language = types.ModuleType("triton.language")
    language.constexpr = object
    triton = types.ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f)
    triton.language = language
    return triton, language


def _load():
    triton, language = _triton_stub()
    catalog = types.ModuleType(_CATALOG)
    catalog.Arch = enum.Enum("Arch", "SM_107")
    catalog.OpWrapper = type("OpWrapper", (), {})
    spec = importlib.util.spec_from_file_location(_STORAGE_NAME, _STORAGE)
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    stubs = {
        "triton": triton,
        "triton.language": language,
        _CATALOG: catalog,
        _STORAGE_NAME: storage,
    }
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_SPEC)]
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    sys.modules[_PACKAGE] = package
    try:
        store = importlib.import_module(f"{_PACKAGE}.ctx_kv_store_paged")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return store, storage


STORE, STORAGE = _load()


@pytest.mark.parametrize(
    "kv_dtype,page_dtype,taken",
    [
        (BF16, BF16, True),
        (FP16, FP16, True),
        (BF16, FP8, True),
        (FP16, FP8, False),
        (BF16, FP16, False),
        (FP16, BF16, False),
        (torch.float32, FP8, False),
    ],
)
def test_page_dtypes_the_op_takes(kv_dtype, page_dtype, taken):
    assert STORE.takes_page_dtype(kv_dtype, page_dtype) is taken


def _args(n, kv_dtype, page_dtype, seed, slots=6, blocks=4):
    """The op's arguments over the manager's per-layer views of one fresh pool, and the pool."""
    pages = slots * blocks + 3
    g = torch.Generator().manual_seed(seed)
    pool = torch.randn((pages, LAYERS, 2, HEADS, PAGE, HEAD_DIM), generator=g).to(page_dtype)
    caches = [pool[:, layer] for layer in range(LAYERS)]
    table = torch.randperm(pages, generator=g)[: slots * blocks].to(torch.int32)
    table = table.view(slots, blocks).contiguous()
    flat = torch.randperm(slots * blocks * PAGE, generator=g)[:n]
    rows = (flat // (blocks * PAGE)).contiguous()
    positions = (flat % (blocks * PAGE)).contiguous()
    k = torch.randn((n, LAYERS, HEADS, HEAD_DIM), generator=g).to(kv_dtype)
    v = torch.randn((n, LAYERS, HEADS, HEAD_DIM), generator=g).to(kv_dtype)
    return [k, v, rows, positions, table, PAGE, caches], pool


@pytest.mark.parametrize("kv_dtype,page_dtype", [(BF16, BF16), (FP16, FP16), (BF16, FP8)])
def test_is_valid_takes(kv_dtype, page_dtype):
    args, _ = _args(7, kv_dtype, page_dtype, seed=1)
    STORE.ctx_kv_store_paged.is_valid(*args)


@pytest.mark.parametrize("case", ["fp16_kv_fp8_pages", "bf16_kv_fp16_pages", "mixed_pages"])
def test_is_valid_refuses(case):
    if case == "fp16_kv_fp8_pages":
        args, _ = _args(7, FP16, FP8, seed=2)
    elif case == "bf16_kv_fp16_pages":
        args, _ = _args(7, BF16, FP16, seed=2)
    else:
        args, _ = _args(7, BF16, FP8, seed=2)
        bf16_args, _ = _args(7, BF16, BF16, seed=2)
        args[6] = args[6][:-1] + bf16_args[6][-1:]
    with pytest.raises(ValueError, match="caches must be"):
        STORE.ctx_kv_store_paged.is_valid(*args)


def _by_entry(pool):
    """``[pages, page_size, layers, 2, heads, head_dim]``: one row per stored entry."""
    return pool.permute(0, 4, 1, 2, 3, 5)


def test_reference_into_fp8_pages_is_the_storage_cast_of_the_bf16_store():
    n = 40
    args, fp8_pool = _args(n, BF16, FP8, seed=3)
    bf16_args, bf16_pool = _args(n, BF16, BF16, seed=3)
    # Past E4M3's range, infinities and NaN go through the same cast.
    k, v = args[0] * 300, args[1] * 300
    k.view(-1)[:4] = torch.tensor([1e4, -1e4, float("inf"), float("nan")], dtype=BF16)
    v.view(-1)[:3] = torch.tensor([-float("inf"), 449.0, -449.0], dtype=BF16)
    args[0], args[1] = k, v
    bf16_args[0], bf16_args[1] = k, v
    before = fp8_pool.clone()
    STORE.ctx_kv_store_paged.reference(*args)
    STORE.ctx_kv_store_paged.reference(*bf16_args)

    rows, positions, table = args[2], args[3], args[4]
    pages = table[rows.long(), positions.long() // PAGE].long()
    entries = positions.long() % PAGE
    written = torch.zeros(fp8_pool.shape[0], PAGE, dtype=torch.bool)
    written[pages, entries] = True
    got = _by_entry(fp8_pool)[written]
    want = STORAGE.to_kv_storage(_by_entry(bf16_pool)[written], FP8)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    # Saturated, not overflowed: every written value is finite but the one NaN that went in.
    assert int(got.float().isnan().sum()) == 1
    assert float(got.float().nan_to_num(0.0).abs().max()) == STORAGE.E4M3_MAX
    # Nothing else changed.
    assert torch.equal(
        _by_entry(fp8_pool)[~written].view(torch.uint8),
        _by_entry(before)[~written].view(torch.uint8),
    )
