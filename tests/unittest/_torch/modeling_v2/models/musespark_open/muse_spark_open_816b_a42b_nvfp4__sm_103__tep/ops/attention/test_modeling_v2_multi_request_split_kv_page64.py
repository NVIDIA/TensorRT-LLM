# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``MUSESPARK_OPEN_MULTI_REQUEST_SPLIT_KV_PAGE64`` of ``multi_request_split_kv``, on the host only.

The selection site (``kernels/multi_request_split_kv/attention.py``) and the op's host side
(``ops/attention/multi_request_split_kv.py``) are loaded from their files with ``torch``,
``triton`` and the op catalog stubbed for the load; no kernel is built or launched. The tests pin:

* gate off (unset, empty or ``0``): the selection site takes a cache of 32-token blocks and
  declines every other block size with ``another cache view``; the build key and the install
  step are the single 32-token build;
* gate on: the 64-token cache is taken too, its build is part of the compiled key, ``install``
  compiles both builds, and a capture needs the 64-token build compiled first;
* the kernel's 64-token index math (each block read as its two 32-token halves) addresses, for
  every page, the same K and V boxes as the 32-token path over a pool that holds the same keys.

The kernel itself needs a GPU: see ``test_modeling_v2_multi_request_split_kv_page64_gpu.py``.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
ENV = "MUSESPARK_OPEN_MULTI_REQUEST_SPLIT_KV_PAGE64"
PAGE, D, HKV = 32, 64, 2


class _Dtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


F8, BF16, I32, F32 = _Dtype("fp8"), _Dtype("bf16"), _Dtype("i32"), _Dtype("f32")
_DEVICE = types.SimpleNamespace(index=0, type="cuda")
_STATE = {"capturing": False}
_LOG = []


class _Tensor:
    """The host-side reads of a tensor the selection site makes: shape, dtype, layout."""

    def __init__(self, shape, dtype=F8, contiguous=True):
        self.shape, self.dtype, self._contiguous, self.device = (
            tuple(shape),
            dtype,
            contiguous,
            _DEVICE,
        )

    def dim(self):
        return len(self.shape)

    def numel(self):
        return int(np.prod(self.shape))

    def is_contiguous(self):
        return self._contiguous


def _stub_modules():
    torch = types.ModuleType("torch")
    torch.Tensor, torch.float8_e4m3fn, torch.bfloat16 = _Tensor, F8, BF16
    torch.int32, torch.float32 = I32, F32
    torch.nn = types.SimpleNamespace(Module=object)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_current_stream_capturing=lambda: _STATE["capturing"],
        current_device=lambda: 0,
        get_device_capability=lambda *_: (10, 3),
    )
    triton = types.ModuleType("triton")
    triton.jit = lambda *a, **k: a[0] if len(a) == 1 and callable(a[0]) else (lambda f: f)
    triton.language = types.ModuleType("triton.language")
    triton.language.constexpr = int
    catalog = types.ModuleType("tensorrt_llm._torch._experimental.modeling_v2.catalog._op")
    catalog.OpWrapper = type("OpWrapper", (), {})
    catalog.Arch = types.SimpleNamespace(SM_103=103)
    catalog.true_fp32_matmul = None
    utils = types.ModuleType("tensorrt_llm._torch.attention.backends.fmha.utils")
    utils.get_bmm1_scale = lambda *a, **k: 0.125
    logger = types.ModuleType("tensorrt_llm.logger")
    logger.logger = types.SimpleNamespace(
        info=lambda *a, **k: _LOG.append(a[0]), warning=lambda *a, **k: None
    )
    return {
        "torch": torch,
        "triton": triton,
        "triton.language": triton.language,
        catalog.__name__: catalog,
        utils.__name__: utils,
        logger.__name__: logger,
    }


def _load():
    """``(ops module, selection-site module)`` of the target, loaded from their files."""
    tag = "_page64_under_test"
    stubs = _stub_modules()
    package = {}
    for name in (
        tag,
        f"{tag}.kernels",
        f"{tag}.kernels.multi_request_split_kv",
        f"{tag}.ops",
        f"{tag}.ops.attention",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        package[name] = module
    attention = types.ModuleType(f"{tag}.attention")
    attention.trtllm_gen_fmha = lambda *_: types.SimpleNamespace(_enable_pdl=False)
    forward = types.ModuleType(f"{tag}.attention_forward")
    forward.decode_reachable = lambda *a, **k: True
    forward.intercepting = lambda *a, **k: lambda f: f
    package[attention.__name__], package[forward.__name__] = attention, forward
    replaced = {**stubs, **package}
    saved = {name: sys.modules.get(name) for name in replaced}
    sys.modules.update(replaced)
    loaded = []
    try:
        for name, relative in (
            (
                f"{tag}.ops.attention.multi_request_split_kv",
                "ops/attention/multi_request_split_kv.py",
            ),
            (
                f"{tag}.kernels.multi_request_split_kv.attention",
                "kernels/multi_request_split_kv/attention.py",
            ),
        ):
            spec = importlib.util.spec_from_file_location(name, _TARGET / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(sys.modules[name.rpartition(".")[0]], name.rpartition(".")[2], module)
            loaded.append(module)
    finally:
        for name in [*replaced, *(m.__name__ for m in loaded)]:
            if saved.get(name) is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
    return loaded[0], loaded[1]


MR, AT = _load()


def _kwargs(pool, kv_heads=4, **override):
    kwargs = dict(
        query=_Tensor((16, 8 * kv_heads, 64), F8),
        kv_cache=(pool, pool),
        workspace_buffer=None,
        block_tables=_Tensor((2, 2, 8), I32),
        seq_lens=_Tensor((2,), I32),
        max_seq_len=100,
        bmm1_scale=_Tensor((1,), F32),
        bmm2_scale=_Tensor((1,), F32),
        window_left=-1,
        out=_Tensor((16, 8 * kv_heads, 64), BF16),
        sinks=None,
        kv_layout="HND",
        enable_pdl=False,
        backend="trtllm-gen",
        q_len_per_req=8,
        max_q_len=None,
        cum_seq_lens_q=None,
        kv_cache_sf=None,
        uses_shared_paged_kv_idx=False,
        bmm1_scale_log2=None,
        multi_ctas_kv_counter_buffer=None,
    )
    kwargs.update(override)
    return kwargs


def _declines(kv_heads, pool):
    return AT.MultiRequestCall(kv_heads=kv_heads, window=0).decline(_kwargs(pool, kv_heads))


def _tip_accepts(kv_heads, pool):
    """The selection site's pool test before the 64-token cache: 32-token blocks only."""
    return (
        pool.dtype is F8
        and pool.dim() == 4
        and tuple(pool.shape[1:]) == (kv_heads, 32, 64)
        and pool.is_contiguous()
    )


def _pools(kv_heads):
    for page, heads, dtype, contiguous, rank in itertools.product(
        (8, 16, 31, 32, 33, 64, 128), (2, 4, 8), (F8, BF16), (True, False), (4, 3)
    ):
        shape = (50, heads, page, 64) if rank == 4 else (50, heads, page)
        yield _Tensor(shape, dtype, contiguous)
    yield _Tensor((50, kv_heads, 32, 128))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    _STATE["capturing"] = False
    _LOG.clear()
    yield
    _STATE["capturing"] = False


@pytest.mark.parametrize("value", [None, "", "0", "true"])
def test_gate_off_takes_32_token_pools_only(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert AT.cache_pages() == (32,)
    for kv_heads in (2, 4):
        for pool in _pools(kv_heads):
            expected = None if _tip_accepts(kv_heads, pool) else "another cache view"
            assert _declines(kv_heads, pool) == expected, (kv_heads, pool.shape, pool.dtype)


def test_gate_on_takes_32_and_64_token_pools(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert AT.cache_pages() == (32, 64)
    for kv_heads in (2, 4):
        assert _declines(kv_heads, _Tensor((50, kv_heads, 32, 64))) is None
        assert _declines(kv_heads, _Tensor((50, kv_heads, 64, 64))) is None
        for page in (8, 16, 31, 33, 96, 128):
            assert _declines(kv_heads, _Tensor((50, kv_heads, page, 64))) == "another cache view"
        assert _declines(kv_heads, _Tensor((50, kv_heads, 64, 64), contiguous=False))
        assert _declines(kv_heads, _Tensor((50, kv_heads, 64, 64), BF16))
        assert _declines(kv_heads, _Tensor((50, kv_heads, 64, 128)))


def test_compiled_key_carries_the_cache_page():
    key = MR.compiled_key(0, True, 4, 0, q_fp8=True, device_scale=True, requests=64)
    assert len(key) == 10 and key[9] == 32
    wide = MR.compiled_key(0, True, 4, 0, q_fp8=True, device_scale=True, requests=64, cache_page=64)
    assert wide[:9] == key[:9] and wide[9] == 64


def test_a_capture_needs_the_precompiled_64_token_build(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(MR, "_compiled_keys", set())
    pool = _Tensor((50, 4, 64, 64))
    call = AT.MultiRequestCall(kv_heads=4, window=0)
    kwargs = _kwargs(pool)
    assert call.decline(kwargs) is None  # outside a capture nothing is compiled ahead
    _STATE["capturing"] = True
    reason = "multi_request_split_kv was not compiled before this capture"
    assert call.decline(kwargs) == reason
    MR._compiled_keys.add(
        MR.compiled_key(0, False, 4, 0, q_fp8=True, device_scale=True, requests=2, cache_page=32)
    )
    assert call.decline(kwargs) == reason  # the 32-token build does not serve a 64-token pool
    MR._compiled_keys.add(
        MR.compiled_key(0, False, 4, 0, q_fp8=True, device_scale=True, requests=2, cache_page=64)
    )
    assert call.decline(kwargs) is None


class _Attention:
    is_sliding = False
    num_key_value_heads, num_heads, head_dim, attention_window_size = 4, 32, 64, 0
    attn = object()

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx


def _install(monkeypatch):
    calls = []
    monkeypatch.setattr(MR, "precompile", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(AT, "trtllm_gen_fmha", lambda attn: types.SimpleNamespace(_enable_pdl=True))
    monkeypatch.setattr(AT, "get_bmm1_scale", lambda attn: 0.125)
    monkeypatch.setattr(AT, "decode_reachable", lambda: None)
    model = types.SimpleNamespace(
        layers=[types.SimpleNamespace(self_attn=_Attention(i)) for i in range(4)]
    )
    assert AT.install(model, False) == 4
    return calls, list(_LOG)


def test_install_compiles_the_32_token_build_only_with_the_gate_off(monkeypatch):
    calls, log = _install(monkeypatch)
    assert [k["cache_page"] for _, k in calls] == [32]
    assert len(log) == 1 and log[0].endswith("q=E4M3 + device scales)")
    assert "cache blocks" not in log[0]


def test_install_compiles_both_builds_with_the_gate_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    calls, log = _install(monkeypatch)
    assert [k["cache_page"] for _, k in calls] == [32, 64]
    assert calls[0][0] == calls[1][0]  # the same build otherwise
    assert log[0].endswith("q=E4M3 + device scales, cache blocks of [32, 64] tokens)")


def test_compile_passes_the_cache_page_next_to_the_ring_depth(monkeypatch):
    seen = []
    monkeypatch.setattr(MR, "_compiled_kernel", lambda *a, **k: seen.append(k))
    monkeypatch.setattr(MR, "ring_stages", lambda device, nq: 20)
    MR._compile(MR.compiled_key(0, True, 4, 0, q_fp8=True, device_scale=True, cache_page=64))
    MR._compile(MR.compiled_key(0, True, 4, 0, q_fp8=True, device_scale=True))
    assert [k["cache_page"] for k in seen] == [64, 32]
    assert all(k["stages"] == 20 for k in seen)


# The loader of ``MultiRequestSplitKvKernel`` over a 64-token pool, emulated: K is one head's
# 32-key box, V a pair's two heads' box, at the indices below; SUB 1 is the 32-token path.
def _k_box(flat, index, count):  # K view: (32 keys, 64 dims, count), strides (64, 1, 32 * 64)
    if not 0 <= index < count:
        return None  # a negative index lies outside the view: the TMA fills zeros
    return flat[index * PAGE * D : (index + 1) * PAGE * D].reshape(PAGE, D)


def _v_box(flat, index, count, head_stride, batch_stride):  # V view: ((64, 2 heads), 32, count)
    if not 0 <= index < count:
        return None
    keys = np.arange(PAGE)[:, None] * D + np.arange(D)
    return np.stack([flat[index * batch_stride + h * head_stride + keys] for h in range(HKV)])


@pytest.mark.parametrize("kv_heads", [2, 4])
def test_64_token_views_address_the_keys_of_the_32_token_path(kv_heads):
    rng = np.random.default_rng(0)
    pairs, blocks = kv_heads // HKV, 50
    pool64 = rng.integers(0, 256, size=(blocks, kv_heads, 2 * PAGE, D), dtype=np.uint8)
    # block 2 B + s of the 32-token pool is half s of block B of the 64-token pool
    pool32 = pool64.reshape(blocks, kv_heads, 2, PAGE, D).transpose(0, 2, 1, 3, 4)
    pool32 = np.ascontiguousarray(pool32.reshape(2 * blocks, kv_heads, PAGE, D))
    flat64, flat32 = pool64.reshape(-1), pool32.reshape(-1)
    k_count64, k_count32 = 2 * kv_heads * blocks, kv_heads * 2 * blocks
    for length in (1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 1000):
        pages32 = -(-length // PAGE)
        table64 = rng.permutation(blocks)[: 2 * -(-length // 64)].reshape(2, -1)
        table32 = np.stack([2 * table64[:, j // 2] + j % 2 for j in range(pages32)], 1)
        for j, pair in itertools.product(range(pages32), range(pairs)):
            k_id, v_id, half = table64[0, j // 2], table64[1, j // 2], j % 2
            k_box = 2 * HKV * (pairs * k_id + pair) + half
            v_box = 2 * HKV * (pairs * v_id + pair) + half
            for h in range(HKV):
                new = _k_box(flat64, k_box + 2 * h, k_count64)
                old = _k_box(flat32, HKV * (pairs * table32[0, j] + pair) + h, k_count32)
                assert new is not None and np.array_equal(new, old), (length, j, pair, h)
                keys = pool64[k_id, HKV * pair + h, half * PAGE : (half + 1) * PAGE]
                assert np.array_equal(new, keys)
            new = _v_box(flat64, v_box, k_count64, 2 * PAGE * D, PAGE * D)
            old = _v_box(
                flat32, pairs * table32[1, j] + pair, pairs * 2 * blocks, PAGE * D, HKV * PAGE * D
            )
            assert np.array_equal(new, old), (length, j, pair)
            assert v_box * PAGE * D + 3 * PAGE * D <= flat64.size  # the second head stays inside
    # a padding page id (-1) is a negative box index for every pair, head and half
    for pair, half in itertools.product(range(pairs), range(2)):
        assert all(2 * HKV * (-pairs + pair) + half + 2 * h < 0 for h in range(HKV))
        assert 2 * HKV * (-pairs + pair) + half < 0


def test_the_kernel_carries_the_index_math_emulated_above():
    kernel = (_TARGET / "ops/attention/multi_request_split_kv_kernel.py").read_text()
    for fragment in (
        "rk = table[b, 0, src // SUB]",
        "rv = table[b, 1, src // SUB]",
        "half = (p_lo + idx) % SUB",
        "kbox = SUB * HKV * kpair + half",
        "vbox = SUB * HKV * vpair + half",
        "tKgK[None, 0, 0, kbox + SUB * h]",
        "tVgV[None, 0, 0, vbox]",
        "kbox = HKV * kpair",
        "vbox = vpair",
        "self.sub = cache_page // PAGE",
    ):
        assert fragment in kernel, fragment
    assert kernel.count("cutlass.const_expr(SUB == 1)") == 3  # the 32-token path is compile-time
