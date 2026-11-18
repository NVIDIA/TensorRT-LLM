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
"""The DFlash worker over an FP8 draft KV pool (``TRTLLM_DFLASH_FP8_DRAFT_KV``).

The KV cache manager stores the drafter's layers in FP8 E4M3 when the knob is set; the
worker keeps computing the context K/V in the drafter's activation dtype (BF16) and casts
them on store. The tests pin:

* the pool binding: under the knob an FP8 pool binds to a BF16 drafter, not to an FP16 one,
  and every layer must have the same dtype; without the knob it does not bind; a BF16 pool
  binds as before;
* the compute dtype of the context writes: BF16 over an FP8 pool, the cache's own otherwise;
* the paged append: K and V reach the append in the pool's dtype through the saturating cast,
  and unchanged (the same tensors) over a BF16 pool;
* the step store computes ``precompute_context_kv`` on BF16 input over an FP8 pool (CUDA).
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dflash_kv_storage import to_kv_storage

FP8_KNOB = "TRTLLM_DFLASH_FP8_DRAFT_KV"
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16
PAGE, NKV, HD, LAYERS = 32, 2, 64, 3


class _Manager:
    """The two members ``_managed_ctx_pool`` reads: the draft layer ids and their pages."""

    def __init__(self, dtypes):
        self.draft_layer_ids = tuple(range(len(dtypes)))
        self._pages = [torch.zeros(5, 2, NKV, PAGE, HD).to(dtype) for dtype in dtypes]

    def get_buffers(self, layer_id, kv_layout="NHD"):
        assert kv_layout == "HND"
        return self._pages[layer_id]


def _bind(dtypes, dtype):
    worker = object.__new__(DFlashWorker)
    return worker._managed_ctx_pool(_Manager(dtypes), LAYERS, NKV, HD, dtype)


@pytest.fixture(autouse=True)
def _fp8_draft_kv_requested(monkeypatch):
    # The manager stores the draft layers in FP8 only under the knob; the tests below assume it.
    monkeypatch.setenv(FP8_KNOB, "1")


@pytest.mark.parametrize("pool_dtype", [BF16, FP8])
def test_bf16_drafter_binds_a_bf16_or_fp8_pool(pool_dtype):
    pool = _bind([pool_dtype] * LAYERS, BF16)
    assert pool is not None and [layer.dtype for layer in pool] == [pool_dtype] * LAYERS


@pytest.mark.parametrize(
    "dtypes,dtype",
    [
        ([FP8] * LAYERS, torch.float16),
        ([FP8, BF16, BF16], BF16),
        ([BF16, FP8, FP8], BF16),
        ([torch.float16] * LAYERS, BF16),
    ],
    ids=["fp16_drafter", "fp8_then_bf16", "bf16_then_fp8", "fp16_pool"],
)
def test_other_pools_are_not_bound(dtypes, dtype):
    assert _bind(dtypes, dtype) is None


@pytest.mark.parametrize("knob", [None, "0", "true"], ids=["unset", "0", "true"])
def test_without_the_knob_only_a_bf16_pool_binds(monkeypatch, knob):
    if knob is None:
        monkeypatch.delenv(FP8_KNOB, raising=False)
    else:
        monkeypatch.setenv(FP8_KNOB, knob)
    assert _bind([FP8] * LAYERS, BF16) is None
    pool = _bind([BF16] * LAYERS, BF16)
    assert pool is not None and [layer.dtype for layer in pool] == [BF16] * LAYERS


def _paged_worker(pool_dtype, compute_dtype=BF16):
    worker = object.__new__(DFlashWorker)
    worker._ctx_paged = True
    worker._ctx_kv_buf = [torch.zeros(5, 2, NKV, PAGE, HD).to(pool_dtype) for _ in range(LAYERS)]
    worker._ctx_k_buf = None
    worker._ctx_compute_dtype = compute_dtype
    return worker


def test_compute_dtype_of_the_context_writes():
    assert _paged_worker(FP8)._ctx_store_compute_dtype() == BF16
    # A pool in the activation dtype is its own compute dtype; the drafter's is not read.
    assert _paged_worker(BF16, compute_dtype=None)._ctx_store_compute_dtype() == BF16
    dense = object.__new__(DFlashWorker)
    dense._ctx_paged = False
    dense._ctx_k_buf = torch.zeros(1, dtype=BF16)
    dense._ctx_compute_dtype = None
    assert dense._ctx_store_compute_dtype() == BF16


@pytest.mark.parametrize("pool_dtype", [BF16, FP8])
def test_paged_append_gets_the_pool_dtype(monkeypatch, pool_dtype):
    worker = _paged_worker(pool_dtype)
    worker._ctx_kv_last_page_len = torch.full((2,), PAGE, dtype=torch.int32)
    calls = []

    def append(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(worker, "_get_ctx_paged_append", lambda: append, raising=False)
    monkeypatch.setattr(worker, "_ctx_paged_index_args", lambda: (None, None), raising=False)
    generator = torch.Generator().manual_seed(0)
    k = (torch.randn(4, LAYERS, NKV, HD, generator=generator) * 300).to(BF16)
    v = (torch.randn(4, LAYERS, NKV, HD, generator=generator) * 300).to(BF16)
    rows = torch.zeros(4, dtype=torch.long)
    positions = torch.arange(4, dtype=torch.long)
    worker._store_context_kv_paged(k, v, rows, positions)
    assert len(calls) == LAYERS
    for layer, call in enumerate(calls):
        assert call["paged_kv_cache"] is worker._ctx_kv_buf[layer]
        for name, source in (("append_key", k), ("append_value", v)):
            stored = call[name]
            assert stored.dtype == pool_dtype and stored.is_contiguous()
            expected = to_kv_storage(source[:, layer], pool_dtype)
            assert torch.equal(stored.view(torch.uint8), expected.contiguous().view(torch.uint8))
    if pool_dtype == FP8:
        assert float(calls[0]["append_key"].float().abs().max()) <= 448.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the step store builds CUDA tensors")
def test_step_store_computes_in_bf16_over_an_fp8_pool(monkeypatch):
    worker = _paged_worker(FP8)
    worker._ctx_base = None
    worker._ctx_block_tables = None
    worker._ctx_keep = 0
    worker._max_ctx = 4096
    seen = {}

    class _Drafter:
        def project_target_hidden(self, hidden):
            return hidden

        def precompute_context_kv(self, projected, positions):
            seen["dtype"] = projected.dtype
            rows = projected.shape[0]
            k = torch.ones(rows, LAYERS, NKV, HD, dtype=projected.dtype, device="cuda")
            return k, k.clone()

    stored = {}

    def store(k, v, rows, positions):
        stored["dtypes"] = (k.dtype, v.dtype)

    monkeypatch.setattr(worker, "_store_context_kv_paged", store, raising=False)
    gen_hs = torch.zeros(2 * 4, 16, dtype=BF16, device="cuda")
    slots = torch.arange(2, dtype=torch.long, device="cuda")
    ctx_len = torch.tensor([10, 20], dtype=torch.long, device="cuda")
    worker._store_step_context(_Drafter(), gen_hs, slots, slots, ctx_len)
    assert seen["dtype"] == BF16
    # The worker hands BF16 K/V to the paged store, which casts them for the pool.
    assert stored["dtypes"] == (BF16, BF16)
