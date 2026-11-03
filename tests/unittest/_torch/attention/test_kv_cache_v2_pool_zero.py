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
"""KVCacheManagerV2's GPU pools start zeroed.

The CuTe DSL MLA decode loads whole 128-key tiles: for a request whose length is not a multiple of 128
it reads the rest of its last page and the pages the following page-table entries name, masks those
keys' scores, and multiplies their value rows by a zero probability. A NaN bit pattern in such a row
(FP8 E4M3 0x7F / 0xFF) makes the output NaN. The pools are mapped from memory that is never cleared,
so the manager zeroes them once at construction.

* Every byte of every GPU pool, and of the target and draft layer views, reads zero after construction,
  also when the memory held 0xFF bytes before; the zero covers every byte the pools expose.
* A decode over a target (dense MLA) or a draft layer's pool whose last tile reads past the request
  returns exactly what it returns over a clean pool holding the same keys, and NaN once the unwritten
  bytes hold 0xFF (the decode does read them).
"""

import math

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

KV_LORA, ROPE = 512, 64
LATENT = KV_LORA + ROPE
TILE = 128
MAX_SEQ_LEN = 2048


def _sm100f() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA cache pools")
needs_sm100f = pytest.mark.skipif(not _sm100f(), reason="CuTe DSL MLA decode needs SM100f")


def _manager(page: int):
    """One FP8 MLA (K-only, 576-wide latent) target layer and one FP8 DSpark draft layer."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    return KVCacheManagerV2(
        KvCacheConfig(max_tokens=16 * MAX_SEQ_LEN, enable_block_reuse=False),
        CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=LATENT,
        tokens_per_block=page,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(),
        dtype=DataType.FP8,
        is_disagg=True,
        draft_num_layers=1,
        draft_num_kv_heads=1,
        draft_head_dim=LATENT,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=9,
        draft_attention_backend="CUTEDSL",
    )


def _raw(base: int, size: int) -> torch.Tensor:
    return convert_to_torch_tensor(TensorWrapper(base, torch.uint8, [size]))


def _nonzero_bytes(view: torch.Tensor) -> int:
    try:
        raw = view.view(torch.uint8)
    except RuntimeError:
        raw = view.contiguous().view(torch.uint8)
    return int(torch.count_nonzero(raw).item())


def _layer_pool(manager, pool: str, page: int) -> torch.Tensor:
    """The layer's pool as ``[pages, page, 576]`` FP8 (one latent per token)."""
    view = manager.get_buffers(0) if pool == "target" else manager.get_draft_buffers(0)
    assert view.is_contiguous() and view.dtype == torch.float8_e4m3fn
    return view.view(-1, page, LATENT)


@needs_gpu
@pytest.mark.parametrize("page", [32, 64])
def test_gpu_pools_read_zero_before_the_first_request(page):
    torch.cuda.init()
    # The memory the pools map may have held anything before; leave 0xFF behind.
    stale = torch.full((1 << 30,), 0xFF, dtype=torch.uint8, device="cuda")
    del stale
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    manager = _manager(page)
    try:
        regions = manager._gpu_pool_regions()
        assert len(regions) >= 2 and all(size > 0 for _, _, size in regions)
        for _, base, size in regions:
            assert _nonzero_bytes(_raw(base, size)) == 0
        assert _nonzero_bytes(manager.get_buffers(0)) == 0
        assert _nonzero_bytes(manager.get_draft_buffers(0)) == 0
        # Every byte the layer views expose lies in the zeroed regions.
        for _, base, size in regions:
            _raw(base, size).fill_(0xFF)
        manager._zero_gpu_pool_memory()
        assert _nonzero_bytes(manager.get_buffers(0)) == 0
        assert _nonzero_bytes(manager.get_draft_buffers(0)) == 0
        assert not manager.check_gpu_pool_growth()
    finally:
        manager.shutdown()


def _decode(cache, table, lengths, heads, seq_q, page):
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    batch = lengths.shape[0]
    g = torch.Generator(device="cuda").manual_seed(11)
    q = (torch.randn(batch, seq_q, heads, LATENT, device="cuda", generator=g) * 0.1).to(
        torch.float8_e4m3fn
    )
    out = torch.full((batch, seq_q, heads, KV_LORA), float("nan"), device="cuda").bfloat16()
    workspace = torch.zeros(
        CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
            heads, seq_q, KV_LORA, batch, cutlass.Float32
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
        q[..., :KV_LORA].permute(2, 3, 1, 0),
        q[..., KV_LORA:].permute(2, 3, 1, 0),
        cache[..., :KV_LORA].permute(1, 2, 0),
        cache[..., KV_LORA:].permute(1, 2, 0),
        table.t(),
        lengths,
        out.permute(2, 3, 1, 0),
        workspace,
        heads,
        seq_q,
        page,
        torch.tensor([1.0 / math.sqrt(192)], device="cuda"),
        torch.tensor([1.0], device="cuda"),
        batch,
        None,
        None,
    )
    torch.cuda.synchronize()
    return out


@needs_sm100f
@pytest.mark.parametrize("page", [32, 64])
@pytest.mark.parametrize(
    "pool,heads,seq_q", [("target", 128, 1), ("draft", 64, 8)], ids=["target_mla", "dspark_draft"]
)
def test_over_reading_decode_on_the_zeroed_pool(page, pool, heads, seq_q):
    manager = _manager(page)
    try:
        cache = _layer_pool(manager, pool, page)
        num_pages = cache.shape[0]
        # Lengths off the 128-key tile: the last tile covers the rest of the last page and the pages of the
        # next table entries, which point at pages no request wrote.
        lengths = [3 * TILE + 37, TILE + page // 2 + 1]
        width = max(-(-n // TILE) * TILE // page for n in lengths) + 1
        g = torch.Generator(device="cuda").manual_seed(page)
        perm = torch.randperm(num_pages, device="cuda", generator=g)
        assert perm.numel() >= len(lengths) * width
        table = perm[: len(lengths) * width].view(len(lengths), width).to(torch.int32).contiguous()
        written = torch.zeros(num_pages, page, dtype=torch.bool, device="cuda")
        for b, n in enumerate(lengths):
            for p in range(-(-n // page)):
                written[table[b, p], : min(page, n - p * page)] = True
        values = (torch.randn(num_pages, page, LATENT, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        # FP8 has no index_put kernel: write the bytes.
        rows = written.nonzero(as_tuple=True)
        cache.view(torch.uint8)[rows] = values.view(torch.uint8)[rows]
        clean = torch.zeros_like(cache)
        clean.view(torch.uint8)[rows] = values.view(torch.uint8)[rows]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")

        got = _decode(cache, table, lens, heads, seq_q, page)
        ref = _decode(clean, table, lens, heads, seq_q, page)
        assert torch.isfinite(got.float()).all()
        assert torch.equal(got, ref)
        # Control: the bytes no request wrote are read; 0xFF there poisons the output.
        cache.view(torch.uint8)[~written] = 0xFF
        poisoned = _decode(cache, table, lens, heads, seq_q, page)
        assert not torch.isfinite(poisoned.float()).all()
    finally:
        manager.shutdown()
