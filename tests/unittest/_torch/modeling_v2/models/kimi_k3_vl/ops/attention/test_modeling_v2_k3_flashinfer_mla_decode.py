# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""flashinfer_mla_decode (family op) on Kimi K3's latent cache: the generation requests of a mixed step (one context
request first) against the fp32 latent MQA over the rows read back through the KV cache manager's block ids, within
the bf16 band; the multi-CTA counter is left zeroed."""

import pytest
import torch
from _kimi_k3_mla_cache_env import LATENT, PAGE, ROPE, ROW, LatentCache

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (
    flashinfer_mla_decode as fi,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0)
    or not fi.available(),
    reason="certified on sm_100 with FlashInfer",
)

MAX_REQUESTS = 16
CONTEXT_LEN = 20


@pytest.mark.parametrize(
    "cell", fi.flashinfer_mla_decode.CELLS, ids=[c.why[:40] for c in fi.flashinfer_mla_decode.CELLS]
)
def test_cells(cell):
    spec = cell.spec
    heads, s = spec["num_heads"], spec["seq_len_q"]
    lens = list(spec["kv_lens"])
    r = len(lens)
    env = LatentCache(MAX_REQUESTS, 512)
    try:
        # Request 0 is the context request; requests 1..r generate.
        env.fill([CONTEXT_LEN] + lens, seed=r * 10 + s)
        md = env.metadata(
            [CONTEXT_LEN] + [s] * r,
            [0] + [n - s for n in lens],
            1,
            [CONTEXT_LEN] + [n - s for n in lens],
        )
        manager = env.manager
        local = manager.layer_offsets[0]
        kv_cache, block_tables = fi.paged_cache(
            md.host_kv_cache_pool_pointers,
            md.host_kv_cache_pool_mapping,
            md.kv_cache_block_offsets,
            local,
            PAGE,
            ROW,
            int(manager.impl.get_page_index_upper_bound(local, "key")),
            1,
            r,
            torch.bfloat16,
        )
        need = fi.workspace_bytes(
            torch.bfloat16, r * s, md.max_context_length, MAX_REQUESTS, heads, ROW, ROPE
        )
        workspace = torch.zeros(need, dtype=torch.int8, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        counter = torch.zeros(
            fi.counter_bytes(heads, MAX_REQUESTS, sms), dtype=torch.uint8, device="cuda"
        )
        gen = torch.Generator(device="cuda").manual_seed(r + s)
        query = (torch.randn(r, s, heads, ROW, generator=gen, device="cuda") * 0.3).bfloat16()
        out = torch.empty(r, s, heads, LATENT, dtype=torch.bfloat16, device="cuda")
        scale = fi.bmm1_scale(128, ROPE, 1.0)
        seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
        with validating(fi.flashinfer_mla_decode):
            fi.flashinfer_mla_decode(
                query, kv_cache, workspace, block_tables, seq_lens, max(lens), out, 128, LATENT, ROPE,
                scale, fi.enable_pdl(), counter,
            )  # fmt: skip
        torch.cuda.synchronize()
        want = fi.flashinfer_mla_decode.reference(
            query, [env.rows[i + 1] for i in range(r)], out, scale, LATENT
        )
        torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-2)
        assert not bool(counter.any()), "the kernel left the multi-CTA counter dirty"
    finally:
        env.shutdown()
