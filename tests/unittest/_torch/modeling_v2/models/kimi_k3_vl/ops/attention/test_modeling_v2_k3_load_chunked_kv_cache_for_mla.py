# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""load_chunked_kv_cache_for_mla (family op) on Kimi K3's latent cache: every chunk the attention metadata plans for
a chunked-prefill step comes back bit for bit the rows the cache holds at those positions, read through the KV cache
manager's own block ids (not the op's addressing)."""

import pytest
import torch
from _kimi_k3_mla_cache_env import LATENT, LatentCache

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.load_chunked_kv_cache_for_mla import (  # noqa: E501
    load_chunked_kv_cache_for_mla as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="certified on sm_100",
)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_cells(cell):
    spec = cell.spec
    cached = list(spec["cached"])
    new = [50, 30, 9]
    env = LatentCache(len(cached), 512)
    try:
        env.fill([c + n for c, n in zip(cached, new)], seed=11)
        md = env.metadata(
            new,
            cached,
            len(cached),
            new,
            chunked_prefill=True,
            chunk_size=spec["chunk"] // 4,
        )
        assert md.chunked_loop_num >= 2
        manager = env.manager
        seen = [0] * len(cached)
        for loop in range(md.chunked_loop_num):
            total = int(md.host_cu_chunked_seq_len[loop, len(cached)])
            with validating(op):
                kv, k_pe = op(
                    torch.bfloat16,
                    len(cached),
                    total,
                    md.cu_chunked_seq_len[loop],
                    md.chunked_global_offset[loop],
                    md.kv_cache_block_offsets,
                    manager.kv_cache_pool_pointers,
                    manager.kv_cache_pool_mapping,
                    None,
                    manager.layer_offsets[0],
                    LATENT,
                    64,
                    64,
                    int(md.max_chunk_len_per_loop[loop]),
                    manager.max_seq_len,
                    1,
                    spec["quant_mode"],
                )
            want_kv, want_pe = op.reference(
                [env.rows[r][: cached[r]] for r in range(len(cached))],
                md.host_cu_chunked_seq_len[loop],
                md.host_chunked_global_offset[loop],
                LATENT,
            )
            assert torch.equal(kv, want_kv) and torch.equal(k_pe, want_pe), f"chunk {loop}"
            cu = md.host_cu_chunked_seq_len[loop].tolist()
            for r in range(len(cached)):
                seen[r] += cu[r + 1] - cu[r]
        assert seen == cached, "the chunks do not cover every cached row exactly once"
    finally:
        env.shutdown()
