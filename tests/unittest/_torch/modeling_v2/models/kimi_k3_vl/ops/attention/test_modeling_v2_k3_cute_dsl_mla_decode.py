# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cute_dsl_mla_decode and mla_split_kv_plan (family ops) on Kimi K3's latent cache.

1. The split-KV plan the device kernel writes is the host mirror's, bit for bit.
2. The decode output, with and without the plan, against the fp32 latent MQA over the rows read back through the KV
   cache manager's block ids: within the bf16 band; the planned and unplanned outputs agree within the same band
   (they split the softmax differently).
"""

import pytest
import torch
from _kimi_k3_mla_cache_env import LATENT, PAGE, ROPE, ROW, LatentCache

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (
    cute_dsl_mla_decode as cute,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0)
    or not cute.available(),
    reason="certified on sm_100 with the CuTe DSL runtime",
)

MAX_REQUESTS = 16


@pytest.mark.parametrize(
    "cell", cute.mla_split_kv_plan.CELLS, ids=[c.why[:40] for c in cute.mla_split_kv_plan.CELLS]
)
def test_split_plan_cells(cell):
    spec = cell.spec
    lens = torch.tensor(spec["kv_lens"], dtype=torch.int32, device="cuda")
    plan = torch.full((MAX_REQUESTS,), -7, dtype=torch.int32, device="cuda")
    rows, waves = cute.mla_split_kv_plan.geometry(spec["num_heads"], spec["seq_len_q"])
    got = cute.mla_split_kv_plan(lens, plan, lens.numel(), rows, waves)
    want = cute.mla_split_kv_plan.reference(lens, plan, lens.numel(), rows, waves)
    assert torch.equal(got, want)
    assert bool((plan[lens.numel() :] == -7).all()), "the plan wrote past its requests"


@pytest.mark.parametrize(
    "cell", cute.cute_dsl_mla_decode.CELLS, ids=[c.why[:40] for c in cute.cute_dsl_mla_decode.CELLS]
)
def test_decode_cells(cell):
    spec = cell.spec
    heads, s = spec["num_heads"], spec["seq_len_q"]
    lens = list(spec["kv_lens"])
    r = len(lens)
    env = LatentCache(MAX_REQUESTS, 512)
    try:
        env.fill(lens, seed=r * 10 + s)
        md = env.metadata([s] * r, [n - s for n in lens], 0, [n - s for n in lens])
        manager = env.manager
        pool_idx = int(md.host_kv_cache_pool_mapping[manager.layer_offsets[0], 0])
        table = md.kv_cache_block_offsets[pool_idx, :r, 0, :].to(torch.int32).contiguous().t()
        seqs = torch.tensor(lens, dtype=torch.int32, device="cuda")
        need = cute.workspace_bytes(heads, s, LATENT, MAX_REQUESTS)
        workspace = torch.zeros(need, dtype=torch.int8, device="cuda")
        gen = torch.Generator(device="cuda").manual_seed(r + s)
        q = (torch.randn(r * s, heads, ROW, generator=gen, device="cuda") * 0.3).bfloat16()
        rows, waves = cute.mla_split_kv_plan.geometry(heads, s)
        plan = cute.mla_split_kv_plan(
            seqs, torch.empty(MAX_REQUESTS, dtype=torch.int32, device="cuda"), r, rows, waves
        )
        scale = (128 + ROPE) ** -0.5
        outs = []
        for split in (plan, None):
            out = torch.empty(r * s, heads * LATENT, dtype=torch.bfloat16, device="cuda")
            args = (q, out, env.pool(), table, seqs, workspace, heads, s, PAGE, LATENT, ROPE, scale)
            with validating(cute.cute_dsl_mla_decode):
                cute.cute_dsl_mla_decode(*args, MAX_REQUESTS, split)
            outs.append(out)
        # The reference reads the rows the test wrote, through the manager's block ids.
        ref_table = torch.zeros_like(table)
        for i in range(r):
            blocks = env.blocks(i)
            ref_table[: len(blocks), i] = torch.tensor(blocks, dtype=torch.int32)
        want = cute.cute_dsl_mla_decode.reference(
            q,
            outs[0],
            env.pool(),
            ref_table,
            seqs,
            workspace,
            heads,
            s,
            PAGE,
            LATENT,
            ROPE,
            scale,
            MAX_REQUESTS,
            None,
        )
        for out in outs:
            torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(outs[0].float(), outs[1].float(), rtol=2e-2, atol=2e-2)
    finally:
        env.shutdown()
