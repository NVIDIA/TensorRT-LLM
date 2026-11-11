# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``megamoe_stage_routes`` writes a MegaMoE bucket's route staging byte for byte as torch copies
and fills do: the routes widened into rows ``[0, n)``, -1 and 0 in the rows after them, the
weight columns past ``top_k`` untouched. A launch without rows, such as the priming of the other
buckets while tuning, stages -1 and 0 in every row."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl

ROWS, TOP_K, WEIGHT_COLUMNS = 16, 6, 8


def _garbage(shape, dtype, generator):
    """A buffer of random bytes, as a bucket holds after earlier launches."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    return raw.view(dtype).view(shape).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    "id_dtype,out_dtype",
    [(torch.int32, torch.int64), (torch.int32, torch.int32), (torch.int64, torch.int64)],
)
@pytest.mark.parametrize("rows", [0, 1, 7, ROWS])
def test_stage_routes_matches_torch_copies_and_fills(id_dtype, out_dtype, rows):
    gen = torch.Generator().manual_seed(rows)
    info = torch.iinfo(torch.int32)
    ids = torch.randint(-1, 384, (rows, TOP_K), dtype=id_dtype, generator=gen)
    if rows:
        ids[0, :2] = torch.tensor([info.max, info.min])  # widening keeps the sign
    ids = ids.cuda()
    weights = torch.rand(rows, TOP_K, generator=gen).cuda()
    ids_out = _garbage((ROWS, TOP_K), out_dtype, gen)
    weights_buf = _garbage((ROWS, WEIGHT_COLUMNS), torch.float32, gen)
    want_ids, want_weights = ids_out.clone(), weights_buf.clone()
    want_ids[:rows].copy_(ids)
    want_weights[:rows, :TOP_K].copy_(weights)
    want_ids[rows:].fill_(-1)
    want_weights[rows:, :TOP_K].zero_()

    torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_buf[:, :TOP_K])

    assert torch.equal(ids_out.view(torch.uint8), want_ids.view(torch.uint8))
    assert torch.equal(weights_buf.view(torch.uint8), want_weights.view(torch.uint8))


def test_bucket_priming_stages_a_launch_without_rows():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._maxt_buckets = [8, 16]
    moe.ep_size, moe.ep_rank = 4, 0
    moe._wait_rebalance_routes = lambda replica_plan=None: None
    bufs = SimpleNamespace(
        topk_idx_local=torch.full((8, TOP_K), 5, dtype=torch.int64),
        topk_weights=torch.ones(8, TOP_K),
        activation=None,
        activation_sf=None,
        combine_output=None,
        shared_workspace=None,
        peer_offsets=(0,),
    )
    moe._acquire_buffers = lambda **kwargs: bufs
    launches = []
    moe._launch_megamoe_kernel = lambda **kwargs: launches.append(kwargs)

    moe._prime_ladder_buckets(
        live_T=16,
        top_k=TOP_K,
        hidden=16,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
        weight_view=None,
        primed=[],
    )

    assert [(launch["launch_max_T"], launch["num_tokens"]) for launch in launches] == [(8, 0)]
    assert (bufs.topk_idx_local == -1).all() and not bufs.topk_weights.any()
