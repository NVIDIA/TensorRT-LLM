# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GQAMIX: a mixed step's sparse generation rows (MsaDecodeFmha) through ops/gqamix_sparse_decode.

The copy must write the in-tree kernel's output bit for bit on a mixed step's [num_ctx:] slices.
The shared MsaDecodeFmha hook must change nothing while it is unset.
"""

import pytest
import torch

import tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode as in_tree
from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_103__tp4.ops import (  # noqa: E501
        gqamix_sparse_decode as copy,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

HEADS = 16  # query heads per rank at tp4, one KV head
D, PAGE, TOPK, QLEN, SF_VEC = 128, 128, 16, 4, 16


# GQAMIX rides MsaDecodeFmha.sparse_decode_hook, an engine hook main's MsaDecodeFmha does not have yet:
# _gqamix_register leaves such a library alone and the in-tree decode runs.


def _mixed_step(nreq, seq_len, row_first, seed):
    """A mixed step's generation rows: NVFP4 caches, FP8 q, and block-table / length slices past row_first."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = seq_len // PAGE
    pages = (nreq + row_first) * nb
    block_table = torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(-1, nb)
    seq_lens = torch.full((nreq + row_first,), seq_len, device="cuda", dtype=torch.int32)
    total_q = nreq * QLEN
    q = torch.randn(total_q, HEADS, D, device="cuda", generator=g).to(torch.float8_e4m3fn)
    kv = [
        torch.randint(
            0, 256, (pages, 1, PAGE, D // 2), device="cuda", generator=g, dtype=torch.uint8
        )
        for _ in "kv"
    ]
    sc = [
        (torch.rand(pages, 1, PAGE, D // SF_VEC, device="cuda", generator=g) + 0.5).to(
            torch.float8_e4m3fn
        )
        for _ in "kv"
    ]
    topk = torch.stack(
        [torch.randperm(nb, device="cuda", generator=g)[:TOPK] for _ in range(total_q)]
    )
    args = (q, kv[0], kv[1], topk.to(torch.int32).view(total_q, 1, TOPK).permute(1, 0, 2))
    args += (block_table[row_first:], seq_lens[row_first:])
    kwargs = dict(
        sm_scale=D**-0.5,
        decode_query_len=QLEN,
        k_block_scale=sc[0],
        v_block_scale=sc[1],
        k_global_scale=torch.tensor([0.02], device="cuda"),
        v_global_scale=torch.tensor([0.03], device="cuda"),
    )
    return args, kwargs, total_q


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in copy._SM100F_CAPABILITIES,
    reason="needs a CUDA device with the NVFP4 sparse decode kernels",
)
@pytest.mark.parametrize("nreq,row_first", [(1, 0), (5, 3), (8, 1)])
@pytest.mark.parametrize("seq_len", [16384, 131072])
def test_the_copy_writes_the_in_tree_output_on_a_mixed_step(nreq, row_first, seq_len) -> None:
    args, kwargs, total_q = _mixed_step(nreq, seq_len, row_first, 31 * nreq + row_first)
    ref = torch.empty(total_q, HEADS, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(ref)
    in_tree.minimax_m3_sparse_attn_decode(*args, output=ref, **kwargs)
    copy.minimax_m3_sparse_attn_decode(*args, output=out, **kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(ref.float()).all()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
