# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_helix_kept_gather entry (helix kept-entry NVFP4 -> E4M3 gather, Rubin)."""

import zlib

import pytest
import torch

# l0_gb300 collects this directory: any GPU but sm_107 skips the module before a family import.
if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("certified on sm_107 only", allow_module_level=True)

import tensorrt_llm  # noqa: E402,F401  (loads the C++ custom ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_helix_kept_gather import (  # noqa: E402,E501
    HEAD_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_helix_kept_gather import (  # noqa: E402,E501
    rubin_helix_kept_gather as op,
)
from tensorrt_llm._utils import get_sm_version  # noqa: E402

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the Rubin family entries are certified on sm_107",
)

#: The global scale with TRTLLM_NVFP4_MLA_KV_CACHE_AMAX unset (amax 100).
_DEQUANT = 100.0 / (448.0 * 6.0)
_EDGES = [0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 513, 1023, 1024, 2047, 2048, 300]
_SCRATCH_FILL = 0x5A
_COMPACT_FILL = 12345
_NUM_LAYERS = 3
_LAYER = 1


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _build(spec, seed: int) -> dict:
    """Helix-shaped rows: each row keeps ``len`` distinct ascending local positions first, then -1."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, num_req, local_len = spec["rows"], spec["num_req"], spec["local_len"]
    top_k, tpb, residual = spec["top_k"], spec["tokens_per_block"], spec["residual_dim"]
    if spec["lens"] == "typical":
        lens_list = torch.randint(180, 340, (rows,), generator=gen, device="cuda").tolist()
    elif spec["lens"] == "full":
        lens_list = [top_k] * rows
    else:
        lens_list = [_EDGES[i % len(_EDGES)] for i in range(rows)]
    pages_per_req = -(-local_len // tpb)
    num_blocks = num_req * pages_per_req + 16
    num_pool_tokens = num_blocks * _NUM_LAYERS * tpb
    perm = torch.randperm(num_blocks, generator=gen, device="cuda").to(torch.int32)
    block_table = perm[: num_req * pages_per_req].view(num_req, pages_per_req).contiguous()
    topk = torch.full((rows, top_k), -1, dtype=torch.int32, device="cuda")
    for r, n in enumerate(lens_list):
        n = min(n, local_len)
        if n > 0:
            pos = torch.randperm(local_len, generator=gen, device="cuda")[:n].sort().values
            topk[r, :n] = pos.to(torch.int32)
    if spec["holes"]:
        # -1 holes, block ids past the table, a padded table entry, a page past the pool.
        for r, n in enumerate(lens_list):
            if n >= 8:
                topk[r, 1] = -1
                topk[r, 3] = pages_per_req * tpb + 5
        block_table[0, 0] = -1
        block_table[-1, 1] = num_blocks * 4
        topk[0, 0] = 3
        topk[-1, 0] = tpb + 7
    data_bytes, scale_bytes = (HEAD_DIM + residual) // 2, (HEAD_DIM + residual) // 16
    data = torch.randint(
        0, 256, (num_pool_tokens, data_bytes), dtype=torch.uint8, device="cuda", generator=gen
    )
    # Finite non-negative E4M3 block scales (0x00..0x7E), the writer's range.
    scale = torch.randint(
        0, 0x7F, (num_pool_tokens, scale_bytes), dtype=torch.uint8, device="cuda", generator=gen
    )
    scratch = torch.full((rows, top_k, HEAD_DIM), _SCRATCH_FILL, dtype=torch.uint8, device="cuda")
    return dict(
        topk_indices=topk,
        row_lens=torch.tensor(lens_list, dtype=torch.int32, device="cuda"),
        req_idx=(torch.arange(rows, device="cuda") * num_req // rows).to(torch.int32),
        block_table=block_table,
        tokens_per_block=tpb,
        stride_factor=_NUM_LAYERS * tpb,
        layer_offset=_LAYER,
        data_pool=data.view(-1),
        scale_pool=scale.view(-1),
        dequant_scale=torch.tensor([_DEQUANT], dtype=torch.float32, device="cuda"),
        num_pool_tokens=num_pool_tokens,
        scratch=scratch.view(torch.float8_e4m3fn),
        compact_indices=torch.full_like(topk, _COMPACT_FILL),
        residual_dim=residual,
    )


def _two_step(args):
    """The two-step path: the C++ remap of every entry, then the C++ gather of every pair."""
    topk = args["topk_indices"]
    rows, top_k = topk.shape
    glob = torch.ops.trtllm.convert_req_index_to_global(
        args["req_idx"],
        args["block_table"],
        topk,
        args["tokens_per_block"],
        top_k,
        args["stride_factor"],
        args["layer_offset"],
    )
    pointers = torch.zeros((1, 2, 2), dtype=torch.int64)
    pointers[0, 0, 0] = args["data_pool"].data_ptr()
    pointers[0, 0, 1] = args["scale_pool"].data_ptr()
    mapping = torch.zeros((1, 2), dtype=torch.int32)
    out = torch.zeros((rows, top_k, HEAD_DIM), dtype=torch.float8_e4m3fn, device="cuda")
    compact = torch.empty_like(topk)
    torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
        pointers,
        mapping,
        glob,
        out,
        compact,
        args["dequant_scale"],
        0,
        args["residual_dim"],
        args["num_pool_tokens"],
    )
    return out, compact


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, _seed(cell))
    ref_scratch, ref_compact = op.reference(**args)
    cpp_scratch, cpp_compact = _two_step(args)
    scratch, compact = op(**args)
    torch.cuda.synchronize()
    assert scratch.data_ptr() == args["scratch"].data_ptr()
    assert compact.data_ptr() == args["compact_indices"].data_ptr()
    # The whole of both outputs, bytes the entry must not write included.
    assert torch.equal(compact, ref_compact)
    assert torch.equal(scratch.view(torch.uint8), ref_scratch.view(torch.uint8))
    # What the sparse MLA reads (row-length columns, valid scratch rows) equals the two-step path.
    top_k = compact.shape[1]
    inside = torch.arange(top_k, device="cuda")[None, :] < args["row_lens"].clamp(0, top_k)[:, None]
    assert torch.equal(compact[inside], cpp_compact[inside])
    valid = compact >= 0
    assert torch.equal(scratch.view(torch.uint8)[valid], cpp_scratch.view(torch.uint8)[valid])
