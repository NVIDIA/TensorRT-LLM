# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m3_fused_score_select entry: MSA index scoring and top-16 block selection of a decode
span of at most two requests in one launch (``torch.ops.trtllm.minimax_m3_tp8_fused_score_select``).

Every cell is held bit for bit to the entry's ``reference`` (``m3_select_long.reference`` over the per-KV-head maxima
of ``m3_index_score.reference``) and to the pipeline it replaces in the target: this target's ``m3_index_score`` (the
prefetching kernel at ``SPLIT_K``), the per-KV-head amax and ``m3_select_long``, on the same inputs. The selection is a
table of block ids, so the claim is exact.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops import (  # noqa: E501
    m3_fused_score_select as fss,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_fused_score_select import (  # noqa: E501
    MAX_REQUESTS,
    scratch_shape,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_fused_score_select import (  # noqa: E501
    m3_fused_score_select as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_index_score import (  # noqa: E501
    SPLIT_K,
    m3_index_score,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_select_long import (  # noqa: E501
    TOPK,
    m3_select_long,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401 — needs the path declared above

assert torch.cuda.is_available(), "m3_fused_score_select requires a CUDA device"

PAGE = 128
HEADS = 1  # index heads per rank at TP8
HEAD_DIM = 128


def _n_valid(seq_lens, qlen: int) -> torch.Tensor:
    """Each query row's causal valid-block count, as the metadata stages it (``per_token_valid_blocks``)."""
    vals = [
        (max(seq - qlen + offset + 1, 0) + PAGE - 1) // PAGE
        for seq in seq_lens
        for offset in range(qlen)
    ]
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _inputs(spec, seed: int, dtype=torch.float8_e4m3fn):
    """Random FP8 index Q [T, 1, 128] and paged index-K [pages, 128, 128] behind a shuffled block table (as the index
    scorer's test builds them), the request lengths and the rows' valid-block counts."""
    seq_lens, qlen = spec["seq_lens"], spec["qlen"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )
    idx_q = torch.randn(batch * qlen, HEADS, HEAD_DIM, device="cuda", generator=g).to(dtype)
    index_k = torch.randn(pages, PAGE, HEAD_DIM, device="cuda", generator=g).to(dtype)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return idx_q, index_k, block_table, seq, _n_valid(seq_lens, qlen)


def _out(total_q: int, num_kv_heads: int) -> torch.Tensor:
    """The [total_q, num_kv_heads, 16] view of a head-major backing, as the target hands it over."""
    return torch.full((num_kv_heads, total_q, TOPK), -7, dtype=torch.int32, device="cuda").permute(
        1, 0, 2
    )


def _args(spec, inputs, out, scratch, counters):
    idx_q, index_k, block_table, seq, n_valid = inputs
    return (
        idx_q,
        index_k,
        block_table,
        seq,
        n_valid,
        out,
        scratch,
        counters,
        spec["qlen"],
        spec["num_kv_heads"],
        spec["init_blocks"],
        spec["local_blocks"],
        spec["max_tiles"],
        False,
    )


def _workspace(max_tiles: int):
    """Scratch filled with garbage (the op needs no initialization) and zeroed counters."""
    scratch = torch.full(scratch_shape(max_tiles), float("nan"), device="cuda")
    counters = torch.zeros(MAX_REQUESTS, dtype=torch.int32, device="cuda")
    return scratch, counters


def _pipeline(spec, inputs) -> torch.Tensor:
    """What the target runs on a wider span: this target's index scorer (prefetching kernel), the per-KV-head amax over
    its index heads, and the long-row selector."""
    idx_q, index_k, block_table, seq, n_valid = inputs
    total_q = idx_q.shape[0]
    max_tiles, kv = spec["max_tiles"], spec["num_kv_heads"]
    backing = torch.full(
        (HEADS, max_tiles, total_q), -float("inf"), dtype=torch.float32, device="cuda"
    )
    m3_index_score(
        idx_q, index_k, block_table, seq, backing.transpose(1, 2), spec["qlen"], SPLIT_K, False
    )
    per_kv = backing.view(kv, HEADS // kv, max_tiles, total_q).amax(dim=1)
    table = _out(total_q, kv)
    return m3_select_long(
        per_kv, n_valid, table, spec["init_blocks"], spec["local_blocks"], False, False
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    inputs = _inputs(spec, zlib.crc32(cell.why.encode()) % 2**31)
    assert len(spec["seq_lens"]) == spec["batch"]
    total_q = inputs[0].shape[0]
    out = _out(total_q, spec["num_kv_heads"])
    scratch, counters = _workspace(spec["max_tiles"])
    args = _args(spec, inputs, out, scratch, counters)
    certifying(op)
    op(*args)
    ref = op.reference(*args)
    torch.cuda.synchronize()

    assert out.dtype is torch.int32 and out.shape == (total_q, spec["num_kv_heads"], TOPK)
    assert torch.equal(out, ref), "the selected blocks differ from the reference"
    assert not bool(counters.any()), "the arrival counters must be back at zero after the call"
    assert torch.equal(out, _pipeline(spec, inputs)), (
        "the selected blocks differ from m3_index_score + per-KV amax + m3_select_long"
    )

    # A second call on the same scratch and counters (as the next sparse layer makes) selects the same.
    again = _out(total_q, spec["num_kv_heads"])
    op(*_args(spec, inputs, again, scratch, counters))
    torch.cuda.synchronize()
    assert torch.equal(again, out) and not bool(counters.any())


@pytest.mark.parametrize("cell", op.CELLS[:2], ids=[c.why[:44] for c in op.CELLS[:2]])
def test_the_target_wrapper_is_the_entry(cell) -> None:
    """``ops/m3_fused_score_select``: the geometry check admits the cell, the scratch and counters are created
    once per (device, engine block count), and the wrapper's table is the entry's."""
    spec = cell.spec
    inputs = _inputs(spec, 3)
    idx_q, index_k, block_table, seq, n_valid = inputs
    max_tiles = spec["max_tiles"]
    assert fss.supported(idx_q, index_k, spec["qlen"], spec["num_kv_heads"], max_tiles) is None
    bufs = fss.buffers(idx_q.device, max_tiles)
    assert bufs is not None and fss.buffers(idx_q.device, max_tiles) is bufs
    table = fss.fused_score_select(
        idx_q,
        index_k,
        block_table,
        seq,
        n_valid,
        decode_query_len=spec["qlen"],
        num_kv_heads=spec["num_kv_heads"],
        init_blocks=spec["init_blocks"],
        local_blocks=spec["local_blocks"],
        max_tiles=max_tiles,
        scratch=bufs[0],
        counters=bufs[1],
    )
    torch.cuda.synchronize()
    ref = op.reference(*_args(spec, inputs, _out(idx_q.shape[0], 1), *_workspace(max_tiles)))
    assert torch.equal(table, ref)
    assert not bool(bufs[1].any())


def test_the_geometry_check_refuses_what_the_kernel_does_not_take() -> None:
    spec = op.CELLS[0].spec
    idx_q, index_k, *_ = _inputs(spec, 5)
    assert "not FP8" in fss.supported(idx_q.bfloat16(), index_k, 4, 1, 8192)
    assert "geometry" in fss.supported(idx_q, index_k, 5, 1, 8192)
    wide_q = idx_q.expand(-1, 4, -1).contiguous()
    assert "geometry" in fss.supported(wide_q, index_k, 4, 1, 8192)
    assert "stage ring" in fss.supported(idx_q, index_k, 4, 1, 16384)


def test_the_dispatch_range_is_the_kernel_limit() -> None:
    assert dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS == MAX_REQUESTS
    assert {c.spec["batch"] for c in op.CELLS} == set(dispatch.FUSED_SCORE_SELECT.keys)


def test_guard_refuses_nonzero_counters() -> None:
    spec = op.CELLS[0].spec
    inputs = _inputs(spec, 7)
    scratch, counters = _workspace(spec["max_tiles"])
    counters[0] = 1
    with pytest.raises(AssertionError, match="counters must be zero"):
        with validating(op):
            op(*_args(spec, inputs, _out(inputs[0].shape[0], 1), scratch, counters))


def test_guard_refuses_a_request_past_its_block_table() -> None:
    spec = op.CELLS[4].spec
    idx_q, index_k, block_table, seq, n_valid = _inputs(spec, 9)
    seq = seq.clone()
    seq[1] = block_table.shape[1] * PAGE + 1
    scratch, counters = _workspace(spec["max_tiles"])
    with pytest.raises(AssertionError, match="past its block-table row"):
        with validating(op):
            op(
                *_args(
                    spec,
                    (idx_q, index_k, block_table, seq, n_valid),
                    _out(idx_q.shape[0], 1),
                    scratch,
                    counters,
                )
            )
