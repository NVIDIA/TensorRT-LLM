# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 cute_dsl_fp8_paged_mqa_logits entry (CuTe DSL paged logits).

The keys live on random physical pages of a [pages, P, 1, 132] cache view; the schedule comes from
DeepGEMM's schedule builder over the cell's atom rows, as the K3.1 DSA metadata builds it. Only
each row's causal keys are compared. The entry writes a caller buffer; the registered op it
replaces must agree with it bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    cute_dsl_fp8_paged_mqa_logits as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    logits_row_stride,
)
from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

_SENTINEL = 12345.0


def _out(rows: int, max_context_len: int, row_stride: int | None = None) -> torch.Tensor:
    """A sentinel-filled caller buffer and the [rows, max_context_len] view the op takes."""
    stride = logits_row_stride(max_context_len) if row_stride is None else row_stride
    base = torch.full((rows + 1, stride), _SENTINEL, dtype=torch.float32, device="cuda")
    return base[:rows, :max_context_len]


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n = spec["batch"], spec["next_n"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    kv_cache, table = inputs.paged_cache(
        lens.tolist(), spec["tokens_per_block"], spec["data"], gen, spec["page_stride_pad"]
    )
    atoms = spec["num_next_n_atoms"]
    if atoms > 1:
        offsets = torch.tensor(inputs.atom_ctx_offsets(next_n, atoms), dtype=torch.int32)
        schedule_lens = (lens[:, None] - offsets).reshape(-1, 1)
    else:
        schedule_lens = lens.reshape(-1, 1)
    schedule = get_paged_mqa_logits_metadata(
        schedule_lens.cuda(), inputs.SCHEDULE_BLOCK_KV, inputs.num_sms()
    )
    q = inputs.queries(batch * next_n, spec["data"], gen).view(
        batch, next_n, inputs.NUM_INDEX_HEADS, inputs.HEAD_DIM
    )
    return dict(
        q=q,
        kv_cache=kv_cache,
        weights=inputs.weights(batch * next_n, spec["data"], gen),
        context_lens=lens.cuda(),
        block_table=table,
        schedule_meta=schedule,
        max_context_len=spec["max_context_len"],
        num_next_n_atoms=atoms,
        atom=spec["atom"],
        out=_out(batch * next_n, spec["max_context_len"]),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = op(**args)
    assert out.data_ptr() == args["out"].data_ptr(), "the entry must write the caller's buffer"
    first = out.clone()
    again = op(**dict(args, out=_out(*out.shape)))
    ref = op.reference(**args)
    registered = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
        args["q"],
        args["kv_cache"],
        args["weights"],
        args["context_lens"],
        args["block_table"],
        args["schedule_meta"],
        args["max_context_len"],
        num_next_n_atoms=args["num_next_n_atoms"],
        atom=args["atom"],
    )
    torch.cuda.synchronize()
    batch, next_n = args["q"].shape[:2]
    assert out.shape == (batch * next_n, args["max_context_len"]) and out.dtype == torch.float32
    causal = torch.isfinite(ref)
    assert torch.equal(first[causal], again[causal]), "two identical calls disagree"
    assert torch.equal(first[causal], registered[causal]), "the registered op computes otherwise"
    # Neither the slack columns of each row nor the spare row past the view are written.
    rows, width, stride = out.shape[0], out.shape[1], out.stride(0)
    if stride > width:
        assert (out.as_strided((rows, stride - width), (stride, 1), width) == _SENTINEL).all()
    assert (out.as_strided((1, stride), (stride, 1), rows * stride) == _SENTINEL).all()
    if cell.spec["data"] == "int":
        assert torch.equal(first[causal], ref[causal])
        return
    inputs.check_paged_logits(first, ref, args, next_n, args["context_lens"])


def test_row_stride_at_the_unrounded_width():
    """A row stride of exactly max_context_len (not a multiple of 256) computes the same logits."""
    spec = dict(op.CELLS[2].spec, max_context_len=10000)
    args = _build(spec, seed=11)
    rows = args["q"].shape[0] * args["q"].shape[1]
    with validating(op):
        tight = op(**dict(args, out=_out(rows, 10000, row_stride=10000))).clone()
        padded = op(**dict(args, out=_out(rows, 10000)))
    torch.cuda.synchronize()
    causal = torch.isfinite(op.reference(**args))
    assert torch.equal(tight[causal], padded[causal])


def test_guard_refuses_a_short_out():
    args = _build(op.CELLS[2].spec, seed=5)
    rows = args["q"].shape[0] * args["q"].shape[1]
    with pytest.raises(AssertionError, match="out must be"):
        with validating(op):
            op(**dict(args, out=_out(rows - 1, args["max_context_len"])))


def test_guard_refuses_a_context_past_max_context_len():
    args = _build(op.CELLS[0].spec, seed=3)
    with pytest.raises(AssertionError, match="max_context_len"):
        with validating(op):
            op(**dict(args, max_context_len=1024, out=_out(1, 1024)))


def test_guard_refuses_unpacked_page_rows():
    args = _build(op.CELLS[0].spec, seed=4)
    cache = args["kv_cache"]
    wide = torch.zeros(
        cache.shape[0], cache.shape[1], 1, 2 * cache.shape[3], dtype=torch.uint8, device="cuda"
    )
    with pytest.raises(AssertionError, match="packed"):
        with validating(op):
            op(**dict(args, kv_cache=wide[..., : cache.shape[3]]))


from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E402,E501
    cute_dsl_fp8_paged_mqa_logits_pdl as pdl_op,
)


@pytest.mark.parametrize("cell", pdl_op.CELLS, ids=[c.why[:48] for c in pdl_op.CELLS])
def test_pdl_certified_cells(cell, certifying):  # noqa: F811
    """The PDL launch of a native tile writes the non-PDL entry's logits bit for bit, also inside
    a CUDA graph behind a producer kernel."""
    certifying(pdl_op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out = pdl_op(**args)
    assert out.data_ptr() == args["out"].data_ptr()
    first = out.clone()
    plain = op(**dict(args, out=_out(*out.shape)))
    ref = pdl_op.reference(**args)
    torch.cuda.synchronize()
    causal = torch.isfinite(ref)
    assert torch.equal(first[causal], plain[causal]), "the PDL launch computes otherwise"
    if cell.spec["data"] == "int":
        assert torch.equal(first[causal], ref[causal])
    else:
        inputs.check_paged_logits(first, ref, args, args["q"].shape[1], args["context_lens"])


@pytest.mark.parametrize("cell_index", [0, 2, -1])
def test_pdl_graph_replay_behind_a_producer(cell_index):
    """Inside a CUDA graph right after a kernel that writes q, the PDL launch reads the new q."""
    cell = pdl_op.CELLS[cell_index]
    args = _build(cell.spec, inputs.seed_of(cell))
    expected = op(**dict(args, out=_out(*args["out"].shape))).clone()
    q_src = args["q"].clone()
    args["q"].zero_()
    graph_out = _out(*args["out"].shape)
    pdl_op(**dict(args, q=q_src, out=graph_out))  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        args["q"].copy_(q_src)
        pdl_op(**dict(args, out=graph_out))
    args["q"].zero_()
    graph_out.fill_(_SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    ref = op.reference(**dict(args, q=q_src))
    causal = torch.isfinite(ref)
    assert torch.equal(graph_out[causal], expected[causal])
