# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for TSEL: the PrefillTarget's top-16 block select through the #20629 prefill tile select
(torch.ops.trtllm.minimax_m31_select_prefill_tile), registered on the MSA indexer's selector hook.

The table must equal minimax_m3_select_blocks' (head-major backing) bit for bit, ties included; outside its rows /
blocks range the hook returns None so the built-in select runs. Skipped when the op is not built (no overlay library).
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4 import (
    modeling as m,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    select_blocks_from_maxscore,
)

assert torch.cuda.is_available(), "TSEL requires a CUDA device"

BLOCK = 128
op = m._tsel_op()
needs_op = pytest.mark.skipif(
    op is None, reason="minimax_m31_select_prefill_tile is not built (or not sm_103)"
)


def _inputs(ctx_end, rows, ties, seed, heads=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = (ctx_end + BLOCK - 1) // BLOCK
    pos = torch.arange(ctx_end - rows, ctx_end, device="cuda")
    nvb = ((pos + BLOCK) // BLOCK).to(torch.int32)
    if ties:
        scores = torch.randint(0, 8, (heads, blocks, rows), generator=g, device="cuda").float()
    else:
        scores = torch.randn(heads, blocks, rows, generator=g, device="cuda")
    return scores.contiguous(), nvb


def _ref(scores, nvb, init_blocks, local_blocks):
    return select_blocks_from_maxscore(
        scores,
        topk=16,
        n_valid_blocks=nvb,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        head_major_output=True,
    )


@needs_op
@pytest.mark.parametrize(
    "ctx_end,rows", [(16384, 16384), (65536, 4096), (163840, 2048), (165000, 1160), (1024, 256)]
)
@pytest.mark.parametrize("ties", [False, True], ids=["random", "ties"])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 3)])
def test_table_equals_select_blocks(ctx_end, rows, ties, init_blocks, local_blocks) -> None:
    scores, nvb = _inputs(ctx_end, rows, ties, ctx_end + rows)
    got = m._tsel_select(scores, nvb, init_blocks, local_blocks)
    assert got is not None
    ref = _ref(scores, nvb, init_blocks, local_blocks)
    assert (
        got.shape == ref.shape
        and got.stride() == ref.stride()
        and got.dtype == ref.dtype == torch.int32
    )
    assert torch.equal(got, ref)


@needs_op
def test_two_kv_heads_and_minus_one_tail() -> None:
    """Rows with fewer than 16 valid blocks pad with -1; two KV heads keep the head-major backing."""
    scores, nvb = _inputs(2048, 2048, False, 7, heads=2)
    ref = _ref(scores, nvb, 0, 1)
    got = m._tsel_select(scores, nvb, 0, 1)
    assert got.shape == ref.shape and got.stride() == ref.stride() and got.dtype == ref.dtype
    assert torch.equal(got, ref)
    assert (got[:100] == -1).any()


@needs_op
def test_host_n_valid_blocks_are_converted() -> None:
    """Callers hand either the device int32 step view or the host per_token_valid_blocks() (int64) tensor."""
    scores, nvb = _inputs(8192, 512, False, 5)
    got = m._tsel_select(scores, nvb.cpu().long(), 0, 1)
    assert torch.equal(got, _ref(scores, nvb, 0, 1))


def test_outside_the_gate_keeps_the_built_in_select() -> None:
    scores, nvb = _inputs(4096, m.TSEL_MIN_ROWS - 1, False, 3)
    assert m._tsel_select(scores, nvb, 0, 1) is None
    wide = torch.zeros(1, m.TSEL_MAX_BLOCKS + 1, 512, device="cuda")
    assert m._tsel_select(wide, torch.ones(512, dtype=torch.int32, device="cuda"), 0, 1) is None
    assert m._tsel_select(scores.transpose(1, 2).contiguous().transpose(1, 2), nvb, 0, 1) is None


def test_register_sets_the_hook_once_and_only_where_it_exists() -> None:
    indexer = types.SimpleNamespace(selector_hook=None)
    m._tsel_register(types.SimpleNamespace(indexer=indexer))
    assert indexer.selector_hook is m._tsel_select
    other = object()
    indexer.selector_hook = other
    m._tsel_register(types.SimpleNamespace(indexer=indexer))
    assert indexer.selector_hook is other
    m._tsel_register(
        types.SimpleNamespace(indexer=types.SimpleNamespace())
    )  # an indexer without the hook
    m._tsel_register(types.SimpleNamespace())  # a backend without an indexer


def test_the_shared_indexer_hook_is_inert_unset_or_declining() -> None:
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import MsaIndexer

    cfg = types.SimpleNamespace(init_blocks=0, local_blocks=1, num_index_heads=1, num_kv_heads=1)
    indexer = MsaIndexer(cfg)
    assert indexer.selector_hook is None
    scores, nvb = _inputs(4096, 300, False, 11)
    ref = _ref(scores, nvb, 0, 1)
    assert torch.equal(indexer._select(scores, nvb), ref)
    indexer.selector_hook = lambda *a: None
    assert torch.equal(indexer._select(scores, nvb), ref)


@needs_op
def test_fake_impl_matches_the_op() -> None:
    """The registered fake (catalog m31_fake_ops) returns what the op returns (nothing) and leaves out as is."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops

    assert "minimax_m31_select_prefill_tile" in m31_fake_ops._REGISTERED
    scores, nvb = _inputs(16384, 1160, False, 7)
    out = torch.empty(1160, 1, 16, dtype=torch.int32, device="cuda")  # (rows, heads, 16)
    assert op(scores, nvb, out, 0, 1) is None
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fout = mode.from_tensor(out)
        assert op(mode.from_tensor(scores), mode.from_tensor(nvb), fout, 0, 1) is None
    assert (tuple(fout.shape), fout.dtype) == (tuple(out.shape), out.dtype)
