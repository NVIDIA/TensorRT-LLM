# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused context-K/V side chain (``drafter_kv_side_chain``,
``drafter_fc_early_fork``) over a dense arena and over pages.

From its fork point and from its early part (``_side_chain_early``), the fused side chain
must leave the drafter's context bit for bit as the worker's ``_store_step_context`` leaves
it (unmasked, as the fork's built-in side chain calls it): over a dense full-length arena
(``_fused_side_chain``) and over the KV cache manager's pages (``_fused_side_chain_paged``,
with and without ``paged_ctx_kv_store``). The paged prologue's op must match its reference,
and the side chain must leave the arenas it does not write to the worker.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep import (  # noqa: E501
    spec_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as fork_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    ctx_kv_prep_paged as prep_mod,
)
from tensorrt_llm._torch.models import modeling_dflash
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(modeling_dflash._flashinfer_rope is None, reason="requires flashinfer RoPE"),
]

LAYERS = 5
KV_HEADS = 2
HEAD_DIM = 128
HIDDEN = 256
CAPTURE = 3
TARGET_HIDDEN = 192
K = 7
PAGE = 32
BLOCKS = 12
SLOTS = 70


class _Identity:
    """The sharded ``fc``'s gather over a TP group of one."""

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        return local


def _drafter(seed):
    """A drafter with the attributes ``precompute_context_kv`` and the side chain read.

    Its projection is the sharded ``fc`` (``spec_worker.sharded_project_target_hidden``) over
    one rank, so the early part takes it.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    drafter = object.__new__(DFlashForCausalLM)
    drafter.__dict__.update(
        _num_attn_layers=LAYERS,
        _num_kv_heads=KV_HEADS,
        _head_dim=HEAD_DIM,
        _fused_kv_weight=randn(LAYERS * 2 * KV_HEADS * HEAD_DIM, HIDDEN, scale=0.05),
        _fused_kv_bias=None,
        _k_norm_stacked=randn(LAYERS, HEAD_DIM),
        _k_norm_eps=1e-6,
        _input_ln_eps=None,
        _rope_dummy_q=None,
        _cos_sin_cache_fp32=torch.randn(8192, HEAD_DIM, generator=g, device="cuda"),
        _is_neox=True,
        fc=types.SimpleNamespace(
            weight=randn(HIDDEN, TARGET_HIDDEN * CAPTURE, scale=0.05).contiguous(), bias=None
        ),
        hidden_norm=lambda x: torch.nn.functional.rms_norm(x, (HIDDEN,), eps=1e-6),
        _fc_shard=spec_worker.ShardedFc(
            rank=0, rows=HIDDEN, gather=_Identity(), enable_pdl=False, world=1, max_rows=None
        ),
    )
    drafter.__dict__["project_target_hidden"] = types.MethodType(
        spec_worker.sharded_project_target_hidden, drafter
    )
    return drafter


def _lengths(num_gens, seed, capacity):
    """Distinct slots for the batch rows, and context lengths that keep every row in capacity."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch_to_slot = torch.randperm(SLOTS, generator=g, device="cuda")[:num_gens].contiguous()
    ctx_len = torch.randint(0, capacity - K, (SLOTS,), generator=g, device="cuda")
    return batch_to_slot, ctx_len


def _paged_state(num_gens, seed):
    """A worker state over the manager's pages (one view per layer), and its pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = num_gens * BLOCKS + 3
    pool = torch.randn((pages, LAYERS, 2, KV_HEADS, PAGE, HEAD_DIM), generator=g, device="cuda")
    pool = pool.to(torch.bfloat16)
    # Every (row, block) its own page, so no two entries share a destination.
    table = torch.randperm(pages, generator=g, device="cuda")[: num_gens * BLOCKS]
    table = table.to(torch.int32).view(num_gens, BLOCKS).contiguous()
    batch_to_slot, ctx_len = _lengths(num_gens, seed + 1, BLOCKS * PAGE)
    state = dict(
        _ctx_paged=True,
        _ctx_base=None,
        _ctx_keep=False,
        _max_ctx=BLOCKS * PAGE,
        _ctx_kv_buf=[pool[:, layer] for layer in range(LAYERS)],
        _ctx_page_size=PAGE,
        _ctx_block_tables=table,
        _ctx_block_counts=torch.full((num_gens,), BLOCKS, dtype=torch.int64, device="cuda"),
        _ctx_block_indptr=torch.arange(
            0, (num_gens + 1) * BLOCKS, BLOCKS, dtype=torch.int32, device="cuda"
        ),
        _ctx_page_table=None,
        _ctx_kv_indptr=None,
        _ctx_kv_last_page_len=torch.full((num_gens,), PAGE, dtype=torch.int32, device="cuda"),
        _ctx_paged_append=None,
        _ctx_len=ctx_len,
        _batch_to_slot=batch_to_slot,
    )
    return state, pool


def _dense_state(num_gens, seed):
    """A worker state over a dense full-length arena, and its K and V arenas."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    max_ctx = BLOCKS * PAGE
    shape = (SLOTS, LAYERS, max_ctx, KV_HEADS, HEAD_DIM)
    k_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    v_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    batch_to_slot, ctx_len = _lengths(num_gens, seed + 1, max_ctx)
    state = dict(
        _ctx_paged=False,
        _ctx_base=None,
        _ctx_keep=False,
        _max_ctx=max_ctx,
        _ctx_k_buf=k_buf,
        _ctx_v_buf=v_buf,
        _ctx_block_tables=None,
        _ctx_len=ctx_len,
        _batch_to_slot=batch_to_slot,
    )
    return state, (k_buf, v_buf)


def _worker(cls, state, store_op=False):
    worker = object.__new__(cls)
    worker.__dict__.update(state)
    if cls is not DFlashWorker:
        fork = object.__new__(fork_mod.ContextKvFork)
        fork.paged_store_on = store_op
        worker.__dict__["_kv_fork"] = fork
    return worker


def _spec_metadata(num_gens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 2)
    captured = torch.randn(
        (num_gens * (K + 1) + 5, TARGET_HIDDEN * CAPTURE + 64), generator=g, device="cuda"
    ).to(torch.bfloat16)
    return types.SimpleNamespace(
        captured_hidden_states=captured, hidden_size=TARGET_HIDDEN, num_capture_layers=CAPTURE
    )


def _builtin(state, drafter, spec_metadata, num_gens):
    """The fork's built-in side chain: the worker's store, unmasked."""
    worker = _worker(DFlashWorker, state)
    slots = worker._batch_to_slot[:num_gens]
    gen_hs = spec_metadata.captured_hidden_states[: num_gens * (K + 1), : TARGET_HIDDEN * CAPTURE]
    rows = torch.arange(num_gens, device="cuda")
    worker._store_step_context(drafter, gen_hs, slots, rows, worker._ctx_len[slots])


@pytest.mark.parametrize("num_gens", [1, 3, 8])
@pytest.mark.parametrize("store_op", [False, True], ids=["appends", "store_op"])
@pytest.mark.parametrize("early", [False, True], ids=["fork_point", "early"])
def test_the_paged_side_chain_is_the_worker_store(num_gens, store_op, early):
    seed = num_gens * 11 + 1
    drafter = _drafter(seed)
    spec_metadata = _spec_metadata(num_gens, seed)
    want_state, want_pool = _paged_state(num_gens, seed)
    _builtin(want_state, drafter, spec_metadata, num_gens)

    state, pool = _paged_state(num_gens, seed)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state, store_op)
    assert fork_mod._side_chain_refusal(worker, drafter) is None
    part = None
    if early:
        refreshed = []
        part = fork_mod._side_chain_early(
            worker, drafter, spec_metadata, num_gens, K, lambda: refreshed.append(True)
        )
        assert part is not None and refreshed == [True]
    fork_mod._fused_side_chain_paged(worker, drafter, spec_metadata, num_gens, K, part)
    torch.cuda.synchronize()
    assert torch.equal(pool, want_pool)


@pytest.mark.parametrize("num_gens", [1, 3, 8])
@pytest.mark.parametrize("early", [False, True], ids=["fork_point", "early"])
def test_the_dense_side_chain_is_the_worker_store(num_gens, early):
    seed = num_gens * 13 + 2
    drafter = _drafter(seed)
    spec_metadata = _spec_metadata(num_gens, seed)
    want_state, want_arenas = _dense_state(num_gens, seed)
    _builtin(want_state, drafter, spec_metadata, num_gens)

    state, arenas = _dense_state(num_gens, seed)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state)
    assert fork_mod._side_chain_refusal(worker, drafter) is None
    part = None
    if early:
        part = fork_mod._side_chain_early(
            worker, drafter, spec_metadata, num_gens, K, lambda: pytest.fail("no table to decode")
        )
        assert part is not None
    fork_mod._fused_side_chain(worker, drafter, spec_metadata, num_gens, K, part)
    torch.cuda.synchronize()
    assert torch.equal(arenas[0], want_arenas[0]) and torch.equal(arenas[1], want_arenas[1])


@pytest.mark.parametrize("num_gens", [1, 5, 64])
def test_the_paged_prologue_matches_its_reference(num_gens):
    g = torch.Generator(device="cuda").manual_seed(num_gens)
    rows, layers = K + 1, LAYERS
    slots = torch.randperm(SLOTS + 64, generator=g, device="cuda")[:num_gens].contiguous()
    ctx_len = torch.randint(0, 3 * PAGE, (SLOTS + 64,), generator=g, device="cuda")
    # Extents of 0..2 pages: some rows clamp to their allocation's last column, some to 0.
    counts = torch.randint(0, 3, (num_gens + 2,), generator=g, device="cuda")
    outs = [
        (
            torch.empty(num_gens * rows, dtype=torch.int64, device="cuda"),
            torch.empty(num_gens * rows, dtype=torch.int64, device="cuda"),
            torch.empty(num_gens * rows * layers, dtype=torch.int32, device="cuda"),
        )
        for _ in range(2)
    ]
    args = (ctx_len, slots, counts)
    prep_mod.ctx_kv_prep_paged.is_valid(*args, *outs[0], rows, layers, PAGE)
    prep_mod.ctx_kv_prep_paged(*args, *outs[0], rows, layers, PAGE)
    prep_mod.ctx_kv_prep_paged.reference(*args, *outs[1], rows, layers, PAGE)
    for got, want in zip(*outs):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    "case, reason",
    [
        ("windowed", "windowed"),
        ("private_pages", "private pages"),
        ("int32_extents", "not int64"),
    ],
)
def test_the_side_chain_leaves_other_arenas_to_the_worker(case, reason):
    drafter = _drafter(5)
    state, _ = _paged_state(2, 5)
    if case == "windowed":
        state["_ctx_base"] = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    elif case == "private_pages":
        state["_ctx_page_table"], state["_ctx_block_tables"] = state["_ctx_block_tables"], None
    else:
        state["_ctx_block_counts"] = state["_ctx_block_counts"].to(torch.int32)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state)
    assert reason in fork_mod._side_chain_refusal(worker, drafter)
