# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paged drafter of a generation server reads no drafter context row below its window.

A context server that keeps only a window of the drafter context
(``KVCacheManagerV2.enable_draft_context_window``) leaves the draft rows below each prompt's last
``KEEP`` positions unwritten, and for pages a later prompt inherits its reuse guard promises the
rows of the window ``[P - WINDOW, P)`` only. An attention kernel that loaded a context row below
the window, even one it then masks, would turn a NaN there into a NaN output (``0 * NaN`` in the
probabilities times V). At the served geometry -- a TP4 rank's 16 query and 2 KV heads of width
128, KV blocks of 32 tokens, a block of ``k + 1 = 8`` query rows, the drafter's 2048-token
two-sided window, the first block decode after the context, window starts inside a page -- the
output and the appended block are bit for bit the same whether the rows below the window hold
valid values or NaN:

* ``dflash_window_attention`` (with and without PDL) and ``dflash_window_attention_wide`` over
  pages, the block's K/V appended in the call;
* the drafter's block decode (``drafter_block_decode``, with and without
  ``wide_drafter_window_attention``) over per-layer views of one interleaved paged pool, as a
  generation server's drafter reads the KV cache manager's draft layers.

Two NaN regions are checked: the band from one KV tile below the kept rows up to the window
(``[P - KEEP - TILE, P - WINDOW)``), and every row below the window, where the table entries of
the pages wholly below the window also name a NaN page that is no request's (a generation server
may hold no page there).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.wide_drafter_window_attention import (  # noqa: E501
    attention as wide_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    dflash_window_attention_wide as wide_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    WINDOW as WINDOW_SIZE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec.dflash_window_attention import (  # noqa: E501
    dflash_window_attention,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.speculative import dflash_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

#: A TP4 rank of the released drafter: query and KV heads, head width; the block of k + 1 rows.
Q_HEADS, KV_HEADS, HEAD_DIM, BLOCK = 16, 2, 128, 8
#: The KV block size of the served caches.
PAGE = 32
#: The drafter's window, and the rows a context server keeps: the window, the reuse rewind and
#: one block.
WINDOW = 2048
KEEP = WINDOW + 512 + PAGE
#: The longest KV tile any of the kernels could load below its first key.
TILE = 128
#: Prompt lengths: the first key read (P - WINDOW + 1) and the window's start (P - WINDOW) inside
#: a page, except 2111 and 4127 (first key on a page boundary) and 2112 (window start on one);
#: 1500 has no row below the window.
CONTEXTS = (2600, 5001, 2049, 3333, 10_000, 2111, 1500, 4127, 2112)
NAN_REGIONS = ("band", "below-window")


def _randn(gen, *shape):
    return torch.randn(*shape, device="cuda", generator=gen).bfloat16()


def _contexts(batch: int):
    """``batch`` prompt lengths cycling through ``CONTEXTS``, each cycle 37 positions later."""
    return [CONTEXTS[i % len(CONTEXTS)] + 37 * (i // len(CONTEXTS)) for i in range(batch)]


def _paged(gen, contexts, layers):
    """A pool ``[pages, layers, K/V, Hkv, page, D]`` of random rows and a page table whose rows
    list each request's pages in token order, scattered; request ``b`` is on table row
    ``rows - 1 - b``, so its table row is not its batch index."""
    batch = len(contexts)
    per_row = -(-(max(contexts) + BLOCK) // PAGE)
    rows = batch + 1
    table = torch.randperm(rows * per_row + 3, device="cuda", generator=gen)
    table = table[: rows * per_row].view(rows, per_row).int()
    pool = _randn(gen, rows * per_row + 3, layers, 2, KV_HEADS, PAGE, HEAD_DIM)
    table_rows = (rows - 1 - torch.arange(batch, device="cuda")).int()
    return pool, table, table_rows


def _poison(pool, table, table_rows, contexts, region: str):
    """NaN into every layer's K and V at the ``region`` rows of each request. Below the window,
    the table entries of the pages wholly below it then name a page no table row lists, all NaN.
    Returns ``(rows poisoned, the table to read through)``."""
    poisoned = 0
    remapped = table.clone()
    unused = torch.ones(pool.shape[0], dtype=torch.bool, device=pool.device)
    unused[table.flatten().long()] = False
    spare = int(unused.nonzero()[0])
    for b, context in enumerate(contexts):
        end = context - WINDOW
        begin = max(0, context - KEEP - TILE) if region == "band" else 0
        if end <= begin:
            continue
        cols = torch.arange(begin, end, device="cuda")
        row = int(table_rows[b])
        pages = table[row].long()[cols // PAGE]
        pool[pages, :, :, :, cols % PAGE] = float("nan")
        poisoned += end - begin
        if region == "below-window":
            remapped[row, : end // PAGE] = spare
    if region == "below-window":
        pool[spare] = float("nan")
    return poisoned, remapped


def _block_rows(pool, table, table_rows, contexts):
    """Every layer's K/V rows of each request's block columns ``[P, P + BLOCK)``."""
    out = []
    for b, context in enumerate(contexts):
        cols = torch.arange(context, context + BLOCK, device="cuda")
        pages = table[int(table_rows[b])].long()[cols // PAGE]
        out.append(pool[pages, :, :, :, cols % PAGE])
    return torch.cat(out)


# --------------------------------------------------------------------------
# The window attention ops over pages
# --------------------------------------------------------------------------
def test_the_window_is_the_released_drafters():
    assert WINDOW_SIZE == (WINDOW - 1, WINDOW - 1)


@pytest.mark.parametrize("region", NAN_REGIONS)
@pytest.mark.parametrize(
    "op_name, batch, pdl",
    [("single", 8, True), ("single", 8, False), ("wide", 8, True), ("wide", 24, True),
     ("wide", 64, True)],
)  # fmt: skip
def test_the_window_ops_read_no_row_below_the_window(op_name, batch, pdl, region):
    """Over pages, with the block's K/V appended: the output and the appended block with NaN
    below the window are bitwise those with valid rows there."""
    op = wide_ops.dflash_window_attention_wide if op_name == "wide" else dflash_window_attention
    gen = torch.Generator(device="cuda").manual_seed(batch)
    contexts = _contexts(batch)
    pool, table, table_rows = _paged(gen, contexts, layers=1)
    q = _randn(gen, batch, BLOCK, Q_HEADS, HEAD_DIM) * 2
    k_new = _randn(gen, batch * BLOCK, KV_HEADS, HEAD_DIM)
    v_new = _randn(gen, batch * BLOCK, KV_HEADS, HEAD_DIM)
    seqlens = (torch.tensor(contexts, device="cuda") + BLOCK).int()

    def run(pool, table):
        k_pages = pool[:, 0, 0].transpose(1, 2)
        v_pages = pool[:, 0, 1].transpose(1, 2)
        out = op(
            q,
            k_pages,
            v_pages,
            seqlens,
            table_rows,
            WINDOW_SIZE,
            None,
            pdl,
            k_new=k_new,
            v_new=v_new,
            page_table=table,
        )
        torch.cuda.synchronize()
        return out

    valid_pool, nan_pool = pool.clone(), pool.clone()
    count, nan_table = _poison(nan_pool, table, table_rows, contexts, region)
    assert count > 0
    valid = run(valid_pool, table)
    poisoned = run(nan_pool, nan_table)
    assert torch.isfinite(valid.float()).all()
    assert torch.isfinite(poisoned.float()).all(), "a row below the window reached the output"
    assert torch.equal(poisoned, valid), (poisoned.float() - valid.float()).abs().max().item()
    assert torch.equal(
        _block_rows(nan_pool, nan_table, table_rows, contexts),
        _block_rows(valid_pool, table, table_rows, contexts),
    )


# --------------------------------------------------------------------------
# The drafter's block decode over a paged pool
# --------------------------------------------------------------------------
def _semantics():
    path = Path(__file__).parents[4] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    return sem


def _served_rank_drafter(monkeypatch, wide: bool):
    """The speculative tests' tiny DFlash 2 drafter at a TP4 rank's head geometry with the
    released drafter's window, with the generation server's block decode installed."""
    sem = _semantics()
    for key, value in (
        ("num_attention_heads", Q_HEADS),
        ("num_key_value_heads", KV_HEADS),
        ("head_dim", HEAD_DIM),
        ("sliding_window", WINDOW),
        ("max_position_embeddings", 16384),
    ):
        monkeypatch.setitem(sem.TINY, key, value)
    config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(dict(sem._tiny_weights(True)))
    drafter._build_fused_kv_buffers()
    block_decode.install(drafter, True, True)
    if wide:
        wide_attention.install(drafter)
    for layer in range(drafter._num_attn_layers):
        assert drafter._resolve_block_attention(layer) == (False, WINDOW_SIZE)
    return drafter, sem.TINY["hidden_size"]


@torch.no_grad()
@pytest.mark.parametrize("region", NAN_REGIONS)
@pytest.mark.parametrize("wide", [False, True], ids=["single", "wide"])
def test_the_paged_block_decode_reads_no_row_below_the_window(monkeypatch, wide, region):
    """The first block decode after the context, over per-layer views of one interleaved paged
    pool: the drafter's output and the appended blocks with NaN below the window are bitwise
    those with valid rows there."""
    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    drafter, hidden = _served_rank_drafter(monkeypatch, wide)
    layers = drafter._num_attn_layers
    gen = torch.Generator(device="cuda").manual_seed(7)
    contexts = list(CONTEXTS)
    pool, table, table_rows = _paged(gen, contexts, layers)
    ctx = torch.tensor(contexts, device="cuda")
    step = dict(
        noise_embedding=_randn(gen, len(contexts), BLOCK, hidden),
        query_positions=ctx[:, None] + torch.arange(BLOCK, device="cuda"),
        num_ctx_per_req=ctx,
        ctx_cache_batch_idx=table_rows.long(),
    )
    if wide:
        assert drafter.__dict__[wide_attention.ATTRIBUTE].takes(len(contexts) * BLOCK)

    def run(pool, table):
        out = drafter.dflash_forward(
            **step,
            ctx_k_cache=None,
            ctx_v_cache=None,
            ctx_kv_cache=[pool[:, layer] for layer in range(layers)],
            ctx_page_table=table,
        )
        torch.cuda.synchronize()
        return out

    first_pool, nan_pool, second_pool = pool.clone(), pool.clone(), pool.clone()
    count, nan_table = _poison(nan_pool, table, table_rows, contexts, region)
    assert count > 0
    valid = run(first_pool, table)
    poisoned = run(nan_pool, nan_table)
    again = run(second_pool, table)
    assert torch.equal(again, valid), "the block decode is not deterministic on these inputs"
    assert torch.isfinite(valid.float()).all()
    assert torch.isfinite(poisoned.float()).all(), "a row below the window reached the output"
    assert torch.equal(poisoned, valid), (poisoned.float() - valid.float()).abs().max().item()
    assert torch.equal(
        _block_rows(nan_pool, nan_table, table_rows, contexts),
        _block_rows(first_pool, table, table_rows, contexts),
    )
