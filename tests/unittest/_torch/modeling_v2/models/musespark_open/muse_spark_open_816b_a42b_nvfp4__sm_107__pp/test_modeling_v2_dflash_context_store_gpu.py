# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 PP target's one-launch drafter-context store on a GPU.

* The store kernel (``page_store.store``) against the built-in store statements
  (``DFlashWorker._store_context_kv_paged``: per drafter layer a K and a V copy and
  flashinfer's ``append_paged_kv_cache``), with the released drafter's shape (5 layers,
  8 KV heads, head dim 128, 32-token pages): three requests per step at 1, 31, 32, 33, 2,592
  and 8,192 rows, block-table rows out of batch order, permuted pages, first positions off a
  page boundary, on a dense arena and on page-major per-layer views. The whole pool is
  compared bit for bit (random bit patterns, NaNs included, in the K/V and in the pages).
* The context-server worker's ``_store_prefill_context`` against the built-in one, on twin
  workers over the same steps (fresh, continued and restarted requests, a reuse-skipped
  prefix, an allocation overflow), on the private arena and on the manager's block tables:
  the pages bit for bit, the slot lengths and the host bookkeeping equal.
* Launches: the worker's store runs one store kernel and no append per step.
"""

from __future__ import annotations

import importlib.util
import types
from importlib import import_module
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PP = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__pp"
)
LAYERS, HEADS, HEAD_DIM, PAGE = 5, 8, 128, 32
ROWS = [1, 31, 32, 33, 2592, 8192]


def _modules():
    return (
        import_module(f"{_PP}.kernels.dflash_context.store"),
        import_module(f"{_PP}.kernels.dflash_context.page_store"),
        import_module(f"{_PP}.kernels.dflash_context.worker"),
    )


def _bits(*shape, generator):
    bits = torch.randint(
        -(1 << 15), 1 << 15, shape, dtype=torch.int16, device="cuda", generator=generator
    )
    return bits.view(torch.bfloat16)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


# --------------------------------------------------------------------------
# The store kernel against the built-in store statements
# --------------------------------------------------------------------------
def _pool(layout: str, pages: int, generator):
    """``(storage, per-layer page views)`` holding random bits."""
    if layout == "arena":
        storage = _bits(LAYERS, pages, 2, HEADS, PAGE, HEAD_DIM, generator=generator)
        return storage, lambda s: [s[layer] for layer in range(LAYERS)]
    storage = _bits(pages, LAYERS, 2, HEADS, PAGE, HEAD_DIM, generator=generator)
    return storage, lambda s: [s[:, layer] for layer in range(LAYERS)]


def _builtin_store(caches, table, run, k, kv_flat):
    """``DFlashWorker._store_context_kv_paged`` on one request, with the manager's tables."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    width = table.shape[1]
    rows = k.shape[0]
    owner = types.SimpleNamespace(
        _ctx_block_tables=table,
        _ctx_block_indptr=torch.arange(
            0, (table.shape[0] + 1) * width, width, dtype=torch.int32, device="cuda"
        ),
        _ctx_page_table=None,
        _ctx_page_size=PAGE,
        _ctx_kv_buf=caches,
        _ctx_kv_last_page_len=torch.full((table.shape[0],), PAGE, dtype=torch.int32, device="cuda"),
        _ctx_paged_append=None,
    )
    owner._get_ctx_paged_append = lambda: DFlashWorker._get_ctx_paged_append(owner)
    owner._ctx_paged_index_args = lambda: DFlashWorker._ctx_paged_index_args(owner)
    # What precompute_context_kv returns for V: the contiguous V half of the GEMM rows.
    v = kv_flat.view(rows, LAYERS, 2, HEADS, HEAD_DIM)[:, :, 1].contiguous()
    DFlashWorker._store_context_kv_paged(
        owner,
        k,
        v,
        torch.full((rows,), run.table_row, dtype=torch.long, device="cuda"),
        torch.arange(run.first, run.first + rows, dtype=torch.long, device="cuda"),
    )


@torch.no_grad()
@pytest.mark.parametrize("layout", ["arena", "page_major"])
@pytest.mark.parametrize("rows", ROWS)
def test_store_kernel_writes_the_built_in_pages(rows, layout):
    store, page_store, _ = _modules()
    generator = torch.Generator(device="cuda").manual_seed(rows)
    # Three requests: block-table rows out of batch order; first positions on, just after
    # and just before a page boundary; one short request between two long ones.
    runs = [
        store.Run(offset=0, rows=rows, table_row=2, first=0),
        store.Run(offset=rows, rows=31, table_row=0, first=5),
        store.Run(offset=rows + 31, rows=rows, table_row=3, first=3 * PAGE - 1),
    ]
    width = -(-(rows + 3 * PAGE) // PAGE) + 1
    table_rows = 4
    pages = table_rows * width
    table = (
        torch.randperm(pages, device="cuda", generator=generator)
        .to(torch.int32)
        .view(table_rows, width)
    )
    held = [
        (
            _bits(run.rows, LAYERS, HEADS, HEAD_DIM, generator=generator),
            _bits(run.rows, LAYERS * 2 * HEADS * HEAD_DIM, generator=generator),
        )
        for run in runs
    ]
    storage, views = _pool(layout, pages, generator)
    expected = storage.clone()
    for run, (k, kv_flat) in zip(runs, held):
        _builtin_store(views(expected), table, run, k, kv_flat)
    reference = storage.clone()
    store.reference_store(
        runs, [k for k, _ in held], [kv for _, kv in held], table, views(reference)
    )

    caches = views(storage)
    buffer = store.host_table(
        runs, [(k.data_ptr(), kv.data_ptr()) for k, kv in held], LAYERS, pin=True
    )
    meta, _ = store.split_table(buffer.to("cuda", non_blocking=True), len(runs), 0)
    addresses = torch.tensor([cache.data_ptr() for cache in caches], device="cuda")
    page_store.store(
        meta,
        addresses,
        table,
        caches[0],
        num_runs=len(runs),
        max_rows=max(run.rows for run in runs),
        layers=LAYERS,
        heads=HEADS,
        head_dim=HEAD_DIM,
    )
    torch.cuda.synchronize()
    assert _same_bits(reference, expected)
    assert _same_bits(storage, expected)


# --------------------------------------------------------------------------
# The worker's store against the built-in store, on twin workers
# --------------------------------------------------------------------------
MAX_SEQ = 20000


def _tiny_drafter():
    """The speculative tests' tiny DFlash 2 drafter, with positions for the long requests."""
    from transformers import Qwen3Config

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    path = Path(__file__).parents[4] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    config = dict(sem.TINY)
    config["max_position_embeddings"] = 2 * MAX_SEQ
    config["dflash_config"] = dict(sem.RELEASED_DFLASH_CONFIG)
    model_config = ModelConfig(
        pretrained_config=Qwen3Config.from_dict(config), attn_backend="TRTLLM"
    )
    drafter = DFlashForCausalLM(model_config, dflash_attention_backend="TRTLLM").to("cuda")
    drafter.load_weights(dict(sem._tiny_weights(True)))
    return drafter, sem


def _workers(drafter, tables: str, capacities: dict):
    """``(built-in worker, context-server worker)`` with the same paged arena and tables."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    _, _, worker_mod = _modules()
    workers = []
    for _ in range(2):
        worker = DFlashWorker(
            DFlashDecodingConfig(max_draft_len=3, attention_backend="TRTLLM"), Mapping()
        )
        worker._lazy_init_ctx_buffers(
            drafter,
            types.SimpleNamespace(max_num_requests=8),
            types.SimpleNamespace(kv_cache_manager=None, max_seq_len=MAX_SEQ),
            None,
        )
        assert worker._ctx_paged
        workers.append(worker)
    if tables == "manager":
        # The manager's per-iteration tables: rows by batch index, pages permuted over the
        # pool, every request's allocation from its cache's capacity; reused prefixes are
        # addressable.
        slots, width = workers[0]._ctx_page_table.shape
        generator = torch.Generator(device="cuda").manual_seed(5)
        table = (
            torch.randperm(slots * width, device="cuda", generator=generator)
            .to(torch.int32)
            .view(slots, width)
        )
        for worker in workers:
            worker._ctx_block_tables = table.clone()
            worker._ctx_block_indptr = torch.arange(
                0, (slots + 1) * width, width, dtype=torch.int32, device="cuda"
            )
            worker._ctx_kv_manager = types.SimpleNamespace(
                kv_cache_map={
                    req_id: types.SimpleNamespace(capacity=cap if cap else width * PAGE)
                    for req_id, cap in capacities.items()
                }
            )
            worker._ctx_reuse_addressable = True
    worker_mod.install(workers[1])
    return workers


def _step_inputs(requests, width: int):
    """``(spec metadata, attention metadata, positions, rows)`` of one context step of
    ``(request id, first position, rows)`` requests, with random captures."""
    lens = [rows for _, _, rows in requests]
    host_pos = [first + t for _, first, rows in requests for t in range(rows)]
    total = len(host_pos)
    captures = torch.randn(total, width, device="cuda").to(torch.bfloat16)
    meta = types.SimpleNamespace(
        request_ids=[req_id for req_id, _, _ in requests],
        host_position_ids=host_pos,
        max_num_requests=8,
        get_hidden_states=lambda num: captures[:num],
    )
    attn = types.SimpleNamespace(
        num_seqs=len(requests),
        num_contexts=len(requests),
        num_ctx_tokens=total,
        _seq_lens=torch.tensor(lens, dtype=torch.int32),
        kv_cache_manager=None,
        max_seq_len=MAX_SEQ,
    )
    positions = torch.tensor(host_pos, dtype=torch.int32, device="cuda")
    return meta, attn, positions, total


def _run_step(workers, drafter, requests, width: int):
    """Store one context step on both workers: the built-in store, then the worker's."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    meta, attn, positions, total = _step_inputs(requests, width)
    builtin, fast = workers
    DFlashWorker._store_prefill_context(builtin, drafter, meta, attn, positions, total)
    fast._store_prefill_context(drafter, meta, attn, positions, total)
    torch.cuda.synchronize()


def _assert_twins(workers):
    builtin, fast = workers
    assert _same_bits(fast._ctx_kv_buf, builtin._ctx_kv_buf)
    assert torch.equal(fast._ctx_len, builtin._ctx_len)
    assert fast._ctx_len_host == builtin._ctx_len_host
    assert fast._req_to_slot == builtin._req_to_slot
    assert fast._req_ctx_pos == builtin._req_ctx_pos
    assert list(fast._free_slots) == list(builtin._free_slots)


@torch.no_grad()
@pytest.mark.parametrize("tables", ["arena", "manager"])
@pytest.mark.parametrize("rows", ROWS)
def test_worker_store_is_the_built_in_store(rows, tables):
    from tensorrt_llm._torch.speculative import dflash_attention

    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    store, _, _ = _modules()
    torch.manual_seed(rows)
    drafter, sem = _tiny_drafter()
    width = len(drafter.target_layer_ids) * sem.TINY["hidden_size"]
    manager = tables == "manager"
    capacities = {11: 0, 12: 0, 13: 0, 14: 0, 15: 2 * PAGE}
    workers = _workers(drafter, tables, capacities)
    assert store.refusal(workers[1], drafter) is None
    # Fresh requests; with the manager's tables request 13 had [0, 70) reused.
    _run_step(workers, drafter, [(11, 0, rows), (12, 0, 33), (13, 70 if manager else 0, 1)], width)
    _assert_twins(workers)
    # The context-server worker ran its own store, not the built-in one.
    assert workers[1]._page_store_logged == {None}
    assert workers[1]._ctx_kv_buf.view(torch.int16).any()
    # Request 11 continues across a page boundary, 14 is new, 12 restarts at position 5.
    _run_step(workers, drafter, [(11, rows, 31), (14, 0, 32), (12, 5, 40)], width)
    _assert_twins(workers)
    if manager:
        # Request 15 does not fit its allocation: neither store writes it.
        _run_step(workers, drafter, [(15, 0, 2 * PAGE + 1), (14, 32, 1)], width)
        _assert_twins(workers)
        assert 15 not in workers[1]._req_to_slot


# --------------------------------------------------------------------------
# Launches
# --------------------------------------------------------------------------
@torch.no_grad()
def test_worker_store_runs_one_store_kernel_and_no_append():
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    from tensorrt_llm._torch.speculative import dflash_attention
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    drafter, sem = _tiny_drafter()
    width = len(drafter.target_layer_ids) * sem.TINY["hidden_size"]
    requests = [(21, 0, 2592), (22, 0, 33)]

    def kernels(fast: bool):
        workers = _workers(drafter, "arena", {})
        _run_step(workers, drafter, requests, width)  # warm-up: lazy builds, kernel compiles
        worker = _workers(drafter, "arena", {})[1 if fast else 0]
        store = type(worker)._store_prefill_context if fast else DFlashWorker._store_prefill_context
        meta, attn, positions, total = _step_inputs(requests, width)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            store(worker, drafter, meta, attn, positions, total)
            torch.cuda.synchronize()
        return [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]

    builtin = kernels(False)
    fast = kernels(True)
    print(f"device activities of one step, 2 requests: built-in {len(builtin)}, fast {len(fast)}")
    assert sum("AppendPagedKVCache" in name for name in builtin) == 2 * drafter._num_attn_layers
    assert not any("AppendPagedKVCache" in name for name in fast)
    assert sum("_draft_kv_page_store_kernel" in name for name in fast) == 1
    assert len(fast) < len(builtin)
