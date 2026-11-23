# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm107 PP target's DFlash drafter context on GPUs.

* Capture integrity: the four stages of the 15/16/16/15 split carry the captures of the
  drafter's target layers to the last stage (``carry.CaptureCarry``) with the real DFlash
  metadata and its ``inplace_slice_copy`` tap. Every stage's capture buffer is NaN-filled
  before each step, and the last stage's assembled columns are compared, bit for bit, with
  the residuals each *sending* stage dumped for its own target layers, keyed by layer id
  (never with the last stage's own buffer). On one GPU the stages run one after another over
  a queue; on four GPUs (``mpirun -n 4``) over the PP transport, NCCL and the buffered ring.
* The context-server worker runs no drafter forward on steps with generation rows (the
  engine's warm-up and decode-graph capture), and the built-in step on context-only steps.
* The reduced drafter's context K/V is bitwise the full drafter's, and the drafter K/V the
  last stage's worker appends into its paged arena from the carried captures equals an
  offline recompute from the sending stages' dumps.
* A drafter context window: the drafter K/V rows a generation server's window reads are
  bitwise the rows of the whole write, the rows below the kept positions are never written,
  and over the PP transport the last stage's kept-row captures equal the sending stages'
  dumps at the kept rows.

The tiny DFlash 2 drafter of the speculative tests stands in for the released one.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_PP = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__pp"
)
NUM_LAYERS = 62
SPLIT_15_16_16_15 = [15, 16, 16, 15]
TARGET_LAYERS = (1, 16, 30, 44, 59)
DTYPE = torch.bfloat16


def _modules():
    from importlib import import_module

    return (
        import_module(f"{_PP}.stages"),
        import_module(f"{_PP}.kernels.dflash_context.route"),
        import_module(f"{_PP}.kernels.dflash_context.carry"),
        import_module(f"{_PP}.kernels.dflash_context.worker"),
        import_module(f"{_PP}.kernels.dflash_context.slim"),
    )


# --------------------------------------------------------------------------
# A stage text model with the built-in stage contract and stub layers
# --------------------------------------------------------------------------
class _Norm(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return (residual * self.scale).to(DTYPE)


class _Layer(nn.Module):
    """One residual update with the built-in decoder layer's call contract and DFlash tap."""

    def __init__(self, index: int, dumps: dict) -> None:
        super().__init__()
        self.layer_idx = index
        self.pre_qkv_proj_norm = _Norm(0.5 + index / 128)
        self.dumps = dumps

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        **kwargs,
    ):
        attn = (
            precomputed_attn_input
            if precomputed_attn_input is not None
            else self.pre_qkv_proj_norm(hidden_states)
        )
        hidden_states = torch.tanh(hidden_states * 0.9 + attn.float() * 0.3 + self.layer_idx / 50)
        self.dumps[self.layer_idx] = hidden_states.clone()
        spec_metadata = kwargs.get("spec_metadata")
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)
        if next_layer_norm is None:
            return hidden_states
        return hidden_states, next_layer_norm(hidden_states)


class _Stage(nn.Module):
    def __init__(self, stage: int, hidden: int, vocab: int, partition=SPLIT_15_16_16_15) -> None:
        super().__init__()
        stages = _modules()[0]
        ranges = stages.stage_ranges(NUM_LAYERS, 4, partition)
        self._pp_stage = ranges[stage]
        self.dumps: dict = {}
        self.layers = nn.ModuleList(_Layer(i, self.dumps) for i in range(NUM_LAYERS))
        self.model_config = types.SimpleNamespace(
            pretrained_config=types.SimpleNamespace(hidden_size=hidden)
        )
        self.dtype = DTYPE
        self.m_emb = 1.0
        generator = torch.Generator().manual_seed(0)
        self.embed_tokens = nn.Embedding(vocab, hidden).requires_grad_(False)
        self.embed_tokens.weight.copy_(torch.randn(vocab, hidden, generator=generator))
        self.embed_norm = nn.Identity()
        self.norm = nn.Identity()


def _metadata(layers, hidden: int, max_rows: int):
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm.llmapi import DFlashDecodingConfig

    config = DFlashDecodingConfig(max_draft_len=7)
    meta = DFlashSpecMetadata(
        max_draft_len=7,
        max_total_draft_tokens=7,
        spec_dec_mode=config.spec_dec_mode,
        max_num_requests=4,
        layers_to_capture=list(layers),
        hidden_size=hidden,
        max_num_tokens=max_rows,
        dtype=DTYPE,
    )
    return meta


class _Queue:
    def __init__(self) -> None:
        self.edges: dict = {}
        self.stage = 0

    def send(self, tensors) -> None:
        self.edges.setdefault(self.stage, []).append([t.clone() for t in tensors])

    def recv(self, tensors) -> None:
        for tensor, value in zip(tensors, self.edges[self.stage - 1].pop(0)):
            assert tensor.shape == value.shape and tensor.dtype == value.dtype
            tensor.copy_(value)


def _window_metadata(layers, hidden, max_rows, lengths, starts, keep_from):
    meta = _metadata(layers, hidden, max_rows)
    meta.captured_hidden_states.fill_(float("nan"))
    meta.draft_context_keep_from = list(keep_from)
    meta.host_position_ids = torch.tensor(
        [position for start, n in zip(starts, lengths) for position in range(start, start + n)],
        dtype=torch.int32,
    )
    return meta


def _window_attn(lengths):
    return types.SimpleNamespace(
        num_contexts=len(lengths), seq_lens=torch.tensor(lengths, dtype=torch.int32)
    )


def _owner(route, layer: int) -> int:
    return next(s.stage for s in route.stages if layer in route.layers_of(s.own))


def _run_chain_one_gpu(layers, hidden, rows, max_rows, vocab=97, partition=SPLIT_15_16_16_15):
    stages, route_mod, carry_mod, _, _ = _modules()
    route = route_mod.capture_route(layers, stages.stage_ranges(NUM_LAYERS, 4, partition), hidden)
    queue = _Queue()
    input_ids = (torch.arange(rows, device="cuda") * 7) % vocab
    metas, dumps = [], []
    for stage in range(4):
        queue.stage = stage
        core = _Stage(stage, hidden, vocab, partition).cuda()
        meta = _metadata(layers, hidden, max_rows)
        meta.captured_hidden_states.fill_(float("nan"))
        carry = carry_mod.CaptureCarry(route, stage, send=queue.send, recv=queue.recv)
        carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
        metas.append(meta)
        dumps.append(dict(core.dumps))
    torch.cuda.synchronize()
    return route, metas, dumps


@torch.no_grad()
@pytest.mark.parametrize("rows", [1, 300, 4096])
def test_capture_integrity_on_one_gpu(rows):
    hidden = 256
    route, metas, dumps = _run_chain_one_gpu(TARGET_LAYERS, hidden, rows, max_rows=4096)
    step = metas[3].get_hidden_states(rows)
    assert torch.isfinite(step.float()).all(), "a capture column was read before it was written"
    for index, layer in enumerate(TARGET_LAYERS):
        sent = dumps[_owner(route, layer)][layer].to(DTYPE)
        got = step[:, index * hidden : (index + 1) * hidden]
        assert torch.equal(got.view(torch.int16), sent.view(torch.int16)), layer


# --------------------------------------------------------------------------
# Four GPUs, the PP transport
# --------------------------------------------------------------------------
def _pp4_main(buffered: bool) -> None:
    """One rank of ``mpirun -n 4``: two context steps of the 15/16/16/15 split over the PP
    transport; per-layer checksums of the sending stages' dumps against the last stage's
    assembled columns."""
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.communicator import init_pp_comm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=4, rank=rank, gpus_per_node=4, pp_size=4, pp_partition=SPLIT_15_16_16_15
    )
    init_pp_comm(mapping, buffered=buffered)
    stages, route_mod, carry_mod, _, _ = _modules()
    hidden, max_rows, vocab = 8192, 8192, 97
    route = route_mod.capture_route(
        TARGET_LAYERS, stages.stage_ranges(NUM_LAYERS, 4, SPLIT_15_16_16_15), hidden
    )
    core = _Stage(rank, hidden, vocab).cuda()
    meta = _metadata(TARGET_LAYERS, hidden, max_rows)
    carry = carry_mod.CaptureCarry(route, rank)
    for rows in (8192, 1234):
        meta.captured_hidden_states.fill_(float("nan"))
        core.dumps.clear()
        input_ids = (torch.arange(rows, device="cuda") * 7) % vocab
        with torch.no_grad():
            carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
        torch.cuda.synchronize()

        def checksum(t: torch.Tensor) -> int:
            return int(t.contiguous().view(torch.int16).to(torch.int64).sum().item())

        own = {
            layer: checksum(core.dumps[layer].to(DTYPE))
            for layer in TARGET_LAYERS
            if layer in core.dumps
        }
        assembled = None
        if rank == 3:
            step = meta.get_hidden_states(rows)
            assert torch.isfinite(step.float()).all()
            assembled = {
                layer: checksum(step[:, i * hidden : (i + 1) * hidden])
                for i, layer in enumerate(TARGET_LAYERS)
            }
        everyone = comm.gather(own, root=3)
        if rank == 3:
            sent = {layer: value for part in everyone for layer, value in part.items()}
            assert sent == assembled, (rows, buffered, sent, assembled)
    comm.Barrier()
    if rank == 0:
        print(f"PP4 capture integrity PASS (buffered={buffered})")


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
@pytest.mark.parametrize("buffered", [False, True], ids=["nccl", "buffered_ring"])
def test_capture_integrity_pp4(buffered):
    env = dict(os.environ)
    result = subprocess.run(
        [
            "mpirun",
            "--allow-run-as-root",
            "-n",
            "4",
            sys.executable,
            __file__,
            "pp4",
            "1" if buffered else "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "PASS" in result.stdout
    # The ring carried every message: no fallback to NCCL for an oversize boundary.
    assert "such messages go through NCCL" not in result.stdout + result.stderr


# --------------------------------------------------------------------------
# The context-server worker
# --------------------------------------------------------------------------
class _CountingDrafter:
    def __init__(self) -> None:
        self.forwards = 0

    def dflash_forward(self, *args, **kwargs):
        self.forwards += 1
        raise AssertionError("a context server ran a drafter forward")


def test_context_worker_runs_no_drafter_forward_on_generation_rows(monkeypatch):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    worker_mod = _modules()[3]
    k, batch = 7, 4
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=k), Mapping())
    assert worker_mod.refusal(worker) is None
    worker_mod.install(worker)
    assert isinstance(worker, worker_mod.ContextServerDFlashWorker)
    accepted = torch.zeros(batch, k + 1, dtype=torch.int32, device="cuda")
    num_accepted = torch.ones(batch, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        worker, "sample_and_accept_draft_tokens", lambda *a, **kw: (accepted, num_accepted)
    )
    meta = types.SimpleNamespace(
        runtime_draft_len=k,
        is_cuda_graph=True,
        request_ids=None,
        batch_indices_cuda=torch.arange(batch, dtype=torch.int32, device="cuda"),
        use_rejection_sampling=False,
        draft_probs=None,
    )
    attn = types.SimpleNamespace(
        num_seqs=batch, num_contexts=0, kv_cache_manager=None, has_spec_dec_saved_state=False
    )
    logits = torch.zeros(batch * (k + 1), 64, device="cuda")
    drafter = _CountingDrafter()
    out = worker.forward(None, None, None, logits, attn, meta, draft_model=drafter)
    assert drafter.forwards == 0
    assert out["next_draft_tokens"].shape == (batch, k) and not out["next_draft_tokens"].any()
    assert out["next_new_tokens"].shape == (batch, k + 1)


# --------------------------------------------------------------------------
# The reduced drafter, and the drafter K/V pages
# --------------------------------------------------------------------------
def _tiny_drafter(backend: str = "TRTLLM"):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    path = Path(__file__).parents[4] / "speculative" / "hw_agnostic" / "test_dflash2_semantics.py"
    spec = importlib.util.spec_from_file_location("_dflash2_semantics", path)
    sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sem)
    config = ModelConfig(pretrained_config=sem._tiny_config(True), attn_backend="TRTLLM")
    drafter = DFlashForCausalLM(config, dflash_attention_backend=backend).to("cuda")
    drafter.load_weights(dict(sem._tiny_weights(True)))
    return drafter, sem


@torch.no_grad()
def test_reduced_drafter_context_kv_is_the_full_drafters():
    slim = _modules()[4]
    drafter, sem = _tiny_drafter()
    hidden = sem.TINY["hidden_size"]
    captures = torch.randn(37, len(drafter.target_layer_ids) * hidden, device="cuda").to(DTYPE)
    positions = torch.arange(100, 137, device="cuda")
    k_full, v_full = drafter.precompute_context_kv(
        drafter.project_target_hidden(captures), positions
    )
    released = slim.slim(drafter)
    assert released > 0
    # Left: the context write's ``fc`` / ``hidden_norm``, and the embedding and LM head a
    # drafter borrows from the target (registered under ``draft_model_full``; this
    # standalone drafter holds its own untied copies of both), which ``slim`` keeps.
    own = {"fc.weight", "hidden_norm.weight"}
    borrowed = ("draft_model_full.model.embed_tokens.", "draft_model_full.lm_head.")
    kept = {name for name, _ in drafter.named_parameters()}
    assert own <= kept
    assert all(name.startswith(borrowed) for name in kept - own), sorted(kept - own)
    assert all(any(name.startswith(prefix) for name in kept) for prefix in borrowed)
    k, v = drafter.precompute_context_kv(drafter.project_target_hidden(captures), positions)
    assert torch.equal(k, k_full) and torch.equal(v, v_full)


@torch.no_grad()
def test_drafter_kv_pages_equal_an_offline_recompute_from_the_sending_stages():
    """The last stage's worker appends the drafter K/V of the carried captures into its paged
    arena; read back by page, they equal ``precompute_context_kv`` over the captures rebuilt
    from the sending stages' dumps (layer order), with the reduced drafter."""
    from tensorrt_llm._torch.speculative import dflash_attention
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    _, _, _, worker_mod, slim = _modules()
    drafter, sem = _tiny_drafter()
    slim.slim(drafter)
    hidden = sem.TINY["hidden_size"]
    # One target layer per captured column block of the drafter's ``fc``: the first
    # stage's (three hops to the last stage) and the third stage's (one hop).
    layers = (1, 44)
    assert len(layers) == len(drafter.target_layer_ids)
    rows, request_id = 45, 7
    route, metas, dumps = _run_chain_one_gpu(layers, hidden, rows, max_rows=64)
    meta = metas[3]
    meta.request_ids = [request_id]
    meta.host_position_ids = list(range(rows))
    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=3, attention_backend="TRTLLM"), Mapping()
    )
    worker_mod.install(worker)
    attn = types.SimpleNamespace(
        num_seqs=1,
        num_contexts=1,
        num_ctx_tokens=rows,
        _seq_lens=torch.tensor([rows], dtype=torch.int32),
        kv_cache_manager=None,
        max_seq_len=256,
    )
    worker._lazy_init_ctx_buffers(drafter, meta, attn, None)
    assert worker._ctx_paged
    positions = torch.arange(rows, device="cuda")
    worker._store_prefill_context(drafter, meta, attn, positions, rows)
    torch.cuda.synchronize()

    sent = torch.cat([dumps[_owner(route, layer)][layer].to(DTYPE) for layer in layers], dim=1)
    k_ref, v_ref = drafter.precompute_context_kv(drafter.project_target_hidden(sent), positions)
    slot = worker._req_to_slot[request_id]
    page = worker._ctx_page_size
    pages = worker._ctx_page_table[slot, positions // page].long()
    offsets = positions % page
    for layer in range(k_ref.shape[1]):
        pool = worker._ctx_kv_buf[layer]
        assert torch.equal(pool[pages, 0, :, offsets, :], k_ref[:, layer]), layer
        assert torch.equal(pool[pages, 1, :, offsets, :], v_ref[:, layer]), layer


# --------------------------------------------------------------------------
# A drafter context window
# --------------------------------------------------------------------------
def _run_window_chain_one_gpu(layers, hidden, lengths, starts, keep_from, max_rows, vocab=97):
    """The four stages of one context step over a queue, each planned first (the shell's
    forward); returns the last stage's carry and metadata."""
    stages, route_mod, carry_mod, _, _ = _modules()
    route = route_mod.capture_route(
        layers, stages.stage_ranges(NUM_LAYERS, 4, SPLIT_15_16_16_15), hidden
    )
    queue = _Queue()
    rows = sum(lengths)
    input_ids = (torch.arange(rows, device="cuda") * 7) % vocab
    carry = meta = None
    for stage in range(4):
        queue.stage = stage
        core = _Stage(stage, hidden, vocab).cuda()
        meta = _window_metadata(layers, hidden, max_rows, lengths, starts, keep_from)
        carry = carry_mod.CaptureCarry(
            route,
            stage,
            send=queue.send,
            recv=queue.recv,
            raw_send=queue.send,
            raw_recv=queue.recv,
            eager_break=lambda fn: fn,
        )
        assert carry.plan_step(_window_attn(lengths), meta) is not None
        carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
    torch.cuda.synchronize()
    return carry, meta


def _context_worker(worker_mod, drafter, meta, attn, carry=None):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=3, attention_backend="TRTLLM"), Mapping()
    )
    worker_mod.install(worker, carry)
    worker._lazy_init_ctx_buffers(drafter, meta, attn, None)
    assert worker._ctx_paged
    for pool in worker._ctx_kv_buf:
        pool.fill_(float("nan"))
    return worker


@torch.no_grad()
@pytest.mark.parametrize("kept_store", ["one-launch", "built-in"])
def test_window_drafter_rows_equal_the_whole_write(monkeypatch, kept_store):
    """Two context requests of one step, each keeping its last ``keep`` positions: the drafter
    K/V rows of every position the generation window reads (the last ``window``) are bitwise
    the whole write's; the kept rows below the window equal it too; no row below the kept
    positions is written; the slots' lengths and chunk positions are the whole write's.

    The whole write takes the built-in calls; the window write takes the one-launch store or
    the built-in calls (``kept_store``)."""
    from tensorrt_llm._torch.speculative import dflash_attention

    if (reason := dflash_attention.dflash_trtllm_gen_unavailability_reason()) is not None:
        pytest.skip(reason)
    worker_mod, slim = _modules()[3], _modules()[4]
    drafter, sem = _tiny_drafter()
    slim.slim(drafter)
    hidden = sem.TINY["hidden_size"]
    layers = (1, 44)
    lengths, starts, ids = [45, 30], [0, 0], [7, 8]
    window, keep = 16, 24
    keep_from = [max(0, n - keep) for n in lengths]
    rows = sum(lengths)
    attn = types.SimpleNamespace(
        num_seqs=2,
        num_contexts=2,
        num_ctx_tokens=rows,
        _seq_lens=torch.tensor(lengths, dtype=torch.int32),
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        kv_cache_manager=None,
        max_seq_len=256,
    )

    # The whole write: the step without keep positions.
    _, metas, _ = _run_chain_one_gpu(layers, hidden, rows, max_rows=128)
    whole_meta = metas[3]
    whole_meta.request_ids = ids
    whole_meta.host_position_ids = torch.tensor(
        [p for s, n in zip(starts, lengths) for p in range(s, s + n)], dtype=torch.int32
    )
    positions = whole_meta.host_position_ids.to("cuda")
    forced = "the built-in store is forced"
    whole = _context_worker(worker_mod, drafter, whole_meta, attn)
    with monkeypatch.context() as patch:
        patch.setattr(worker_mod, "store_override", lambda cls: forced)
        whole._store_prefill_context(drafter, whole_meta, attn, positions, rows)
    assert whole._page_store_logged == {forced}

    # The window's write.
    carry, meta = _run_window_chain_one_gpu(layers, hidden, lengths, starts, keep_from, 128)
    meta.request_ids = ids
    kept = _context_worker(worker_mod, drafter, meta, attn, carry)
    with monkeypatch.context() as patch:
        if kept_store == "built-in":
            patch.setattr(worker_mod, "store_override", lambda cls: forced)
        kept._store_prefill_context(drafter, meta, attn, positions, rows)
    assert kept._page_store_logged == ({None} if kept_store == "one-launch" else {forced})
    torch.cuda.synchronize()

    def rows_of(worker, request, positions, layer, kv):
        slot = worker._req_to_slot[request]
        pages = worker._ctx_page_table[slot, positions // worker._ctx_page_size].long()
        return worker._ctx_kv_buf[layer][pages, kv, :, positions % worker._ctx_page_size, :]

    for request, length, first in zip(ids, lengths, keep_from):
        assert whole._ctx_len_host[whole._req_to_slot[request]] == length
        assert kept._ctx_len_host[kept._req_to_slot[request]] == length
        assert kept._req_ctx_pos[request] == whole._req_ctx_pos[request] == length
        read = torch.arange(length - window, length, device="cuda")
        written = torch.arange(first, length, device="cuda")
        dropped = torch.arange(0, first, device="cuda")
        for layer in range(len(whole._ctx_kv_buf)):
            for kv in (0, 1):
                ref = rows_of(whole, request, read, layer, kv)
                assert torch.isfinite(ref.float()).all()
                got = rows_of(kept, request, read, layer, kv)
                assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
                    "the generation window's rows differ from the whole write's",
                    request,
                    layer,
                    kv,
                )
                assert torch.equal(
                    rows_of(kept, request, written, layer, kv).view(torch.int16),
                    rows_of(whole, request, written, layer, kv).view(torch.int16),
                ), (request, layer, kv)
                if first:
                    assert torch.isnan(rows_of(kept, request, dropped, layer, kv).float()).all()


@pytest.mark.parametrize(
    "floor, carried", [(0, 20 + 0), (None, 40 + 24)], ids=["no-lead", "served-floor"]
)
@torch.no_grad()
def test_window_breaks_read_the_replayed_steps_plan(floor, carried, monkeypatch):
    """A middle stage's window step captured as a breakable prefill graph for one step's plan
    and replayed for another sends what the other step sends eagerly; a replay without a plan
    raises.

    Without lead rows (``floor`` 0) the two steps carry different rows; at the served row floor
    each step is under the floor, so it carries every row of its chunks."""
    from importlib import import_module

    if floor is not None:
        window_mod = import_module(f"{_PP}.kernels.dflash_context.window")
        monkeypatch.setattr(window_mod, "MIN_GEMM_ROWS", floor)
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
        enable_breakable_cuda_graph,
    )

    stages, route_mod, carry_mod, _, _ = _modules()
    hidden, rows, stage, vocab = 256, 64, 1, 97
    route = route_mod.capture_route(
        TARGET_LAYERS, stages.stage_ranges(NUM_LAYERS, 4, SPLIT_15_16_16_15), hidden
    )
    core = _Stage(stage, hidden, vocab).cuda()
    sent = []

    def raw_recv(tensors):
        for index, tensor in enumerate(tensors):
            values = torch.arange(tensor.numel(), device=tensor.device) % (7 + index)
            tensor.copy_(values.reshape(tensor.shape).to(tensor.dtype))

    def raw_send(tensors):
        sent.append([tensor.clone() for tensor in tensors])

    carry = carry_mod.CaptureCarry(route, stage, raw_send=raw_send, raw_recv=raw_recv)
    meta = _metadata(TARGET_LAYERS, hidden, rows)
    input_ids = (torch.arange(rows, device="cuda") * 7) % vocab

    def plan(lengths, starts, keep_from):
        meta.draft_context_keep_from = keep_from
        meta.host_position_ids = torch.tensor(
            [p for s, n in zip(starts, lengths) for p in range(s, s + n)], dtype=torch.int32
        )
        return carry.plan_step(_window_attn(lengths), meta)

    step_a = ([64], [0], [32])
    step_b = ([40, 24], [100, 0], [120, 30])
    assert plan(*step_b).count == carried
    carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
    torch.cuda.synchronize()
    (eager_b,) = sent
    sent.clear()

    plan(*step_a)
    graph, stream = BreakableCUDAGraph(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with (
            enable_breakable_cuda_graph(),
            BreakableCUDAGraphCapture(graph, pool=torch.cuda.graph_pool_handle(), stream=stream),
        ):
            carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert carry._captured_window is True
    sent.clear()

    plan(*step_b)
    with enable_breakable_cuda_graph():
        graph.replay()
    torch.cuda.synchronize()
    (replay_b,) = sent
    assert [t.shape for t in replay_b] == [t.shape for t in eager_b]
    for got, ref in zip(replay_b, eager_b):
        assert got.dtype == ref.dtype and torch.equal(got, ref)

    plan(*step_b[:2], None)
    with pytest.raises(RuntimeError, match="without one"):
        with enable_breakable_cuda_graph():
            graph.replay()
    graph.reset()


def _pp4_window_main(buffered: bool) -> None:
    """One rank of ``mpirun -n 4``: window steps of the 15/16/16/15 split over the PP transport
    (the carry's own transport and eager breaks); per-layer checksums of the sending stages'
    dumps at the kept rows against the last stage's kept-row captures."""
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.communicator import init_pp_comm
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=4, rank=rank, gpus_per_node=4, pp_size=4, pp_partition=SPLIT_15_16_16_15
    )
    init_pp_comm(mapping, buffered=buffered)
    stages, route_mod, carry_mod, _, _ = _modules()
    hidden, max_rows, vocab, keep = 8192, 8192, 97, 2592
    route = route_mod.capture_route(
        TARGET_LAYERS, stages.stage_ranges(NUM_LAYERS, 4, SPLIT_15_16_16_15), hidden
    )
    core = _Stage(rank, hidden, vocab).cuda()
    carry = carry_mod.CaptureCarry(route, rank)
    schedules = (
        ([5000, 3192], [0, 1000], [5000 - keep, 1000 + 3192 - keep]),  # both keep a tail
        ([8192], [40000], [100000]),  # a chunk before its window: nothing kept
        ([1234], [0], [0]),  # a short prompt: everything kept
        # A chunk keeping fewer rows than the row floor carries lead rows ahead of them.
        ([4000, 300, 3000], [0, 0, 0], [4000 - 100, 0, 100000]),
        # A short whole prompt beside a chunk that keeps nothing: that chunk's last rows pad
        # the step up to the row floor.
        ([300, 7892], [0, 0], [0, 100000]),
    )
    for lengths, starts, keep_from in schedules:
        rows = sum(lengths)
        meta = _window_metadata(TARGET_LAYERS, hidden, max_rows, lengths, starts, keep_from)
        plan = carry.plan_step(_window_attn(lengths), meta)
        core.dumps.clear()
        input_ids = (torch.arange(rows, device="cuda") * 7) % vocab
        with torch.no_grad():
            carry.forward_stage(core, None, input_ids, None, None, spec_metadata=meta)
        torch.cuda.synchronize()
        index = plan.index("cuda")

        def checksum(t: torch.Tensor) -> int:
            return int(t.contiguous().view(torch.int16).to(torch.int64).sum().item())

        own = {
            layer: checksum(core.dumps[layer].to(DTYPE).index_select(0, index))
            for layer in TARGET_LAYERS
            if layer in core.dumps
        }
        assembled = None
        if rank == 3:
            window = carry.window_capture(meta.captured_hidden_states)
            assert window is not None and window[0] == plan
            captures = window[1]
            if plan.count == 0:
                assert captures is None
                captures = torch.zeros(0, hidden * len(TARGET_LAYERS), dtype=DTYPE, device="cuda")
            assembled = {
                layer: checksum(captures[:, i * hidden : (i + 1) * hidden])
                for i, layer in enumerate(TARGET_LAYERS)
            }
        everyone = comm.gather(own, root=3)
        if rank == 3:
            sent = {layer: value for part in everyone for layer, value in part.items()}
            assert sent == assembled, (lengths, buffered, sent, assembled)
    comm.Barrier()
    if rank == 0:
        print(f"PP4 window capture integrity PASS (buffered={buffered})")


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
@pytest.mark.skipif(shutil.which("mpirun") is None, reason="needs mpirun")
@pytest.mark.parametrize("buffered", [False, True], ids=["nccl", "buffered_ring"])
def test_window_capture_integrity_pp4(buffered):
    env = dict(os.environ)
    result = subprocess.run(
        [
            "mpirun",
            "--allow-run-as-root",
            "-n",
            "4",
            sys.executable,
            __file__,
            "pp4w",
            "1" if buffered else "0",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "PASS" in result.stdout
    assert "such messages go through NCCL" not in result.stdout + result.stderr


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "pp4":
        _pp4_main(len(sys.argv) > 2 and sys.argv[2] == "1")
    elif len(sys.argv) >= 2 and sys.argv[1] == "pp4w":
        _pp4_window_main(len(sys.argv) > 2 and sys.argv[2] == "1")
