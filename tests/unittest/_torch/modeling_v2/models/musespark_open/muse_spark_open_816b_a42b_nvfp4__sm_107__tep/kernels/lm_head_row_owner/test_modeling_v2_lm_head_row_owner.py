# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests of ``lm_head_row_owner``: four MPI ranks on one sm107 node, one GPU each.

Run on every rank alike (the tests run collectives in order)::

    srun -n 4 --mpi=pmix python -m pytest -p no:cacheprovider -q <this file>

Every comparison is against the built-in path on the same inputs: ``Linear.forward`` (local
GEMM + NCCL all-gather + ``cat``), the materialize soft cap, then
``SpecWorkerBase._sample_and_accept_draft_tokens_rejection``. ``ROW_OWNER_LM_HEAD`` may name a
checkpoint directory whose ``lm_head.weight`` shard each rank loads (read-only); otherwise the
weight is random.

With ``python <this file> hang`` (four ranks) rank 3 skips its GEMM and the other ranks' owner
polls must trap within a short timeout instead of waiting (the job step exits non-zero).
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

import pytest
import torch

T = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
HIDDEN = 8192
VOCAB = 202048
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0
K = 7
WIDTH = K + 1


def _mpi():
    try:
        from tensorrt_llm._utils import mpi_rank, mpi_world_size

        return mpi_rank(), mpi_world_size()
    except Exception:  # noqa: BLE001 - no MPI: the tests skip
        return 0, 1


RANK, WORLD = _mpi()


def _usable() -> bool:
    return (
        WORLD == 4
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 1
        and torch.cuda.get_device_capability() == (10, 7)
    )


pytestmark = pytest.mark.skipif(not _usable(), reason="needs four MPI ranks on sm107 GPUs")


def _owner_module():
    import importlib

    return importlib.import_module(f"{T}.kernels.lm_head_row_owner.owner")


class _Env:
    """The ranks' shared fixtures: the LM head, its owner head, and a fake worker."""

    def __init__(self) -> None:
        from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
        from tensorrt_llm.mapping import Mapping

        torch.cuda.set_device(RANK % torch.cuda.device_count())
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.mapping = Mapping(world_size=4, rank=RANK, tp_size=4, gpus_per_node=4)
        self.owner_mod = _owner_module()
        lm_head = Linear(
            HIDDEN,
            VOCAB,
            bias=False,
            dtype=torch.bfloat16,
            mapping=self.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        ).cuda()
        shard = VOCAB // 4
        ckpt = os.environ.get("ROW_OWNER_LM_HEAD")
        with torch.no_grad():
            if ckpt:
                from safetensors import safe_open

                index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
                name = index["weight_map"]["lm_head.weight"]
                with safe_open(os.path.join(ckpt, name), framework="pt", device="cpu") as f:
                    rows = f.get_slice("lm_head.weight")[RANK * shard : (RANK + 1) * shard]
                lm_head.weight.copy_(rows.to(self.device))
            else:
                g = torch.Generator(device=self.device).manual_seed(1000 + RANK)
                lm_head.weight.copy_(
                    (torch.randn(lm_head.weight.shape, generator=g, device=self.device) * 0.02).to(
                        torch.bfloat16
                    )
                )
        self.lm_head = lm_head
        # The served install runs with grad mode on and a weight that requires grad.
        lm_head.weight.requires_grad_(True)
        self.owner = self.owner_mod.RowOwnerHead(self.mapping, lm_head, MULTIPLIER, SOFT_CAP)
        lm_head.weight.requires_grad_(False)
        self.gen = torch.Generator(device=self.device).manual_seed(77)

    # -------------------------------------------------------------- inputs
    def inputs(self, batch: int, dense: bool = False, padding: int = 0, seed: int = 0):
        """Rank-identical step inputs: hidden rows, draft tokens and rows, spec metadata."""
        from tensorrt_llm._torch.speculative.compact_draft_probs import (
            CompactDraftProbs,
            DeferredDraftProbs,
        )
        from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

        g = torch.Generator(device=self.device).manual_seed(seed * 1009 + batch)
        rows = batch * WIDTH
        x = (torch.randn((rows, HIDDEN), generator=g, device=self.device) * 2.0).to(torch.bfloat16)
        draft_tokens = torch.randint(0, VOCAB, (batch, K), generator=g, device=self.device)
        slots = 128
        dummy = slots - 1
        slot_ids = torch.randperm(slots - 1, generator=g, device=self.device)[:batch]
        if padding:
            slot_ids[-padding:] = dummy
        store = CompactDraftProbs(slots, K, VOCAB, self.device)
        lanes = store.lanes
        # Distinct ids within a record (a repeated id would make the densify scatter's winner
        # unspecified on both paths): lane l holds base + l * stride, lane 0 the draft token.
        base = torch.randint(0, VOCAB, (slots, K, 1), generator=g, device=self.device)
        base[slot_ids, :, 0] = draft_tokens
        stride = VOCAB // lanes
        ids = (base + torch.arange(lanes, device=self.device) * stride) % VOCAB
        vals = torch.rand((slots, K, lanes), generator=g, device=self.device) * 0.02
        # Half the requests' drafts carry a tiny q (accepted), the others 0.6 (mostly rejected).
        tiny = torch.rand((slots,), generator=g, device=self.device) < 0.5
        vals[:, :, 0] = torch.where(tiny[:, None], 1e-9, 0.6)
        store.ids.copy_(ids)
        store.vals.copy_(vals)
        draft = DeferredDraftProbs(store, slot_ids, K, VOCAB)
        if dense:
            draft = draft.materialize()
        temps = torch.full((rows,), 1.0, device=self.device)
        temps[: rows // 2] = 0.7
        md = types.SimpleNamespace(
            is_ragged_verify=False,
            total_verify_tokens=None,
            verify_lens=None,
            qo_indptr=None,
            temperatures=temps,
            top_ks=torch.zeros((rows,), dtype=torch.int32, device=self.device),
            top_ps=torch.ones((rows,), device=self.device),
            min_ps=torch.zeros((rows,), device=self.device),
            advanced_sampling_mode=AdvancedSamplingMode.FULL,
            batch_slot_ids=slot_ids,
            dummy_slot_row=dummy,
            request_seeds=torch.randint(
                0, 1 << 40, (slots,), generator=g, device=self.device, dtype=torch.int64
            ),
            request_offsets=torch.randint(
                0, 1 << 20, (slots,), generator=g, device=self.device, dtype=torch.int64
            ),
            rng_accept_slot=3,
            full_draft_probs=None,
            is_cuda_graph=False,
        )
        return x, draft_tokens, draft, md

    def worker(self, force: float = 0.0):
        from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

        cls = type(
            "FakeWorker",
            (),
            {
                "_compute_probs_from_logits": SpecWorkerBase.__dict__["_compute_probs_from_logits"],
                "_zero_padding_rows": SpecWorkerBase.__dict__["_zero_padding_rows"],
                "_rng_state_per_request": SpecWorkerBase._rng_state_per_request,
                "_apply_force_accepted_tokens": SpecWorkerBase._apply_force_accepted_tokens,
                "_sample_and_accept_draft_tokens_rejection": (
                    SpecWorkerBase._sample_and_accept_draft_tokens_rejection
                ),
            },
        )
        w = cls()
        w._d2t = None
        w.force_num_accepted_tokens = force
        return w

    # -------------------------------------------------------------- the two paths
    def builtin(self, x, draft_tokens, draft, md, force: float = 0.0):
        from tensorrt_llm._torch.modules.logit_soft_cap import soft_capped_logits
        from tensorrt_llm._torch.speculative.compact_draft_probs import mark_private_verify_logits

        gathered = self.lm_head(x)
        capped = mark_private_verify_logits(soft_capped_logits(gathered, MULTIPLIER, SOFT_CAP))
        accepted, counts = self.worker(force)._sample_and_accept_draft_tokens_rejection(
            capped, draft_tokens, draft, 0, draft_tokens.shape[0], md
        )
        return gathered, accepted, counts

    def owned(self, x, draft_tokens, draft, md, force: float = 0.0):
        placeholder = self.owner.produce(self.lm_head, x)
        head = types.SimpleNamespace(release_row_owner=lambda: None)
        accepted, counts = self.owner.verify(
            self.worker(force), head, placeholder, draft_tokens, draft, draft_tokens.shape[0], md
        )
        return accepted, counts

    def pages_clean(self) -> bool:
        """Whether every word of this rank's pages is the sentinel (after a device sync)."""
        torch.cuda.synchronize()
        own = self.owner.pages.owned(self.owner_mod.SLOTS, VOCAB)
        return bool((own == -1).all().item())

    def barrier(self) -> None:
        from tensorrt_llm._torch.distributed import allgather

        torch.cuda.synchronize()
        allgather(torch.zeros(1, dtype=torch.int32, device=self.device), self.mapping, dim=0)
        torch.cuda.synchronize()


_ENV = None


@pytest.fixture(scope="module")
def env() -> _Env:
    global _ENV
    if _ENV is None:
        _ENV = _Env()
    return _ENV


def _all_ranks(flag: bool, env: _Env) -> bool:
    from tensorrt_llm._torch.distributed.communicator import Distributed

    return all(Distributed.get(env.mapping).tp_allgather(bool(flag)))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def test_owner_slices_cover_the_batch():
    owner = _owner_module()
    for batch in range(owner.MIN_REQUESTS, owner.MAX_REQUESTS + 1):
        slices = [owner.owner_slice(batch, r) for r in range(4)]
        q = slices[0][0]
        assert q >= owner.MIN_SLICE and q * WIDTH >= 136
        assert sum(n for _, _, n in slices) == batch
        assert all(first == r * q for r, (_, first, _) in enumerate(slices))
        assert all(n >= 1 for _, _, n in slices)


def test_census_covers_the_band(env):
    owner = env.owner_mod
    assert env.owner.batches == frozenset(range(owner.MIN_REQUESTS, owner.MAX_REQUESTS + 1))
    assert env.pages_clean()
    env.barrier()


def test_owner_cap_is_the_built_in_soft_cap_and_resets(env):
    """Rank-local: words written into this rank's own rows, every special BF16 value."""
    from tensorrt_llm._torch.modules.logit_soft_cap import soft_cap_constants, soft_capped_logits

    owner = env.owner_mod
    q, live = 20, 150
    g = torch.Generator(device=env.device).manual_seed(5 + RANK)
    words = (torch.randn((live, VOCAB), generator=g, device=env.device) * 30).to(torch.bfloat16)
    special = torch.tensor(
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FFF, 0x7FC1, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x0080],
        dtype=torch.int32,
    ).to(torch.int16)
    flat = words.view(torch.int16).view(-1)
    flat[: special.numel()] = special.to(env.device)
    flat[-special.numel() :] = special.to(env.device)
    pages = env.owner.pages.owned(q, VOCAB)
    pages[:live].copy_(words.view(torch.int16))
    out = torch.full((q * WIDTH, VOCAB), 7.0, device=env.device)
    owner._ops().owner_cap(
        pages,
        live,
        out,
        VOCAB,
        *soft_cap_constants(MULTIPLIER, SOFT_CAP),
        env.owner.status,
        10_000,
        True,
    )
    reference = soft_capped_logits(words, MULTIPLIER, SOFT_CAP)
    assert torch.equal(_bits(out[:live]), _bits(reference))
    assert bool((out[live:] == 0).all())
    assert env.pages_clean()
    env.barrier()


def test_chain_row_base_is_the_stock_kernel(env):
    """Rank-local: the whole batch at row base 0, and slices at their row base, vs flashinfer's kernel."""
    from tensorrt_llm._torch.speculative.interface import rejection_sampling_one_model

    owner = env.owner_mod
    module = owner.chain_module()
    g = torch.Generator(device=env.device).manual_seed(11)
    batch = 80
    target = torch.softmax(
        torch.randn((batch, WIDTH, VOCAB), generator=g, device=env.device) * 3, dim=-1
    )
    draft = torch.softmax(torch.randn((batch, K, VOCAB), generator=g, device=env.device) * 3, -1)
    tokens = torch.multinomial(draft.view(-1, VOCAB), 1, generator=g).view(batch, K).int()
    seed = torch.full((batch,), 12345, dtype=torch.int64, device=env.device)
    offset = torch.full((batch,), 64, dtype=torch.int64, device=env.device)
    stock_tokens, stock_counts = rejection_sampling_one_model(
        draft, tokens, target, True, seed, offset
    )

    def run(first, count):
        out = torch.empty((count, WIDTH), dtype=torch.int32, device=env.device)
        acc = torch.zeros(count, dtype=torch.int32, device=env.device)
        emitted = torch.zeros(count, dtype=torch.int32, device=env.device)
        module.chain_speculative_sampling_row_base(
            draft[first : first + count].contiguous(),
            tokens[first : first + count].contiguous(),
            target[first : first + count].contiguous(),
            out,
            acc,
            emitted,
            True,
            seed,
            offset,
            first,
        )
        return out, emitted + 1

    whole = run(0, batch)
    assert torch.equal(whole[0], stock_tokens) and torch.equal(whole[1], stock_counts)
    for first, count in ((0, 20), (20, 20), (40, 20), (60, 20), (17, 9)):
        part = run(first, count)
        assert torch.equal(part[0], stock_tokens[first : first + count])
        assert torch.equal(part[1], stock_counts[first : first + count])
    env.barrier()


@pytest.mark.parametrize(
    "batch,dense,padding,force",
    [
        (65, False, 0, 0.0),
        (66, False, 3, 0.0),
        (68, True, 0, 0.0),
        (73, False, 5, 0.0),
        (77, False, 0, 3.0),
        (80, False, 8, 0.0),
        (84, True, 0, 0.0),
        (93, False, 0, 0.0),
        (95, False, 2, 3.0),
        (96, False, 0, 0.0),
    ],
)
def test_owner_verify_is_the_built_in_verify(env, batch, dense, padding, force):
    x, draft_tokens, draft, md = env.inputs(batch, dense=dense, padding=padding)
    _, accepted, counts = env.builtin(x, draft_tokens, draft, md, force)
    got_accepted, got_counts = env.owned(x, draft_tokens, draft, md, force)
    same = torch.equal(got_accepted, accepted) and torch.equal(got_counts, counts)
    clean = env.pages_clean()
    assert _all_ranks(same and clean, env), (batch, same, clean)
    # Both outcomes of the chain are exercised.
    if force == 0.0:
        assert int((counts == WIDTH).sum()) > 0 and int((counts == 1).sum()) > 0


@pytest.mark.parametrize("batch", [65, 79, 96])
def test_gather_is_the_built_in_row_and_resets(env, batch):
    x, draft_tokens, draft, md = env.inputs(batch, seed=3)
    gathered, _, _ = env.builtin(x, draft_tokens, draft, md)
    placeholder = env.owner.produce(env.lm_head, x)
    env.owner.gather_into(placeholder)
    same = torch.equal(_bits(placeholder), _bits(gathered))
    clean = env.pages_clean()
    assert _all_ranks(same and clean, env), (batch, same, clean)


@pytest.mark.parametrize("batch", [65, 88])
def test_drain_resets_the_pages(env, batch):
    x, _, _, _ = env.inputs(batch, seed=4)
    env.owner.produce(env.lm_head, x)
    env.owner.drain(batch * WIDTH)
    clean = env.pages_clean()
    env.barrier()
    assert _all_ranks(clean, env)


def test_target_head_consumers(env):
    """``TargetHead``'s materialize, drain and verify routing with a row-owned row pending."""
    import importlib

    from tensorrt_llm._torch.modules.logit_soft_cap import soft_capped_logits

    target_head = importlib.import_module(f"{T}.kernels.split_sampler.target_head")
    head = object.__new__(target_head.TargetHead)
    head.__dict__.update(
        pending=None,
        pending_inbox=False,
        pending_local=None,
        pending_s2=False,
        rs_pending=None,
        row_owner=env.owner,
        pending_row_owner=False,
        multiplier=MULTIPLIER,
        soft_cap=SOFT_CAP,
    )
    batch = 72
    x, draft_tokens, draft, md = env.inputs(batch, seed=9)
    gathered, _, _ = env.builtin(x, draft_tokens, draft, md)
    # materialize: the gathered row from the pages, then the built-in soft cap.
    head.pending = env.owner.produce(env.lm_head, x)
    head.pending_row_owner = True
    capped = head.materialize(head.pending)
    ok = torch.equal(_bits(capped), _bits(soft_capped_logits(gathered, MULTIPLIER, SOFT_CAP)))
    ok = ok and head.pending is None and not head.pending_row_owner and env.pages_clean()
    # drain: nothing consumed the row.
    head.pending = env.owner.produce(env.lm_head, x)
    head.pending_row_owner = True
    head.drain()
    ok = ok and head.pending is None and not head.pending_row_owner and env.pages_clean()
    # routing: the verify takes the row unmaterialized only on a rejection step without penalties.
    w = types.SimpleNamespace(_can_use_rejection_sampling=lambda m: m.use_rs)
    head.pending_row_owner = True
    ok = ok and head.row_owner_verifies(w, types.SimpleNamespace(use_rs=True), 0, batch)
    ok = ok and not head.row_owner_verifies(w, types.SimpleNamespace(use_rs=False), 0, batch)
    ok = ok and not head.row_owner_verifies(
        w, types.SimpleNamespace(use_rs=True, enable_penalty=True), 0, batch
    )
    head.pending_row_owner = False
    ok = ok and not head.row_owner_verifies(w, types.SimpleNamespace(use_rs=True), 0, batch)
    env.barrier()
    assert _all_ranks(ok, env)


def test_graph_replay_with_changing_inputs(env):
    """Captured produce + verify for two batch sizes, replayed alternately with new inputs."""
    graphs = {}
    static_inputs = {}
    for batch in (80, 73):
        x, draft_tokens, draft, md = env.inputs(batch, seed=20)
        # The capture reads these tensors; replays copy new inputs into them.
        static_inputs[batch] = (x, draft_tokens, draft, md)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            env.owned(x, draft_tokens, draft, md)  # warm-up (eager)
        torch.cuda.current_stream().wait_stream(stream)
        env.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = env.owned(x, draft_tokens, draft, md)
        graphs[batch] = (graph, outputs)
        env.barrier()
        assert env.pages_clean()
    ok = True
    for step, batch in enumerate((80, 73, 73, 80, 73, 80, 80, 73)):
        graph, (accepted, counts) = graphs[batch]
        x, draft_tokens, draft, md = static_inputs[batch]
        nx, ntok, ndraft, nmd = env.inputs(batch, seed=100 + step, padding=step % 3)
        x.copy_(nx)
        draft_tokens.copy_(ntok)
        draft.store.ids.copy_(ndraft.store.ids)
        draft.store.vals.copy_(ndraft.store.vals)
        draft.slot_ids.copy_(ndraft.slot_ids)
        for name in ("temperatures", "request_seeds", "request_offsets"):
            getattr(md, name).copy_(getattr(nmd, name))
        env.barrier()
        graph.replay()
        torch.cuda.synchronize()
        _, ref_accepted, ref_counts = env.builtin(x, draft_tokens, draft, md)
        ok = ok and torch.equal(accepted, ref_accepted) and torch.equal(counts, ref_counts)
        ok = ok and env.pages_clean()
        # An eager step of another size between replays.
        if step % 2:
            other = 65 + (step * 7) % 32
            ex, et, ed, em = env.inputs(other, seed=200 + step)
            _, ra, rc = env.builtin(ex, et, ed, em)
            ga, gc = env.owned(ex, et, ed, em)
            ok = ok and torch.equal(ga, ra) and torch.equal(gc, rc) and env.pages_clean()
    env.barrier()
    assert _all_ranks(ok, env)


def test_a_sequence_of_batch_sizes_and_consumers(env):
    owner = env.owner_mod
    g = torch.Generator().manual_seed(4242)
    ok = True
    log = []
    for step in range(30):
        batch = int(torch.randint(owner.MIN_REQUESTS, owner.MAX_REQUESTS + 1, (1,), generator=g))
        kind = ("verify", "gather", "drain")[int(torch.randint(0, 3, (1,), generator=g))]
        x, draft_tokens, draft, md = env.inputs(batch, seed=300 + step, padding=step % 4)
        gathered, accepted, counts = env.builtin(x, draft_tokens, draft, md)
        if kind == "verify":
            ga, gc = env.owned(x, draft_tokens, draft, md)
            same = torch.equal(ga, accepted) and torch.equal(gc, counts)
        elif kind == "gather":
            placeholder = env.owner.produce(env.lm_head, x)
            env.owner.gather_into(placeholder)
            same = torch.equal(_bits(placeholder), _bits(gathered))
        else:
            env.owner.produce(env.lm_head, x)
            env.owner.drain(batch * WIDTH)
            same = True
        clean = env.pages_clean()
        log.append((batch, kind, same, clean))
        ok = ok and same and clean
    env.barrier()
    assert _all_ranks(ok, env), log


# ------------------------------------------------------------------ hang check (not pytest)
def _hang() -> int:
    """Rank 3 skips its GEMM: ranks 0-2 must trap within the lowered timeout."""
    owner = _owner_module()
    owner.POLL_TIMEOUT_MS = 5000
    env = _Env()
    x, _, _, _ = env.inputs(80, seed=1)
    env.barrier()
    start = time.monotonic()
    if RANK != 3:
        env.owner.produce(env.lm_head, x)
        env.owner.drain(80 * WIDTH)
        try:
            torch.cuda.synchronize()
        except RuntimeError as exc:
            print(
                f"[hang-check] rank {RANK}: trapped after {time.monotonic() - start:.1f} s: {exc}",
                flush=True,
            )
            return 3
        print(f"[hang-check] rank {RANK}: the poll returned without rank 3's rows", flush=True)
        return 1
    time.sleep(60)
    print("[hang-check] rank 3: still alive after 60 s (peers should have trapped)", flush=True)
    return 0


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "hang":
    os._exit(_hang())
