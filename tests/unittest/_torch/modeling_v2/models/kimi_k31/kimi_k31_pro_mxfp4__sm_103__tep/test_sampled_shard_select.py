# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep target's sampled DFlash2 draft from the vocab shard equals the gathered-logits path, on a TP group.

Four ranks of one GB300 tray, the engine ``AllReduce`` module of the group (MNNVL: both gathers are the Lamport
kernels). On every rank, the gathered path (``DFlashWorker._draft_block_logits``: the Lamport vocab gather of the
LM head shard, then ``dflash2_sample_draft``: the full-vocab top-32, the sampled walk, the proposal store) and the
shard path (``DFlashWorker._k31_sampled_selector_tokens``: the shard's Lamport-mode top-32 pairs, their Lamport
all-gather, the merge, the same walk) must return the same draft tokens and leave the same proposals (candidate ids,
q, tokens) bit for bit: drafter-like logits, ties, and rows whose 32nd largest logit is a mix of signed zeros and
Lamport sentinel words; sampled, greedy and edge temperatures; eager and replayed from a CUDA graph.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys
import types

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the selection kernels are CuTe DSL kernels", allow_module_level=True)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
VOCAB = 163840
RANK = 256
HIDDEN = 7168
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 3)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


def _full_logits(kind: str, rows: int, seed: int) -> torch.Tensor:
    """``[rows, VOCAB]`` BF16 block logits, identical on every rank."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "ties":
        x = 0.25 * torch.randint(-3, 9, (rows, VOCAB), generator=g, device="cuda").float()
        return x.bfloat16()
    if kind == "randn":
        return (2.0 + 3.0 * torch.randn(rows, VOCAB, generator=g, device="cuda")).bfloat16()
    # "zeros": 12 positive logits, 200 zeros (sentinel words, stray +0.0 / -0.0), the rest negative.
    positives, zeros = 12, 200
    count = positives + zeros
    x = (-1.0 - torch.rand(rows, VOCAB, generator=g, device="cuda")).bfloat16()
    bits = x.view(torch.int16)
    even = torch.rand(rows, VOCAB // 2, generator=g, device="cuda").argsort(dim=-1)[:, :count] * 2
    j = torch.arange(count, device="cuda").expand(rows, count)
    x.scatter_(1, even[:, :positives] + (j[:, :positives] & 1), (1.0 + j[:, :positives]).bfloat16())
    ze, zo, kind_ = even[:, positives:], j[:, positives:] & 1, j[:, positives:] % 4
    neg = torch.full_like(ze, -(2**15), dtype=torch.int16)
    pos = torch.zeros_like(ze, dtype=torch.int16)
    lo, hi = bits.gather(1, ze), bits.gather(1, ze + 1)
    lo = torch.where(kind_ == 0, pos, torch.where(kind_ == 2, neg, lo))
    lo = torch.where((kind_ == 3) & (zo == 0), pos, lo)
    hi = torch.where((kind_ == 0) | (kind_ == 1), neg, hi)
    hi = torch.where((kind_ == 3) & (zo == 1), pos, hi)
    bits.scatter_(1, ze, lo)
    bits.scatter_(1, ze + 1, hi)
    return x


def _temperatures(kind: str, n: int, g) -> torch.Tensor:
    if kind == "hot":
        return 0.3 + 1.5 * torch.rand(n, generator=g, device="cuda")
    temps = torch.rand(n, generator=g, device="cuda")
    temps[::3] = 0.0  # greedy rows in a sampling batch
    temps[1::5] = 1e-5
    return temps


def _rank_case(case: dict) -> dict:
    import tensorrt_llm
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping

    modeling = __import__(
        f"{_PKG}.kimi_k31_pro_mxfp4__sm_103__tep.modeling", fromlist=["K31DFlashWorker"]
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
        dflash2_sample_draft,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    reg = vars(sys.modules.setdefault("_k31_sampled_shard", types.ModuleType("_k31_sampled_shard")))
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    if "ar" not in reg:
        reg["ar"] = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16
        )
        g = torch.Generator(device="cuda").manual_seed(7)
        selector = types.SimpleNamespace(
            top_k=32,
            predecessor_codebook=(
                0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")
            ).bfloat16(),
            successor_codebook=(
                0.05 * torch.randn(VOCAB, RANK, generator=g, device="cuda")
            ).bfloat16(),
            hidden_projection=torch.nn.Linear(
                HIDDEN, RANK, bias=False, device="cuda", dtype=torch.bfloat16
            ),
        )
        with torch.no_grad():
            selector.hidden_projection.weight.normal_(0.0, 0.02, generator=g)
        reg["selector"] = selector
    ar, selector = reg["ar"], reg["selector"]
    n, K = case["n"], case["K"]
    rows = n * K
    shard = VOCAB // WORLD
    full = _full_logits(case["logits"], rows, case["seed"])
    local = full[:, rank * shard : (rank + 1) * shard].contiguous()
    g = torch.Generator(device="cuda").manual_seed(case["seed"] + 1)
    hidden = (0.5 * torch.randn(rows, HIDDEN, generator=g, device="cuda")).bfloat16()
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda", dtype=torch.int32)
    slots = 2 * n + 1

    lm_head = types.SimpleNamespace(
        tp_mode=TensorParallelMode.COLUMN,
        gather_output=True,
        padding_size=0,
        gather_output_sizes=None,
        tp_size=WORLD,
        tp_rank=rank,
        mapping=mapping,
        bias=None,
        # The LM head shard of these block logits.
        apply_linear=lambda x, bias: local.clone(),
    )
    draft_model = types.SimpleNamespace(
        lm_head=lm_head, candidate_selector=selector, has_candidate_selector=True, lilicorr=None
    )
    spec_metadata = types.SimpleNamespace(
        use_rejection_sampling=True,
        is_all_greedy_sample=False,
        wants_advanced_draft_sampling=True,
        runtime_draft_len=K,
        max_draft_len=K,
        draft_probs=None,
        num_seq_slots=slots - 1,
        max_num_requests=slots - 1,
        request_seeds=torch.randint(0, 2**62, (n,), generator=g, device="cuda"),
        request_offsets=torch.randint(0, 2**40, (n,), generator=g, device="cuda"),
        request_temperatures=_temperatures(case["temps"], n, g),
        batch_slot_ids=torch.randperm(slots, generator=g, device="cuda")[:n],
        rng_draft_slot=lambda step: 3 + step,
    )

    def worker():
        w = modeling.K31DFlashWorker.__new__(modeling.K31DFlashWorker)
        torch.nn.Module.__init__(w)
        w._pair_allreduce = ar
        w._d2t = None
        w.mapping = mapping
        w.spec_config = types.SimpleNamespace(max_draft_len=K)
        w._draft_model_ref = lambda: draft_model
        return w

    def gathered(w):
        gen_logits = modeling.K31DFlashWorker._draft_block_logits(w, draft_model, hidden, None)
        return dflash2_sample_draft(
            w,
            draft_model,
            gen_logits.reshape(n, K, -1),
            hidden.view(n, K, -1),
            anchor,
            spec_metadata,
            0,
            n,
        )

    def shard_path(w):
        out = w._k31_sampled_selector_tokens(
            draft_model, hidden, {"first_prev_tokens": anchor}, spec_metadata, 0, n, n, K
        )
        assert out is not None and out[1] == VOCAB
        return out[0]

    with torch.inference_mode():
        wg, ws = worker(), worker()
        tok_g = gathered(wg)
        tok_s = shard_path(ws)
        ar(
            torch.randn(8, HIDDEN, device="cuda").bfloat16()
        )  # an engine all-reduce on the same workspace
        wr = worker()
        shard_path(wr)  # warm-up (allocates the store) before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tok_r = shard_path(wr)
        wr._dflash2_proposals.candidate_ids.zero_()
        wr._dflash2_proposals.q.zero_()
        wr._dflash2_proposals.tokens.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()

    def same(a, b):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        return bool(torch.equal(a, b))

    stores = []
    for w in (wg, ws, wr):
        st = w._dflash2_proposals
        stores.append((st.candidate_ids, st.q, st.tokens))
    return dict(
        rank=rank,
        tokens=same(tok_g, tok_s),
        replay=same(tok_g, tok_r),
        store=all(same(a, b) for a, b in zip(stores[0], stores[1])),
        store_replay=all(same(a, b) for a, b in zip(stores[0], stores[2])),
        drawn=int((stores[0][2] >= 0).sum()),
    )


_CASES = [
    dict(n=7, K=4, logits="randn", temps="mixed", seed=11),  # the sampled TEP8 B7 block
    dict(n=16, K=4, logits="randn", temps="hot", seed=12),  # the sampled TEP16 B16 block
    dict(n=8, K=7, logits="ties", temps="mixed", seed=13),
    dict(n=4, K=4, logits="zeros", temps="hot", seed=14),
    dict(n=6, K=5, logits="zeros", temps="mixed", seed=15),
]


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "case", _CASES, ids=[f"{c['logits']}-{c['n']}x{c['K']}-{c['temps']}" for c in _CASES]
)
def test_sampled_shard_select_matches_the_gathered_path(case, mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(_rank_case, [case] * WORLD))
    for r in results:
        assert r["tokens"] and r["store"] and r["replay"] and r["store_replay"], r
        assert r["drawn"] == case["n"] * case["K"], r
