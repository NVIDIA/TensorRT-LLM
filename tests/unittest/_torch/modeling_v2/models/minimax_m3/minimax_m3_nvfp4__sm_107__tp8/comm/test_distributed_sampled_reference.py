# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real TP2/4/8 sampled verification against a full-vocabulary CPU oracle."""

import gc
import hashlib
import importlib
import json
import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("distributed sampling requires CUDA", allow_module_level=True)
torch.cuda.set_device(
    int(os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
)

# Select the local GPU before TensorRT-LLM initializes its CUDA libraries.
from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

_WORLD = mpi_world_size()
_COMM = mpi_comm()
if _WORLD not in (2, 4, 8):
    if os.environ.get("M3_REQUIRE_DISTRIBUTED_SAMPLE_REFERENCE") == "1":
        raise RuntimeError(f"expected TP2/4/8 MPI ranks, got {_WORLD}")
    pytest.skip("run with two, four or eight MPI ranks", allow_module_level=True)
_CAPABILITIES = _COMM.allgather(torch.cuda.get_device_capability())
if any(capability != (10, 7) for capability in _CAPABILITIES):
    if os.environ.get("M3_REQUIRE_DISTRIBUTED_SAMPLE_REFERENCE") == "1":
        raise RuntimeError(f"expected SM107 on every rank: {_CAPABILITIES}")
    pytest.skip("SM107 required on every rank", allow_module_level=True)

# The hardware-specific module must be imported after collective eligibility.
_SA = importlib.import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
    f"minimax_m3_nvfp4__sm_107__tp{_WORLD}.ops.m3_sampled_accept"
)
_CALL_POLICY = {}
if _WORLD == 4:
    _POLICY = importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_nvfp4__sm_107__tp4.modeling"
    )
    assert _POLICY.SAMPLED_ACCEPT_SM107TP4
    _CALL_POLICY = dict(
        one_round_k=_POLICY.SAMPLED_ACCEPT_1R_K,
        prewait_params=_POLICY.SAMPLED_ACCEPT_PREWAIT_PARAMS,
        early_trigger=_POLICY.SAMPLED_ACCEPT_EARLY_TRIGGER,
        ilp_walk=_POLICY.SAMPLED_ACCEPT_ILP_WALK,
    )
_VOCAB = 200064
_VS = _VOCAB // _WORLD
_DRAFT = 3
_DRAWS = 1024
_ALPHA = 1e-7


def _collective_check(ok: bool, reason: str) -> None:
    failures = _COMM.allgather(None if ok else reason)
    assert not any(failures), f"distributed check failed: {failures}"


def _digest(tensors: tuple[torch.Tensor, ...]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        h.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _probabilities(logits: torch.Tensor, temperatures: torch.Tensor, top_p: float) -> torch.Tensor:
    """Full vocabulary, float64, with ties kept as one nucleus group."""
    result = torch.zeros_like(logits, dtype=torch.float64)
    for row in range(logits.shape[0]):
        scaled = logits[row].double() / (float(temperatures[row]) + 1e-6)
        weights = torch.exp(scaled - scaled.max())
        values, order = weights.sort(descending=True)
        unique, groups, counts = torch.unique_consecutive(
            values, return_inverse=True, return_counts=True
        )
        group_mass = unique * counts
        strictly_above = group_mass.cumsum(0) - group_mass
        keep = (strictly_above <= top_p * weights.sum())[groups]
        indices = order[keep]
        result[row, indices] = weights[indices] / weights[indices].sum()
    return result


@pytest.mark.parametrize("nreq", (1, 2))
@pytest.mark.parametrize("top_p", (0.5, 0.95))
@pytest.mark.parametrize("graph_mode", (False, True), ids=("eager", "graph"))
def test_distributed_sampled_reference(nreq: int, top_p: float, graph_mode: bool) -> None:
    rank = mpi_rank()
    if _WORLD == 8:
        _collective_check(_SA.ONE_ROUND_K == 82, "exercise production TP8 one-round K82")
    device = torch.device("cuda", torch.cuda.current_device())
    rows = nreq * (_DRAFT + 1)
    mapping = Mapping(
        world_size=_WORLD,
        rank=rank,
        gpus_per_node=torch.cuda.device_count(),
        tp_size=_WORLD,
    )
    # Quantize deterministic host inputs before copying to CUDA: no scatter races.
    local = torch.full((rows, _VS), -30.0, dtype=torch.bfloat16)
    hot = (torch.arange(12) * 97 + rank * 43 + 5) % _VS
    local[:, hot] = (torch.linspace(4.0, 0.0, 12) + rank * 0.5).to(local.dtype)
    shards = _COMM.allgather(local.float().numpy())
    full_logits = torch.cat([torch.from_numpy(shard) for shard in shards], dim=1)
    temperatures_cpu = torch.linspace(0.8, 1.2, rows)
    probabilities = _probabilities(full_logits, temperatures_cpu, top_p)
    masses = torch.stack([probabilities[:, i * _VS : (i + 1) * _VS].sum(1) for i in range(_WORLD)])
    _collective_check(bool((masses > 0).sum(0).min() >= 2), "nucleus must span multiple ranks")
    _collective_check(
        bool((masses.max(0).values - masses.min(0).values).max() > 0.02),
        "test requires unequal shard masses",
    )
    logits = local.to(device)
    temperatures = temperatures_cpu.to(device)
    top_ps = torch.full((rows,), top_p, device=device)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=device)
    seeds = torch.arange(rows, dtype=torch.int64, device=device) * 7919 + 20261101
    offsets = torch.zeros(rows, dtype=torch.int64, device=device)
    draft = torch.zeros(nreq, _DRAFT, dtype=torch.int32, device=device)
    batch = torch.arange(nreq, dtype=torch.int32, device=device)
    next_new = torch.full((nreq, _DRAFT + 1), -7, dtype=torch.int32, device=device)
    ws = (
        _SA.SampledAcceptWorkspace.create(mapping, device, _CALL_POLICY["one_round_k"])
        if _WORLD == 4
        else _SA.SampledAcceptWorkspace.create(mapping, device)
    )
    graph = None

    def call() -> tuple[torch.Tensor, ...]:
        result = _SA.sampled_accept(
            logits,
            rank * _VS,
            mapping,
            draft,
            batch,
            temperatures,
            top_ks,
            top_ps,
            None,
            seeds,
            offsets,
            0,
            next_new=next_new,
            ws=ws,
            use_pdl=True,
            **_CALL_POLICY,
        )
        assert result is not None
        return (*result, next_new, ws.tokens[:rows])

    try:
        warm = call()
        torch.cuda.synchronize()
        warm_cpu = tuple(t.cpu() for t in warm)
        peers = _COMM.allgather(_digest(warm_cpu))
        _collective_check(len(set(peers)) == 1, "warm outputs disagree")
        draft.copy_(warm[-1].reshape(nreq, _DRAFT + 1)[:, :_DRAFT])
        outputs = None
        if graph_mode:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                outputs = call()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        records = (
            torch.empty((_DRAWS, nreq, _DRAFT + 1), dtype=torch.int32, device=device),
            torch.empty((_DRAWS, nreq), dtype=torch.int32, device=device),
            torch.empty((_DRAWS, nreq), dtype=torch.int64, device=device),
            torch.empty((_DRAWS, nreq, _DRAFT + 1), dtype=torch.int32, device=device),
            torch.empty((_DRAWS, rows), dtype=torch.int32, device=device),
        )
        for draw in range(_DRAWS):
            offsets.fill_(draw * 1000)
            if graph_mode:
                graph.replay()
            else:
                outputs = call()
            for destination, output in zip(records, outputs):
                destination[draw].copy_(output)
        torch.cuda.synchronize()
        observed = tuple(t.cpu() for t in records)
        peers = _COMM.allgather(_digest(observed))
        _collective_check(len(set(peers)) == 1, "ranks disagree on live outputs")
        accepted, num_acc, gather_ids, new_tokens, tokens = observed
        expected_accepted = tokens.reshape(_DRAWS, nreq, _DRAFT + 1)
        matches = (expected_accepted[:, :, :_DRAFT] == draft.cpu()).int()
        expected_num = 1 + matches.cumprod(2).sum(2)
        expected_gather = torch.arange(nreq)[None, :] * (_DRAFT + 1) + expected_num - 1
        expected_new = tokens.gather(1, expected_gather.long())
        _collective_check(torch.equal(accepted, expected_accepted), "accepted tokens differ")
        _collective_check(torch.equal(num_acc, expected_num.int()), "strict prefix counts differ")
        _collective_check(torch.equal(gather_ids, expected_gather.long()), "gather IDs differ")
        _collective_check(torch.equal(new_tokens[:, :, 0], expected_new), "next tokens differ")
        _collective_check(bool((new_tokens[:, :, 1:] == -7).all()), "unused output columns changed")
        _collective_check(ws.counters.cpu().tolist() == [0, _DRAWS + 1], "call parity differs")
        distances = []
        for row in range(rows):
            counts = torch.bincount(tokens[:, row].long(), minlength=_VOCAB).double()
            expected = probabilities[row]
            outside = counts[expected == 0].sum().item()
            support = int((expected > 0).sum())
            # E[TV] <= .5 sqrt((K-1)/N); bounded-difference tail <= alpha per row.
            band = 0.5 * math.sqrt((support - 1) / _DRAWS) + math.sqrt(
                math.log(1 / _ALPHA) / (2 * _DRAWS)
            )
            tv = 0.5 * (counts / _DRAWS - expected).abs().sum().item()
            _collective_check(outside == 0, f"row{row}: sampled outside full-vocabulary nucleus")
            _collective_check(tv <= band, f"row{row}: TV{tv} exceeds preregistered band{band}")
            distances.append(dict(row=row, support=support, tv=tv, band=band))
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        world=_WORLD,
                        nreq=nreq,
                        top_p=top_p,
                        graph=graph_mode,
                        draws=_DRAWS,
                        rank_agreement=True,
                        exact_acceptance=True,
                        both_halves=True,
                        one_round_k=_CALL_POLICY.get(
                            "one_round_k", getattr(_SA, "ONE_ROUND_K", None)
                        ),
                        prewait_params=_CALL_POLICY.get(
                            "prewait_params", getattr(_SA, "PREWAIT_PARAMS", None)
                        ),
                        early_trigger=_CALL_POLICY.get(
                            "early_trigger", getattr(_SA, "EARLY_TRIGGER", None)
                        ),
                        distances=distances,
                    )
                ),
                flush=True,
            )
    finally:
        torch.cuda.synchronize()
        graph = None
        gc.collect()
        _COMM.Barrier()
        ws.uc = None
        ws.handle = None
        gc.collect()
        _COMM.Barrier()
        if ws.comm is not None:
            ws.comm.Free()
