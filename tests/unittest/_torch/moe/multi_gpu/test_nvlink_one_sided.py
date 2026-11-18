# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLinkOneSided path selection and model-shaped round trips with independent references.

``Case.mode`` is the transport policy. Model cases run the default (``auto``: sentinel for the
steps it can carry, fence for the rest); the round-sequence cases also run fence-only and with
sentinel requested, and the skew cases with sentinel. The sentinel transport places received rows
at (source rank, token) instead of compacting them, which the dispatch check accepts because it
identifies tokens by their encoded routing weights.
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal
from unittest.mock import patch

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_no_mnnvl_fabric_handles

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
    SENTINEL_MAX_TOKENS_ENV,
    SYNC_PROTOCOL_ENV,
    NVLinkOneSided,
    sentinel_row,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden_size: int
    num_experts: int
    top_k: int
    # activation format of MoE for post-quant dispatch
    dispatch_dtype: Literal["bf16", "blockwise_fp8", "mxfp8", "nvfp4"]


MODELS = {
    model.name: model
    for model in (
        ModelShape("gpt_oss", 2880, 128, 4, "mxfp8"),
        ModelShape("deepseek_v3", 7168, 256, 8, "blockwise_fp8"),
        ModelShape("deepseek_r1_nvfp4", 7168, 256, 8, "nvfp4"),
        ModelShape("qwen3_5_397b_a17b", 4096, 512, 10, "bf16"),
        ModelShape("deepseek_v4_flash", 4096, 256, 6, "mxfp8"),
        ModelShape("deepseek_v4_pro", 7168, 384, 6, "mxfp8"),
        ModelShape(
            "kimi_k3", 3584, 896, 16, "mxfp8"
        ),  # K3 communicates its latent MoE width, not the model's 7168-wide states.
    )
}


# Sentinel payload format of each dispatch format the tests generate.
SENTINEL_FORMATS = {
    "bf16": "BF16",
    "blockwise_fp8": "FP8_BLOCK128",
    "mxfp8": "MXFP8",
    "nvfp4": "NVFP4",
}


def _expert_bounds(num_experts: int, ep_size: int, rank: int) -> tuple[int, int]:
    base, remainder = divmod(num_experts, ep_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


class Routing(Enum):
    """Synthetic expert routing patterns for communication tests.

    SPREAD distributes routes across all ranks, LOCAL selects the source
    rank's experts, and HOTSPOT selects only rank 0's experts.
    """

    SPREAD = "spread"
    LOCAL = "local"
    HOTSPOT = "hotspot"

    def make_expert_ids(
        self,
        model: ModelShape,
        tokens: tuple[int, ...],
        rank: int,
        round_index: int,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """Build int32 global expert IDs shaped [tokens[rank], model.top_k]."""
        ep_size = len(tokens)
        count = tokens[rank]
        token = torch.arange(count, device=device, dtype=torch.int64)[:, None]
        choice = torch.arange(model.top_k, device=device, dtype=torch.int64)[None, :]
        if self is Routing.LOCAL:
            owners = torch.full((count, model.top_k), rank, device=device, dtype=torch.int64)
        elif self is Routing.HOTSPOT:
            owners = torch.zeros((count, model.top_k), device=device, dtype=torch.int64)
        else:
            owners = (token + choice + rank + round_index) % ep_size
        bounds = [_expert_bounds(model.num_experts, ep_size, r) for r in range(ep_size)]
        starts = torch.tensor([b[0] for b in bounds], device=device)
        sizes = torch.tensor([b[1] - b[0] for b in bounds], device=device)
        return (starts[owners] + (token + choice + 3 * round_index) % sizes[owners]).int()


@dataclass(frozen=True)
class Round:
    """One dispatch → simulated expert computation → combine iteration.

    Attributes:
        tokens: Local token counts indexed by EP rank; zero-token ranks still
            participate in communication.
        routing: Strategy used to construct expert IDs; see Routing.
        delay_rank: Rank to delay on the GPU after dispatch, before expert
            computation, to exercise rank skew. -1 disables the delay.
        delay_before_dispatch_rank: Rank to delay on the GPU before dispatch, so
            that its peers run ahead into this round. -1 disables the delay.
    """

    tokens: tuple[int, ...]
    routing: Routing
    delay_rank: int
    delay_before_dispatch_rank: int = -1


@dataclass(frozen=True)
class Case:
    """A workload and execution settings sharing one communicator across rounds.

    Attributes:
        model: Expert layout, hidden width, and dispatch payload format.
        rounds: Ordered iterations with no host synchronization between them.
            All token-count tuples must describe the same EP group size.
        mode: Automatic transport selection (sentinel for small steps, fence for
            the rest), forced fence, or requested sentinel with fence for the steps
            it cannot carry; platform and payload restrictions still apply to
            sentinel.
        payload_in_workspace: Copy simulated expert outputs into the communication workspace
            before combine, instead of passing an external tensor.
        fp8_combine: Use FP8 on the combine wire; the returned output remains BF16.
        graph: Capture and replay the entire round sequence in one CUDA graph.
        pdl: Enable programmatic dependent launch for communication kernels.
        eplb: Gather and verify expert-load statistics alongside dispatch.
        sentinel_max_tokens: Token limit of the sentinel transport; None keeps its default.
        bf16_weights: Dispatch bf16 routing weights, as TRTLLM-Gen backends do.
    """

    model: ModelShape
    rounds: tuple[Round, ...]
    mode: Literal["auto", "fence", "sentinel"]
    payload_in_workspace: bool
    fp8_combine: bool
    graph: bool
    pdl: bool
    eplb: bool
    sentinel_max_tokens: int | None = None
    bf16_weights: bool = False

    @property
    def ep_size(self) -> int:
        """Number of participating EP workers."""
        return len(self.rounds[0].tokens)

    @property
    def runtime_max_num_tokens_per_rank(self) -> int:
        """Allocation-time per-rank token limit covering every round."""
        return max(max(r.tokens) for r in self.rounds)


def _make_inputs(case: Case, rank: int, round_index: int) -> tuple[torch.Tensor, ...]:
    import torch

    model = case.model
    spec = case.rounds[round_index]
    count = spec.tokens[rank]
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank * 97 + round_index)
    # Positive, exactly representable values keep cancellation and saturation out
    # of the communication reference. Scales vary by row and channel block.
    x = torch.randint(1, 17, (max(count, 1), model.hidden_size), generator=generator, device="cuda")
    row_scale = 2.0 ** ((torch.arange(max(count, 1), device="cuda") + round_index) % 3)
    block_size = 128 if model.dispatch_dtype == "blockwise_fp8" else 32
    block_scale = 2.0 ** ((torch.arange(model.hidden_size, device="cuda") // block_size) % 3)
    x = (x.float() / 16 * row_scale[:, None] * block_scale[None, :]).to(torch.bfloat16)
    if model.dispatch_dtype == "blockwise_fp8":
        from tensorrt_llm.quantization.utils.fp8_utils import fp8_quantize_1x128_sf_transpose

        payload, sf = fp8_quantize_1x128_sf_transpose(x, use_ue8m0=False)
        # Dispatch requires contiguous token-major scales, not GEMM's column-major layout.
        sf = sf[:count].contiguous()
        assert payload.dtype == torch.float8_e4m3fn
        assert sf.dtype == torch.float32
        assert sf.shape == (count, model.hidden_size // 128)
    elif model.dispatch_dtype == "mxfp8":
        payload, sf = torch.ops.trtllm.mxfp8_quantize(x, False, 32)
        sf = sf.view(x.shape[0], -1)[:count]
    elif model.dispatch_dtype == "nvfp4":
        global_scale = torch.ones((), dtype=torch.float32, device="cuda")
        payload, sf = torch.ops.trtllm.fp4_quantize(x, global_scale, 16, False, False)
        sf = sf.view(x.shape[0], -1)[:count]
    else:
        payload, sf = x, None
    payload = payload[:count]

    slots = spec.routing.make_expert_ids(model, spec.tokens, rank, round_index)
    token = torch.arange(count, device="cuda", dtype=torch.int64)[:, None]
    choice = torch.arange(model.top_k, device="cuda", dtype=torch.int64)[None, :]
    # The first routing weight also identifies the source token independently
    # of the implementation's compact send indices and receive counters.
    identity = rank * case.runtime_max_num_tokens_per_rank + token + 1
    weights = (identity * (choice + 1)).float() / (
        case.runtime_max_num_tokens_per_rank * case.ep_size * 32
    )
    if case.bf16_weights:
        weights = weights.bfloat16()
    return payload, sf, slots, weights


def _dequantize(payload: torch.Tensor, sf: torch.Tensor | None, mode: str) -> torch.Tensor:
    if mode == "bf16":
        return payload.float()
    if mode == "blockwise_fp8":
        values = payload.view(torch.float8_e4m3fn).float()
        blocks = values.reshape(values.shape[0], values.shape[1] // 128, 128)
        return (blocks * sf[..., None]).flatten(1)
    if mode == "mxfp8":
        values = payload.view(torch.float8_e4m3fn).float()
        exponents = sf.view(torch.uint8).int() - 127
        return torch.ldexp(
            values.reshape(values.shape[0], values.shape[1] // 32, 32), exponents[..., None]
        ).flatten(1)
    packed = payload.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).long()
    # Decode E2M1 on-device without a host lookup-table copy during graph capture.
    mantissa = (codes & 1).float()
    exponent = (codes >> 1) & 3
    magnitude = torch.where(
        exponent == 0, mantissa * 0.5, torch.ldexp(1.0 + mantissa * 0.5, exponent - 1)
    )
    values = torch.where(codes < 8, magnitude, -magnitude)
    scales = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * scales


def _expert_output(
    payload: torch.Tensor,
    sf: torch.Tensor | None,
    slots: torch.Tensor,
    weights: torch.Tensor,
    case: Case,
    rank: int,
) -> torch.Tensor:
    start, end = _expert_bounds(case.model.num_experts, case.ep_size, rank)
    owned = (slots >= start) & (slots < end)
    gain = torch.where(owned, weights * (1.0 + (slots % 7).float() / 8), 0).sum(dim=-1)
    values = _dequantize(payload, sf, case.model.dispatch_dtype)
    # Padded receive slots have invalid expert IDs and unspecified payload bytes.
    values = torch.where(owned.any(dim=-1)[:, None], values, 0)
    return (values * gain[:, None]).to(torch.bfloat16)


def _cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    # Transport FP8 as raw bytes: its storage is not supported by the legacy
    # torch serialization used by MPI pickle. References reinterpret the bytes.
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.uint8)
    return tensor.cpu()


def _run_worker(case: Case) -> dict:
    # Import locally so cloudpickle does not serialize torch's dynamic ops namespace.
    import os

    import torch

    # Selection is read by each communicator constructor, not cached by the MPI pool.
    os.environ[SYNC_PROTOCOL_ENV] = case.mode
    os.environ[SENTINEL_MAX_TOKENS_ENV] = (
        "" if case.sentinel_max_tokens is None else str(case.sentinel_max_tokens)
    )

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    supported = MnnvlMemory.supports_mnnvl()
    sm_major = torch.cuda.get_device_capability()[0]
    quant_supported = (
        case.model.dispatch_dtype == "bf16"
        or (case.model.dispatch_dtype == "blockwise_fp8" and sm_major >= 9)
        or sm_major >= 10
    )
    reasons = MPI.COMM_WORLD.allgather((supported, quant_supported))
    if not all(item[0] for item in reasons):
        return {"skip": "NVLink one-sided is not supported on every participating GPU"}
    if not all(item[1] for item in reasons):
        return {
            "skip": f"{case.model.dispatch_dtype} payload generation is unsupported on a participating GPU"
        }

    mapping = Mapping(
        rank=rank, world_size=case.ep_size, tp_size=case.ep_size, moe_ep_size=case.ep_size
    )
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=case.model.num_experts,
        top_k=case.model.top_k,
        max_num_tokens_per_rank=case.runtime_max_num_tokens_per_rank,
        hidden_size=case.model.hidden_size,
        dtype=torch.bfloat16,
        payload_in_workspace=case.payload_in_workspace,
        use_low_precision_combine=case.fp8_combine,
        num_experts=case.model.num_experts // 2 if case.eplb else None,
        sentinel_payload_format=SENTINEL_FORMATS[case.model.dispatch_dtype],
    )
    try:
        # Whether every precondition of the sentinel transport holds, from the case alone.
        try:
            sentinel_row(
                SENTINEL_FORMATS[case.model.dispatch_dtype],
                case.model.hidden_size,
                case.model.top_k,
            )
            sentinel_shape_ok = True
        except RuntimeError:
            sentinel_shape_ok = False
        # The kernels are available exactly on the GPUs the transport is validated on, so a build
        # or gate regression that drops them fails here instead of turning the sentinel cases into
        # skips.
        sentinel_available = bool(torch.ops.trtllm.moe_a2a_sentinel_available())
        assert sentinel_available == (
            torch.cuda.get_device_capability() in ((10, 0), (10, 3), (10, 7))
        ), f"sentinel availability {sentinel_available} on {torch.cuda.get_device_capability()}"
        sentinel_expected = (
            case.mode in ("auto", "sentinel")
            and sentinel_available
            and sentinel_shape_ok
            and not case.fp8_combine
            and case.model.num_experts % case.ep_size == 0
        )
        if case.mode == "sentinel" and not sentinel_expected:
            return {"skip": "the sentinel transport does not support this case on this platform"}
        inputs = [_make_inputs(case, rank, i) for i in range(len(case.rounds))]
        stats = None
        if case.eplb:
            stats = (
                torch.arange(case.model.num_experts // 2, dtype=torch.int32, device="cuda")
                + rank * 1000
            )
        dispatch_op = torch.ops.trtllm.moe_a2a_dispatch
        combine_op = torch.ops.trtllm.moe_a2a_combine
        launches = []
        combine_offsets = []

        def record_dispatch(*args, **kwargs):
            # Record the actual op arguments, not just the wrapper's capability.
            use_sentinel = kwargs.get("use_sentinel", False)
            launches.append(("dispatch", bool(use_sentinel)))
            result = dispatch_op(*args, **kwargs)
            combine_offsets.append(result[1])
            return result

        def record_combine(*args, **kwargs):
            use_sentinel = kwargs.get("use_sentinel", False)
            launches.append(("combine", bool(use_sentinel)))
            return combine_op(*args, **kwargs)

        def sequence() -> list[dict]:
            outputs = []
            for i, (spec, tensors) in enumerate(zip(case.rounds, inputs, strict=True)):
                payload, sf, slots, weights = tensors
                counts = list(spec.tokens)
                comm.prepare_dispatch(slots, counts)
                if rank == spec.delay_before_dispatch_rank:
                    torch.cuda._sleep(200_000)
                recv = comm.dispatch(payload, sf, slots, weights, counts, eplb_local_stats=stats)
                # Combine clears dispatch state; consume statistics at the same
                # point as the MoE scheduler, before starting expert computation.
                eplb = comm.get_eplb_gathered_statistics().clone() if case.eplb else None
                if rank == spec.delay_rank:
                    torch.cuda._sleep(200_000)
                # Detailed dispatch snapshots are omitted in the race sequence;
                # even device copies can perturb the overlap being exercised.
                snapshot = (
                    tuple(None if t is None else t.clone() for t in recv)
                    if len(inputs) == 1
                    else None
                )
                expert_out = _expert_output(*recv, case, rank)
                if case.payload_in_workspace:
                    workspace_out = comm.get_combine_payload_tensor_in_workspace(
                        max(counts), case.model.hidden_size, torch.bfloat16
                    )
                    workspace_out.view_as(expert_out).copy_(expert_out)
                    expert_out = workspace_out
                combined = comm.combine(expert_out, all_rank_max_num_tokens=max(counts))
                outputs.append({"combined": combined.clone(), "dispatch": snapshot, "eplb": eplb})
            return outputs

        with (
            patch.object(torch.ops.trtllm, "moe_a2a_dispatch", record_dispatch),
            patch.object(torch.ops.trtllm, "moe_a2a_combine", record_combine),
        ):
            # No host reads, barriers, or synchronizations between rounds.
            MPI.COMM_WORLD.Barrier()
            if case.graph:
                sequence()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = sequence()
                launches = launches[-2 * len(inputs) :]
                for _ in range(3):
                    graph.replay()
            else:
                outputs = sequence()
            torch.cuda.synchronize()

        # Region boundaries must stay fixed across runtime token counts and paths.
        assert len(set(combine_offsets)) == 1, combine_offsets
        result = {
            "rank": rank,
            "inputs": [tuple(_cpu(t) for t in tensors) for tensors in inputs],
            "outputs": [
                {
                    "combined": _cpu(o["combined"]),
                    "dispatch": None
                    if o["dispatch"] is None
                    else tuple(_cpu(t) for t in o["dispatch"]),
                    "eplb": _cpu(o["eplb"]),
                }
                for o in outputs
            ],
            "launches": launches,
            "sentinel_max_tokens": comm.sentinel_max_tokens,
            "sentinel_expected": sentinel_expected,
        }
        MPI.COMM_WORLD.Barrier()
        return result
    finally:
        comm.destroy()


def verify_dispatch(case: Case, results: list[dict], receiver: int, round_index: int) -> None:
    """Check received tokens, payload bytes, routing metadata, and padding against original inputs."""
    payload, sf, slots, weights = results[receiver]["outputs"][round_index]["dispatch"]
    maximum = max(case.rounds[round_index].tokens)
    payload = payload.view(case.ep_size, maximum, -1)
    sf = None if sf is None else sf.view(case.ep_size, maximum, -1)
    slots = slots.view(case.ep_size, maximum, -1)
    weights = weights.view(case.ep_size, maximum, -1)
    begin, end = _expert_bounds(case.model.num_experts, case.ep_size, receiver)
    for source in range(case.ep_size):
        original, original_sf, original_slots, original_weights = results[source]["inputs"][
            round_index
        ]
        wanted = ((original_slots >= begin) & (original_slots < end)).any(dim=1)
        expected_ids = torch.where(wanted)[0]
        valid = (slots[source] >= 0).any(dim=1)
        observed_ids = (
            (weights[source, valid, 0] * (case.runtime_max_num_tokens_per_rank * case.ep_size * 32))
            .round()
            .long()
            - source * case.runtime_max_num_tokens_per_rank
            - 1
        )
        order = observed_ids.argsort()
        torch.testing.assert_close(observed_ids[order], expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(
            slots[source, ~valid], torch.full_like(slots[source, ~valid], -1)
        )
        torch.testing.assert_close(
            slots[source, valid][order], original_slots[expected_ids], rtol=0, atol=0
        )
        torch.testing.assert_close(
            weights[source, valid][order], original_weights[expected_ids], rtol=0, atol=0
        )
        torch.testing.assert_close(
            payload[source].view(torch.uint8)[valid][order],
            original.view(torch.uint8)[expected_ids],
            rtol=0,
            atol=0,
        )
        if sf is not None:
            torch.testing.assert_close(
                sf[source].view(torch.uint8)[valid][order],
                original_sf.view(torch.uint8)[expected_ids],
                rtol=0,
                atol=0,
            )


def verify_combine(case: Case, results: list[dict], rank: int, round_index: int) -> None:
    """Check combined output against original inputs, including per-rank BF16/FP8 rounding."""
    payload, sf, slots, weights = results[rank]["inputs"][round_index]
    spec = case.rounds[round_index]
    reference = torch.zeros((spec.tokens[rank], case.model.hidden_size), dtype=torch.float32)
    for expert_rank in range(case.ep_size):
        contribution = _expert_output(payload, sf, slots, weights, case, expert_rank)
        if case.fp8_combine:
            contribution = contribution.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        reference += contribution.float()
    output = results[rank]["outputs"][round_index]["combined"]
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(
        output,
        reference.to(torch.bfloat16),
        rtol=0.016,
        atol=0.002,
        msg=lambda detail: (
            f"rank={rank}, round={round_index}, tokens={spec.tokens}, "
            f"mode={case.mode}, graph={case.graph}\n{detail}"
        ),
    )


def _assert_results(case: Case, results: list[dict]) -> None:
    for rank in range(case.ep_size):
        result = results[rank]
        assert len(result["outputs"]) == len(case.rounds)
        sentinel_max_tokens = result["sentinel_max_tokens"]
        assert (sentinel_max_tokens > 0) == result["sentinel_expected"], (
            f"rank {rank}: sentinel limit {sentinel_max_tokens}, expected it "
            f"{'on' if result['sentinel_expected'] else 'off'}"
        )
        if case.sentinel_max_tokens is not None and sentinel_max_tokens:
            assert sentinel_max_tokens == min(
                case.sentinel_max_tokens, case.runtime_max_num_tokens_per_rank
            )
        expected_launches = []
        for i, spec in enumerate(case.rounds):
            verify_combine(case, results, rank, i)
            if result["outputs"][i]["dispatch"] is not None:
                verify_dispatch(case, results, rank, i)
            # The sentinel transport takes every step it can carry; fence takes the rest.
            sentinel = max(spec.tokens) <= sentinel_max_tokens and not case.eplb
            expected_launches.extend((("dispatch", sentinel), ("combine", sentinel)))
            if case.eplb:
                expected = torch.stack(
                    [
                        torch.arange(case.model.num_experts // 2, dtype=torch.int32) + r * 1000
                        for r in range(case.ep_size)
                    ]
                )
                torch.testing.assert_close(result["outputs"][i]["eplb"], expected, rtol=0, atol=0)
        assert result["launches"] == expected_launches


@pytest.fixture(scope="module")
def mpi_pools() -> Iterator[dict[tuple[int, bool], MPIPoolExecutor]]:
    """Reuse imported workers; each case still creates and destroys its communicator."""
    pools: dict[tuple[int, bool], MPIPoolExecutor] = {}
    try:
        yield pools
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True, cancel_futures=True)


def _run(case: Case, pools: dict[tuple[int, bool], MPIPoolExecutor]) -> None:
    if torch.cuda.device_count() < case.ep_size:
        pytest.skip(f"requires {case.ep_size} GPUs")
    # PDL may be cached in native code, so different settings need separate workers.
    # The transport mode is Python-side and is set on every worker before constructing the comm.
    env = {
        "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB": "",
        "TRTLLM_ENABLE_PDL": "1" if case.pdl else "0",
        "TRTLLM_MOE_COMM_TIMEOUT_SEC": "30",
        "TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC": "60",
    }
    key = (case.ep_size, case.pdl)
    if key not in pools:
        pools[key] = MPIPoolExecutor(case.ep_size, env=env)
    executor = pools[key]
    healthy = False
    try:
        results = list(executor.map(_run_worker, [case] * case.ep_size))
        skipped = [r["skip"] for r in results if "skip" in r]
        if skipped:
            assert len(skipped) == case.ep_size, "inconsistent platform support across ranks"
            pytest.skip(skipped[0])
        # Task submission order is independent of the MPI rank that executes it.
        results.sort(key=lambda result: result["rank"])
        assert [result["rank"] for result in results] == list(range(case.ep_size))
        _assert_results(case, results)
        healthy = True
    finally:
        # A failed collective or reference check must not contaminate another case.
        if not healthy:
            pools.pop(key).shutdown(wait=True, cancel_futures=True)


def _skew_rounds(ep_size: int, num_rounds: int, sentinel_max_tokens: int) -> tuple[Round, ...]:
    """Rounds with varying, uneven token counts (zero included), every routing pattern, and
    rotating delays; every eighth round exceeds ``sentinel_max_tokens`` on one rank."""
    rounds = []
    patterns = (Routing.SPREAD, Routing.LOCAL, Routing.HOTSPOT)
    for i in range(num_rounds):
        tokens = [(7 * i + 13 * rank) % (sentinel_max_tokens + 1) for rank in range(ep_size)]
        if i % 8 == 7:
            tokens[i % ep_size] = sentinel_max_tokens + 1 + i % 5
        rounds.append(
            Round(
                tokens=tuple(tokens),
                routing=patterns[i % len(patterns)],
                delay_rank=i % ep_size,
                delay_before_dispatch_rank=(i + 1 + i // ep_size) % ep_size,
            )
        )
    return tuple(rounds)


CASES = [
    # Model coverage uses equal token counts on every rank.
    *[
        pytest.param(
            Case(
                model=model,
                rounds=(
                    Round(tokens=(num_tokens,) * ep_size, routing=Routing.SPREAD, delay_rank=-1),
                ),
                mode="auto",
                payload_in_workspace=False,
                fp8_combine=False,
                graph=False,
                pdl=True,
                eplb=False,
            ),
            id=f"model-{model.name}-ep{ep_size}-tokens{num_tokens}",
        )
        for model, ep_size in (
            (MODELS["gpt_oss"], 4),
            (MODELS["deepseek_v3"], 8),
            (MODELS["deepseek_r1_nvfp4"], 8),
            (MODELS["qwen3_5_397b_a17b"], 8),
            (MODELS["deepseek_v4_flash"], 4),
            (MODELS["deepseek_v4_pro"], 8),
            (MODELS["kimi_k3"], 8),
        )
        for num_tokens in (1, 128, 1024)
    ],
    # bf16 routing weights, as TRTLLM-Gen backends dispatch them: a sentinel step carries them
    # bit for bit (few tokens per rank, so the identity encoded in the first weight survives bf16).
    pytest.param(
        Case(
            model=MODELS["deepseek_r1_nvfp4"],
            rounds=(Round(tokens=(9, 5, 3, 1), routing=Routing.SPREAD, delay_rank=-1),),
            mode="auto",
            payload_in_workspace=False,
            fp8_combine=False,
            graph=False,
            pdl=True,
            eplb=False,
            bf16_weights=True,
        ),
        id="bf16-weights-ep4",
    ),
    # Cover BF16/FP8 combine with external/workspace-resident input payloads.
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=(Round(tokens=(9, 5), routing=Routing.SPREAD, delay_rank=-1),),
                mode="auto",
                payload_in_workspace=payload_in_workspace,
                fp8_combine=fp8_combine,
                graph=False,
                pdl=True,
                eplb=False,
            ),
            id=(
                f"combine-{'fp8' if fp8_combine else 'bf16'}-"
                f"{'workspace' if payload_in_workspace else 'external'}"
            ),
        )
        for fp8_combine in (False, True)
        for payload_in_workspace in (False, True)
    ],
    # Reuse one four-rank communicator across changing token counts and peer dependencies,
    # including zero-token ranks and delayed ranks. Counts 128/129 straddle a sentinel
    # limit of 128; graph cases replay the entire sequence.
    *[
        pytest.param(
            Case(
                model=MODELS["deepseek_v3"],
                rounds=(
                    Round((3, 0, 2, 1), Routing.LOCAL, 0),
                    Round((1, 129, 0, 5), Routing.SPREAD, 1),
                    Round((0, 5, 3, 1), Routing.HOTSPOT, 2),
                    Round((128, 3, 1, 0), Routing.SPREAD, 3),
                    Round((5, 2, 0, 3), Routing.LOCAL, 0),
                    Round((1, 0, 4, 2), Routing.SPREAD, 1),
                    # Local-only routing gives rank 0 no token dependency on rank 1.
                    # Combine on rank 0 must still wait for rank 1 until rank 1 consumes its dispatched inputs.
                    # Otherwise, rank 0's next dispatch would abrupt the data that rank 1 is consuming.
                    Round((1, 128, 0, 0), Routing.LOCAL, 1),
                    Round((129, 1, 0, 0), Routing.SPREAD, 0),
                ),
                mode=mode,
                payload_in_workspace=True,
                fp8_combine=False,
                graph=graph,
                pdl=True,
                eplb=False,
                sentinel_max_tokens=sentinel_max_tokens,
            ),
            id=(
                f"round-sequence-ep4-{mode}-{'graph' if graph else 'eager'}"
                + ("" if sentinel_max_tokens is None else f"-sentinel{sentinel_max_tokens}")
            ),
        )
        for mode, graph, sentinel_max_tokens in (
            ("auto", False, None),
            ("auto", True, None),
            ("fence", True, None),
            # A sentinel limit of 128 hands the 129-token rounds to fence, so one
            # sequence and one graph mix the transports.
            ("auto", True, 128),
            ("sentinel", False, 128),
            ("sentinel", True, 128),
        )
    ],
    # Back-to-back sentinel rounds under rank skew: no host synchronization or fence round
    # between most of them, a rotating rank delayed before dispatch and another before
    # combine, and every eighth round above the sentinel limit (a fence round). Every round's
    # combine is checked, so a stale or early landing row in any round shows up.
    *[
        pytest.param(
            Case(
                model=MODELS[model],
                rounds=_skew_rounds(ep_size, num_rounds=48, sentinel_max_tokens=128),
                mode="sentinel",
                payload_in_workspace=True,
                fp8_combine=False,
                graph=graph,
                pdl=True,
                eplb=False,
                sentinel_max_tokens=128,
            ),
            id=f"sentinel-skew-ep{ep_size}-{model}-{'graph' if graph else 'eager'}",
        )
        for ep_size in (4, 8)
        for model, graph in (("qwen3_5_397b_a17b", False), ("deepseek_r1_nvfp4", True))
    ],
    # Gather rank-distinct EPLB statistics from all four ranks, including the rank
    # with zero input tokens. This checks statistics transport, not expert migration.
    pytest.param(
        Case(
            model=MODELS["deepseek_v3"],
            rounds=(Round(tokens=(9, 0, 3, 1), routing=Routing.SPREAD, delay_rank=-1),),
            mode="auto",
            payload_in_workspace=False,
            fp8_combine=False,
            graph=False,
            pdl=True,
            eplb=True,
        ),
        id="eplb-statistics",
    ),
    # Use 129 experts with the DeepSeek V3 payload shape, split across two ranks (65/64),
    # to exercise remainder-aware ownership in dispatch and the combine reference.
    pytest.param(
        Case(
            model=replace(MODELS["deepseek_v3"], num_experts=129),
            rounds=(Round(tokens=(9, 3), routing=Routing.SPREAD, delay_rank=-1),),
            mode="auto",
            payload_in_workspace=False,
            fp8_combine=False,
            graph=False,
            pdl=True,
            eplb=False,
        ),
        id="non-divisible-experts",
    ),
]


@pytest.mark.parametrize("case", CASES)
def test_nvlink_one_sided(case: Case, mpi_pools: dict[tuple[int, bool], MPIPoolExecutor]) -> None:
    _run(case, mpi_pools)


# ============================================================================
# CUDA graphs under an execution state
# ============================================================================


def _run_graph_replay_worker(failed_rank: int) -> dict:
    """Capture one round trip with every rank active, then replay it without ``failed_rank``."""
    # Import locally so cloudpickle does not serialize torch's dynamic ops namespace.
    import torch

    from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
    from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth

    rank = MPI.COMM_WORLD.Get_rank()
    ep_size = MPI.COMM_WORLD.Get_size()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    if not all(MPI.COMM_WORLD.allgather(MnnvlMemory.supports_mnnvl())):
        return {"skip": "NVLink one-sided is not supported on every participating GPU"}

    hidden, capacity, slots_per_rank = 256, 8, 2
    health = EPGroupHealth(ep_size)
    state = MoeExecutionState(health, rank, torch.device("cuda", rank))
    mapping = Mapping(rank=rank, world_size=ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=ep_size * slots_per_rank,
        top_k=1,
        max_num_tokens_per_rank=capacity,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        enable_wide_ep_fault_tolerance=True,
    )
    comm.bind_execution_state(state)
    everyone = list(range(ep_size))
    active = [r for r in everyone if r != failed_rank]
    x = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
    slots = torch.empty(capacity, 1, dtype=torch.int32, device="cuda")
    weights = torch.ones(capacity, 1, device="cuda")
    errors = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            errors.append(f"rank {rank}: {what}")

    def set_inputs(owners: list[int], seed: int) -> None:
        """Route this rank's tokens, in turn, to the first slot of each of ``owners``."""
        generator = torch.Generator().manual_seed(seed * ep_size + rank)
        x.copy_(torch.randn(capacity, hidden, generator=generator).bfloat16())
        owner = torch.tensor(owners)[torch.arange(capacity) % len(owners)]
        slots.copy_((owner * slots_per_rank)[:, None].int())

    def round_trip() -> torch.Tensor:
        """The experts return their input, so a complete round combines each token to itself."""
        recv = comm.dispatch(x, None, slots, weights, [capacity] * ep_size)
        return comm.combine(recv[0])

    try:
        set_inputs(everyone, 0)
        check(torch.equal(round_trip(), x), "eager round combined wrong")
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = round_trip()
        set_inputs(everyone, 1)
        graph.replay()
        torch.cuda.synchronize()
        check(torch.equal(output, x), "replay with every rank combined wrong")
        MPI.COMM_WORLD.Barrier()

        # The failed rank replays nothing more; the others install the mask without it and
        # replay the graph captured before.
        health.mark_failed(failed_rank)
        if rank in active:
            address = state.active_rank_mask.data_ptr()
            state.install_mask()
            check(state.active_rank_mask.data_ptr() == address, "the mask moved")
            for seed in (2, 3):
                set_inputs(active, seed)
                graph.replay()
                torch.cuda.synchronize()
                check(torch.equal(output, x), f"replay {seed} without rank {failed_rank} wrong")
        MPI.COMM_WORLD.Barrier()
        return {"rank": rank, "errors": errors, "sentinel_max_tokens": comm.sentinel_max_tokens}
    finally:
        comm.destroy()


def test_graph_replay_follows_installed_mask() -> None:
    """A CUDA graph captured with every rank active replays without a rank masked out later."""
    ep_size, failed_rank = 4, 2
    # Wide-EP fault tolerance runs only on MNNVL memory shared through fabric handles.
    skip_no_mnnvl_fabric_handles(ep_size)
    # A replay that waited for the masked rank would end at this kernel timeout.
    executor = MPIPoolExecutor(ep_size, env={"TRTLLM_MOE_COMM_TIMEOUT_SEC": "60"})
    try:
        results = list(executor.map(_run_graph_replay_worker, [failed_rank] * ep_size))
        skipped = [r["skip"] for r in results if "skip" in r]
        if skipped:
            pytest.skip(skipped[0])
        assert [e for r in results for e in r["errors"]] == []
        # The sentinel transport takes no active-rank mask, so every step takes the fence one.
        assert all(r["sentinel_max_tokens"] == 0 for r in results)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


# ============================================================================
# Abort of bound collectives stuck on a rank that stopped
# ============================================================================


def _run_abort_stuck_collective_worker(failed_rank: int, stop_before: str, round_flag: int) -> dict:
    """Abort a round from ``round_flag`` whose ``failed_rank`` stops before ``stop_before``."""
    # Import locally so cloudpickle does not serialize torch's dynamic ops namespace.
    import time

    import torch

    from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
    from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
    from tensorrt_llm.bindings import internal

    rank = MPI.COMM_WORLD.Get_rank()
    ep_size = MPI.COMM_WORLD.Get_size()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    if not all(MPI.COMM_WORLD.allgather(MnnvlMemory.supports_mnnvl())):
        return {"skip": "NVLink one-sided is not supported on every participating GPU"}

    hidden, capacity, slots_per_rank = 256, 8, 2
    waited = int(internal.thop.MOE_A2A_ABORT_WAITED_FLAG_INDEX)
    health = EPGroupHealth(ep_size)
    mapping = Mapping(rank=rank, world_size=ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=ep_size * slots_per_rank,
        top_k=1,
        max_num_tokens_per_rank=capacity,
        hidden_size=hidden,
        dtype=torch.bfloat16,
        enable_wide_ep_fault_tolerance=True,
    )
    everyone = list(range(ep_size))
    active = [r for r in everyone if r != failed_rank]
    errors = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            errors.append(f"rank {rank}: {what}")

    def dispatch(senders: list[int], seed: int) -> tuple[torch.Tensor, tuple]:
        """Dispatch this rank's tokens, each to the first slot of one of ``senders``."""
        generator = torch.Generator().manual_seed(seed * ep_size + rank)
        x = torch.randn(capacity, hidden, generator=generator).bfloat16().cuda()
        owner = torch.tensor(senders)[torch.arange(capacity) % len(senders)]
        slots = (owner * slots_per_rank)[:, None].int().cuda()
        weights = torch.ones(capacity, 1, device="cuda")
        counts = [capacity if r in senders else 0 for r in everyone]
        return x, comm.dispatch(x, None, slots, weights, counts)

    def round_trip(senders: list[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The experts return their input, so a complete round combines each token to itself."""
        x, recv = dispatch(senders, seed)
        return x, comm.combine(recv[0], all_rank_max_num_tokens=capacity)

    try:
        state = MoeExecutionState(health, rank, torch.device("cuda"))
        comm.bind_execution_state(state)
        MPI.COMM_WORLD.Barrier()
        x, combined = round_trip(everyone, 0)
        check(torch.equal(combined, x), "healthy round combined wrong")
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()

        # Round 1: the dispatch waits for flag value round_flag + 1 and the combine for
        # round_flag + 2, modulo 2**32. A wait publishes 1 for a flag value that wrapped to 0.
        flag_val_offset = comm.moe_a2a_metainfo[comm.FLAG_VAL_OFFSET_INDEX].item()
        comm.workspace[comm.ep_rank, flag_val_offset : flag_val_offset + 4].view(torch.int32).fill_(
            round_flag - 2**32 if round_flag >= 2**31 else round_flag
        )
        if rank == failed_rank:
            if stop_before == "combine":
                dispatch(everyone, 1)
                torch.cuda.synchronize()
                comm.reset_state()
        else:
            x, combined = round_trip(everyone, 1)
            done = torch.cuda.Event()
            done.record()
            stuck_on = (round_flag + (1 if stop_before == "dispatch" else 2)) % 2**32 or 1
            deadline = time.monotonic() + 10
            while (
                state.waited_flags().get(failed_rank) != stuck_on
                and not done.query()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            check(
                state.waited_flags().get(failed_rank) == stuck_on,
                f"no waited flag for rank {failed_rank}: "
                f"{state.abort[waited : waited + ep_size].tolist()}",
            )
            check(not done.query(), "the round completed without the stopped rank")
            check(not state.aborted(), "aborted before the request")
            state.request_abort()
            start = time.monotonic()
            torch.cuda.synchronize()
            check(time.monotonic() - start < 10, "the abort did not release the waiting collective")
            check(state.abort[:2].tolist() == [1, 1], f"abort state {state.abort[:2].tolist()}")
            check(not state.abort[waited:].any(), "waited flags left after the abort")
            # The combine of an incomplete round takes only the tokens this rank routed to itself.
            expected = torch.zeros_like(x)
            own = torch.arange(capacity, device=x.device) % ep_size == rank
            expected[own] = x[own]
            check(torch.equal(combined, expected), "the incomplete round combined wrong")
        MPI.COMM_WORLD.Barrier()

        health.mark_failed(failed_rank)
        if rank in active:
            state.rearm()
        MPI.COMM_WORLD.Barrier()
        if rank in active:
            state.install_mask()
            for seed in (2, 3):
                x, combined = round_trip(active, seed)
                check(torch.equal(combined, x), f"round {seed} after the recovery combined wrong")
            torch.cuda.synchronize()
            check(not state.aborted(), "a round after the recovery aborted")
        MPI.COMM_WORLD.Barrier()
        return {"rank": rank, "errors": errors}
    finally:
        comm.destroy()


@pytest.mark.parametrize("round_flag", [2, 2**32 - 2], ids=["no_wrap", "flag_wraps"])
@pytest.mark.parametrize("stop_before", ["dispatch", "combine"])
def test_abort_releases_bound_collectives_waiting_on_a_stopped_rank(
    stop_before: str, round_flag: int
) -> None:
    """Abort a bound dispatch or combine waiting on a stopped rank, then recover without it.

    With ``flag_wraps`` the combine waits for round flag value 0, which is still published.
    """
    ep_size, failed_rank = 4, 2
    # Wide-EP fault tolerance runs only on MNNVL memory shared through fabric handles.
    skip_no_mnnvl_fabric_handles(ep_size)
    executor = MPIPoolExecutor(ep_size, env={"TRTLLM_MOE_COMM_TIMEOUT_SEC": "30"})
    try:
        args = [(failed_rank, stop_before, round_flag)] * ep_size
        results = list(executor.starmap(_run_abort_stuck_collective_worker, args))
        skipped = [r["skip"] for r in results if "skip" in r]
        if skipped:
            pytest.skip(skipped[0])
        assert [e for r in results for e in r["errors"]] == []
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
