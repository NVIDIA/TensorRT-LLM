# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLinkOneSided path selection and model-shaped round trips with independent references.

Every round trip runs with the fence and sentinel transports (``Case.mode``); the sentinel
transport places received rows at (source rank, token) instead of compacting them, which the
dispatch check accepts because it identifies tokens by their encoded routing weights.
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

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
    SENTINEL_MAX_TOKENS_ENV,
    SYNC_PROTOCOL_ENV,
    NVLinkOneSided,
    NVLinkOneSidedConfigError,
    default_sentinel_max_tokens,
    get_sentinel_max_tokens,
    get_sync_protocol,
    sentinel_payload_format,
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
        sentinel_expected = (
            case.mode in ("auto", "sentinel")
            and bool(torch.ops.trtllm.moe_a2a_sentinel_available())
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
# Allocation-time workspace layout (no MPI workers)
# ============================================================================


@pytest.mark.cpu_only
def test_workspace_layout() -> None:
    from tensorrt_llm.bindings import internal as _tllm_internal

    thop = _tllm_internal.thop

    def field(layout: torch.Tensor, name: str) -> int:
        return int(layout[getattr(thop, f"MOE_A2A_{name}")])

    ep_size, top_k, capacity, hidden_size = 8, 6, 128, 7168
    layout = NVLinkOneSided._make_workspace_layout(
        ep_size, top_k, capacity, hidden_size, torch.bfloat16, None, 0
    )
    dispatch_start = field(layout, "DISPATCH_PAYLOAD_OFFSET_INDEX")
    dispatch_end = dispatch_start + field(layout, "DISPATCH_PAYLOAD_SIZE_INDEX")
    combine_start = field(layout, "COMBINE_INPUT_OFFSET_INDEX")
    combine_end = combine_start + field(layout, "COMBINE_INPUT_SIZE_INDEX")
    total = field(layout, "WORKSPACE_SIZE_INDEX")
    assert field(layout, "TOPK_TARGET_INDICES_OFFSET_INDEX") < dispatch_start
    assert dispatch_end <= field(layout, "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX") < combine_start
    assert field(layout, "COMBINE_INPUT_SIZE_INDEX") == ep_size * capacity * hidden_size * 2
    assert dispatch_start % 256 == combine_start % 256 == total % 256 == 0
    assert combine_end == total

    # Routing metadata reserves the configured top-k, not the maximum supported top-k.
    routes_start = field(layout, "TOPK_TARGET_RANKS_OFFSET_INDEX")
    indices_start = field(layout, "TOPK_TARGET_INDICES_OFFSET_INDEX")
    assert indices_start - routes_start == capacity * top_k * 4
    with pytest.raises(RuntimeError, match="capacity"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(ep_size, capacity, top_k, -1, 256)
    with pytest.raises(RuntimeError, match="overflows"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size, capacity, top_k, (1 << 63) - 512, 1024
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("payload_format", ["BF16", "MXFP8", "FP8_BLOCK128", "NVFP4"])
def test_sentinel_workspace_layout(payload_format: str) -> None:
    from tensorrt_llm.bindings import internal as _tllm_internal

    thop = _tllm_internal.thop

    def field(layout: torch.Tensor, name: str) -> int:
        return int(layout[getattr(thop, f"MOE_A2A_{name}")])

    ep_size, top_k, capacity, hidden_size, sentinel_tokens = 8, 8, 256, 7168, 160
    base = NVLinkOneSided._make_workspace_layout(
        ep_size, top_k, capacity, hidden_size, torch.bfloat16, None, 0
    )
    for name in ("SENTINEL_MAX_NUM_TOKENS_INDEX", "SENTINEL_RECV_SIZE_INDEX"):
        assert field(base, name) == 0
    layout = base.clone()
    torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
        layout,
        sentinel_tokens,
        int(getattr(thop, f"MOE_A2A_SENTINEL_FORMAT_{payload_format}")),
        hidden_size,
    )
    # The other regions keep their offsets; the sentinel regions follow them in order.
    num_base_fields = int(thop.MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX)
    assert torch.equal(
        layout[: int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)],
        base[: int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)],
    )
    assert num_base_fields == int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX) + 1
    starts = [
        field(layout, f"SENTINEL_{name}_OFFSET_INDEX")
        for name in ("STATE", "LANDING", "COMBINE_LANDING", "RECV")
    ]
    assert field(base, "WORKSPACE_SIZE_INDEX") <= starts[0]
    assert starts == sorted(starts) and all(start % 256 == 0 for start in starts)
    row = sentinel_row(payload_format, hidden_size, top_k)
    rows = ep_size * sentinel_tokens
    # One bf16 partial row per (source rank, token), padded to the workspace alignment.
    assert starts[3] - starts[2] >= rows * hidden_size * 2
    assert field(layout, "SENTINEL_RECV_SIZE_INDEX") >= rows * (
        row.hidden_bytes + row.scale_bytes + 2 * top_k * 4
    )
    assert field(layout, "WORKSPACE_SIZE_INDEX") == starts[3] + field(
        layout, "SENTINEL_RECV_SIZE_INDEX"
    )
    assert field(layout, "SENTINEL_MAX_NUM_TOKENS_INDEX") == sentinel_tokens
    assert field(layout, "SENTINEL_HIDDEN_SIZE_INDEX") == hidden_size

    # Only a layout fresh from moe_a2a_get_workspace_layout can be extended, once.
    with pytest.raises(RuntimeError, match="must be a layout from moe_a2a_get_workspace_layout"):
        torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
            layout, sentinel_tokens, int(thop.MOE_A2A_SENTINEL_FORMAT_BF16), hidden_size
        )
    with pytest.raises(RuntimeError, match="token capacity"):
        torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
            base.clone(), capacity + 1, int(thop.MOE_A2A_SENTINEL_FORMAT_BF16), hidden_size
        )


@pytest.mark.cpu_only
def test_sentinel_row_geometry() -> None:
    hidden_size, top_k = 7168, 8
    # (hidden-state bytes, scale bytes, dispatch blocks per token) per row.
    assert sentinel_row("BF16", hidden_size, top_k)[:3] == (2 * hidden_size, 0, 7)
    assert sentinel_row("MXFP8", hidden_size, top_k)[:3] == (hidden_size, hidden_size // 32, 4)
    assert sentinel_row("FP8_BLOCK128", hidden_size, top_k)[:3] == (
        hidden_size,
        hidden_size // 32,
        4,
    )
    assert sentinel_row("NVFP4", hidden_size, top_k)[:3] == (hidden_size // 2, hidden_size // 16, 2)
    # The payload format must split into whole 16-byte vectors.
    with pytest.raises(RuntimeError, match="hidden_size % 256"):
        sentinel_row("NVFP4", 2880, top_k)
    with pytest.raises(RuntimeError, match="hidden_size % 8"):
        sentinel_row("BF16", 4100, top_k)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("payload_format", "hidden_size", "top_k", "ep_size", "expected"),
    [
        # Limits FlashInfer's sentinel transport reports on GB300 (152 SMs).
        # One wave holds fewer than 159 tokens: the floor.
        pytest.param("BF16", 7168, 8, 8, 159, id="dsv3-bf16-ep8"),
        # The 64 MiB bound on the sentinel regions.
        pytest.param("NVFP4", 7168, 8, 8, 371, id="dsv3-nvfp4-ep8"),
        # The 64 MiB bound below the floor: the floor.
        pytest.param("NVFP4", 7168, 8, 64, 159, id="dsv3-nvfp4-ep64"),
        # The one-wave bound.
        pytest.param("MXFP8", 3584, 16, 8, 455, id="kimi-k3-mxfp8-ep8"),
    ],
)
def test_default_sentinel_max_tokens(
    payload_format: str, hidden_size: int, top_k: int, ep_size: int, expected: int
) -> None:
    row = sentinel_row(payload_format, hidden_size, top_k)
    assert default_sentinel_max_tokens(ep_size, 152, row) == expected


@pytest.mark.cpu_only
def test_sentinel_payload_format() -> None:
    from tensorrt_llm.quantization.mode import QuantAlgo

    assert sentinel_payload_format(QuantAlgo.NVFP4) == "NVFP4"
    assert sentinel_payload_format(QuantAlgo.NVFP4_AWQ) == "NVFP4"  # a calibration of NVFP4
    assert sentinel_payload_format(QuantAlgo.W4A8_MXFP4_MXFP8) == "MXFP8"
    assert sentinel_payload_format(QuantAlgo.MXFP8) == "MXFP8"
    # FP8 block-scale backends quantize after dispatch.
    assert sentinel_payload_format(QuantAlgo.FP8_BLOCK_SCALES) == "BF16"
    assert sentinel_payload_format(None) == "BF16"
    # FP8 per-tensor rows carry no block scales: no sentinel format.
    assert sentinel_payload_format(QuantAlgo.FP8) is None


@pytest.mark.cpu_only
def test_sync_protocol_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNC_PROTOCOL_ENV, raising=False)
    monkeypatch.delenv(SENTINEL_MAX_TOKENS_ENV, raising=False)
    assert get_sync_protocol() == "auto"
    assert get_sentinel_max_tokens() is None
    # The removed CFT pin keeps selecting fence for one release.
    monkeypatch.setenv("TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT", "0")
    assert get_sync_protocol() == "fence"
    monkeypatch.setenv(SYNC_PROTOCOL_ENV, " Sentinel ")
    assert get_sync_protocol() == "sentinel"  # an explicit protocol wins
    for name, value, parse in (
        (SYNC_PROTOCOL_ENV, "off", get_sync_protocol),
        (SENTINEL_MAX_TOKENS_ENV, "0", get_sentinel_max_tokens),
        (SENTINEL_MAX_TOKENS_ENV, "many", get_sentinel_max_tokens),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(NVLinkOneSidedConfigError):
            parse()


@pytest.mark.cpu_only
def test_sentinel_step_selection() -> None:
    from types import SimpleNamespace

    hidden_size, top_k = 7168, 8
    comm = SimpleNamespace(
        _sentinel_row=sentinel_row("NVFP4", hidden_size, top_k),
        _sentinel_format="NVFP4",
        sentinel_max_tokens=160,
        top_k=top_k,
    )
    tokens = 4
    hidden = torch.empty((tokens, hidden_size // 2), dtype=torch.uint8)
    sf = torch.empty((tokens, hidden_size // 16), dtype=torch.uint8)
    weights = torch.empty((tokens, top_k), dtype=torch.float32)

    def carries(*args) -> bool:
        return NVLinkOneSided.use_sentinel_for_dispatch(comm, *args)

    assert carries(160, hidden, sf, weights, None)
    assert carries(160, hidden, sf, None, None)
    assert not carries(161, hidden, sf, weights, None)  # above the token limit
    assert not carries(1, hidden, sf, weights, torch.zeros(4))  # EPLB statistics
    assert not carries(1, hidden, None, weights, None)  # no scales
    assert not carries(
        1, torch.empty((tokens, hidden_size), dtype=torch.bfloat16), None, weights, None
    )
    assert carries(1, hidden, sf, weights.bfloat16(), None)  # bf16 weights (TRTLLM-Gen)
    assert not carries(1, hidden, sf, weights.half(), None)  # other 2-byte weights
    comm._sentinel_row, comm.sentinel_max_tokens = None, 0
    assert not carries(1, hidden, sf, weights, None)


# ============================================================================
# Dispatch completion flags for the host watchdog
# ============================================================================


def _run_watchdog_worker(_unused: int) -> dict:
    """Dispatch once with a watchdog armed, and report what it saw."""
    # Import locally so cloudpickle does not serialize torch's dynamic ops namespace.
    import os

    import torch

    # The default policy: a watchdog communicator must keep the sentinel transport off by itself.
    os.environ.pop(SYNC_PROTOCOL_ENV, None)

    from tensorrt_llm._torch.moe.fused_moe.wide_ep_ft import EPGroupHealth

    rank = MPI.COMM_WORLD.Get_rank()
    ep_size = MPI.COMM_WORLD.Get_size()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()

    if not all(MPI.COMM_WORLD.allgather(MnnvlMemory.supports_mnnvl())):
        return {"skip": "NVLink one-sided is not supported on every participating GPU"}

    timeouts: list = []
    mapping = Mapping(rank=rank, world_size=ep_size, tp_size=ep_size, moe_ep_size=ep_size)
    comm = NVLinkOneSided(
        mapping=mapping,
        num_slots=ep_size * 2,
        top_k=1,
        max_num_tokens_per_rank=8,
        hidden_size=512,
        dtype=torch.bfloat16,
        ep_group_health=EPGroupHealth(ep_size),
        alltoall_watchdog_timeout_s=5.0,
        alltoall_watchdog_poll_interval_s=0.02,
        alltoall_watchdog_on_timeout=timeouts.append,
    )
    try:
        num_tokens = 4
        slots = torch.randint(0, ep_size * 2, (num_tokens, 1), dtype=torch.int32, device="cuda")
        payload = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
        scales = torch.ones(num_tokens, 1, dtype=torch.float32, device="cuda")
        comm.dispatch(payload, None, slots, scales, [num_tokens] * ep_size)
        torch.cuda.synchronize()

        # The flags the watchdog polls live in this rank's own workspace row.
        offset = comm.moe_a2a_metainfo[comm.DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX].item()
        flags = (
            comm.workspace[comm.ep_rank, offset : offset + ep_size * 4]
            .view(torch.int32)
            .cpu()
            .tolist()
        )
        flag_val_offset = comm.moe_a2a_metainfo[comm.FLAG_VAL_OFFSET_INDEX].item()
        expected = int(
            comm.workspace[comm.ep_rank, flag_val_offset : flag_val_offset + 4]
            .view(torch.int32)
            .cpu()
            .item()
        )
        MPI.COMM_WORLD.barrier()
        return {
            "rank": rank,
            "flags": flags,
            "expected_flag": expected,
            "timeouts": [t.missing_ranks for t in timeouts],
            "sentinel_max_tokens": comm.sentinel_max_tokens,
        }
    finally:
        comm.destroy()


@pytest.mark.parametrize("ep_size", [2, 4])
def test_dispatch_publishes_watchdog_flags(ep_size: int) -> None:
    """Dispatch must publish completion flags so the watchdog sees live peers.

    The host watchdog observes only the completion flag table. Without the
    flags a healthy collective is reported as every peer missing.
    """
    if torch.cuda.device_count() < ep_size:
        pytest.skip(f"requires {ep_size} GPUs")
    executor = MPIPoolExecutor(ep_size)
    try:
        results = list(executor.map(_run_watchdog_worker, range(ep_size)))
        skipped = [r["skip"] for r in results if "skip" in r]
        if skipped:
            pytest.skip(skipped[0])
        results.sort(key=lambda r: r["rank"])
        for result in results:
            rank = result["rank"]
            expected = result["expected_flag"]
            assert result["flags"] == [expected] * ep_size, (
                f"rank {rank}: dispatch completion flags {result['flags']} did not all "
                f"reach {expected}; the watchdog would report healthy peers as missing"
            )
            assert result["timeouts"] == [], (
                f"rank {rank}: watchdog reported {result['timeouts']} on a healthy collective"
            )
            # Sentinel steps write no completion flags, so the watchdog keeps them off.
            assert result["sentinel_max_tokens"] == 0, f"rank {rank}: sentinel on with a watchdog"

    finally:
        executor.shutdown(wait=True, cancel_futures=True)
