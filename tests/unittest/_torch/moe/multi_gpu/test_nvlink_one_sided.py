# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVLinkOneSided path selection and model-shaped round trips with independent references."""

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
    FORCE_CFT_ENV,
    NVLinkOneSided,
    _cft_device_support_reason,
    _get_nvidia_driver_version,
    cft_driver_is_supported,
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
    """

    tokens: tuple[int, ...]
    routing: Routing
    delay_rank: int


@dataclass(frozen=True)
class Case:
    """A workload and execution settings sharing one communicator across rounds.

    Attributes:
        model: Expert layout, hidden width, and dispatch payload format.
        rounds: Ordered iterations with no host synchronization between them.
            All token-count tuples must describe the same EP group size.
        mode: Automatic CFT selection, forced fence, or requested CFT; platform
            and payload restrictions still apply to CFT.
        payload_in_workspace: Copy simulated expert outputs into the communication workspace
            before combine, instead of passing an external tensor.
        fp8_combine: Use FP8 on the combine wire; the returned output remains BF16.
        graph: Capture and replay the entire round sequence in one CUDA graph.
        pdl: Enable programmatic dependent launch for communication kernels.
        eplb: Gather and verify expert-load statistics alongside dispatch.
    """

    model: ModelShape
    rounds: tuple[Round, ...]
    mode: Literal["auto", "fence", "cft"]
    payload_in_workspace: bool
    fp8_combine: bool
    graph: bool
    pdl: bool
    eplb: bool

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
    os.environ[FORCE_CFT_ENV] = {"auto": "", "fence": "0", "cft": "1"}[case.mode]

    rank = MPI.COMM_WORLD.Get_rank()
    torch.cuda.set_device(rank)
    MnnvlMemory.initialize()
    supported = MnnvlMemory.supports_mnnvl()
    cft_reason = None
    if case.mode != "fence":
        if not cft_driver_is_supported(_get_nvidia_driver_version()):
            cft_reason = "CFT requires driver 615 or newer"
        else:
            cft_reason = _cft_device_support_reason()
    sm_major = torch.cuda.get_device_capability()[0]
    quant_supported = (
        case.model.dispatch_dtype == "bf16"
        or (case.model.dispatch_dtype == "blockwise_fp8" and sm_major >= 9)
        or sm_major >= 10
    )
    reasons = MPI.COMM_WORLD.allgather((supported, cft_reason, quant_supported))
    if not all(item[0] for item in reasons):
        return {"skip": "NVLink one-sided is not supported on every participating GPU"}
    if case.mode == "cft" and any(item[1] for item in reasons):
        return {"skip": str(reasons)}
    if not all(item[2] for item in reasons):
        return {
            "skip": f"{case.model.dispatch_dtype} payload generation is unsupported on a participating GPU"
        }
    if case.mode == "auto" and len({item[1] is None for item in reasons}) != 1:
        return {"skip": "automatic CFT selection requires consistent capability across ranks"}

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
    )
    try:
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
            # Record the actual op argument, not just the wrapper's capability.
            launches.append(("dispatch", bool(args[10])))
            result = dispatch_op(*args, **kwargs)
            combine_offsets.append(result[1])
            return result

        def record_combine(*args, **kwargs):
            launches.append(("combine", bool(args[11])))
            return combine_op(*args, **kwargs)

        def sequence() -> list[dict]:
            outputs = []
            for i, (spec, tensors) in enumerate(zip(case.rounds, inputs, strict=True)):
                payload, sf, slots, weights = tensors
                counts = list(spec.tokens)
                comm.prepare_dispatch(slots, counts)
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
            "cft_capable": comm.can_use_cft_counted_writes,
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
        expected_launches = []
        for i, spec in enumerate(case.rounds):
            payload, sf, slots, weights = result["inputs"][i]
            verify_combine(case, results, rank, i)
            if result["outputs"][i]["dispatch"] is not None:
                verify_dispatch(case, results, rank, i)
            requested = (
                result["cft_capable"]
                and case.mode != "fence"
                and (case.mode == "cft" or max(spec.tokens) <= 128)
            )
            # Calculate wire eligibility independently from the production helper.
            payloads = (payload, slots, weights) if sf is None else (payload, sf, slots, weights)
            dispatch_cft = requested and all(
                t.shape[1] * t.element_size() % 16 == 0 for t in payloads
            )
            combine_cft = (
                requested and case.model.hidden_size * (1 if case.fp8_combine else 2) % 16 == 0
            )
            expected_launches.extend((("dispatch", dispatch_cft), ("combine", combine_cft)))
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
    # CFT mode is Python-side and is set on every worker before constructing the comm.
    env = {
        "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH": "128",
        "TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE": "128",
        "TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB": "",
        "TRTLLM_ENABLE_PDL": "1" if case.pdl else "0",
        "TRTLLM_NVLINK_ONE_SIDED_A2A_TIMEOUT_SEC": "30",
        "TRTLLM_NVLINK_ONE_SIDED_A2A_WARMUP_TIMEOUT_SEC": "60",
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
    # including zero-token ranks and delayed ranks. Counts 128/129 straddle the
    # automatic CFT threshold; graph cases replay the entire sequence.
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
            ),
            id=f"round-sequence-ep4-{mode}-{'graph' if graph else 'eager'}",
        )
        for mode, graph in (
            ("auto", False),
            ("auto", True),
            ("cft", True),
            ("fence", True),
        )
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


@pytest.mark.parametrize("cft,fp8_combine", [(False, False), (True, False), (True, True)])
def test_workspace_layout(cft: bool, fp8_combine: bool) -> None:
    from tensorrt_llm.bindings import internal as _tllm_internal

    thop = _tllm_internal.thop

    def field(layout: torch.Tensor, name: str) -> int:
        return int(layout[getattr(thop, f"MOE_A2A_{name}")])

    ep_size, top_k, capacity, hidden_size = 8, 6, 128, 7168
    layout = NVLinkOneSided._make_workspace_layout(
        ep_size, top_k, capacity, hidden_size, torch.bfloat16, None, 0, cft, fp8_combine
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
    assert total == NVLinkOneSided.calculate_required_workspace_size(
        ep_size,
        top_k,
        capacity,
        hidden_size,
        torch.bfloat16,
        can_use_cft_counted_writes=cft,
        use_low_precision_combine=fp8_combine,
    )
    if cft:
        recv_start = field(layout, "COMBINE_RECV_OFFSET_INDEX")
        recv_bytes = field(layout, "COMBINE_RECV_SIZE_INDEX")
        assert combine_end <= recv_start and recv_start % 256 == 0
        assert recv_bytes == ep_size * capacity * hidden_size * (1 if fp8_combine else 2)
        assert recv_start + recv_bytes == total
        assert (
            dispatch_end
            <= field(layout, "COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX")
            < combine_start
        )
    else:
        assert field(layout, "COMBINE_RECV_OFFSET_INDEX") == 0
        assert field(layout, "COMBINE_RECV_SIZE_INDEX") == 0
        assert field(layout, "COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX") == 0
        assert combine_end == total

    # Routing metadata reserves the configured top-k, not the maximum supported top-k.
    routes_start = field(layout, "TOPK_TARGET_RANKS_OFFSET_INDEX")
    indices_start = field(layout, "TOPK_TARGET_INDICES_OFFSET_INDEX")
    assert indices_start - routes_start == capacity * top_k * 4
    with pytest.raises(RuntimeError, match="capacity"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(ep_size, capacity, top_k, -1, 256, 0)
    with pytest.raises(RuntimeError, match="overflows"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size, capacity, top_k, (1 << 63) - 512, 1024, 0
        )


# ============================================================================
# CFT path selection and capability checks
# ============================================================================


@pytest.mark.parametrize(
    ("force_env", "driver_version", "num_tokens", "device_supported", "expected"),
    [
        pytest.param(None, "615.00", 128, True, True, id="auto-at-threshold"),
        pytest.param(None, "615.00", 129, True, False, id="auto-above-threshold"),
        pytest.param("0", "615.00", 128, True, False, id="force-fence"),
        pytest.param("1", "615.00", 129, True, True, id="force-cft-above-threshold"),
        pytest.param("1", b"614.99", 128, True, False, id="force-cft-old-driver"),
        pytest.param("1", None, 128, True, False, id="force-cft-nvml-error"),
        pytest.param("1", "615.00", 128, False, False, id="force-cft-unsupported-device"),
        pytest.param("invalid", "615.00", 129, True, False, id="invalid-env-uses-auto"),
    ],
)
def test_cft_selection(
    monkeypatch: pytest.MonkeyPatch,
    force_env: str | None,
    driver_version: str | bytes | None,
    num_tokens: int,
    device_supported: bool,
    expected: bool,
) -> None:
    from tensorrt_llm._torch.modules.fused_moe.communication import nvlink_one_sided

    if force_env is None:
        monkeypatch.delenv(FORCE_CFT_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_CFT_ENV, force_env)

    def query_driver_version() -> str | bytes:
        if driver_version is None:
            raise nvlink_one_sided.pynvml.NVMLError(nvlink_one_sided.pynvml.NVML_ERROR_UNKNOWN)
        return driver_version

    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvlink_one_sided.pynvml, "nvmlSystemGetDriverVersion", query_driver_version)
    force_cft = nvlink_one_sided.get_force_cft()
    version = nvlink_one_sided._get_nvidia_driver_version()
    assert version == (
        driver_version.decode() if isinstance(driver_version, bytes) else driver_version
    )
    can_use_cft = (
        nvlink_one_sided.resolve_cft_counted_writes(force_cft, version) and device_supported
    )
    assert nvlink_one_sided.should_use_cft(can_use_cft, force_cft, 128, num_tokens) is expected


@pytest.mark.parametrize(
    ("capability", "unsupported_index"),
    [
        pytest.param((9, 0), None, id="hopper"),
        pytest.param((10, 3), None, id="supported"),
        pytest.param((10, 3), 0, id="no-fabric-handle"),
        pytest.param((10, 3), 1, id="no-unicast-endpoint"),
        pytest.param((10, 3), 2, id="no-counted-ops"),
    ],
)
def test_cft_device_support(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    unsupported_index: int | None,
) -> None:
    from tensorrt_llm._torch.modules.fused_moe.communication import nvlink_one_sided

    cuda = nvlink_one_sided.cuda
    attributes = (
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
    )
    unsupported = None if unsupported_index is None else attributes[unsupported_index]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda,
        "cuDeviceGetAttribute",
        lambda attribute, device: (cuda.CUresult.CUDA_SUCCESS, int(attribute != unsupported)),
    )
    reason = nvlink_one_sided._cft_device_support_reason()
    if capability[0] < 10:
        assert "SM90" in reason
    elif unsupported is None:
        assert reason is None
    else:
        assert unsupported.name in reason
