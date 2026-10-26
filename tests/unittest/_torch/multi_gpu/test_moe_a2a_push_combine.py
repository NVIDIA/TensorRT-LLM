# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-GPU tests of the SM107 Push-Combine strategy (NVLinkOneSidedTilePushCombine).

Each rank dispatches NVFP4 activations with DeepSeek-R1 MoE shapes (hidden 7168,
top-8 of 256 experts), runs a synthetic MoE whose output is a function of the
received routing only, and combines. The Push-Combine output is checked against
a torch emulation of its tile-wise packed-NVFP4 record quantizer and against the
stock NVLinkOneSided BF16 combine, over several rounds (exercising the A2A
round-flag/parity protocol) and under CUDA-graph replay.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm as tllm
from tensorrt_llm._mnnvl_utils import MnnvlMemory
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.modules.fused_moe.communication.tile_tma_forced_loader import (
    create_push_combine_strategy,
    get_packaged_push_combine_dso_path,
    resolve_push_combine_dso_path,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

HIDDEN = 7168
TOP_K = 8
NUM_EXPERTS = 256
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expert_table(device):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    return (torch.randn(NUM_EXPERTS, HIDDEN, generator=generator) * 0.25).to(device)


def _make_inputs(num_tokens, seed, rank, device):
    generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + rank)
    x_fp4 = torch.randint(0, 256, (num_tokens, HIDDEN // 2), dtype=torch.uint8, generator=generator)
    x_sf = torch.randint(0, 256, (num_tokens, HIDDEN // 16), dtype=torch.uint8, generator=generator)
    slots = torch.rand(num_tokens, NUM_EXPERTS, generator=generator).argsort(dim=1)[:, :TOP_K]
    scales = torch.rand(num_tokens, TOP_K, generator=generator)
    scales = scales / scales.sum(dim=1, keepdim=True)
    return (
        x_fp4.to(device),
        x_sf.to(device),
        slots.to(torch.int32).contiguous().to(device),
        scales.contiguous().to(device),
    )


def _partial_for_rank(slots, scales, owner_rank, ep_size, table):
    """bf16 sum of the owner rank's experts' contributions, with a shape-independent k order."""
    experts_per_rank = NUM_EXPERTS // ep_size
    lo, hi = owner_rank * experts_per_rank, (owner_rank + 1) * experts_per_rank
    acc = torch.zeros(slots.shape[0], HIDDEN, dtype=torch.float32, device=slots.device)
    for k in range(TOP_K):
        expert = slots[:, k].long()
        weight = ((expert >= lo) & (expert < hi)).float() * scales[:, k].float()
        acc += weight[:, None] * table[expert.clamp(0, NUM_EXPERTS - 1)]
    return acc.to(torch.bfloat16)


def _round_e2m1(values):
    grid = torch.tensor(E2M1_VALUES, device=values.device)
    magnitude = values.abs().clamp(max=6.0)
    distance = (magnitude[..., None] - grid).abs()
    lower = distance.argmin(dim=-1)
    upper = (lower + 1).clamp(max=len(E2M1_VALUES) - 1)
    tie = (distance.gather(-1, upper[..., None]) == distance.gather(-1, lower[..., None])).squeeze(
        -1
    ) & (upper != lower)
    index = torch.where(tie & (lower % 2 == 1), upper, lower)
    return grid[index] * torch.sign(values)


def _quant_dequant_record(x_bf16):
    """Torch emulation of the K1 tile-NVFP4 record quantizer followed by the K4 decode."""
    x = x_bf16.float().view(-1, HIDDEN // 64, 4, 16)
    block_amax = x.abs().amax(dim=-1)
    tile_amax = block_amax.amax(dim=-1, keepdim=True)
    min_amax = torch.tensor(0x06280000, dtype=torch.int32).view(torch.float32).item()
    effective_amax = torch.clamp(tile_amax, min=min_amax)
    multiplier = 2688.0 / effective_amax
    global_dequant = effective_amax / 2688.0
    block_scale = (multiplier * (block_amax / 6.0)).to(torch.float8_e4m3fn).float()
    output_scale = torch.where(
        block_amax != 0, 1.0 / (block_scale / multiplier), torch.zeros_like(block_scale)
    )
    dequant = _round_e2m1(x * output_scale[..., None]) * (block_scale * global_dequant)[..., None]
    dequant = torch.where(tile_amax[..., None] == 0, torch.zeros_like(dequant), dequant)
    return dequant.view(-1, HIDDEN)


def _expected_outputs(slots, scales, ep_size, table):
    """(Push-Combine emulation, NVLinkOneSided expectation), both BF16 [tokens, 7168]."""
    num_tokens = slots.shape[0]
    target_rank = slots.long() // (NUM_EXPERTS // ep_size)
    partials = torch.stack(
        [_partial_for_rank(slots, scales, r, ep_size, table) for r in range(ep_size)]
    )
    dequants = torch.stack([_quant_dequant_record(p) for p in partials])
    rows = torch.arange(num_tokens, device=slots.device)
    seen = torch.zeros(num_tokens, ep_size, dtype=torch.bool, device=slots.device)
    acc_push = torch.zeros(num_tokens, HIDDEN, dtype=torch.float32, device=slots.device)
    acc_ref = torch.zeros_like(acc_push)
    for k in range(TOP_K):
        rank_k = target_rank[:, k]
        first = (~seen.gather(1, rank_k[:, None]).squeeze(1)).float()[:, None]
        seen.scatter_(1, rank_k[:, None], True)
        acc_ref += first * partials[rank_k, rows].float()
        acc_push += first * dequants[rank_k, rows]
    return acc_push.to(torch.bfloat16), acc_ref.to(torch.bfloat16)


def _dispatch_moe_combine(
    strategy, inputs, all_num_tokens, payload_in_workspace, rank, ep_size, table
):
    _, _, recv_slots, recv_scales = strategy.dispatch(
        hidden_states=inputs[0],
        hidden_states_sf=inputs[1],
        token_selected_slots=inputs[2],
        token_final_scales=inputs[3],
        all_rank_num_tokens=all_num_tokens,
    )
    runtime_max = max(all_num_tokens)
    moe_output = _partial_for_rank(recv_slots, recv_scales, rank, ep_size, table)
    strategy.payload_in_workspace = payload_in_workspace
    if payload_in_workspace:
        payload = strategy.get_combine_payload_tensor_in_workspace(
            runtime_max, HIDDEN, torch.bfloat16
        )
        payload.view(-1, HIDDEN).copy_(moe_output.view(-1, HIDDEN))
    else:
        payload = moe_output
    return strategy.combine(payload, all_rank_max_num_tokens=runtime_max)


def _check_outputs(push_output, onesided_output, slots, scales, ep_size, table):
    expected_push, expected_onesided = _expected_outputs(slots, scales, ep_size, table)
    # The stock BF16 combine matches its fp32-accumulate expectation up to the
    # accumulation order (a rare 1-ulp BF16 difference).
    diff = (onesided_output.float() - expected_onesided.float()).norm()
    assert (diff / expected_onesided.float().norm()).item() < 1e-4
    # Push-Combine against its quantizer emulation: the emulation uses exact division where
    # the kernel uses rcp.approx, so a few E2M1 codes may round differently; any routing or
    # slot error would corrupt whole rows instead.
    mismatch = (push_output != expected_push).float()
    assert mismatch.mean().item() < 0.02, mismatch.mean().item()
    assert mismatch.mean(dim=-1).max().item() < 0.05, mismatch.mean(dim=-1).max().item()
    diff = (push_output.float() - expected_push.float()).norm()
    assert (diff / expected_push.float().norm()).item() < 0.02
    # NVFP4 combine against the BF16 combine: NVFP4 quantization error only.
    diff = (push_output.float() - onesided_output.float()).norm()
    assert (diff / onesided_output.float().norm()).item() < 0.15


def _run_push_combine_rank(ep_size, token_counts, rounds, graph_replays, env):
    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda")
    saved_env = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        mapping = Mapping(rank=rank, tp_size=ep_size, moe_ep_size=ep_size, world_size=ep_size)
        max_tokens = max(token_counts)
        push = create_push_combine_strategy(
            dso_path=resolve_push_combine_dso_path(),
            mapping=mapping,
            num_slots=NUM_EXPERTS,
            top_k=TOP_K,
            max_num_tokens_per_rank=max_tokens,
            payload_in_workspace=False,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
        )
        onesided = NVLinkOneSided(
            mapping, NUM_EXPERTS, TOP_K, max_tokens, hidden_size=HIDDEN, dtype=torch.bfloat16
        )
        # Built once, outside any CUDA-graph capture (it is a host-to-device copy).
        table = _expert_table(device)
        seed = 0
        for _ in range(rounds):
            for num_tokens in token_counts:
                for payload_in_workspace in (False, True):
                    seed += 1
                    all_num_tokens = [
                        max(1, num_tokens - r * (num_tokens // 8)) for r in range(ep_size)
                    ]
                    inputs = _make_inputs(all_num_tokens[rank], seed, rank, device)
                    push_output = _dispatch_moe_combine(
                        push, inputs, all_num_tokens, payload_in_workspace, rank, ep_size, table
                    )
                    onesided_output = _dispatch_moe_combine(
                        onesided, inputs, all_num_tokens, payload_in_workspace, rank, ep_size, table
                    )
                    _check_outputs(
                        push_output, onesided_output, inputs[2], inputs[3], ep_size, table
                    )

        # CUDA graph: dispatch + MoE + combine captured once, replayed with fresh inputs.
        num_tokens = token_counts[-1]
        all_num_tokens = [num_tokens] * ep_size
        static_inputs = [t.clone() for t in _make_inputs(num_tokens, 777, rank, device)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _dispatch_moe_combine(push, static_inputs, all_num_tokens, False, rank, ep_size, table)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _dispatch_moe_combine(
                push, static_inputs, all_num_tokens, False, rank, ep_size, table
            )
        for replay in range(graph_replays):
            for dst, src in zip(
                static_inputs, _make_inputs(num_tokens, 1000 + replay, rank, device)
            ):
                dst.copy_(src)
            torch.cuda.synchronize()
            MPI.COMM_WORLD.Barrier()
            graph.replay()
            torch.cuda.synchronize()
            onesided_output = _dispatch_moe_combine(
                onesided, static_inputs, all_num_tokens, False, rank, ep_size, table
            )
            _check_outputs(
                static_output, onesided_output, static_inputs[2], static_inputs[3], ep_size, table
            )
        del graph
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        push.destroy()
        onesided.destroy()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return True


def _push_combine_supported() -> str:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return "Push-Combine kernels target SM107 only"
    dso_path = os.environ.get("TRTLLM_TILE_PUSH_COMBINE_DSO", get_packaged_push_combine_dso_path())
    if not os.path.isfile(dso_path):
        return f"Push-Combine DSO not built: {dso_path}"
    try:
        MnnvlMemory.initialize()
        if not MnnvlMemory.supports_mnnvl():
            return "MNNVL not supported on this system"
    except Exception as error:
        return f"MNNVL not supported on this system: {error}"
    return ""


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs at least 4 GPUs")
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="default"),
        pytest.param(
            {
                "TRTLLM_PUSH_D0_FOLD": "1",
                "TRTLLM_PUSH_K4_V2": "1",
                "TRTLLM_PUSH_K23_PDL": "1",
                "TRTLLM_PUSH_K23_DEADLINE": "1",
            },
            id="fold_k4v2_k23pdl_deadline",
        ),
    ],
)
@pytest.mark.parametrize(
    "mpi_pool_executor,token_counts",
    [
        (4, [1, 64]),
        (4, [256, 1024]),
        (2, [64, 333]),
    ],
    indirect=["mpi_pool_executor"],
)
def test_push_combine_matches_reference(mpi_pool_executor, token_counts, env):
    reason = _push_combine_supported()
    if reason:
        pytest.skip(reason)
    ep_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _run_push_combine_rank,
        *zip(*[(ep_size, token_counts, 2, 3, env)] * ep_size),
    )
    assert all(results)
