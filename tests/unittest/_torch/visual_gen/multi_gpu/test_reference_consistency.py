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

"""Cross-rank consistency of the tp_size == 1 reference models built inside multi-rank workers.

Every visual_gen multi-GPU parity test constructs its "single-GPU" reference on each
rank. With ``TLLM_DISABLE_MPI=1`` a ``Mapping(world_size=1, rank=0, tp_size=1)`` built
there reads its ``tp_rank`` from the live process group, so on ranks >= 1 the
reference's always-ROW ``MLP.down_proj`` drops its bias and the reference differs from
rank to rank (see ``_visual_gen_dist_utils.single_rank_llm_mapping``). The tests here
pin the contract the parity tests rely on: a reference built through
``test_wan_tp._make_model_config(..., tp_size=1)`` is bitwise identical on every rank.

The CPU test runs under gloo and needs no GPU; the NCCL tests need 2 GPUs.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_reference_consistency.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from datetime import timedelta
from typing import Any, Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm.mapping import Mapping

from ._visual_gen_dist_utils import single_rank_llm_mapping, spawn_with_retry
from .test_wan_tp import _WAN_T2V_TEST_CONFIG, _make_model_config, _stabilize_model_weights

# Bounds every collective so a rank that never reaches one (because it failed first)
# surfaces as a test failure within minutes rather than the backend default hang
# (30 min for gloo, 10 min for NCCL).
_PG_TIMEOUT = timedelta(seconds=120)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers
# =============================================================================


def _worker(rank: int, world_size: int, backend: str, test_fn: Callable, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=_PG_TIMEOUT)
    ok = False
    try:
        test_fn(rank, world_size)
        ok = True
    finally:
        if dist.is_initialized():
            # Rendezvous only on success: a failed rank must exit promptly so that
            # mp.spawn tears its peers down instead of every rank waiting on the others.
            if ok:
                dist.barrier()
            dist.destroy_process_group()


def _run_distributed(world_size: int, backend: str, test_fn: Callable) -> None:
    if backend == "nccl" and torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs, have {torch.cuda.device_count()}")
    spawn_with_retry(
        lambda port: mp.spawn(
            _worker,
            args=(world_size, backend, test_fn, port),
            nprocs=world_size,
            join=True,
        )
    )


def _all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


def _all_gather_object(obj: Any) -> list[Any]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def _rank_coordinates(mapping: Mapping) -> tuple[int, int, int]:
    return (mapping.tp_rank, mapping.pp_rank, mapping.cp_rank)


# =============================================================================
# CPU (gloo): the helper's contract, the call site that uses it, and the MLP it protects
# =============================================================================


def _logic_single_rank_mapping_cpu(rank: int, world_size: int) -> None:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.mlp import MLP

    # Product behaviour, recorded for the log but not asserted: this is what the
    # parity tests used to hand to their reference model.
    plain = Mapping(world_size=1, rank=0, tp_size=1)
    print(
        f"rank {rank}: Mapping(world_size=1, rank=0, tp_size=1) -> "
        f"tp_rank={plain.tp_rank} pp_rank={plain.pp_rank} cp_rank={plain.cp_rank}"
    )

    fixed = single_rank_llm_mapping()
    assert isinstance(fixed, Mapping)
    assert fixed.tp_group == [0]
    assert (fixed.tp_size, fixed.pp_size, fixed.cp_size) == (1, 1, 1)
    assert (fixed.world_size, fixed.rank) == (1, 0)

    # The shared MLP is what the Wan / FLUX.1 / LTX-2 references run through; its
    # down_proj is ROW-parallel unconditionally, so this is the bias that goes missing.
    torch.manual_seed(0)
    mlp = MLP(
        hidden_size=8,
        intermediate_size=16,
        bias=True,
        activation=torch.nn.functional.gelu,
        dtype=torch.float32,
        config=ModelConfig(mapping=fixed),
        reduce_output=False,
    )
    # Linear leaves its parameters uninitialised; seed them identically on every rank.
    with torch.no_grad():
        for param in mlp.parameters():
            param.uniform_(-0.1, 0.1)
    torch.manual_seed(1)
    x = torch.randn(2, 8)
    with torch.no_grad():
        out = mlp(x)

    # Rank-dependent facts are gathered before being asserted so that every rank
    # reaches the collectives whatever it observed locally.
    facts = dict(
        helper=_rank_coordinates(fixed),
        # The parity tests reach the helper through their _make_model_config; guard the
        # call site, not only the helper.
        call_site=_rank_coordinates(_make_model_config(_WAN_T2V_TEST_CONFIG, tp_size=1).mapping),
        down_proj_tp_rank=mlp.down_proj.tp_rank,
    )
    expected = dict(helper=(0, 0, 0), call_site=(0, 0, 0), down_proj_tp_rank=0)
    for other_rank, other_facts in enumerate(_all_gather_object(facts)):
        assert other_facts == expected, (
            f"rank {other_rank}: the tp_size == 1 reference must look like rank 0 of a "
            f"1-process job, got {other_facts}"
        )
    for other_rank, other in enumerate(_all_gather(out)):
        assert torch.equal(other, out), f"rank {rank}: MLP output differs from rank {other_rank}"


# =============================================================================
# NCCL (2 GPUs): the tiny Wan T2V reference of test_wan_tp.py
# =============================================================================


def _build_wan_reference(device: torch.device, plain_mapping: bool = False):
    """Build the Wan T2V reference exactly as test_wan_tp.py does on every rank."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    config = _make_model_config(_WAN_T2V_TEST_CONFIG, tp_size=1)
    if plain_mapping:
        # What the parity tests assigned before single_rank_llm_mapping() existed.
        config.mapping = config.visual_gen_mapping.to_llm_mapping()
    torch.manual_seed(123)
    model = WanTransformer3DModel(config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)
    return model


def _wan_reference_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(456)
    return dict(
        hidden_states=torch.randn(1, 16, 2, 4, 4, device=device, dtype=torch.bfloat16) * 0.1,
        timestep=torch.tensor([0.5], device=device, dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(1, 8, 128, device=device, dtype=torch.bfloat16) * 0.1,
    )


def _logic_fixed_reference_identical_across_ranks(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    model = _build_wan_reference(device)
    with torch.no_grad():
        out = model(**_wan_reference_inputs(device))

    tp_ranks = _all_gather_object(model.blocks[0].ffn.down_proj.tp_rank)
    gathered = _all_gather(out)
    assert tp_ranks == [0] * world_size, f"reference down_proj.tp_rank per rank: {tp_ranks}"
    for other_rank, other in enumerate(gathered):
        assert torch.equal(other, out), (
            f"rank {rank}: reference output differs from rank {other_rank} "
            f"(max abs diff {(other.float() - out.float()).abs().max().item():.3e})"
        )


def _logic_plain_mapping_reference_is_rank_dependent(rank: int, world_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    inputs = _wan_reference_inputs(device)
    plain_model = _build_wan_reference(device, plain_mapping=True)
    fixed_model = _build_wan_reference(device)
    with torch.no_grad():
        plain_out = plain_model(**inputs)
        fixed_out = fixed_model(**inputs)

    plain_tp_ranks = _all_gather_object(plain_model.blocks[0].ffn.down_proj.tp_rank)
    gathered = _all_gather(plain_out)
    max_abs_diff = (plain_out.float() - gathered[0].float()).abs().max().item()
    print(
        f"rank {rank}: plain-mapping reference down_proj.tp_rank per rank {plain_tp_ranks}, "
        f"vs rank 0 max abs diff {max_abs_diff:.3e}"
    )
    if all(r == 0 for r in plain_tp_ranks):
        # Nothing left to demonstrate: a world_size=1 Mapping now looks like rank 0
        # everywhere, so the parity tests no longer need single_rank_llm_mapping().
        print(
            f"rank {rank}: Mapping(world_size=1) resolves tp_rank to 0 on every rank; "
            "single_rank_llm_mapping() can be retired"
        )
        return

    assert fixed_model.blocks[0].ffn.down_proj.tp_rank == 0
    assert plain_tp_ranks[0] == 0 and all(r > 0 for r in plain_tp_ranks[1:]), plain_tp_ranks
    if rank == 0:
        # Same tp_rank and same construction: the two models must agree here.
        assert torch.equal(plain_out, fixed_out)
    else:
        assert not torch.equal(plain_out, fixed_out), (
            f"rank {rank}: expected the plain mapping to drop the ROW bias"
        )
        assert not torch.equal(plain_out, gathered[0]), (
            f"rank {rank}: expected the plain-mapping reference to differ from rank 0"
        )


# =============================================================================
# Test classes
# =============================================================================


@pytest.mark.cpu_only
class TestSingleRankMappingCPU:
    def test_single_rank_mapping_cpu(self):
        """The helper, the call site that uses it and a real MLP are rank-independent under gloo."""
        _run_distributed(2, "gloo", _logic_single_rank_mapping_cpu)


class TestWanReferenceConsistency:
    def test_fixed_reference_identical_across_ranks(self):
        """The Wan T2V reference of test_wan_tp.py is bitwise equal on 2 GPUs."""
        _run_distributed(2, "nccl", _logic_fixed_reference_identical_across_ranks)

    def test_plain_mapping_reference_is_rank_dependent(self):
        """The same reference built with a plain Mapping(world_size=1) differs on rank 1.

        Demonstrates the behaviour that motivates the helper. Self-retiring: once
        ``Mapping(world_size=1)`` resolves ``tp_rank`` to 0 on every rank, the test only
        logs that the helper can be retired and passes without asserting a divergence.
        """
        _run_distributed(2, "nccl", _logic_plain_mapping_reference_is_rank_dependent)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
