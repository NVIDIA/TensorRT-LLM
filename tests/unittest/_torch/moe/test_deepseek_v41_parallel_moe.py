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
"""DS-V4.1 native EP: routing, quantized GEMMs and MPI collectives.

Run ordinary pytest to spawn local workers, or launch one pytest process per MPI
rank for multiple nodes. External MPI uses node-major ranks and all local GPUs
visible to each process; select cases matching the MPI world size. No checkpoint
is needed: every rank loads the same unsharded MXFP4 weights and compares against
unsharded quantized expert MLPs.
"""

import os
import pickle
import sys
from collections.abc import Iterator

import cloudpickle
import pytest
import torch
import torch.nn.functional as F
from _torch.moe.quantize_utils import MXFP4MXFP8QuantizeUtil
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4Gate
from tensorrt_llm._torch.moe.fused_moe import BaseMoeRoutingMethod, SwigluActivation, create_moe
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._utils import get_sm_version, mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_NUM_EXPERTS = 384
_TOP_K = 6
_HIDDEN_SIZE = 5120
_INTERMEDIATE_SIZE = 2304
_CLAMP = 10.0
_ROUTED_SCALE = 1.5

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


class _ReferenceRouting(BaseMoeRoutingMethod):
    def __init__(self, bias: torch.Tensor) -> None:
        super().__init__()
        self.bias = bias
        self.top_k = _TOP_K

    def apply(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.softplus(router_logits.float()).sqrt()
        indices = (scores + self.bias).topk(_TOP_K, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * _ROUTED_SCALE
        return indices.to(torch.int32), weights


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths
    os.environ["TRTLLM_FORCE_COMM_METHOD"] = "ALLGATHER"
    os.environ["ENABLE_PERFECT_ROUTER"] = "0"
    # Keep torch backend proxies out of the initializer's pickle.
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False


def _local_mpi_topology() -> tuple[int, int]:
    local_comm = mpi_comm().Split_type(MPI.COMM_TYPE_SHARED)
    try:
        local_rank, local_size = local_comm.Get_rank(), local_comm.Get_size()
    finally:
        local_comm.Free()
    assert len(set(mpi_comm().allgather(local_size))) == 1, "Requires equal ranks per node"
    assert mpi_rank() % local_size == local_rank, "Requires node-major MPI rank placement"
    return local_rank, local_size


@pytest.fixture(scope="module")
def parallel_moe_executor(request: pytest.FixtureRequest) -> Iterator[MPIPoolExecutor | None]:
    world_size = request.param
    if mpi_world_size() > 1:
        if mpi_world_size() != world_size:
            pytest.skip(f"Requires MPI world size {world_size}, got {mpi_world_size()}")
        local_rank, local_size = _local_mpi_topology()
        reason = None
        if not torch.cuda.is_available() or torch.cuda.device_count() < local_size:
            reason = f"Requires {local_size} visible CUDA GPUs per node"
        else:
            torch.cuda.set_device(local_rank)
            if get_sm_version() not in (100, 103):
                reason = "DS-V4.1 MXFP4/MXFP8 regression requires Blackwell SM100/SM103"
        reasons = mpi_comm().allgather(reason)
        if any(reasons):
            pytest.skip(f"MPI GPU requirements not met: {reasons}")
        with pytest.MonkeyPatch.context() as settings:
            settings.setenv("TRTLLM_FORCE_COMM_METHOD", "ALLGATHER")
            settings.setenv("ENABLE_PERFECT_ROUTER", "0")
            settings.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
            yield None
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} visible CUDA GPUs")
    if get_sm_version() not in (100, 103):
        pytest.skip("DS-V4.1 MXFP4/MXFP8 regression requires Blackwell SM100/SM103")
    if world_size == 1:
        with pytest.MonkeyPatch.context() as settings:
            settings.setenv("TRTLLM_FORCE_COMM_METHOD", "ALLGATHER")
            settings.setenv("ENABLE_PERFECT_ROUTER", "0")
            settings.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
            yield None
        return
    with MPIPoolExecutor(
        max_workers=world_size, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        if not hasattr(executor, "num_workers"):
            executor.num_workers = world_size
        yield executor


def _run_native_ep() -> None:
    rank = mpi_rank()
    world_size = mpi_world_size()
    local_rank, local_size = _local_mpi_topology()
    torch.cuda.set_device(local_rank)
    supported = mpi_comm().allgather(NVLinkOneSided.is_platform_supported())
    if not all(supported):
        pytest.skip(f"Requires NVLink/MNNVL support on every EP rank: {supported}")
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=local_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
    model_config = ModelConfig(
        pretrained_config=PretrainedConfig(
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=_INTERMEDIATE_SIZE,
            num_experts=_NUM_EXPERTS,
            torch_dtype=torch.bfloat16,
        ),
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="CUTLASS",
        max_num_tokens=256,
    )

    with (
        torch.device(f"cuda:{local_rank}"),
        torch.inference_mode(),
        pytest.MonkeyPatch.context() as env,
    ):
        env.setenv("TRTLLM_FORCE_COMM_METHOD", "NVLINK_ONE_SIDED")
        torch.manual_seed(1701)
        gate = DeepseekV4Gate(
            _HIDDEN_SIZE,
            _NUM_EXPERTS,
            top_k=_TOP_K,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=_ROUTED_SCALE,
            is_hashed=False,
            dtype=torch.bfloat16,
        )
        gate.weight.copy_(torch.randn_like(gate.weight) / _HIDDEN_SIZE**0.5)
        gate.e_score_correction_bias.copy_(torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05)
        reference_routing = _ReferenceRouting(gate.e_score_correction_bias)
        quantizer = MXFP4MXFP8QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=quant_config,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=_CLAMP,
        )
        weights = quantizer.create_weights(input_hidden_alignment=128)
        reference = quantizer.create_ref_module(reference_routing)
        reference.load_weights([weights])

        with create_moe(
            routing_method=gate.routing_method,
            model_config=model_config,
            reduce_results=True,
            activation=SwigluActivation(clamp=_CLAMP),
        ) as moe:
            assert isinstance(moe.backend, CutlassFusedMoE)
            moe.load_weights([weights])
            moe.post_load_weights()
            del weights
            assert isinstance(moe.comm, NVLinkOneSided)

            token_counts = [list(range(1, world_size + 1)), [0, *range(2, world_size + 1)]]
            for counts in token_counts:
                torch.manual_seed(1702)
                global_x = torch.randn((sum(counts), _HIDDEN_SIZE), dtype=torch.bfloat16)
                global_selected, _ = reference_routing.apply(
                    F.linear(global_x.float(), gate.weight.float())
                )
                assert set(
                    (global_selected // (_NUM_EXPERTS // world_size)).flatten().tolist()
                ) == set(range(world_size))
                start = sum(counts[:rank])
                x = global_x[start : start + counts[rank]]
                # Final-window replay can leave a peer empty before the gate.
                logits = gate(x)
                reference_logits = F.linear(x.float(), gate.weight.float())
                torch.testing.assert_close(logits, reference_logits, rtol=1e-4, atol=1e-4)
                if x.shape[0]:
                    selected, scales = gate.routing_method.apply(logits)
                    expected_selected, expected_scales = reference_routing.apply(reference_logits)
                    torch.testing.assert_close(selected, expected_selected, rtol=0, atol=0)
                    torch.testing.assert_close(scales, expected_scales, rtol=1e-4, atol=1e-5)

                expected = reference(x, reference_logits)
                actual = moe(x, logits, all_rank_num_tokens=counts)
                torch.cuda.synchronize()
                assert actual.shape == expected.shape == x.shape
                assert torch.isfinite(actual).all()
                # Runtime fallback must not turn native EP coverage into AllGather coverage.
                assert isinstance(moe.comm, NVLinkOneSided)
                returned_counts = mpi_comm().allgather(actual.shape[0])
                assert returned_counts == counts
                assert moe.comm._dispatch_state["phase"] == "idle"
                if rank == 0 and 0 in counts:
                    print(
                        f"Idle-rank NVLinkOneSided collective completed: input={counts}, "
                        f"output={returned_counts}",
                        flush=True,
                    )
                if x.numel():
                    reference.check_accuracy(actual, expected)


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize(
    "parallel_moe_executor",
    [pytest.param(2, id="world2"), pytest.param(8, id="world8")],
    indirect=True,
)
def test_deepseek_v41_native_ep(parallel_moe_executor: MPIPoolExecutor | None) -> None:
    """Pure EP on one NVLink fabric, including cross-node MNNVL for eight ranks."""
    if parallel_moe_executor is None:
        _run_native_ep()
        return
    results = [
        parallel_moe_executor.submit(_run_native_ep)
        for _ in range(parallel_moe_executor.num_workers)
    ]
    assert all(result.result() is None for result in results)
