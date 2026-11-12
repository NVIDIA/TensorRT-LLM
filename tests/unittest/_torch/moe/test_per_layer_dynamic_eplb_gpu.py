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
"""Per-layer dynamic EPLB on MegaMoE CuteDSL, end to end on four GPUs.

Every forward runs the whole path: HALO-Q exchanges the routing counts, plans
the hot experts, copies their weights into helper slots with an in-switch
multicast and remaps the routes, and MegaMoE computes on the remapped slots.
Three quarters of the tokens pick the experts of one rank, so the plan must
use helper slots.  Three forwards cover a fresh plan, the same plan again with
its helpers already resident, and a different plan.  A second case removes the
pair team objects, so an owner outside a rank pair writes its share of the
copy with unicast stores, as at EP16.  After each forward every rank checks

- the output against the NVFP4 reference MoE;
- every route: a home slot must be the expert's own, and a helper slot must
  hold a byte-exact copy of the expert's weights in every plane;
- that some route used a helper slot.

The path needs two or more EP ranks with NVLink multicast, so one GPU cannot
run it; the planner alone is tested on one GPU by moeRebalanceHaloQTest.
"""

import os
import pickle
import sys

import cloudpickle
import pytest
import torch
import torch.distributed as dist
from _torch.moe.moe_test_utils import MoeBackendType, MoeModelConfig, should_skip_megamoe_cutedsl
from _torch.moe.quantize_utils import get_test_quant_params
from cuda.bindings import driver as cuda
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancerIterContext,
    maybe_create_moe_load_balancer,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
from tensorrt_llm._utils import is_sm_100f, mpi_comm, mpi_rank
from tensorrt_llm.llmapi.llm_args import MoePerLayerDynamicLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

EP_SIZE = 4
MODEL = MoeModelConfig(num_experts=8, top_k=2, hidden_size=512, intermediate_size=512)
HELPER_SLOTS_PER_RANK = 2
TOKENS_PER_RANK = 128
# The rank whose experts the skewed tokens pick, per forward.
HOT_RANKS = (0, 0, 1)


def _skip_reason():
    if torch.cuda.device_count() < EP_SIZE:
        return f"needs {EP_SIZE} GPUs"
    if not is_sm_100f():
        return "MegaMoE CuteDSL runs on SM10x only"
    reason = should_skip_megamoe_cutedsl(
        MoeBackendType.MEGAMOE_CUTEDSL,
        quant_algo=QuantAlgo.NVFP4,
        dtype=torch.bfloat16,
        model_config=MODEL,
    )
    if reason:
        return reason
    # Helper slots are fabric-handle memory bound to an NVLink multicast object.
    for attribute in (
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
    ):
        result, supported = cuda.cuDeviceGetAttribute(attribute, 0)
        if result != cuda.CUresult.CUDA_SUCCESS or not supported:
            return f"needs {attribute.name}"
    return None


def _init_worker(custom_paths):
    # Per-layer dynamic EPLB needs the larger launch queues from CUDA start.
    os.environ["CUDA_SCALE_LAUNCH_QUEUES"] = "4x"
    for custom_path in custom_paths:
        if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
            sys.path.append(custom_path)


def _init_process_group(rank, world_size):
    """Join the NCCL WORLD group, which MegaMoE resolves as its EP group."""
    master_addr = "127.0.0.1"
    store = None
    if rank == 0:
        # Bind before publishing the OS-selected port.
        store = dist.TCPStore(
            host_name=master_addr,
            port=0,
            world_size=world_size,
            is_master=True,
            wait_for_workers=False,
        )
    master_port = mpi_comm().bcast(store.port if store is not None else None, root=0)
    if store is None:
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=False,
            wait_for_workers=False,
        )
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size)


def _record_routes(layer):
    """Record the logical experts and physical slots of each route the layer plans."""
    records = []
    route = layer.route

    def recording_route(token_selected_experts, *args, **kwargs):
        physical = route(token_selected_experts, *args, **kwargs)
        records.append((token_selected_experts, physical))
        return physical

    layer.route = recording_route
    return records


def _gather_slot_rows(plane, world_size):
    """Return every rank's slot rows of a weight plane as bytes, rank-major."""
    rows = torch.stack([row.contiguous().reshape(-1).view(torch.uint8) for row in plane])
    gathered = rows.new_empty((world_size * rows.shape[0], rows.shape[1]))
    dist.all_gather_into_tensor(gathered, rows)
    return gathered


def _check_routes(layer, records, world_size):
    """Check the last forward's routes against the slot weights; return the errors.

    Collective: it gathers the weight planes and sums the helper routes.
    """
    errors = []
    if layer.replica_plan is not None:
        errors.append("the replica plan was not released")
    if len(records) != 1:
        errors.append(f"expected one planned route, got {len(records)}")
    if records:
        logical, physical = (t.flatten().to(torch.int64) for t in records[-1])
    else:
        logical = physical = torch.empty(0, dtype=torch.int64, device="cuda")
    records.clear()

    arena = layer.live_weight_arena
    home, slots = arena.home_experts, arena.slot_count
    in_range = (physical >= 0) & (physical < world_size * slots)
    if not in_range.all():
        errors.append(f"{int((~in_range).sum())} routes fall outside the physical slots")
    # The rank-major row of each expert's home slot.
    home_row = logical // home * slots + logical % home
    is_home = in_range & (physical % slots < home)
    if (is_home & (physical != home_row)).any():
        errors.append("a route reaches a home slot of another expert")
    is_helper = in_range & (physical % slots >= home)
    pairs = torch.unique(torch.stack([physical[is_helper], home_row[is_helper]], dim=1), dim=0)
    for name, plane in zip(arena.plane_names, arena.tekit_alias_views):
        rows = _gather_slot_rows(plane, world_size)
        if (rows[pairs[:, 0]] != rows[pairs[:, 1]]).any():
            errors.append(f"a helper slot read by a route holds another {name}")

    helper_routes = is_helper.sum().reshape(1)
    dist.all_reduce(helper_routes)
    if helper_routes.item() == 0:
        errors.append("no route used a helper slot")
    return errors


def _run_rank(world_size: int, team_objects: bool) -> tuple[int, list[str], bool]:
    """Run the forwards on this rank and return its errors and shutdown result."""
    if not team_objects:
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler import geometry

        geometry.MAX_PAIR_TEAM_EP = 2
    rank = mpi_rank()
    _init_process_group(rank, world_size)
    # The test utilities create weights and scales on the default device.
    with torch.device(f"cuda:{rank}"):
        errors, shut_down = _run_forwards(rank, world_size)
    return rank, errors, shut_down


def _run_forwards(rank, world_size):
    dtype = torch.bfloat16
    num_experts = MODEL.num_experts
    home = num_experts // world_size

    # All ranks draw the same tensors, so they build identical weights with one
    # activation scale; each rank then takes its own tokens.
    torch.manual_seed(0)
    all_x = torch.randn(
        (world_size, TOKENS_PER_RANK, MODEL.hidden_size), dtype=dtype, device="cuda"
    )
    all_logits = torch.randn((world_size, TOKENS_PER_RANK, num_experts), dtype=dtype, device="cuda")
    x = all_x[rank]

    routing_method = RenormalizeMoeRoutingMethod(top_k=MODEL.top_k, force_enable_pytorch_op=True)
    quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
        QuantAlgo.NVFP4, all_x, MoeBackendType.MEGAMOE_CUTEDSL
    )
    ref_cls = quant_kwargs.pop("ref_cls")
    quantize_util = quantize_util_cls(
        num_experts=num_experts,
        dtype=dtype,
        intermediate_size=MODEL.intermediate_size,
        hidden_size=MODEL.hidden_size,
        quant_config=quant_config,
        num_local_experts=home,
    )
    weights = quantize_util.create_weights(**quant_kwargs)

    mapping = Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    mapping.rank = rank
    pretrained_config = PretrainedConfig()
    # Any architecture with EPLB support; this one adds no MoE specifics.
    pretrained_config.architectures = ["Qwen3MoeForCausalLM"]
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = MODEL.hidden_size
    pretrained_config.intermediate_size = MODEL.intermediate_size
    pretrained_config.torch_dtype = dtype
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        moe_load_balancer=MoePerLayerDynamicLoadBalancerConfig(
            num_slots=num_experts + world_size * HELPER_SLOTS_PER_RANK
        ),
        max_num_tokens=256,
    )

    errors = []
    with (
        # Outputs stay live through the checks.
        nccl_window_tensor_scope(x),
        maybe_create_moe_load_balancer(model_config, mapping) as balancer,
        create_moe(
            routing_method=routing_method, reduce_results=True, model_config=model_config
        ) as moe,
    ):
        moe.create_weights()
        moe.load_weights([weights])
        moe.post_load_weights()
        (layer,) = balancer.single_layer_load_balancers
        records = _record_routes(layer)
        ref_moe = quantize_util.create_ref_module(routing_method, ref_cls=ref_cls)
        ref_moe.load_weights([weights])
        ref_moe.cuda(f"cuda:{rank}")

        for step, hot_rank in enumerate(HOT_RANKS):
            logits = all_logits[rank].clone()
            logits[: TOKENS_PER_RANK * 3 // 4, hot_rank * home : (hot_rank + 1) * home] += 8
            with torch.inference_mode():
                with MoeLoadBalancerIterContext(balancer):
                    output = moe.forward(
                        x, logits, all_rank_num_tokens=[TOKENS_PER_RANK] * world_size
                    )
                ref_output = ref_moe.forward(x, logits)
            # Peers write this rank's helper slots, so wait for every rank.
            torch.cuda.synchronize()
            dist.barrier()
            try:
                ref_moe.check_accuracy(output, ref_output)
            except AssertionError as error:
                errors.append(f"forward {step}: output: {error}")
            errors += [
                f"forward {step}: {error}" for error in _check_routes(layer, records, world_size)
            ]
        shut_down = balancer.shutdown()
    return errors, shut_down


@pytest.mark.threadleak(enabled=False)  # the MPIPoolExecutor manager thread outlives the test
@pytest.mark.parametrize("team_objects", [True, False], ids=["pair_teams", "pair_unicast"])
def test_per_layer_dynamic_eplb_megamoe_cutedsl(team_objects: bool) -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    with MPIPoolExecutor(
        initializer=_init_worker, initargs=(sys.path,), max_workers=EP_SIZE
    ) as executor:
        results = list(executor.map(_run_rank, [EP_SIZE] * EP_SIZE, [team_objects] * EP_SIZE))
    for rank, errors, shut_down in sorted(results):
        assert not errors, f"rank {rank}: " + "; ".join(errors)
        assert shut_down, f"rank {rank}: per-layer dynamic EPLB did not shut down"
