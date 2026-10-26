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
"""Model construction sizes and registers the MoE layers that DWDP manages (real ConfigurableMoE).

``create_moe``, which every model calls for its MoE layers, builds real ``ConfigurableMoE`` layers
while a real ``DwdpManager`` is entered, as ``py_executor_creator`` enters it before the model is
built.  Each layer decides once its backend exists (``ConfigurableMoE._decide_dwdp``): an NVFP4
layer on the CuTeDSL backend stores only this rank's range of the expert partition, sized before
the backend sync and ``create_weights``, and is registered with the manager; any other layer, such
as a BF16 MTP layer of an NVFP4 checkpoint, keeps every expert and logs one warning.  The backend
resolution needs a GPU of SM100, SM103 or SM107, for the CuTeDSL NVFP4 backend; the weights are
created on the host and no kernel runs.  Only the MPI world under the DWDP group is a stand-in.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import set_global_dense_dwdp_manager
from tensorrt_llm._torch.moe.fused_moe import (
    ConfigurableMoE,
    RenormalizeMoeRoutingMethod,
    configurable_moe,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.cutedsl import TrtllmCutedslGroupedGemmNvfp4Impl
from tensorrt_llm._torch.pyexecutor.dwdp import (
    DwdpManager,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not TrtllmCutedslGroupedGemmNvfp4Impl.sm_support.accepts(get_sm_version()),
    reason="the CuTeDSL NVFP4 MoE backend, which DWDP manages, does not run on this GPU",
)

# Ten routed experts over a DWDP group of four: the balanced partition stores 3, 3, 2 and 2.
NUM_EXPERTS, DWDP_SIZE, HIDDEN, INTERMEDIATE = 10, 4, 256, 256
RANGES = [(0, 3), (3, 6), (6, 8), (8, 10)]
NVFP4_LAYER, BF16_LAYER = 3, 7


@pytest.fixture(autouse=True)
def _no_registered_managers() -> Iterator[None]:
    """Every test starts and ends without a registered DWDP manager."""
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)
    yield
    set_global_dwdp_manager(None)
    set_global_dense_dwdp_manager(None)


def _manager(dwdp_rank: int) -> tuple[DwdpManager, Mapping]:
    """A real DwdpManager of rank ``dwdp_rank``, with the balanced partition, and its mapping.

    The mapping is the one ``py_executor_creator`` gives a tp-1 context worker of the group.  Only
    the MPI world, from which the manager creates the DWDP group, is a stand-in.
    """
    mapping = Mapping(world_size=1, rank=0, tp_size=1, dwdp_size=DWDP_SIZE, dwdp_rank=dwdp_rank)
    with (
        patch("tensorrt_llm._torch.pyexecutor.dwdp.COMM_WORLD") as world,
        patch("tensorrt_llm._torch.pyexecutor.dwdp.dwdp_global_rank", return_value=dwdp_rank),
    ):
        world.Get_size.return_value = DWDP_SIZE
        manager = DwdpManager(
            config=DwdpConfig(dwdp_size=DWDP_SIZE), dist=MagicMock(spec=MPIDist), mapping=mapping
        )
    return manager, mapping


def _create_layer(
    model_config: ModelConfig, layer_idx: int, quant_config: QuantConfig | None
) -> ConfigurableMoE:
    """A MoE layer as a model builds it: ``create_moe`` with the model's config."""
    moe = create_moe(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        dtype=torch.bfloat16,
        model_config=model_config,
        override_quant_config=quant_config,
        layer_idx=layer_idx,
    )
    assert isinstance(moe, ConfigurableMoE)
    return moe


def _layout(module: torch.nn.Module) -> tuple[int, int, int, int, list[int]]:
    """The expert layout of a ConfigurableMoE layer or its backend.

    ``(num_slots, expert_size_per_partition, slot_start, slot_end, initial_local_expert_ids)``.
    """
    return (
        module.num_slots,
        module.expert_size_per_partition,
        module.slot_start,
        module.slot_end,
        list(module.initial_local_expert_ids),
    )


def _expert_rows(backend: torch.nn.Module) -> set[int]:
    """The leading dimensions of the backend's per-expert weights and scales."""
    return {param.shape[0] for param in backend.parameters(recurse=False) if param.dim() > 0}


@pytest.mark.parametrize("dwdp_rank", range(DWDP_SIZE))
def test_model_construction_sizes_and_registers_only_the_layers_dwdp_manages(
    dwdp_rank: int,
) -> None:
    """The NVFP4 layer stores and registers its rank's range; the BF16 layer keeps every expert.

    The wrapper's and the backend's layout and the rows of every expert parameter show that the
    layer was sized before the backend sync and ``create_weights``.
    """
    start, end = RANGES[dwdp_rank]
    manager, mapping = _manager(dwdp_rank)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN
    pretrained_config.intermediate_size = INTERMEDIATE
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16),
        moe_backend="CUTEDSL",
    )
    warnings = []
    with manager, patch.object(configurable_moe.logger, "warning", side_effect=warnings.append):
        managed = _create_layer(model_config, NVFP4_LAYER, None)
        # A BF16 layer of the same NVFP4 model, as a BF16 MTP layer is built.
        unmanaged = _create_layer(model_config, BF16_LAYER, QuantConfig())
        registered = list(manager._registered_layers)
    assert get_global_dwdp_manager() is None

    assert isinstance(managed.backend, TrtllmCutedslGroupedGemmNvfp4Impl)
    assert managed.enable_dwdp
    sized = (NUM_EXPERTS, end - start, start, end, list(range(start, end)))
    assert _layout(managed) == _layout(managed.backend) == sized
    assert _expert_rows(managed.backend) == {end - start}

    assert not unmanaged.enable_dwdp
    every_expert = (NUM_EXPERTS, NUM_EXPERTS, 0, NUM_EXPERTS, list(range(NUM_EXPERTS)))
    assert _layout(unmanaged) == _layout(unmanaged.backend) == every_expert
    assert _expert_rows(unmanaged.backend) == {NUM_EXPERTS}

    assert registered == [NVFP4_LAYER]
    dwdp_warnings = [message for message in warnings if message.startswith("[DWDP]")]
    assert dwdp_warnings == [
        f"[DWDP] MoE layer {BF16_LAYER} ({type(unmanaged.backend).__name__}) keeps all "
        f"{NUM_EXPERTS} experts on every rank: DWDP does not manage it, since "
        f"{type(unmanaged.backend).__name__} does not support DWDP."
    ]
