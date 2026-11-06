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
"""The llm_args the layer-wise benchmark Runner hands to ModelLoader.

No GPU and no checkpoint: ModelLoader is replaced by a stub that records its
arguments and stops construction before any model is built.
"""

import pytest

from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig, MoePerLayerDynamicLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.tools.layer_wise_benchmarks import runner as runner_module

pytestmark = pytest.mark.cpu_only


class _StopBeforeLoad(Exception):
    pass


def test_runner_llm_args_follow_mapping_and_model_defaults(monkeypatch):
    recorded = {}

    class _RecordingModelLoader:
        @staticmethod
        def load_config_and_apply_defaults(checkpoint_dir, llm_args, checkpoint_loader):
            # Which config loader is active while the defaults are resolved.
            recorded["config_loader"] = (
                runner_module.tensorrt_llm._torch.model_config.load_pretrained_config.__name__
            )
            # Stand-in for a model default that the checkpoint would apply.
            return llm_args.model_copy(update={"max_beam_width": 3})

        def __init__(self, **kwargs):
            recorded["kwargs"] = kwargs
            raise _StopBeforeLoad

    monkeypatch.setattr(runner_module, "ModelLoader", _RecordingModelLoader)
    monkeypatch.setattr(runner_module, "_construct_checkpoint_loader", lambda *args: object())

    mapping = Mapping(
        world_size=4,
        rank=2,
        tp_size=4,
        moe_ep_size=4,
        enable_attention_dp=True,
        gpus_per_node=4,
    )
    with pytest.raises(_StopBeforeLoad):
        runner_module.Runner(
            "/nonexistent/checkpoint",
            mapping,
            load_format="DUMMY",
            moe_backend="CUTLASS",
            layer_indices=[0],
            scaled_from=None,
            max_seq_len=128,
            max_num_tokens=128,
            moe_max_num_tokens=128,
            kv_cache_dtype="auto",
            mamba_ssm_cache_dtype="auto",
            use_low_precision_moe_combine=False,
            use_cuda_graph=False,
            vision_config="none",
        )

    llm_args = recorded["kwargs"]["llm_args"]
    assert recorded["kwargs"]["mapping"] is mapping
    # A nonzero rank needs llm_args whose parallelism matches the mapping.
    assert llm_args.tensor_parallel_size == 4
    assert llm_args.moe_expert_parallel_size == 4
    assert llm_args.enable_attention_dp
    assert llm_args.parallel_config.to_mapping().world_size == 4
    # The model defaults reach ModelLoader ...
    assert llm_args.max_beam_width == 3
    # ... and are resolved from the text-only config the model loads with.
    assert recorded["config_loader"] == "load_pretrained_config_text_only"


@pytest.mark.parametrize(
    ("load_balancer", "use_cuda_graph", "error"),
    [
        (MoePerLayerDynamicLoadBalancerConfig(num_slots=408), True, "does not support CUDA graphs"),
        (MoeLoadBalancerConfig(num_slots=384, layer_updates_per_iter=1), False, "static standard"),
    ],
)
def test_runner_rejects_load_balancers_a_layer_slice_cannot_run(
    load_balancer, use_cuda_graph, error
):
    with pytest.raises(ValueError, match=error):
        runner_module._check_load_balancer_support(load_balancer, use_cuda_graph)


@pytest.mark.parametrize(
    ("load_balancer", "use_cuda_graph"),
    [
        (None, True),
        (MoePerLayerDynamicLoadBalancerConfig(num_slots=408), False),
        (MoeLoadBalancerConfig(num_slots=384, layer_updates_per_iter=0), True),
    ],
)
def test_runner_accepts_load_balancers_a_layer_slice_can_run(load_balancer, use_cuda_graph):
    runner_module._check_load_balancer_support(load_balancer, use_cuda_graph)
