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

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
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
    "manager_cls,state_dtype,draft_len,replay_enabled",
    [
        (MixedMambaHybridCacheManager, torch.float32, 3, False),
        (MambaHybridCacheManagerV2, torch.float32, 3, True),
        (MambaHybridCacheManagerV2, torch.bfloat16, 3, True),
        (MambaHybridCacheManagerV2, torch.float16, 3, False),
        (MambaHybridCacheManagerV2, torch.float32, None, False),
    ],
)
def test_kimi_cache_factory_enables_replay_only_for_supported_v2(
    monkeypatch, manager_cls, state_dtype, draft_len, replay_enabled
):
    recorded = {}

    class RecordingManager(manager_cls):
        def __init__(self, *args, **kwargs):
            recorded.update(kwargs)

        def add_dummy_requests(self, *args, **kwargs):
            return []

    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        torch_dtype=torch.bfloat16,
        num_hidden_layers=2,
        kv_lora_rank=32,
        qk_rope_head_dim=8,
        linear_attn_config={"gate_lower_bound": -5.0},
    )
    model_config = SimpleNamespace(
        pretrained_config=config,
        quant_config=SimpleNamespace(kv_cache_quant_algo=None),
        enable_flash_mla=False,
    )
    mamba_params = SimpleNamespace(
        state_size=128,
        conv_kernel=4,
        num_heads=6,
        n_groups=6,
        head_dim=128,
        dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=state_dtype,
    )
    monkeypatch.setattr(runner_module.ModelConfig, "from_pretrained", lambda *_: model_config)
    monkeypatch.setattr(runner_module, "validate_and_set_kv_cache_quant", lambda *_: None)
    monkeypatch.setattr(runner_module, "validate_and_set_mamba_ssm_cache_dtype", lambda *_: None)
    monkeypatch.setattr(runner_module, "get_kv_cache_manager_cls", lambda *_: RecordingManager)
    monkeypatch.setattr(
        runner_module, "extract_mamba_kv_cache_params", lambda *_, **__: mamba_params
    )
    monkeypatch.setattr(
        runner_module, "get_kimi_linear_layer_masks", lambda *_: ([False, True], [True, False])
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.kimi_kda._kda_kernels.kda_mtp_replay_num_spec",
        lambda count: count,
    )

    runner_module.Runner.create_kv_cache_manager(
        "/nonexistent/checkpoint",
        mapping=Mapping(world_size=1),
        tokens_per_block=64,
        max_batch_size=4,
        max_seq_len=128,
        kv_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        layer_indices=[0, 1],
        spec_config=MTPDecodingConfig(max_draft_len=draft_len) if draft_len is not None else None,
    )

    if replay_enabled:
        assert recorded["kda_replay_num_spec"] == draft_len
    else:
        assert "kda_replay_num_spec" not in recorded
    if issubclass(manager_cls, MambaHybridCacheManagerV2):
        assert recorded["conv_state_layout"] == "q_k_v"
        assert "model_type" not in recorded
    else:
        assert recorded["model_type"] == "qwen3_next"
