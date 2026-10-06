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
"""Sanity tests for Nemotron-H LoRA support."""

import os
import tempfile

import pytest
from nemotron_h_lora_utils import create_nemotron_h_lora_adapter
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_80gb, skip_num_gpus_less_than, skip_pre_blackwell

import tensorrt_llm.bindings as _tb
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest

# Weak enough that the adapter stays close to identity, which is all these
# sanity tests need: they assert that a LoRA run produces tokens at all, not
# that the adapter moved them.
_SANITY_STD = (0.01, 0.001)


def _get_lora_config(lora_dir):
    """Get model-specific LoRA config via NemotronHForCausalLM.lora_config()."""
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHForCausalLM

    config = NemotronHForCausalLM.lora_config("")
    config.lora_dir = [lora_dir]
    config.max_lora_rank = 16
    config.max_loras = 2
    return config


@skip_pre_blackwell
@skip_gpu_memory_less_than_80gb
class TestNemotronHLoRA:
    """E2E LoRA tests on the Super-V3 120B NVFP4 model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")

    def _run_generate(self, llm, lora_dir, prompts):
        sampling = SamplingParams(max_tokens=5, temperature=0.0)
        lora_req = [LoRARequest("test-lora", 0, lora_dir)] * len(prompts)
        return llm.generate(prompts, sampling, lora_request=lora_req)

    def test_lora_pp1_sanity(self):
        """LoRA inference with pp_size=1 produces tokens for base and LoRA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = create_nemotron_h_lora_adapter(
                os.path.join(tmpdir, "lora"), self.model_path, std=_SANITY_STD
            )
            with LLM(
                model=self.model_path,
                lora_config=_get_lora_config(lora_dir),
                tensor_parallel_size=1,
                max_batch_size=2,
                max_num_tokens=256,
            ) as llm:
                prompts = ["Hello", "The capital of France is"]
                lora = self._run_generate(llm, lora_dir, prompts)
                assert all(len(o.outputs[0].token_ids) > 0 for o in lora)

    @skip_num_gpus_less_than(2)
    def test_lora_pp2_sanity(self):
        """LoRA inference with pp_size=2 produces tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = create_nemotron_h_lora_adapter(
                os.path.join(tmpdir, "lora"), self.model_path, std=_SANITY_STD
            )
            with LLM(
                model=self.model_path,
                lora_config=_get_lora_config(lora_dir),
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                max_batch_size=2,
                max_num_tokens=256,
            ) as llm:
                lora = self._run_generate(llm, lora_dir, ["Hello", "The capital of France is"])
                assert all(len(o.outputs[0].token_ids) > 0 for o in lora)


def _make_model_config(n_lora_layers):
    """Create a C++ ModelConfig with the given number of LoRA layers."""
    cfg = _tb.ModelConfig(
        vocab_size=32000,
        num_layers=n_lora_layers,
        num_attention_layers=n_lora_layers,
        num_rnn_layers=0,
        num_heads=32,
        hidden_size=4096,
        data_type=_tb.DataType.HALF,
    )
    cfg.set_num_lora_layers(n_lora_layers)
    return cfg


@pytest.mark.parametrize(
    "n_layers,pp_size",
    [
        (88, 4),  # 22 each
        (7, 2),  # 4 + 3
        (7, 3),  # 3 + 2 + 2
        (1, 2),  # Edge: 1 + 0
        (32, 1),  # Single rank
    ],
)
def test_lora_layer_distribution_no_overlap(n_layers, pp_size):
    """C++ getFirstLoraLayer()/getNbLoraLayers(): every layer assigned to exactly one PP rank."""
    cfg = _make_model_config(n_layers)
    assigned = set()
    prev_end = 0
    for rank in range(pp_size):
        first = cfg.first_lora_layer(pp_size, rank)
        count = cfg.num_lora_layers(pp_size, rank)
        assert first == prev_end, f"Rank {rank}: first_layer={first}, expected {prev_end}"
        assert count >= 0, f"Rank {rank}: negative count {count}"
        layers = set(range(first, first + count))
        assert not assigned & layers, (
            f"Rank {rank} overlaps with previous ranks on {assigned & layers}"
        )
        assigned |= layers
        prev_end = first + count
    assert assigned == set(range(n_layers)), f"Missing layers: {set(range(n_layers)) - assigned}"


def test_lora_fallback_with_empty_layer_types_and_pp():
    """Tests for correct layer distribution when no layer types are set."""
    num_layers = 32
    pp_size = 4
    layers_per_rank = num_layers // pp_size

    # No set_num_lora_layers → mNbLoraLayers=0, no layer_types → mLayerTypes empty
    cfg = _tb.ModelConfig(
        vocab_size=32000,
        num_layers=num_layers,
        num_attention_layers=num_layers,
        num_rnn_layers=0,
        num_heads=32,
        hidden_size=4096,
        data_type=_tb.DataType.HALF,
    )

    for rank in range(pp_size):
        assert cfg.first_lora_layer(pp_size, rank) == rank * layers_per_rank
        assert cfg.num_lora_layers(pp_size, rank) == layers_per_rank


def test_moe_latent_lora_modules_use_correct_dimensions():
    """Test for correct dimensions of MoE latent projection layers."""
    hidden, mlp_hidden, moe_latent, tp = 6144, 3072, 1024, 2
    up, down = _tb.LoraModule.create_lora_modules(
        lora_module_names=["moe_latent_fc1", "moe_latent_fc2"],
        hidden_size=hidden,
        mlp_hidden_size=mlp_hidden,
        num_attention_heads=32,
        num_kv_attention_heads=8,
        attention_head_size=128,
        tp_size=tp,
        moe_latent_size=moe_latent,
    )
    global_hidden = hidden * tp

    assert (up.in_dim, up.out_dim) == (global_hidden, moe_latent)
    assert (down.in_dim, down.out_dim) == (moe_latent, global_hidden)
    for m in (up, down):
        assert m.in_tp_split_dim == -1
        assert m.out_tp_split_dim == -1
