# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import torch

import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_afmoe import (
    AfmoeConfig,
    AfmoeForCausalLM,
    AfmoeMoE,
    _validate_routing_config,
)
from tensorrt_llm._torch.models.modeling_utils import (
    MODEL_CLASS_MAPPER_MAPPING,
    MODEL_CLASS_MAPPING,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import KvCacheConfig as LlmKvCacheConfig
from tensorrt_llm.llmapi import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from transformers import AfmoeConfig as HFAfmoeConfig
from transformers.models.afmoe.modeling_afmoe import AfmoeForCausalLM as HFAfmoeForCausalLM

WINDOW_SIZE = 4
NUM_HIDDEN_LAYERS = 4
NUM_DENSE_LAYERS = 1

AFMOE_CONFIG = {
    "architectures": ["AfmoeForCausalLM"],
    "dtype": "bfloat16",
    "hidden_size": 256,
    "intermediate_size": 512,
    "max_position_embeddings": 2048,
    "model_type": "afmoe",
    "moe_intermediate_size": 128,
    "n_group": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 8,
    "num_dense_layers": NUM_DENSE_LAYERS,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_key_value_heads": 2,
    "num_shared_experts": 1,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000,
    "route_scale": 1.0,
    "scoring_func": "sigmoid",
    "sliding_window": WINDOW_SIZE,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "tie_word_embeddings": False,
    "topk_group": 1,
    "vocab_size": 1024,
    "hidden_act": "silu",
    "mup_enabled": False,
}


def _synthetic_tp_mapping(tp_size: int, enable_attention_dp: bool = False) -> Mapping:
    # These tests inspect module TP attributes in one pytest process.  Force
    # the lightweight MPI-topology Mapping even when TLLM_DISABLE_MPI=1 would
    # otherwise require an initialized torch.distributed DeviceMesh.
    with patch("tensorrt_llm.mapping.mpi_disabled", return_value=False):
        return Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=0,
            enable_attention_dp=enable_attention_dp,
        )


def _shutdown_kv_cache_manager(kv_cache_manager: KVCacheManager) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    kv_cache_manager.shutdown()


class TestAfmoeRegistry(unittest.TestCase):
    """Verify AfmoeForCausalLM resolves through _torch auto-model registration."""

    def test_auto_model_registry(self):
        self.assertIn("AfmoeForCausalLM", MODEL_CLASS_MAPPING)
        self.assertIs(MODEL_CLASS_MAPPING["AfmoeForCausalLM"], AfmoeForCausalLM)

    def test_weight_mapper_registry(self):
        self.assertIn("AfmoeForCausalLM_HF", MODEL_CLASS_MAPPER_MAPPING)

    def test_legacy_model_map_does_not_contain_afmoe(self):
        from tensorrt_llm.models import MODEL_MAP

        self.assertNotIn("AfmoeForCausalLM", MODEL_MAP)


class TestAfmoeRoutingValidation(unittest.TestCase):
    """Verify routing assumption guards."""

    def test_valid_sigmoid_config(self):
        config = AfmoeConfig.from_dict(deepcopy(AFMOE_CONFIG))
        _validate_routing_config(config)

    def test_rejects_softmax_scoring(self):
        d = deepcopy(AFMOE_CONFIG)
        d["scoring_func"] = "softmax"
        config = AfmoeConfig.from_dict(d)
        with self.assertRaisesRegex(ValueError, "Only 'sigmoid' is supported"):
            _validate_routing_config(config)

    def test_rejects_disabled_norm_topk(self):
        d = deepcopy(AFMOE_CONFIG)
        d["norm_topk_prob"] = False
        config = AfmoeConfig.from_dict(d)
        with self.assertRaisesRegex(ValueError, "norm_topk_prob"):
            _validate_routing_config(config)

    def test_model_init_rejects_invalid_routing(self):
        d = deepcopy(AFMOE_CONFIG)
        d["scoring_func"] = "softmax"
        config = AfmoeConfig.from_dict(d)
        model_config = ModelConfig(pretrained_config=config)
        with self.assertRaisesRegex(ValueError, "Only 'sigmoid' is supported"):
            AfmoeForCausalLM(model_config)


class TestAfmoeWeightMapper(unittest.TestCase):
    """Verify AfmoeHfWeightMapper key transformations."""

    def setUp(self):
        from tensorrt_llm._torch.models.checkpoints.hf.afmoe_weight_mapper import (
            AfmoeHfWeightMapper,
        )

        self.mapper = AfmoeHfWeightMapper()

    def test_expert_key_remapping(self):
        fake_weights = {
            "model.layers.1.mlp.experts.0.gate_proj.weight": torch.zeros(1),
            "model.layers.1.mlp.experts.0.up_proj.weight": torch.zeros(1),
            "model.layers.1.mlp.experts.0.down_proj.weight": torch.zeros(1),
            "model.layers.1.mlp.experts.3.gate_proj.weight": torch.zeros(1),
            "model.layers.1.mlp.experts.3.up_proj.weight": torch.zeros(1),
            "model.layers.1.mlp.experts.3.down_proj.weight": torch.zeros(1),
        }
        result = self.mapper.preprocess_weights(fake_weights)
        for expert_id in [0, 3]:
            prefix = f"model.layers.1.mlp.experts.{expert_id}"
            self.assertIn(f"{prefix}.w1.weight", result)
            self.assertIn(f"{prefix}.w3.weight", result)
            self.assertIn(f"{prefix}.w2.weight", result)
            self.assertNotIn(f"{prefix}.gate_proj.weight", result)
            self.assertNotIn(f"{prefix}.up_proj.weight", result)
            self.assertNotIn(f"{prefix}.down_proj.weight", result)

    def test_router_gate_rename(self):
        fake_weights = {
            "model.layers.2.mlp.router.gate.weight": torch.zeros(1),
        }
        result = self.mapper.preprocess_weights(fake_weights)
        self.assertIn("model.layers.2.mlp.gate.weight", result)
        self.assertNotIn("model.layers.2.mlp.router.gate.weight", result)

    def test_expert_bias_rename(self):
        fake_weights = {
            "model.layers.2.mlp.expert_bias": torch.zeros(1),
        }
        result = self.mapper.preprocess_weights(fake_weights)
        self.assertIn("model.layers.2.mlp.gate.e_score_correction_bias", result)
        self.assertNotIn("model.layers.2.mlp.expert_bias", result)

    def test_attention_gate_fused_into_q(self):
        # AfmoeAttention uses attn_output_gate=True, so the separate gate_proj
        # is interleaved per head into q_proj and the gate_proj key is dropped.
        num_heads, head_dim, hidden = 8, 32, 256
        self.mapper._model = Mock()
        self.mapper._model.config = Mock(num_attention_heads=num_heads)

        q = torch.arange(num_heads * head_dim * hidden, dtype=torch.float32).reshape(
            num_heads * head_dim, hidden
        )
        gate = q + 0.5
        fake_weights = {
            "model.layers.0.self_attn.q_proj.weight": q,
            "model.layers.0.self_attn.gate_proj.weight": gate,
        }
        result = self.mapper.preprocess_weights(fake_weights)

        self.assertNotIn("model.layers.0.self_attn.gate_proj.weight", result)
        fused = result["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(fused.shape, (2 * num_heads * head_dim, hidden))
        # Per head the layout is [q_head, gate_head]: first head_dim rows are q,
        # next head_dim rows are gate.
        torch.testing.assert_close(fused[:head_dim], q[:head_dim])
        torch.testing.assert_close(fused[head_dim : 2 * head_dim], gate[:head_dim])

    def test_qkv_keys_unchanged_without_gate(self):
        # Without a gate_proj key (e.g. partial weight dicts), q/k/v are untouched.
        fake_weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.k_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.v_proj.weight": torch.zeros(1),
        }
        result = self.mapper.preprocess_weights(fake_weights)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", result)
        self.assertIn("model.layers.0.self_attn.k_proj.weight", result)
        self.assertIn("model.layers.0.self_attn.v_proj.weight", result)

    def test_dense_mlp_keys_unchanged_by_preprocess(self):
        fake_weights = {
            "model.layers.0.mlp.gate_proj.weight": torch.zeros(1),
            "model.layers.0.mlp.up_proj.weight": torch.zeros(1),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(1),
        }
        result = self.mapper.preprocess_weights(fake_weights)
        self.assertIn("model.layers.0.mlp.gate_proj.weight", result)
        self.assertIn("model.layers.0.mlp.up_proj.weight", result)
        self.assertIn("model.layers.0.mlp.down_proj.weight", result)

    def test_is_special_instance_module_for_moe(self):
        from unittest.mock import MagicMock

        from tensorrt_llm._torch.modules.fused_moe.interface import MoE

        mock_moe = MagicMock(spec=MoE)
        mock_moe.__class__ = MoE
        self.assertTrue(self.mapper.is_special_instance_module(mock_moe))

        mock_linear = MagicMock(spec=torch.nn.Linear)
        self.assertFalse(self.mapper.is_special_instance_module(mock_linear))


class TestAfmoeWeightLoading(unittest.TestCase):
    """Verify AfmoeForCausalLM applies mapper preprocessing in the real load hook."""

    def test_load_weights_preprocesses_mapper_weights(self):
        from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

        model = object.__new__(AfmoeForCausalLM)
        raw_weights = {"model.layers.1.mlp.router.gate.weight": torch.zeros(1)}
        processed_weights = {"model.layers.1.mlp.gate.weight": torch.zeros(1)}
        mapper = Mock()
        mapper.preprocess_weights.return_value = processed_weights

        with patch.object(DecoderModelForCausalLM, "load_weights", autospec=True) as load_weights:
            AfmoeForCausalLM.load_weights(model, raw_weights, mapper, allow_partial_loading=True)

        mapper.preprocess_weights.assert_called_once_with(raw_weights)
        load_weights.assert_called_once()
        args, kwargs = load_weights.call_args
        self.assertIs(args[0], model)
        self.assertIs(kwargs["weights"], processed_weights)
        self.assertIs(kwargs["weight_mapper"], mapper)
        self.assertTrue(kwargs["allow_partial_loading"])


class TestAfmoeSanity(unittest.TestCase):
    """Smoke test: build a tiny AFMoE and run a forward pass."""

    def test_afmoe_sanity(self):
        config_dict = deepcopy(AFMOE_CONFIG)
        afmoe_config = AfmoeConfig.from_dict(config_dict)

        model_config = ModelConfig(pretrained_config=afmoe_config, quant_config=QuantConfig())
        dtype = afmoe_config.torch_dtype
        device = torch.device("cuda")
        model = AfmoeForCausalLM(model_config).to(device)

        input_ids = torch.tensor(
            [100, 200, 300, 100, 200, 100, 400, 500], dtype=torch.int, device=device
        )

        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 62, 75]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 128
        head_dim = afmoe_config.hidden_size // afmoe_config.num_attention_heads
        num_layers = afmoe_config.num_hidden_layers
        num_kv_heads = afmoe_config.num_key_value_heads
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(context_sequence_lengths) + 2

        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=len(context_sequence_lengths) + 2,
            max_num_tokens=8192,
        )

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_lengths[i] if i < len(context_sequence_lengths) else 1
            position_id = torch.arange(tokens, tokens + seq_len, device=input_ids.device)
            position_ids.append(position_id)
        position_ids = torch.cat(position_ids).unsqueeze(0)

        try:
            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
                )

            self.assertEqual(len(past_seen_tokens), logits.shape[0])
        finally:
            _shutdown_kv_cache_manager(kv_cache_manager)

    def test_moe_layer_config(self):
        config_dict = deepcopy(AFMOE_CONFIG)
        afmoe_config = AfmoeConfig.from_dict(config_dict)

        device = torch.device("cuda")
        model_config = ModelConfig(pretrained_config=afmoe_config)
        model = AfmoeForCausalLM(model_config).to(device)

        self.assertEqual(len(model.model.layers), NUM_HIDDEN_LAYERS)

        for i in range(NUM_HIDDEN_LAYERS):
            layer = model.model.layers[i]
            if i < NUM_DENSE_LAYERS:
                self.assertFalse(layer.moe_enabled, f"Layer {i} should be dense")
                self.assertNotIsInstance(layer.mlp, AfmoeMoE)
            else:
                self.assertTrue(layer.moe_enabled, f"Layer {i} should be MoE")
                self.assertIsInstance(layer.mlp, AfmoeMoE)


class TestAfmoeEndToEnd(unittest.TestCase):
    """Exercise AFMoE through the PyTorch LLM API with dummy weights."""

    def test_llm_dummy_load_generates_from_token_ids(self):
        if not torch.cuda.is_available():
            self.skipTest("AFMoE LLM API test requires CUDA")

        with tempfile.TemporaryDirectory() as tmp_model_dir:
            with open(f"{tmp_model_dir}/config.json", "w", encoding="utf-8") as f:
                json.dump(AFMOE_CONFIG, f, indent=2)

            prompts = [
                {"prompt_token_ids": [100, 200, 300]},
                {"prompt_token_ids": [101, 202]},
            ]
            sampling_params = SamplingParams(
                max_tokens=2,
                end_id=AFMOE_CONFIG["vocab_size"] - 1,
                pad_id=AFMOE_CONFIG["vocab_size"] - 1,
                detokenize=False,
                ignore_eos=True,
            )

            with LLM(
                model=tmp_model_dir,
                load_format="dummy",
                tensor_parallel_size=1,
                enable_chunked_prefill=False,
                disable_overlap_scheduler=True,
                attn_backend="TRTLLM",
                max_batch_size=len(prompts),
                max_num_tokens=16,
                max_seq_len=64,
                moe_config=MoeConfig(max_num_tokens=64),
                moe_expert_parallel_size=-1,
                moe_tensor_parallel_size=-1,
                enable_attention_dp=False,
                kv_cache_config=LlmKvCacheConfig(enable_block_reuse=False),
            ) as llm:
                outputs = llm.generate(prompts, sampling_params=sampling_params)

        self.assertEqual(len(outputs), len(prompts))
        for prompt, output in zip(prompts, outputs):
            self.assertEqual(output.prompt_token_ids, prompt["prompt_token_ids"])
            self.assertEqual(len(output.outputs), 1)
            self.assertEqual(len(output.outputs[0].token_ids), sampling_params.max_tokens)


class TestAfmoeTPAttributes(unittest.TestCase):
    """Verify TP-related module attributes are wired correctly."""

    def _build_model(self, tp_size):
        config_dict = deepcopy(AFMOE_CONFIG)
        afmoe_config = AfmoeConfig.from_dict(config_dict)
        mapping = _synthetic_tp_mapping(tp_size)
        model_config = ModelConfig(
            pretrained_config=afmoe_config, mapping=mapping, allreduce_strategy="NCCL"
        )
        return AfmoeForCausalLM(model_config)

    def _build_attention_dp_model(self, tp_size):
        config_dict = deepcopy(AFMOE_CONFIG)
        afmoe_config = AfmoeConfig.from_dict(config_dict)
        mapping = _synthetic_tp_mapping(tp_size, enable_attention_dp=True)
        model_config = ModelConfig(
            pretrained_config=afmoe_config, mapping=mapping, allreduce_strategy="NCCL"
        )
        return AfmoeForCausalLM(model_config)

    def test_qkv_is_column_parallel_with_output_gate(self):
        model = self._build_model(tp_size=1)
        for layer in model.model.layers:
            attn = layer.self_attn
            self.assertTrue(attn.attn_output_gate)
            self.assertEqual(attn.qkv_proj.tp_mode, TensorParallelMode.COLUMN)

    def test_moe_experts_no_reduce(self):
        model = self._build_model(tp_size=1)
        for layer in model.model.layers:
            if layer.moe_enabled:
                self.assertFalse(layer.mlp.experts.reduce_results)

    def test_allreduce_created_for_tp2(self):
        model = self._build_model(tp_size=2)
        for layer in model.model.layers:
            if layer.moe_enabled:
                self.assertIsNotNone(
                    layer.mlp.allreduce, "MoE layer should have allreduce for tp_size=2"
                )

    def test_no_allreduce_for_tp1(self):
        model = self._build_model(tp_size=1)
        for layer in model.model.layers:
            if layer.moe_enabled:
                self.assertIsNone(
                    layer.mlp.allreduce, "MoE layer should NOT have allreduce for tp_size=1"
                )

    def test_qkv_output_includes_fused_gate(self):
        # With attn_output_gate=True the query slot is doubled (q + gate) and
        # fused into qkv_proj, so its local output is 2*q_size + 2*kv_size.
        model = self._build_model(tp_size=2)
        for layer in model.model.layers:
            attn = layer.self_attn
            expected_out = attn.q_size * 2 + 2 * attn.kv_size
            actual_out = attn.qkv_proj.weight.shape[0]
            self.assertEqual(
                actual_out,
                expected_out,
                f"qkv_proj local output should be 2*q_size + 2*kv_size = "
                f"{expected_out}, got {actual_out}",
            )

    def test_attention_dp_uses_unsharded_qkv_and_mlp_modules(self):
        model = self._build_attention_dp_model(tp_size=2)

        for layer in model.model.layers:
            attn = layer.self_attn
            self.assertEqual(attn.qkv_proj.tp_size, 1)
            # Unsharded: full heads, q slot doubled by the output gate.
            expected_out = attn.q_size * 2 + 2 * attn.kv_size
            self.assertEqual(attn.qkv_proj.weight.shape[0], expected_out)

            if layer.moe_enabled:
                self.assertIsNone(layer.mlp.allreduce)
                if layer.mlp.shared_experts is not None:
                    self.assertEqual(layer.mlp.shared_experts.gate_up_proj.tp_size, 1)
                    self.assertEqual(layer.mlp.shared_experts.down_proj.tp_size, 1)
            else:
                self.assertEqual(layer.mlp.gate_up_proj.tp_size, 1)
                self.assertEqual(layer.mlp.down_proj.tp_size, 1)

    def test_attention_layer_types(self):
        model = self._build_model(tp_size=1)
        layer_types = AFMOE_CONFIG["layer_types"]
        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if layer_types[i] == "sliding_attention":
                self.assertTrue(attn.is_local_attention)
                self.assertEqual(attn.attention_window_size, WINDOW_SIZE)
            else:
                self.assertFalse(attn.is_local_attention)
                self.assertIsNone(attn.attention_window_size)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestAfmoeAllCloseToHF(unittest.TestCase):
    """Compare TRT-LLM AFMoE context-phase logits against the HF reference.

    Loads the HF model's weights into the TRT-LLM model via AfmoeHfWeightMapper,
    exercising the per-head q/gate fusion (attn_output_gate) and the HF
    fused-expert -> per-expert conversion, then checks logit parity.
    """

    # Field names follow the HF AfmoeConfig schema (released-checkpoint names).
    HF_CONFIG = {
        "hidden_size": 256,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "head_dim": 32,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_dense_layers": NUM_DENSE_LAYERS,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "global_attn_every_n_layers": 4,
        "sliding_window": WINDOW_SIZE,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "route_scale": 1.0,
        "route_norm": True,
        "score_func": "sigmoid",
        "vocab_size": 1024,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }

    @staticmethod
    def _convert_hf_experts(state_dict, moe_intermediate_size):
        """Split HF fused 3D expert params into per-expert gate/up/down weights.

        HF-native AFMoE stores experts as ``experts.gate_up_proj``
        ``[num_experts, 2 * moe_inter, hidden]`` and ``experts.down_proj``
        ``[num_experts, hidden, moe_inter]``; AfmoeHfWeightMapper expects the
        released-checkpoint layout with separate per-expert matrices.
        """
        converted = dict(state_dict)
        gate_up_keys = [k for k in state_dict if k.endswith(".experts.gate_up_proj")]
        for gate_up_key in gate_up_keys:
            prefix = gate_up_key[: -len(".gate_up_proj")]
            gate_up = converted.pop(gate_up_key)
            down = converted.pop(prefix + ".down_proj")
            for expert_idx in range(gate_up.shape[0]):
                converted[f"{prefix}.{expert_idx}.gate_proj.weight"] = gate_up[expert_idx][
                    :moe_intermediate_size
                ].contiguous()
                converted[f"{prefix}.{expert_idx}.up_proj.weight"] = gate_up[expert_idx][
                    moe_intermediate_size:
                ].contiguous()
                converted[f"{prefix}.{expert_idx}.down_proj.weight"] = down[expert_idx].contiguous()
        return converted

    @torch.no_grad()
    def test_afmoe_allclose_to_hf(self):
        from tensorrt_llm._torch.models.checkpoints.hf.afmoe_weight_mapper import (
            AfmoeHfWeightMapper,
        )

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.bfloat16

        hf_config = HFAfmoeConfig(dtype="float32", **self.HF_CONFIG)
        hf_model = HFAfmoeForCausalLM(hf_config).to(dtype).to(device).eval()

        # TRT-LLM needs a couple of routing fields that the HF schema names
        # differently; provide both so AfmoeConfig validates and builds.
        trt_config_dict = dict(self.HF_CONFIG)
        trt_config_dict.update(
            architectures=["AfmoeForCausalLM"],
            model_type="afmoe",
            dtype="bfloat16",
            n_group=1,
            topk_group=1,
            scoring_func=self.HF_CONFIG["score_func"],
            norm_topk_prob=self.HF_CONFIG["route_norm"],
        )
        afmoe_config = AfmoeConfig.from_dict(trt_config_dict)
        model_config = ModelConfig(pretrained_config=afmoe_config)
        model = AfmoeForCausalLM(model_config).to(dtype).to(device)

        weights = self._convert_hf_experts(
            hf_model.state_dict(), self.HF_CONFIG["moe_intermediate_size"]
        )
        weights = {k: v.to(dtype) for k, v in weights.items()}

        weight_mapper = AfmoeHfWeightMapper()
        weight_mapper.init_model_and_config(model, model_config)
        model.load_weights(weights, weight_mapper)
        if hasattr(model, "post_load_weights"):
            model.post_load_weights()

        # Short context: input_len < sliding_window so sliding == full attention.
        input_len = WINDOW_SIZE - 1
        input_ids = torch.tensor([101, 202, 303][:input_len], dtype=torch.int32, device=device)
        position_ids = torch.arange(input_len, dtype=torch.int32, device=device).unsqueeze(0)

        num_blocks, tokens_per_block = 4, 128
        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=num_blocks * tokens_per_block),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=num_blocks * tokens_per_block,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        kv_cache_manager.add_dummy_requests([0], [input_len])
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_len], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            kv_cache_manager=kv_cache_manager,
            request_ids=[0],
            prompt_lens=[input_len],
            max_num_requests=1,
            max_num_tokens=8192,
        )

        try:
            hf_position_ids = position_ids.to(torch.long)
            with torch.inference_mode():
                attn_metadata.prepare()
                logits = model.forward(
                    input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
                )
                ref = hf_model.forward(
                    input_ids=input_ids.unsqueeze(0).long(),
                    position_ids=hf_position_ids,
                    use_cache=False,
                )

            # Loose tolerance: bf16 + token-choice MoE routing amplify per-logit
            # noise (same rationale as the EXAONE-MoE parity test).
            torch.testing.assert_close(logits, ref.logits[:, -1].float(), atol=1.0, rtol=0.5)
        finally:
            _shutdown_kv_cache_manager(kv_cache_manager)


if __name__ == "__main__":
    unittest.main()
