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
"""Tests for the Step3p7 text-generation bring-up (Step3p7ForConditionalGeneration).

TestStep3p7Helpers — pure-Python loader/helper tests (no checkpoint, no GPU):
  - Stacked routed-expert weight splitting (FP8 block-scale and NVFP4 layouts)
  - MTP weight rewriting to the ``mtp_block`` layout
  - ``model.language_model.*`` namespace flattening for the multimodal checkpoint
  - quant-config exclude-module normalization
  - NVFP4 dequant round-tripping
  - MTP shared-head normalization before the output projection

TestStep3p7AutoModelRegistration — model registry verification:
  - ``Step3p7ForConditionalGeneration`` resolves to the VLM entry point
  - ``Step3p5ForCausalLM`` resolves to the text-only causal LM

TestStep3p7Checkpoint — checkpoint-backed config / weight-accounting tests
(requires the Step-3.7-Flash checkpoints under ``LLM_MODELS_ROOT``):
  - Config sanity, per-layer attention/MoE inventory, and FP8 quant config
  - Weight accounting that separates consumed text-path keys from intentionally
    ignored vision and plain-path MTP keys
  - Per-layer head / RoPE / SwiGLU helpers against the real config
  - Default ``MTPDecodingConfig`` resolving to the checkpoint layer count
"""

import json
import os
import types
import unittest

import torch
from parameterized import parameterized
from transformers import PretrainedConfig
from utils.llm_data import llm_models_root

# Resolve the Step3p7 checkpoints under the shared model root (LLM_MODELS_ROOT)
# like the other modeling tests instead of hard-coding a developer workspace
# path. The FP8 block-scale, NVFP4, and BF16 reference checkpoints share the
# same per-layer geometry; only the routed-expert dtype/layout differs.
#
# The FP8 and BF16 checkpoints store text decoder keys directly under
# ``model.layers.*`` and ship 3 plain-path MTP layers (45..47). The NVFP4
# checkpoint is a modelopt export: text keys live under
# ``model.language_model.layers.*``, vision keys under ``model.vision_model.*``,
# and it carries no MTP layers.
STEP3P7_FP8_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash-FP8"))
STEP3P7_NVFP4_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash-NVFP4"))
STEP3P7_BF16_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash"))


def _load_config(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        return json.load(f)


def _load_safetensors_keys(checkpoint_dir: str) -> list:
    import glob

    import safetensors

    all_keys = []
    for f in sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors"))):
        with safetensors.safe_open(f, framework="pt") as h:
            all_keys.extend(h.keys())
    return all_keys


class TestStep3p7Helpers(unittest.TestCase):
    """Pure-Python loader/helper tests — no checkpoint and no GPU required."""

    def test_split_stacked_moe_weights_expands_per_expert_keys(self):
        """Stacked routed-expert tensors expand into per-expert ``w1/w2/w3`` keys.

        Source convention: ``...moe.gate_proj.weight`` shape
        ``(N, intermediate, hidden)``. Target convention (VANILLA MoE backend):
        ``...moe.experts.<e>.w1.weight``. Uses a synthetic 2-layer dict (one
        MoE, one dense) so the test runs in <1s without the real checkpoint.
        """
        from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

        num_experts = 4
        intermediate = 8
        hidden = 16
        text_config = types.SimpleNamespace(
            num_hidden_layers=2,
            moe_num_experts=num_experts,
            moe_layers_enum=[1],  # Only layer 1 is MoE.
        )

        weights = {
            # Layer 0 (dense): no MoE stacked tensors.
            "model.layers.0.mlp.gate_proj.weight": torch.arange(
                intermediate * hidden, dtype=torch.float32
            ).reshape(intermediate, hidden),
            # Layer 1 (MoE): stacked routed-expert tensors.
            "model.layers.1.moe.gate_proj.weight": torch.arange(
                num_experts * intermediate * hidden, dtype=torch.float32
            ).reshape(num_experts, intermediate, hidden),
            "model.layers.1.moe.gate_proj.weight_scale_inv": torch.arange(
                num_experts * 1 * 2, dtype=torch.float32
            ).reshape(num_experts, 1, 2),
            "model.layers.1.moe.up_proj.weight": torch.arange(
                num_experts * intermediate * hidden, dtype=torch.float32
            ).reshape(num_experts, intermediate, hidden)
            + 1.0,
            "model.layers.1.moe.up_proj.weight_scale_inv": torch.arange(
                num_experts * 1 * 2, dtype=torch.float32
            ).reshape(num_experts, 1, 2)
            + 1.0,
            "model.layers.1.moe.down_proj.weight": torch.arange(
                num_experts * hidden * intermediate, dtype=torch.float32
            ).reshape(num_experts, hidden, intermediate),
            "model.layers.1.moe.down_proj.weight_scale_inv": torch.arange(
                num_experts * 2 * 1, dtype=torch.float32
            ).reshape(num_experts, 2, 1),
        }
        original_gate = weights["model.layers.1.moe.gate_proj.weight"]
        original_gate_scale = weights["model.layers.1.moe.gate_proj.weight_scale_inv"]

        layers_split = split_stacked_moe_weights(weights, text_config)
        self.assertEqual(layers_split, 1)

        # Stacked keys must be removed.
        self.assertNotIn("model.layers.1.moe.gate_proj.weight", weights)
        self.assertNotIn("model.layers.1.moe.gate_proj.weight_scale_inv", weights)
        self.assertNotIn("model.layers.1.moe.up_proj.weight", weights)
        self.assertNotIn("model.layers.1.moe.down_proj.weight", weights)

        # Per-expert keys present with correct slicing. Source ``gate_proj`` →
        # ``w1``, ``up_proj`` → ``w3``, ``down_proj`` → ``w2``.
        for expert_id in range(num_experts):
            key_w1 = f"model.layers.1.moe.experts.{expert_id}.w1.weight"
            key_w1_scale = f"model.layers.1.moe.experts.{expert_id}.w1.weight_scale_inv"
            key_w3 = f"model.layers.1.moe.experts.{expert_id}.w3.weight"
            key_w2 = f"model.layers.1.moe.experts.{expert_id}.w2.weight"
            self.assertIn(key_w1, weights)
            self.assertIn(key_w1_scale, weights)
            self.assertIn(key_w3, weights)
            self.assertIn(key_w2, weights)
            self.assertTrue(torch.equal(weights[key_w1], original_gate[expert_id]))
            self.assertTrue(torch.equal(weights[key_w1_scale], original_gate_scale[expert_id]))
            # Per-expert tensor drops the leading expert dim.
            self.assertEqual(weights[key_w1].shape, (intermediate, hidden))
            self.assertEqual(weights[key_w2].shape, (hidden, intermediate))

        # Layer 0 (dense) is untouched.
        self.assertIn("model.layers.0.mlp.gate_proj.weight", weights)

    def test_split_stacked_moe_weights_no_op_when_no_stacked_keys(self):
        """With no stacked MoE keys present, the splitter is a no-op."""
        from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

        text_config = types.SimpleNamespace(
            num_hidden_layers=1,
            moe_num_experts=8,
            moe_layers_enum=[0],
        )
        weights = {"some.other.key": "not_a_tensor"}
        self.assertEqual(split_stacked_moe_weights(weights, text_config), 0)
        self.assertEqual(weights, {"some.other.key": "not_a_tensor"})

    def test_split_stacked_moe_weights_handles_nvfp4_suffixes(self):
        """NVFP4 checkpoints carry ``weight_scale`` / ``weight_scale_2`` /
        ``input_scale`` alongside the packed ``weight``; the splitter must fan
        each suffix out to per-expert keys."""
        from tensorrt_llm._torch.models.modeling_step3p7 import split_stacked_moe_weights

        num_experts = 3
        intermediate = 8
        hidden = 16
        text_config = types.SimpleNamespace(
            num_hidden_layers=1,
            moe_num_experts=num_experts,
            moe_layers_enum=[0],
        )
        weights = {
            # Packed FP4 weight: (E, I, H/2) uint8.
            "model.layers.0.moe.gate_proj.weight": torch.zeros(
                (num_experts, intermediate, hidden // 2), dtype=torch.uint8
            ),
            # Per-16 block scale: (E, I, H/16) fp8_e4m3fn.
            "model.layers.0.moe.gate_proj.weight_scale": torch.ones(
                (num_experts, intermediate, hidden // 16), dtype=torch.float8_e4m3fn
            ),
            # Per-tensor global scale: (E,) float32.
            "model.layers.0.moe.gate_proj.weight_scale_2": torch.arange(
                num_experts, dtype=torch.float32
            ),
            # Per-tensor input scale: (E,) float32.
            "model.layers.0.moe.gate_proj.input_scale": torch.arange(
                num_experts, dtype=torch.float32
            )
            + 10.0,
        }
        layers_split = split_stacked_moe_weights(weights, text_config)
        self.assertEqual(layers_split, 1)
        self.assertNotIn("model.layers.0.moe.gate_proj.weight_scale", weights)
        self.assertNotIn("model.layers.0.moe.gate_proj.weight_scale_2", weights)
        self.assertNotIn("model.layers.0.moe.gate_proj.input_scale", weights)
        for e in range(num_experts):
            self.assertEqual(
                weights[f"model.layers.0.moe.experts.{e}.w1.weight"].shape,
                (intermediate, hidden // 2),
            )
            self.assertEqual(
                weights[f"model.layers.0.moe.experts.{e}.w1.weight_scale"].shape,
                (intermediate, hidden // 16),
            )
            self.assertEqual(
                weights[f"model.layers.0.moe.experts.{e}.w1.weight_scale_2"].item(), float(e)
            )
            self.assertEqual(
                weights[f"model.layers.0.moe.experts.{e}.w1.input_scale"].item(), float(e) + 10.0
            )

    def test_rewrite_mtp_weights_uses_mtp_block_layout(self):
        """Checkpoint MTP keys map to TRT-LLM's ``mtp_block`` layout."""
        from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
        from tensorrt_llm._torch.models.modeling_step3p7 import rewrite_mtp_weights_for_step3p7

        # The rewriter only relies on ``num_hidden_layers`` (45 decoder layers,
        # so MTP layers start at index 45) and ``num_nextn_predict_layers``
        # (3 MTP layers); a lightweight stub avoids loading the real checkpoint.
        text_config = types.SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=3)
        weights = ConsumableWeightsDict(
            {
                "model.layers.45.enorm.weight": torch.tensor([1.0]),
                "model.layers.45.eh_proj.weight": torch.tensor([2.0]),
                "model.layers.45.self_attn.q_proj.weight": torch.tensor([3.0]),
                "model.layers.45.mlp.gate_proj.weight": torch.tensor([4.0]),
                "model.layers.45.transformer.shared_head.norm.weight": torch.tensor([5.0]),
                "model.layers.45.transformer.shared_head.output.weight": torch.tensor([6.0]),
                "model.layers.44.self_attn.q_proj.weight": torch.tensor([7.0]),
            }
        )

        rewritten = rewrite_mtp_weights_for_step3p7(weights, text_config)

        self.assertEqual(rewritten, 4)
        self.assertIn("model.layers.45.enorm.weight", weights)
        self.assertIn("model.layers.45.eh_proj.weight", weights)
        self.assertIn("model.layers.45.mtp_block.self_attn.q_proj.weight", weights)
        self.assertIn("model.layers.45.mtp_block.mlp.gate_proj.weight", weights)
        self.assertIn("model.layers.45.shared_head.norm.weight", weights)
        self.assertIn("model.layers.45.shared_head.output.weight", weights)
        self.assertIn("model.layers.44.self_attn.q_proj.weight", weights)
        self.assertNotIn("model.layers.45.self_attn.q_proj.weight", weights)
        self.assertNotIn("model.layers.45.transformer.shared_head.output.weight", weights)

    def test_rewrite_language_model_keys_flattens_multimodal_namespace(self):
        """The NVFP4 multimodal checkpoint stores text decoder keys under
        ``model.language_model.*``; the loader normalizes them to ``model.*`` so
        the module tree (built around ``model.layers.*``) is addressable. Vision
        keys move out from under ``model.`` to match the ignored prefix list."""
        from tensorrt_llm._torch.models.modeling_step3p7 import rewrite_language_model_keys

        weights = {
            "lm_head.weight": "lm",
            "model.language_model.embed_tokens.weight": "embed",
            "model.language_model.layers.0.input_layernorm.weight": "in_ln",
            "model.language_model.norm.weight": "final_ln",
            "model.vision_model.conv1.weight": "vision",
            "model.vit_large_projector.weight": "projector",
        }
        n = rewrite_language_model_keys(weights)
        # 5 of 6 keys get renamed; lm_head is untouched.
        self.assertEqual(n, 5)
        self.assertIn("lm_head.weight", weights)
        self.assertIn("model.embed_tokens.weight", weights)
        self.assertIn("model.layers.0.input_layernorm.weight", weights)
        self.assertIn("model.norm.weight", weights)
        self.assertIn("vision_model.conv1.weight", weights)
        self.assertIn("vit_large_projector.weight", weights)
        self.assertNotIn("model.language_model.embed_tokens.weight", weights)
        self.assertNotIn("model.vision_model.conv1.weight", weights)

    def test_strip_language_model_prefix_preserves_regex_and_non_matching(self):
        """The quant-config exclude-modules normalizer must replace
        ``model.language_model.`` with ``model.``, leave ``re:`` prefixed
        entries untouched, and leave entries without the segment untouched."""
        from tensorrt_llm._torch.models.modeling_step3p7 import (
            strip_language_model_prefix_from_exclude_modules,
        )

        src = [
            "lm_head",
            "model.language_model.layers.0*",
            "model.language_model.layers.3.moe.gate",
            "re:^model\\.something\\..*",
            "model.vision_model*",
        ]
        out = strip_language_model_prefix_from_exclude_modules(src)
        self.assertEqual(
            out,
            [
                "lm_head",
                "model.layers.0*",
                "model.layers.3.moe.gate",
                "re:^model\\.something\\..*",
                "model.vision_model*",
            ],
        )
        self.assertIsNone(strip_language_model_prefix_from_exclude_modules(None))

    def test_nvfp4_dequant_batched_round_trips_constant_values(self):
        """Sanity check the NVFP4 dequant helper. Packing a constant e2m1 index
        of 4 (= 2.0) and scaling with block_scale=1.0 and global_scale=0.5 must
        produce 1.0 everywhere."""
        from tensorrt_llm._torch.models.modeling_step3p7 import _nvfp4_dequant_batched

        # Each byte holds (high<<4) | low. Encoding index 4 in both nibbles:
        # (4 << 4) | 4 = 68.
        weight = torch.full((2, 8), 68, dtype=torch.uint8)
        block_scale = torch.ones((2, 1), dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor(0.5, dtype=torch.float32)
        out = _nvfp4_dequant_batched(weight, block_scale, global_scale)
        self.assertEqual(out.dtype, torch.bfloat16)
        # 2.0 (e2m1) * 1.0 (block) * 0.5 (global) = 1.0
        self.assertTrue(torch.all(out == 1.0))

        # Encode a negative index (12 = -2.0): byte = (12 << 4) | 4 → high=-2,
        # low=2. Use K=16 so the per-16 block-scale shape works.
        weight2 = torch.full((1, 8), (12 << 4) | 4, dtype=torch.uint8)
        out2 = _nvfp4_dequant_batched(
            weight2,
            torch.ones((1, 1), dtype=torch.float8_e4m3fn),
            torch.tensor(1.0, dtype=torch.float32),
        )
        # Even positions get 2.0 (low nibble), odd positions get -2.0 (high nibble).
        self.assertEqual(out2[0, 0].item(), 2.0)
        self.assertEqual(out2[0, 1].item(), -2.0)

        # 3D batched form: (E=2, M=2, K_half=8) — exercises the per-expert
        # global-scale broadcast path used by the Python-clamp loader.
        w3 = torch.full((2, 2, 8), 68, dtype=torch.uint8)
        s1 = torch.ones((2, 2, 1), dtype=torch.float8_e4m3fn)
        s2 = torch.tensor([0.5, 0.25], dtype=torch.float32)
        out3 = _nvfp4_dequant_batched(w3, s1, s2)
        self.assertTrue(torch.all(out3[0] == 1.0))  # 2.0 * 1.0 * 0.5 = 1.0
        self.assertTrue(torch.all(out3[1] == 0.5))  # 2.0 * 1.0 * 0.25 = 0.5

    def test_mtp_head_normalizes_before_output_projection(self):
        """Step3p7 MTP applies shared-head norm only when producing draft logits."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_step3p7 import Step3p7MTPHead
        from tensorrt_llm._torch.modules import rms_norm as rms_norm_module

        flashinfer_available = rms_norm_module.IS_FLASHINFER_AVAILABLE
        rms_norm_module.IS_FLASHINFER_AVAILABLE = False
        try:
            text_config = PretrainedConfig()
            text_config.hidden_size = 2
            text_config.rms_norm_eps = 0.0
            text_config.torch_dtype = torch.float32
            text_config.vocab_size = 2
            top_config = PretrainedConfig()
            top_config.text_config = text_config
            head = Step3p7MTPHead(ModelConfig(pretrained_config=top_config))

            class CaptureOutput(torch.nn.Module):
                gather_output = True

                def __init__(self):
                    super().__init__()
                    self.seen = None

                def forward(self, hidden_states, **kwargs):
                    del kwargs
                    self.seen = hidden_states.detach().clone()
                    return hidden_states

            output = CaptureOutput()
            head.output = output
            hidden_states = torch.tensor([[3.0, 4.0], [6.0, 8.0]], dtype=torch.float32)

            logits = head(hidden_states, lm_head=None, attn_metadata=None)

            expected = hidden_states[-1:] / hidden_states[-1:].pow(2).mean(-1, keepdim=True).sqrt()
            self.assertTrue(torch.allclose(output.seen, expected))
            self.assertTrue(torch.allclose(logits, expected))
        finally:
            rms_norm_module.IS_FLASHINFER_AVAILABLE = flashinfer_available


class TestStep3p7AutoModelRegistration(unittest.TestCase):
    """Verify the Step3p7 architectures resolve to the expected model classes."""

    def test_auto_model_registered(self):
        from tensorrt_llm._torch.models.modeling_step3p7 import Step3p7ForCausalLM
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VLForConditionalGeneration
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        # Step3p7ForConditionalGeneration is the multimodal entry point; the
        # text-only causal LM remains reachable via "Step3p5ForCausalLM".
        self.assertIs(
            MODEL_CLASS_MAPPING.get("Step3p7ForConditionalGeneration"),
            Step3p7VLForConditionalGeneration,
        )
        self.assertIs(MODEL_CLASS_MAPPING.get("Step3p5ForCausalLM"), Step3p7ForCausalLM)


class TestStep3p7Checkpoint(unittest.TestCase):
    """Config / weight-accounting tests against the real Step-3.7-Flash checkpoints.

    The FP8 block-scale, NVFP4, and BF16 reference checkpoints share the same
    per-layer geometry; only the routed-expert dtype/layout differs. The FP8 and
    BF16 checkpoints store text keys under ``model.layers.*`` with 3 plain-path
    MTP layers; the NVFP4 modelopt export stores text keys under
    ``model.language_model.layers.*`` and carries no MTP layers. The checkpoints
    are expected under ``LLM_MODELS_ROOT`` (as in CI); a missing checkpoint
    surfaces as a test failure rather than a silent skip.
    """

    def _check_text_config(self, text_cfg: dict):
        self.assertEqual(text_cfg["model_type"], "step3p5")
        self.assertEqual(text_cfg["num_hidden_layers"], 45)
        self.assertEqual(text_cfg["hidden_size"], 4096)
        self.assertEqual(text_cfg["vocab_size"], 128896)
        self.assertEqual(text_cfg["num_attention_heads"], 64)
        self.assertEqual(text_cfg["num_attention_groups"], 8)
        self.assertEqual(text_cfg["head_dim"], 128)
        self.assertEqual(text_cfg["moe_num_experts"], 288)
        self.assertEqual(text_cfg["moe_top_k"], 8)
        self.assertEqual(text_cfg["moe_router_scaling_factor"], 3.0)
        self.assertIs(text_cfg["use_head_wise_attn_gate"], True)
        self.assertIs(text_cfg["use_moe_router_bias"], True)
        self.assertIs(text_cfg["need_fp32_gate"], True)

    @parameterized.expand(
        [
            ("fp8", STEP3P7_FP8_DIR),
            ("nvfp4", STEP3P7_NVFP4_DIR),
            ("bf16", STEP3P7_BF16_DIR),
        ]
    )
    def test_config_and_weight_accounting(self, name, checkpoint_dir):
        """Recognize Step3p7 + account for every safetensors key.

        - architectures == ["Step3p7ForConditionalGeneration"], top
          model_type == "step3p7", text model_type == "step3p5"
        - 45 text decoder layers with the documented full/sliding pattern
        - FP8 (fp8 block-scale) and NVFP4 (modelopt) checkpoints carry a
          quantization_config block; the BF16 reference does not
        - All consumed text-path keys can be enumerated; only vision, plain-path
          MTP (layers 45..47, FP8/BF16 only), and the vision projector are
          ignored. The NVFP4 export nests text keys under
          ``model.language_model.*`` and vision keys under ``model.vision_model.*``.
        """
        import re

        is_fp8 = name == "fp8"
        is_nvfp4 = name == "nvfp4"
        config_dict = _load_config(checkpoint_dir)
        safetensors_keys = _load_safetensors_keys(checkpoint_dir)

        # 1. Top-level config sanity.
        self.assertEqual(config_dict["architectures"], ["Step3p7ForConditionalGeneration"])
        self.assertEqual(config_dict["model_type"], "step3p7")
        text_cfg = config_dict["text_config"]
        self._check_text_config(text_cfg)

        # 2. Layer inventory: 45 decoder layers, full at idx 0,4,8,...,44 and
        #    sliding elsewhere. The raw layer_types array can be longer (48
        #    entries) because it also covers the 3 MTP layers (45..47); the
        #    NVFP4 export has no MTP layers so it carries exactly 45 entries.
        layer_types = text_cfg["layer_types"]
        self.assertGreaterEqual(len(layer_types), 45)
        for idx in range(45):
            lt = layer_types[idx]
            if idx % 4 == 0:
                self.assertEqual(lt, "full_attention", f"layer {idx}")
            else:
                self.assertEqual(lt, "sliding_attention", f"layer {idx}")

        # 3. Quant config: FP8 and NVFP4 carry it, BF16 does not. The NVFP4
        #    export uses the modelopt schema (quant_algo + ``ignore`` list with
        #    ``model.language_model.*`` patterns) rather than the fp8 schema
        #    (weight_block_size + ``modules_to_not_convert``).
        if is_fp8:
            quant_cfg = config_dict["quantization_config"]
            self.assertEqual(quant_cfg["quant_method"], "fp8")
            self.assertEqual(quant_cfg["weight_block_size"], [128, 128])
            not_convert = set(quant_cfg["modules_to_not_convert"])
            for layer_idx in range(45):
                if layer_idx < 3:
                    # Dense MLP layers: gate/up/down all bf16.
                    for sub in (
                        "self_attn.q_proj",
                        "mlp.gate_proj",
                        "mlp.up_proj",
                        "mlp.down_proj",
                    ):
                        key = f"model.layers.{layer_idx}.{sub}"
                        self.assertIn(key, not_convert, f"dense layer {layer_idx} {sub}")
                else:
                    # MoE layers: routed gate/up/down are FP8; shared expert is bf16.
                    for sub in (
                        "self_attn.q_proj",
                        "moe.gate",
                        "share_expert.gate_proj",
                        "share_expert.up_proj",
                        "share_expert.down_proj",
                    ):
                        key = f"model.layers.{layer_idx}.{sub}"
                        self.assertIn(key, not_convert, f"MoE layer {layer_idx} {sub}")
        elif is_nvfp4:
            quant_cfg = config_dict["quantization_config"]
            self.assertEqual(quant_cfg["quant_method"], "modelopt")
            self.assertEqual(quant_cfg["quant_algo"], "NVFP4")
            # Routed-expert matmuls quantize to NVFP4; everything else (lm_head,
            # router gate, shared expert, attention) is excluded via the
            # ``ignore`` list keyed off the ``model.language_model.*`` namespace.
            ignore = set(quant_cfg["ignore"])
            self.assertIn("lm_head", ignore)
            self.assertTrue(
                any(e.startswith("model.language_model.layers.") for e in ignore),
                "expected language_model exclude entries in NVFP4 ignore list",
            )
        else:
            self.assertNotIn("quantization_config", config_dict)

        # 4. Weight accounting on the actual safetensors keys. The NVFP4 export
        #    nests the text decoder under ``model.language_model.*`` and the
        #    vision tower under ``model.vision_model.*``; FP8/BF16 use bare
        #    ``model.layers.*`` / ``vision_model.*`` prefixes.
        if is_nvfp4:
            vision_prefixes = ("model.vision_model.", "model.vit_large_projector")
            embed_norm_keys = (
                "model.language_model.embed_tokens.weight",
                "model.language_model.norm.weight",
                "lm_head.weight",
            )
            text_layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
        else:
            vision_prefixes = ("vision_model.", "vit_large_projector")
            embed_norm_keys = (
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
            )
            text_layer_re = re.compile(r"^model\.layers\.(\d+)\.")

        consumed_text_keys = []
        ignored_vision_keys = []
        ignored_mtp_keys = []
        for key in safetensors_keys:
            m = text_layer_re.match(key)
            if m:
                if int(m.group(1)) >= 45:
                    ignored_mtp_keys.append(key)
                else:
                    consumed_text_keys.append(key)
            elif key.startswith(vision_prefixes):
                ignored_vision_keys.append(key)
            elif key in embed_norm_keys:
                consumed_text_keys.append(key)
            else:
                self.fail(f"Unaccounted-for safetensors key: {key}")

        self.assertGreater(len(consumed_text_keys), 0, "no consumed text keys found")
        self.assertGreater(len(ignored_vision_keys), 0, "no ignored vision keys found")
        # FP8/BF16 ship 3 plain-path MTP layers (45..47); the NVFP4 export omits
        # them entirely.
        if is_nvfp4:
            self.assertEqual(len(ignored_mtp_keys), 0, "NVFP4 export should carry no MTP keys")
        else:
            self.assertGreater(len(ignored_mtp_keys), 0, "no ignored MTP keys (expected 45..47)")

    @parameterized.expand(
        [
            ("fp8", STEP3P7_FP8_DIR),
            ("nvfp4", STEP3P7_NVFP4_DIR),
            ("bf16", STEP3P7_BF16_DIR),
        ]
    )
    def test_model_config_resolves_registered_architecture(self, name, checkpoint_dir):
        """``ModelConfig.from_pretrained`` resolves Step3p7 and detects the quant algo."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm.quantization import QuantAlgo

        model_config = ModelConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        pc = model_config.pretrained_config
        self.assertEqual(pc.architectures, ["Step3p7ForConditionalGeneration"])
        self.assertEqual(pc.num_hidden_layers, 45)
        self.assertEqual(pc.vocab_size, 128896)
        self.assertEqual(pc.hidden_size, 4096)
        if name == "fp8":
            self.assertEqual(model_config.quant_config.quant_algo, QuantAlgo.FP8_BLOCK_SCALES)
        elif name == "nvfp4":
            self.assertEqual(model_config.quant_config.quant_algo, QuantAlgo.NVFP4)
            # NVFP4 modelopt export also carries an FP8 KV-cache scheme.
            self.assertEqual(model_config.quant_config.kv_cache_quant_algo, QuantAlgo.FP8)
        else:
            self.assertTrue(
                model_config.quant_config is None or model_config.quant_config.quant_algo is None
            )

    @parameterized.expand(
        [
            ("fp8", STEP3P7_FP8_DIR),
            ("nvfp4", STEP3P7_NVFP4_DIR),
            ("bf16", STEP3P7_BF16_DIR),
        ]
    )
    def test_per_layer_attention_geometry_helpers(self, name, checkpoint_dir):
        """Per-layer head / RoPE / SwiGLU helpers match config arithmetic.

        Runs against all three checkpoints because the per-layer attention
        geometry is identical between them: only the routed-expert dtype/layout
        differs. The layer-45 assertions exercise the out-of-range fallback path
        (the NVFP4 export carries only 45 ``layer_types`` entries; FP8/BF16
        include the 3 MTP layers), which resolves to the same sliding-attention
        geometry either way.
        """
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_step3p7 import (
            _is_moe_layer,
            _layer_attention_type,
            _layer_kv_heads,
            _layer_partial_rotary,
            _layer_query_heads,
            _layer_rope_theta,
            _layer_swiglu_limit,
            _layer_uses_rope_scaling,
        )

        model_config = ModelConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        text_config = model_config.pretrained_config.text_config

        # Full vs sliding head counts.
        self.assertEqual(_layer_query_heads(text_config, 0), 64)
        self.assertEqual(_layer_kv_heads(text_config, 0), 8)
        self.assertEqual(_layer_query_heads(text_config, 1), 96)
        self.assertEqual(_layer_kv_heads(text_config, 1), 8)
        self.assertEqual(_layer_attention_type(text_config, 0), "full_attention")
        self.assertEqual(_layer_attention_type(text_config, 1), "sliding_attention")
        self.assertEqual(_layer_attention_type(text_config, 45), "sliding_attention")
        self.assertEqual(_layer_query_heads(text_config, 45), 96)
        self.assertEqual(_layer_kv_heads(text_config, 45), 8)

        # RoPE per-layer.
        self.assertEqual(_layer_rope_theta(text_config, 0), 5_000_000.0)
        self.assertEqual(_layer_rope_theta(text_config, 1), 10_000.0)
        self.assertEqual(_layer_rope_theta(text_config, 45), 10_000.0)
        self.assertEqual(_layer_partial_rotary(text_config, 0), 0.5)
        self.assertEqual(_layer_partial_rotary(text_config, 1), 1.0)
        self.assertEqual(_layer_partial_rotary(text_config, 45), 1.0)
        self.assertIs(_layer_uses_rope_scaling(text_config, 0), True)
        self.assertIs(_layer_uses_rope_scaling(text_config, 1), False)

        # MoE vs dense.
        self.assertIs(_is_moe_layer(text_config, 0), False)
        self.assertIs(_is_moe_layer(text_config, 1), False)
        self.assertIs(_is_moe_layer(text_config, 2), False)
        self.assertIs(_is_moe_layer(text_config, 3), True)
        self.assertIs(_is_moe_layer(text_config, 44), True)

        # SwiGLU clamp limits: nonzero only on layers 43 and 44 (per config).
        self.assertIsNone(_layer_swiglu_limit(text_config, 0))
        self.assertEqual(_layer_swiglu_limit(text_config, 43), 7.0)
        self.assertEqual(_layer_swiglu_limit(text_config, 44), 7.0)
        self.assertIsNone(_layer_swiglu_limit(text_config, 45))
        self.assertEqual(_layer_swiglu_limit(text_config, 43, shared=True), 16.0)
        self.assertEqual(_layer_swiglu_limit(text_config, 44, shared=True), 16.0)

    def test_mtp_spec_config_defaults_to_checkpoint_layer_count(self):
        """Default ``MTPDecodingConfig`` loads all Step3p7 MTP layers."""
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.models.modeling_step3p7 import _prepare_step3p7_mtp_spec_config
        from tensorrt_llm._torch.speculative import update_spec_config_from_model_config
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        spec_config = MTPDecodingConfig()
        # Unset → None sentinel; resolved to the checkpoint layer count below.
        self.assertIsNone(spec_config.max_draft_len)
        self.assertNotIn("max_draft_len", spec_config.model_fields_set)
        model_config = ModelConfig.from_pretrained(
            STEP3P7_FP8_DIR, trust_remote_code=True, spec_config=spec_config
        )
        update_spec_config_from_model_config(spec_config, model_config.pretrained_config)
        _prepare_step3p7_mtp_spec_config(model_config)

        self.assertEqual(spec_config.num_nextn_predict_layers, 3)
        self.assertEqual(spec_config.max_draft_len, 3)
        self.assertEqual(spec_config.max_total_draft_tokens, 3)
        self.assertTrue(spec_config.spec_dec_mode.is_mtp_vanilla())

        explicit_spec_config = MTPDecodingConfig(max_draft_len=1)
        explicit_model_config = ModelConfig.from_pretrained(
            STEP3P7_FP8_DIR, trust_remote_code=True, spec_config=explicit_spec_config
        )
        update_spec_config_from_model_config(
            explicit_spec_config, explicit_model_config.pretrained_config
        )
        _prepare_step3p7_mtp_spec_config(explicit_model_config)

        self.assertEqual(explicit_spec_config.num_nextn_predict_layers, 3)
        self.assertEqual(explicit_spec_config.max_draft_len, 1)
        self.assertEqual(explicit_spec_config.max_total_draft_tokens, 1)


if __name__ == "__main__":
    unittest.main()
