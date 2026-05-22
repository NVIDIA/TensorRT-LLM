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
"""Unit tests for _compute_num_lora_modules (per-layer LoRA cache sizing).

Verifies that models with variable per-layer structure (e.g. Nemotron-NAS)
get accurate LoRA module-layer counts instead of the uniform over-estimate.
"""

import unittest
from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor._util import _compute_num_lora_modules


def _make_block_config(
    attn_no_op=False, attn_linear=False, ffn_no_op=False, ffn_linear=False, ffn_mult=2.0
):
    return SimpleNamespace(
        attention=SimpleNamespace(
            no_op=attn_no_op, replace_with_linear=attn_linear, n_heads_in_group=8
        ),
        ffn=SimpleNamespace(no_op=ffn_no_op, replace_with_linear=ffn_linear, ffn_mult=ffn_mult),
    )


def _make_pretrained_config(num_hidden_layers, block_configs=None):
    cfg = SimpleNamespace(num_hidden_layers=num_hidden_layers)
    if block_configs is not None:
        cfg.block_configs = block_configs
    return cfg


class TestComputeNumLoraModules(unittest.TestCase):
    def test_uniform_model_no_block_configs(self):
        """Standard model without block_configs uses num_layers x num_modules."""
        config = _make_pretrained_config(32)
        modules = ["attn_q", "attn_k", "attn_v", "mlp_h_to_4h", "mlp_4h_to_h"]
        self.assertEqual(_compute_num_lora_modules(config, modules), 32 * 5)

    def test_all_layers_active(self):
        """block_configs present but all layers fully active gives same as uniform."""
        blocks = [_make_block_config() for _ in range(10)]
        config = _make_pretrained_config(10, blocks)
        modules = ["attn_q", "attn_k", "attn_v", "mlp_h_to_4h", "mlp_4h_to_h"]
        self.assertEqual(_compute_num_lora_modules(config, modules), 10 * 5)

    def test_no_op_ffn_layers_excluded(self):
        """Layers with no_op FFN are excluded from MLP module count."""
        blocks = [
            _make_block_config(),
            _make_block_config(),
            _make_block_config(ffn_no_op=True),
            _make_block_config(ffn_no_op=True),
        ]
        config = _make_pretrained_config(4, blocks)
        modules = ["attn_q", "attn_v", "mlp_h_to_4h", "mlp_4h_to_h"]
        # attn: 4 layers x 2 modules = 8, mlp: 2 layers x 2 modules = 4
        self.assertEqual(_compute_num_lora_modules(config, modules), 12)

    def test_no_op_attn_layers_excluded(self):
        """Layers with no_op attention are excluded from attention module count."""
        blocks = [
            _make_block_config(),
            _make_block_config(attn_no_op=True),
            _make_block_config(attn_no_op=True),
            _make_block_config(),
        ]
        config = _make_pretrained_config(4, blocks)
        modules = ["attn_q", "attn_k", "attn_v", "mlp_gate"]
        # attn: 2 layers x 3 modules = 6, mlp: 4 layers x 1 module = 4
        self.assertEqual(_compute_num_lora_modules(config, modules), 10)

    def test_linear_stub_layers_excluded(self):
        """Layers with replace_with_linear are excluded (LoRA not supported)."""
        blocks = [
            _make_block_config(),
            _make_block_config(attn_linear=True),
            _make_block_config(ffn_linear=True),
            _make_block_config(attn_linear=True, ffn_linear=True),
        ]
        config = _make_pretrained_config(4, blocks)
        modules = ["attn_q", "attn_k", "mlp_h_to_4h"]
        # attn: layers 0,2 active = 2x2 = 4
        # mlp: layers 0,1 active = 2x1 = 2
        self.assertEqual(_compute_num_lora_modules(config, modules), 6)

    def test_nemotron_nas_mini_pattern(self):
        """Reproduces the 14-layer Nemotron-NAS mini config pattern."""
        blocks = [
            _make_block_config(),  # 0: full/full
            _make_block_config(),  # 1: full/full
            _make_block_config(attn_linear=True),  # 2: linear/full
            _make_block_config(attn_no_op=True),  # 3: no_op/full
            _make_block_config(ffn_linear=True),  # 4: full/linear
            _make_block_config(ffn_no_op=True),  # 5: full/no_op
            _make_block_config(),  # 6: full/full
            _make_block_config(),  # 7: full/full
            _make_block_config(),  # 8: full/full
            _make_block_config(attn_linear=True),  # 9: linear/full
            _make_block_config(attn_no_op=True),  # 10: no_op/full
            _make_block_config(ffn_linear=True),  # 11: full/linear
            _make_block_config(ffn_no_op=True),  # 12: full/no_op
            _make_block_config(),  # 13: full/full
        ]
        config = _make_pretrained_config(14, blocks)
        modules = ["attn_q", "attn_k", "attn_v", "mlp_h_to_4h", "mlp_4h_to_h", "mlp_gate"]

        # Layers with LoRA-capable attention (not no_op, not linear):
        #   0,1,4,5,6,7,8,11,12,13 = 10 layers
        # Layers with LoRA-capable MLP (not no_op, not linear):
        #   0,1,2,3,6,7,8,9,10,13 = 10 layers
        expected = 10 * 3 + 10 * 3  # 60
        uniform = 14 * 6  # 84
        result = _compute_num_lora_modules(config, modules)
        self.assertEqual(result, expected)
        self.assertLess(result, uniform)

    def test_attn_only_modules_ignore_ffn(self):
        """When only targeting attention modules, FFN structure is irrelevant."""
        blocks = [
            _make_block_config(),
            _make_block_config(ffn_no_op=True),
            _make_block_config(ffn_linear=True),
        ]
        config = _make_pretrained_config(3, blocks)
        modules = ["attn_q", "attn_k", "attn_v"]
        self.assertEqual(_compute_num_lora_modules(config, modules), 3 * 3)

    def test_other_modules_use_all_layers(self):
        """Unrecognized module names fall back to counting all layers."""
        blocks = [
            _make_block_config(attn_no_op=True, ffn_no_op=True),
            _make_block_config(attn_no_op=True, ffn_no_op=True),
        ]
        config = _make_pretrained_config(2, blocks)
        modules = ["moe_router", "mamba_in_proj"]
        # Neither attn nor mlp, so all layers are counted
        self.assertEqual(_compute_num_lora_modules(config, modules), 2 * 2)

    def test_empty_modules(self):
        """No target modules gives zero."""
        config = _make_pretrained_config(10)
        self.assertEqual(_compute_num_lora_modules(config, []), 0)


if __name__ == "__main__":
    unittest.main()
