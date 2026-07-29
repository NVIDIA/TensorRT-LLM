# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import safetensors.torch
import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.lora import apply_static_lora, load_lora_deltas
from tensorrt_llm.visual_gen.args import LoRAConfig, VisualGenArgs


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        self.qkv_proj = nn.Linear(3, 6, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.qkv_proj.weight)


def test_visual_gen_args_accept_lora_config():
    args = VisualGenArgs(
        lora_config={
            "path": "/tmp/lora.safetensors",
            "strength": 0.5,
            "strip_prefixes": ["transformer."],
        }
    )

    assert isinstance(args.lora_config, LoRAConfig)
    assert args.lora_config.path == "/tmp/lora.safetensors"
    assert args.lora_config.strength == 0.5


def test_apply_static_lora_merges_bf16_delta(tmp_path):
    lora_path = tmp_path / "adapter.safetensors"
    down = torch.tensor([[1.0, 2.0, 3.0]])
    up = torch.tensor([[2.0], [4.0]])
    safetensors.torch.save_file(
        {
            "transformer.proj.lora_A.weight": down,
            "transformer.proj.lora_B.weight": up,
            "transformer.proj.alpha": torch.tensor(2.0),
        },
        str(lora_path),
    )
    module = TinyTransformer()
    config = LoRAConfig(path=str(lora_path), strength=0.5)

    assert apply_static_lora(module, config) == 1

    torch.testing.assert_close(module.proj.weight.float(), up @ down)


def test_load_lora_deltas_fuses_qkv(tmp_path):
    lora_path = tmp_path / "adapter.safetensors"
    tensors = {}
    for name, value in (("to_q", 1.0), ("to_k", 2.0), ("to_v", 3.0)):
        tensors[f"transformer.block.{name}.lora_A.weight"] = torch.ones(1, 3)
        tensors[f"transformer.block.{name}.lora_B.weight"] = torch.full((2, 1), value)
    safetensors.torch.save_file(tensors, str(lora_path))
    module = TinyTransformer()
    module.block = nn.Module()
    module.block.qkv_proj = module.qkv_proj

    deltas = load_lora_deltas(str(lora_path), module)

    assert set(deltas) == {"block.qkv_proj"}
    assert tuple(deltas["block.qkv_proj"].shape) == (6, 3)
