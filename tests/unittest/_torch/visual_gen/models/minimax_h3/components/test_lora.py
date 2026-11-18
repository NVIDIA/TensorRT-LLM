# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.models.minimax_h3.components.linear import Linear
from tensorrt_llm._torch.visual_gen.models.minimax_h3.components.transformer import (
    MiniMaxH3Transformer3DModel,
)
from tensorrt_llm._torch.visual_gen.runtime_lora import apply_runtime_lora
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.visual_gen.args import RuntimeLoRAConfig

from .test_minimax_h3_transformer import (
    _diffusers_golden_inputs,
    _initialize_weights,
    _make_model_config,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@torch.inference_mode()
def test_turbo_lora_matches_premerged_h3(tmp_path: Path) -> None:
    """Compare both output streams with the same adapter merged before model preparation."""
    config = _make_model_config(
        num_layers=1, num_refiner_layers=1, hidden_size=32, ffn_dim=32, attention_head_dim=16
    )
    base = MiniMaxH3Transformer3DModel(config).cuda().eval()
    _initialize_weights(base, scale=0.1)
    actual = MiniMaxH3Transformer3DModel(config).cuda().eval()
    actual.load_state_dict(base.state_dict())
    reference = MiniMaxH3Transformer3DModel(config).cuda().eval()
    reference.load_state_dict(base.state_dict())

    targets = {
        "blocks.0.mlp.fc1": "transformer_blocks.0.ff.gate_up_proj",
        "token_refiner.blocks.0.attn.out_proj": "token_refiner.refiner_blocks.0.attn.to_out.0",
        "final_layer.adaln_proj.linear": "norm_out.linear",
    }
    pairs = {}
    for adapter_name, module_name in targets.items():
        projection = reference.get_submodule(module_name)
        a = torch.zeros(1, projection.in_features)
        a[0, 0] = 1
        b = (torch.arange(projection.out_features) % 2 * 2 - 1).float().unsqueeze(1)
        pairs[f"{adapter_name}.lora_A.weight"] = a
        pairs[f"{adapter_name}.lora_B.weight"] = b
        projection.weight[:, 0].add_(b.flatten().to(projection.weight))

    path = tmp_path / "adapter.safetensors"
    save_file(pairs, str(path))
    for model in (base, actual, reference):
        model.post_load_weights()
    application = apply_runtime_lora(actual, RuntimeLoRAConfig(path=str(path)))
    assert set(application.applied_modules) == set(targets.values())
    assert application.skipped_non_targets == application.skipped_incomplete == 0
    for module_name in targets.values():
        torch.testing.assert_close(
            actual.get_submodule(module_name).weight,
            reference.get_submodule(module_name).weight,
            rtol=0,
            atol=0,
        )

    inputs = _diffusers_golden_inputs("cuda")
    before, expected, output = (model(**inputs) for model in (base, reference, actual))
    for stream in ("sample", "audio_sample"):
        baseline = getattr(before, stream)
        merged = getattr(expected, stream)
        assert not torch.equal(baseline, merged), stream
        torch.testing.assert_close(getattr(output, stream), merged)


@torch.inference_mode()
def test_runtime_lora_uses_h3_projection_tp_slices(tmp_path: Path) -> None:
    """Rank 1 owns two of six heads under TP2 × U2. Fused TP gate/up is ineligible."""
    mapping = Mapping(world_size=2, tp_size=2, rank=1, gpus_per_node=2)

    def projection(mode, width, shard):
        return Linear(
            6,
            width,
            bias=False,
            dtype=torch.float32,
            mapping=mapping,
            tensor_parallel_mode=mode,
            override_tp_sharding=shard,
            reduce_output=False,
        )

    layers = torch.nn.ModuleDict(
        {
            "query": projection("column", 6, (4, 6)),
            "output": projection("row", 6, (4, 6)),
            "gate_up": Linear(
                6,
                12,
                bias=False,
                dtype=torch.float32,
                mapping=mapping,
                tensor_parallel_mode="column",
                reduce_output=False,
                override_tp_sharding={"gate": (4, 6), "up": (4, 6)},
                fused_weight_shard_indices_mapping={"gate": (0, 2), "up": (2, 2)},
            ),
        }
    )
    checkpoint = {}
    for name, layer in layers.items():
        layer.weight.zero_()
        checkpoint[f"{name}.lora_A.weight"] = torch.arange(1, 7, dtype=torch.float32)[None]
        rows = 12 if name == "gate_up" else 6
        checkpoint[f"{name}.lora_B.weight"] = torch.arange(rows, dtype=torch.float32)[:, None]
    path = tmp_path / "adapter.safetensors"
    save_file(checkpoint, str(path))
    report = apply_runtime_lora(layers, RuntimeLoRAConfig(path=str(path), strict=False))
    assert set(report.applied_modules) == {"query", "output"}
    assert report.skipped_non_targets == 1
    delta = checkpoint["query.lora_B.weight"] @ checkpoint["query.lora_A.weight"]
    torch.testing.assert_close(layers["query"].weight, delta[4:])
    torch.testing.assert_close(layers["output"].weight, delta[:, 4:])
    assert layers["gate_up"].weight.count_nonzero().item() == 0
