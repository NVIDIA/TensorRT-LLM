# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.linear import Linear
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.components.transformer import (
    MiniMaxH3Transformer3DModel,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.weights import (
    map_runtime_lora_names,
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
    width = 32
    config = _make_model_config(
        num_layers=1, num_refiner_layers=1, hidden_size=width, ffn_dim=width, attention_head_dim=16
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
        rows = torch.arange(1, projection.out_features + 1, dtype=torch.float32)
        pairs[f"{adapter_name}.lora_A.weight"] = a
        pairs[f"{adapter_name}.lora_B.weight"] = (rows / (2 * width)).unsqueeze(1)

    # Rank-one A selects column zero. Turbo fc1 packs gate then up, and AdaLN packs shift then scale.
    first_half = torch.arange(1, width + 1, device="cuda", dtype=torch.float32) / (2 * width)
    second_half = first_half + 0.5
    gate, up = reference.transformer_blocks[0].ff.gate_up_proj.weight.chunk(2)
    gate[:, 0].add_(first_half)
    up[:, 0].add_(second_half)
    reference.token_refiner.refiner_blocks[0].attn.to_out[0].weight[:, 0].add_(first_half)
    shift, scale = reference.norm_out.linear.weight.chunk(2)
    shift[:, 0].add_(first_half)
    scale[:, 0].add_(second_half)

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
def test_turbo_lora_names_reach_every_h3_projection(tmp_path: Path) -> None:
    """Turbo names reach the trunk and refiner projections; to_q/to_k/to_v fold into qkv_proj."""
    model = MiniMaxH3Transformer3DModel(_make_model_config(num_layers=1, num_refiner_layers=1))
    for parameter in model.parameters():
        parameter.zero_()

    targets = {
        "blocks.0.attn.out_proj": "transformer_blocks.0.attn.to_out.0",
        "blocks.0.mlp.fc1": "transformer_blocks.0.ff.gate_up_proj",
        "blocks.0.mlp.fc2": "transformer_blocks.0.ff.down_proj",
        "blocks.0.adaln_proj.linear": "transformer_blocks.0.adaln_proj.linear",
        "token_refiner.blocks.0.attn.out_proj": "token_refiner.refiner_blocks.0.attn.to_out.0",
        "token_refiner.blocks.0.mlp.fc1": "token_refiner.refiner_blocks.0.ff.gate_up_proj",
        "token_refiner.blocks.0.mlp.fc2": "token_refiner.refiner_blocks.0.ff.down_proj",
        "final_layer.adaln_proj.linear": "norm_out.linear",
    }
    qkv_targets = {
        "blocks.0.attn": "transformer_blocks.0.attn.qkv_proj",
        "token_refiner.blocks.0.attn": "token_refiner.refiner_blocks.0.attn.qkv_proj",
    }
    pairs = {}
    expected = {}

    def add(adapter_name: str, in_features: int, column: torch.Tensor) -> None:
        a = torch.zeros(1, in_features)
        a[0, 0] = 1
        pairs[f"{adapter_name}.lora_A.weight"] = a
        pairs[f"{adapter_name}.lora_B.weight"] = column[:, None]

    for value, (adapter_name, module_name) in enumerate(targets.items(), start=1):
        projection = model.get_submodule(module_name)
        expected[module_name] = torch.full((projection.out_features,), float(value))
        add(adapter_name, projection.in_features, expected[module_name])
    for adapter_name, module_name in qkv_targets.items():
        projection = model.get_submodule(module_name)
        rows = projection.out_features // 3
        columns = [torch.full((rows,), value) for value in (9.0, 10.0, 11.0)]
        for suffix, column in zip(("to_q", "to_k", "to_v"), columns):
            add(f"{adapter_name}.{suffix}", projection.in_features, column)
        expected[module_name] = torch.cat(columns)

    path = tmp_path / "adapter.safetensors"
    save_file(pairs, str(path))
    application = apply_runtime_lora(model, RuntimeLoRAConfig(path=str(path)))

    assert set(application.applied_modules) == set(expected)
    assert application.skipped_non_targets == application.skipped_incomplete == 0
    for module_name, column in expected.items():
        weight = model.get_submodule(module_name).weight
        torch.testing.assert_close(weight[:, 0], column.to(weight.dtype), rtol=0, atol=0)
        assert weight[:, 1:].count_nonzero().item() == 0, module_name


_TURBO_LAYOUT = (
    "blocks.0.mlp.fc1",
    "token_refiner.blocks.0.mlp.fc1",
    "final_layer.adaln_proj.linear",
)


@pytest.mark.parametrize("missing", _TURBO_LAYOUT)
def test_turbo_lora_names_map_only_with_the_full_turbo_layout(missing: str) -> None:
    """Without any one part of the Turbo layout, adapter names stay unmapped."""
    names = {name: name for name in _TURBO_LAYOUT if name != missing}
    assert map_runtime_lora_names(names) == {}


@torch.inference_mode()
def test_runtime_lora_uses_h3_projection_tp_slices(tmp_path: Path, monkeypatch) -> None:
    """Rank 1 owns two of six heads under TP2 × U2. Fused TP gate/up is ineligible."""
    with monkeypatch.context() as patch:
        patch.delenv("TLLM_DISABLE_MPI", raising=False)
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
