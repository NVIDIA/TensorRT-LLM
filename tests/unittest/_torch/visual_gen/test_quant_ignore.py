# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent configuration and executed BF16/quantized routing tests."""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from test_common.visual_gen_quantization import (
    ToyDiT,
    assert_quantization_routes,
    observe_quantization_paths,
)

from tensorrt_llm._torch.visual_gen.attention.backends.utils import is_quant_excluded
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm.visual_gen import LinearQuantConfig
from tensorrt_llm.visual_gen.args import QuantAttentionConfig, VisualGenArgs


@pytest.mark.cpu_only
@pytest.mark.parametrize("steps", [[-1], [True], [1.5], ["1"], "0,1", None])
def test_quant_ignore_config_validation(steps: object) -> None:
    for factory in (
        lambda: LinearQuantConfig(ignore_steps=steps),
        lambda: QuantAttentionConfig(ignore_steps=steps),
        lambda: VisualGenArgs(quant_config={"quant_algo": "NVFP4", "ignore_steps": steps}),
    ):
        with pytest.raises(ValidationError):
            factory()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode",
    ["block", "linear_quant_ignore_steps", "attention_quant_ignore_steps", "independent_steps"],
)
def test_quant_ignore_config_yaml_roundtrip(tmp_path: Path, mode: str) -> None:
    linear_ignore = ["blocks.0.ffn.*"] if mode == "block" else []
    attention_ignore = ["blocks.1.attn1"] if mode == "block" else []
    linear_quant_ignore_steps = (
        [0, 19] if mode in ("linear_quant_ignore_steps", "independent_steps") else []
    )
    attention_quant_ignore_steps = (
        [1, 18] if mode in ("attention_quant_ignore_steps", "independent_steps") else []
    )
    path = tmp_path / "quant_ignore.yaml"
    path.write_text(
        "quant_config:\n  quant_algo: NVFP4\n  dynamic: true\n"
        f"  ignore: {linear_ignore}\n  ignore_steps: {linear_quant_ignore_steps}\n"
        "attention_config:\n  backend: FLASHINFER\n  quant_attention_config:\n"
        "    qk_dtype: nvfp4\n    v_dtype: fp8\n"
        f"    ignore: {attention_ignore}\n    ignore_steps: {attention_quant_ignore_steps}\n"
    )
    args = VisualGenArgs.from_yaml(path)
    recipe = args.attention_config.quant_attention_config
    assert recipe.ignore == attention_ignore
    assert recipe.ignore_steps == attention_quant_ignore_steps
    quant, _, dynamic, activations = DiffusionPipelineConfig.load_diffusion_quant_config(
        args.quant_config
    )
    assert quant.exclude_modules == linear_ignore
    assert quant.ignore_steps == linear_quant_ignore_steps
    assert dynamic and activations
    assert VisualGenArgs.model_validate(args.model_dump()).model_dump() == args.model_dump()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "linear_ignore,attention_ignore,linear_quant_ignore_steps,attention_quant_ignore_steps,typed",
    [
        (["blocks.0.*"], [], [0], [], False),
        (["blocks.0.*"], [], [], [0], True),
        ([], ["blocks.0.*"], [0], [], True),
        ([], ["blocks.0.*"], [], [0], False),
        (["blocks.0.*"], ["blocks.1.*"], [], [], True),
        ([], [], [0], [], False),
        ([], [], [], [1], True),
        ([], [], [0], [1], False),
    ],
)
def test_block_and_step_modes_are_globally_exclusive(
    linear_ignore: list[str],
    attention_ignore: list[str],
    linear_quant_ignore_steps: list[int],
    attention_quant_ignore_steps: list[int],
    typed: bool,
) -> None:
    quant = (
        LinearQuantConfig(
            quant_algo="NVFP4",
            exclude_modules=linear_ignore,
            ignore_steps=linear_quant_ignore_steps,
        )
        if typed
        else {
            "quant_algo": "NVFP4",
            "ignore": linear_ignore,
            "ignore_steps": linear_quant_ignore_steps,
        }
    )
    config = dict(
        quant_config=quant,
        attention_config={
            "backend": "FLASHINFER",
            "quant_attention_config": {
                "qk_dtype": "nvfp4",
                "v_dtype": "fp8",
                "ignore": attention_ignore,
                "ignore_steps": attention_quant_ignore_steps,
            },
        },
    )
    if (linear_ignore or attention_ignore) and (
        linear_quant_ignore_steps or attention_quant_ignore_steps
    ):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VisualGenArgs(**config)
    else:
        VisualGenArgs(**config)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config",
    [
        {"quant_algo": "NVFP4", "dynamic": False, "ignore_steps": [0]},
        {"ignore_steps": [0]},
    ],
)
def test_linear_steps_require_load_time_quantization(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="dynamic weight|explicit quant_algo"):
        DiffusionPipelineConfig.load_diffusion_quant_config(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "algo",
    [
        "FP8",
        "FP8_BLOCK_SCALES",
        "FP8_PER_CHANNEL_PER_TOKEN",
        "NVFP4",
        "W4A16_AWQ",
        "W4A8_AWQ",
        "W8A8_SQ_PER_CHANNEL",
    ],
)
def test_linear_step_quant_algo_support_is_validated_by_loader(algo: str, typed: bool) -> None:
    config = {"quant_algo": algo, "ignore_steps": [0]}
    quant = LinearQuantConfig(**config) if typed else config
    VisualGenArgs(quant_config=quant)
    parsed, _, dynamic, _ = DiffusionPipelineConfig._resolve_quant_config(quant, None)
    assert parsed.quant_algo == algo
    assert parsed.ignore_steps == [0]
    assert dynamic


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "algorithm,activation_dynamic",
    [
        ("FP8", False),
        ("FP8_BLOCK_SCALES", False),
        ("FP8_PER_CHANNEL_PER_TOKEN", True),
        ("NVFP4", True),
    ],
)
def test_typed_step_quantization_uses_shared_parser_defaults(
    algorithm: str, activation_dynamic: bool
) -> None:
    typed = LinearQuantConfig(quant_algo=algorithm, ignore_steps=[0, 2])
    parsed, *modes = DiffusionPipelineConfig._resolve_quant_config(typed, None)
    _, *dict_modes = DiffusionPipelineConfig._resolve_quant_config(
        {"quant_algo": algorithm, "dynamic": True, "ignore_steps": [0, 2]}, None
    )
    assert parsed.model_dump() == typed.model_dump()
    assert modes == dict_modes == [None, True, activation_dynamic]


@pytest.mark.cpu_only
@pytest.mark.parametrize("algorithm", ["NVFP4", "INT8"])
@pytest.mark.parametrize("ignore_steps", [[], [0]])
def test_typed_quantization_preserves_explicit_fields_and_loading_mode(
    algorithm: str, ignore_steps: list[int]
) -> None:
    typed = LinearQuantConfig(
        quant_algo=algorithm,
        group_size=64,
        exclude_modules=["blocks.0.*"] if not ignore_steps else None,
        ignore_steps=ignore_steps,
    )
    original = typed.model_dump()
    parsed, layer_config, dynamic_weights, dynamic_activations = (
        DiffusionPipelineConfig._resolve_quant_config(typed, None)
    )
    assert parsed.model_dump() == original == typed.model_dump()
    assert layer_config is None
    assert dynamic_weights is bool(ignore_steps)
    assert dynamic_activations is (algorithm == "NVFP4" and bool(ignore_steps))


@pytest.mark.cpu_only
@pytest.mark.parametrize("attention_only", [False, True])
def test_ignore_steps_reject_diffusion_caching(attention_only: bool) -> None:
    config = (
        {
            "attention_config": {
                "backend": "CUTEDSL",
                "quant_attention_config": {"ignore_steps": [0]},
            }
        }
        if attention_only
        else {"quant_config": {"quant_algo": "NVFP4", "dynamic": True, "ignore_steps": [0]}}
    )
    with pytest.raises(ValidationError, match="diffusion step caching"):
        VisualGenArgs(
            **config,
            cache_config={"cache_backend": "teacache"},
        )


@pytest.mark.cpu_only
def test_linear_ignore_steps_reject_runtime_lora() -> None:
    with pytest.raises(ValidationError, match="runtime LoRA fusion"):
        VisualGenArgs(
            quant_config={"quant_algo": "NVFP4", "dynamic": True, "ignore_steps": [0]},
            runtime_lora_config={"path": "adapter.safetensors"},
        )


@pytest.mark.cpu_only
def test_attention_steps_do_not_require_linear_quantization_or_restrict_lora() -> None:
    args = VisualGenArgs(
        attention_config={"backend": "CUTEDSL", "quant_attention_config": {"ignore_steps": [0]}},
        runtime_lora_config={"path": "adapter.safetensors"},
    )
    assert args.quant_config is None or args.quant_config.quant_algo is None
    assert args.attention_config.quant_attention_config.ignore_steps == [0]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "pattern,matched",
    [
        ("blocks.1.attn1", True),
        ("blocks.1.*", True),
        ("transformer.blocks.1.attn1", True),
        ("blocks.1", True),
        ("blocks.1.attn1.*", True),
        (r"re:blocks\.1\.attn1$", True),
        ("blocks.10.*", False),
        ("blocks.1.attn2", False),
    ],
)
def test_attention_ignore_paths(pattern: str, matched: bool) -> None:
    recipe = QuantAttentionConfig(ignore=[pattern])
    assert is_quant_excluded(recipe, "transformer.blocks.1.attn1") == matched


@pytest.mark.cpu_only
@pytest.mark.parametrize("component", ["linear", "attention"])
def test_block_quant_ignore_routes(component: str) -> None:
    torch.manual_seed(42)
    model = ToyDiT(
        linear_ignore=("blocks.0.linear",) if component == "linear" else (),
        attention_ignore=("blocks.0.attention",) if component == "attention" else (),
        quantized_linear=component == "linear",
        quantized_attention=component == "attention",
    )
    with observe_quantization_paths(model) as trace:
        for step in range(2):
            trace.step = step
            model(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    expected = {
        (step, f"blocks.{block}.{name}", kind): "quantized"
        if kind == component and block == 1
        else "bf16"
        for step in range(2)
        for block in range(2)
        for name, kind in (("linear", "linear"), ("attention", "attention"))
    }
    assert_quantization_routes(trace, expected)
    assert all("_unquantized_weight" not in block.linear._buffers for block in model.blocks)
