# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-independent configuration and executed BF16/quantized routing tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from test_common.visual_gen_quantization import (
    ToyAttentionBackend,
    ToyDiT,
    assert_quantization_routes,
    observe_quantization_paths,
)

from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention.backends.quant_dispatcher import (
    AttentionQuantDispatcher,
)
from tensorrt_llm._torch.visual_gen.attention.backends.utils import (
    create_attention,
    is_quant_excluded,
)
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.quantization.quant_dispatcher import (
    install_linear_quant_dispatcher,
)
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    DiffusionQuantConfig,
    QuantAttentionConfig,
    VisualGenArgs,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("steps", [[-1], [True], [1.5], ["1"], "0,1", None])
def test_quant_ignore_config_validation(steps: object) -> None:
    for factory in (
        lambda: DiffusionQuantConfig(ignored_steps=steps),
        lambda: QuantAttentionConfig(ignored_steps=steps),
        lambda: VisualGenArgs(quant_config={"quant_algo": "NVFP4", "ignored_steps": steps}),
    ):
        with pytest.raises(ValidationError):
            factory()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "mode",
    ["block", "linear_quant_ignored_steps", "attention_quant_ignored_steps", "independent_steps"],
)
def test_quant_ignore_config_yaml_roundtrip(tmp_path: Path, mode: str) -> None:
    linear_ignore = ["blocks.0.ffn.*"] if mode == "block" else []
    attention_ignore = ["blocks.1.attn1"] if mode == "block" else []
    linear_quant_ignored_steps = (
        [0, 19] if mode in ("linear_quant_ignored_steps", "independent_steps") else []
    )
    attention_quant_ignored_steps = (
        [1, 18] if mode in ("attention_quant_ignored_steps", "independent_steps") else []
    )
    path = tmp_path / "quant_ignore.yaml"
    path.write_text(
        "quant_config:\n  quant_algo: NVFP4\n  dynamic: true\n"
        f"  ignore: {linear_ignore}\n  ignored_steps: {linear_quant_ignored_steps}\n"
        "attention_config:\n  backend: FLASHINFER\n  quant_attention_config:\n"
        "    qk_dtype: nvfp4\n    v_dtype: fp8\n"
        f"    ignore: {attention_ignore}\n    ignored_steps: {attention_quant_ignored_steps}\n"
    )
    args = VisualGenArgs.from_yaml(path)
    recipe = args.attention_config.quant_attention_config
    assert recipe.ignore == attention_ignore
    assert recipe.ignored_steps == attention_quant_ignored_steps
    quant, _, dynamic, activations = DiffusionPipelineConfig.load_diffusion_quant_config(
        args.quant_config
    )
    assert quant.exclude_modules == linear_ignore
    assert quant.ignored_steps == linear_quant_ignored_steps
    assert dynamic and activations
    assert VisualGenArgs.model_validate(args.model_dump()).model_dump() == args.model_dump()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "linear_ignore,attention_ignore,linear_quant_ignored_steps,attention_quant_ignored_steps,typed",
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
    linear_quant_ignored_steps: list[int],
    attention_quant_ignored_steps: list[int],
    typed: bool,
) -> None:
    quant = (
        DiffusionQuantConfig(
            quant_algo="NVFP4",
            exclude_modules=linear_ignore,
            ignored_steps=linear_quant_ignored_steps,
        )
        if typed
        else {
            "quant_algo": "NVFP4",
            "ignore": linear_ignore,
            "ignored_steps": linear_quant_ignored_steps,
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
                "ignored_steps": attention_quant_ignored_steps,
            },
        },
    )
    if (linear_ignore or attention_ignore) and (
        linear_quant_ignored_steps or attention_quant_ignored_steps
    ):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            VisualGenArgs(**config)
    else:
        VisualGenArgs(**config)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config",
    [
        {"quant_algo": "NVFP4", "dynamic": False, "ignored_steps": [0]},
        {"ignored_steps": [0]},
    ],
)
def test_linear_steps_require_load_time_quantization(config: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="dynamic weight|explicit quant_algo"):
        DiffusionPipelineConfig.load_diffusion_quant_config(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize(
    "algo,valid",
    [
        ("FP8", True),
        ("FP8_BLOCK_SCALES", True),
        ("FP8_PER_CHANNEL_PER_TOKEN", True),
        ("NVFP4", True),
        ("W4A16_AWQ", False),
        ("W4A8_AWQ", False),
        ("W8A8_SQ_PER_CHANNEL", False),
    ],
)
def test_linear_ignored_steps_supported_quant_algorithms(
    algo: str, valid: bool, typed: bool
) -> None:
    config = {"quant_algo": algo, "ignored_steps": [0]}
    if valid:
        quant = DiffusionQuantConfig(**config) if typed else config
        VisualGenArgs(quant_config=quant)
    else:
        with pytest.raises(ValidationError, match="requires load-time quantization"):
            quant = DiffusionQuantConfig(**config) if typed else config
            VisualGenArgs(quant_config=quant)
        with pytest.raises(ValidationError, match="requires load-time quantization"):
            DiffusionPipelineConfig.load_diffusion_quant_config(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("attention_only", [False, True])
def test_ignored_steps_reject_diffusion_caching(attention_only: bool) -> None:
    config = (
        {
            "attention_config": {
                "backend": "CUTEDSL",
                "quant_attention_config": {"ignored_steps": [0]},
            }
        }
        if attention_only
        else {"quant_config": {"quant_algo": "NVFP4", "dynamic": True, "ignored_steps": [0]}}
    )
    with pytest.raises(ValidationError, match="diffusion step caching"):
        VisualGenArgs(
            **config,
            cache_config={"cache_backend": "teacache"},
        )


@pytest.mark.cpu_only
def test_linear_ignored_steps_reject_runtime_lora() -> None:
    with pytest.raises(ValidationError, match="runtime LoRA fusion"):
        VisualGenArgs(
            quant_config={"quant_algo": "NVFP4", "dynamic": True, "ignored_steps": [0]},
            runtime_lora_config={"path": "adapter.safetensors"},
        )


@pytest.mark.cpu_only
def test_attention_steps_do_not_require_linear_quantization_or_restrict_lora() -> None:
    args = VisualGenArgs(
        attention_config={"backend": "CUTEDSL", "quant_attention_config": {"ignored_steps": [0]}},
        runtime_lora_config={"path": "adapter.safetensors"},
    )
    assert args.quant_config is None or args.quant_config.quant_algo is None
    assert args.attention_config.quant_attention_config.ignored_steps == [0]


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


@pytest.mark.cpu_only
@pytest.mark.parametrize("component", ["linear", "attention"])
def test_step_quant_ignore_routes(component: str) -> None:
    torch.manual_seed(42)
    model = ToyDiT(
        linear_quant_ignored_steps=(1,) if component == "linear" else (),
        attention_quant_ignored_steps=(1,) if component == "attention" else (),
        quantized_linear=component == "linear",
        quantized_attention=component == "attention",
    )
    state = model.model_config.attention_metadata_state
    with observe_quantization_paths(model) as trace:
        for step in range(3):
            trace.step = step
            state[f"{component}_quant_ignored"] = step == 1
            model(torch.randn(1, 8, 128, dtype=torch.bfloat16))
    expected = {
        (step, f"blocks.{block}.{name}", kind): "quantized"
        if kind == component and step != 1
        else "bf16"
        for step in range(3)
        for block in range(2)
        for name, kind in (("linear", "linear"), ("attention", "attention"))
    }
    assert_quantization_routes(trace, expected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("output_slice", [None, (16, 80)], ids=["whole", "slice"])
def test_linear_quant_ignore_uses_original_bf16_weight(
    output_slice: tuple[int, int] | None,
) -> None:
    model = ToyDiT(linear_quant_ignored_steps=(1,), quantized_attention=False)
    layer = model.blocks[0].linear
    layer.bias = torch.nn.Parameter(torch.randn(128, dtype=torch.bfloat16), requires_grad=False)
    method = layer.quant_method
    original = layer._unquantized_weight
    inputs = torch.randn(2, 128, dtype=torch.bfloat16)
    model.model_config.attention_metadata_state["linear_quant_ignored"] = True
    with (
        patch.object(
            method.unquantized_method, "apply", wraps=method.unquantized_method.apply
        ) as bf16,
        patch.object(
            method.quantized_method, "apply", wraps=method.quantized_method.apply
        ) as quantized,
    ):
        if output_slice is None:
            method.apply(layer, inputs, layer.bias)
        else:
            method.apply_slice(layer, inputs, *output_slice)
        bf16.assert_called_once()
        quantized.assert_not_called()
        actual_weight = bf16.call_args.kwargs["weight"]
        actual_bias = bf16.call_args.args[2]
        assert actual_weight.dtype == torch.bfloat16
        assert actual_bias.dtype == torch.bfloat16
        if output_slice is None:
            assert actual_weight is original
            assert actual_bias is layer.bias
        else:
            assert actual_weight.data_ptr() == original[output_slice[0] :].data_ptr()
            assert actual_weight.shape[0] == output_slice[1] - output_slice[0]
            assert actual_bias.data_ptr() == layer.bias[output_slice[0] :].data_ptr()
            assert actual_bias.shape[0] == actual_weight.shape[0]
        model.model_config.attention_metadata_state["linear_quant_ignored"] = False
        method.apply(layer, inputs, layer.bias)
        quantized.assert_called_once()
        assert bf16.call_count == 1
    assert method.requires_unquantized_activation and not method.quantizes_nvfp4_activations
    assert not layer.use_fused_gemm_allreduce
    layer.to(torch.float32)
    assert layer._unquantized_weight.dtype == torch.float32


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["default", "low_m", "cute_dsl", "cublas"])
def test_bf16_linear_backend_dispatch(backend: str) -> None:
    model = ToyDiT(linear_quant_ignored_steps=(1,), quantized_attention=False)
    layer = model.blocks[0].linear
    layer.use_cute_dsl_bf16_gemm = backend == "cute_dsl"
    layer.use_custom_cublas_mm = backend == "cublas"
    model.model_config.attention_metadata_state["linear_quant_ignored"] = True
    inputs = torch.randn(16, 128, dtype=torch.bfloat16)
    output = torch.empty_like(inputs)
    with (
        patch(
            "tensorrt_llm._torch.modules.linear._should_apply_low_m_gemm",
            return_value=backend == "low_m",
        ),
        patch("tensorrt_llm._torch.modules.linear.is_sm_100f", return_value=True),
        patch("tensorrt_llm._torch.modules.linear.get_sm_version", return_value=100),
        patch("tensorrt_llm._torch.modules.linear.apply_low_m_gemm", return_value=output) as low_m,
        patch("torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell") as cute_dsl,
        patch("torch.ops.trtllm.cublas_mm", return_value=output) as cublas,
        patch("tensorrt_llm._torch.modules.linear.F.linear", return_value=output) as linear,
    ):
        layer(inputs)
    calls = {"default": linear, "low_m": low_m, "cute_dsl": cute_dsl, "cublas": cublas}
    assert {name: call.call_count for name, call in calls.items()} == {
        name: int(name == backend) for name in calls
    }


@pytest.mark.cpu_only
def test_attention_quant_ignore_preserves_backend_contract() -> None:
    from tensorrt_llm._torch.visual_gen.attention.backends.parallel import RingAttention

    model = ToyDiT(attention_quant_ignored_steps=(1,), quantized_linear=False)
    backend = model.blocks[0].attention.attn
    assert isinstance(backend, AttentionQuantDispatcher)
    assert type(backend.unquantized) is type(backend.quantized) is ToyAttentionBackend
    assert backend.unquantized.quant_attention_config is None
    assert (
        backend.quantized.quant_attention_config
        is model.model_config.attention.quant_attention_config
    )
    inputs = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16)
    for quant_ignored in (False, True, False):
        backend.state["attention_quant_ignored"] = quant_ignored
        with (
            patch.object(
                backend.unquantized, "forward_with_lse", wraps=backend.unquantized.forward_with_lse
            ) as bf16,
            patch.object(
                backend.quantized, "forward_with_lse", wraps=backend.quantized.forward_with_lse
            ) as quantized,
        ):
            backend.forward_with_lse(inputs, inputs, inputs)
        assert (bf16.call_count, quantized.call_count) == ((1, 0) if quant_ignored else (0, 1))
    with patch.object(backend.quantized, "support_lse", return_value=False):
        assert not backend.support_lse()
        with pytest.raises(ValueError, match="LSE-capable"):
            RingAttention(backend, process_group=None)
    with patch.object(type(backend.quantized), "preferred_layout", AttentionTensorLayout.HND):
        with pytest.raises(ValueError, match="same tensor layout"):
            AttentionQuantDispatcher(
                backend.quantized, SimpleNamespace(preferred_layout=AttentionTensorLayout.NHD), {}
            )


@pytest.mark.cpu_only
def test_missing_quant_ignore_state_or_unquantized_weight_is_rejected() -> None:
    config = AttentionConfig(
        backend="CUTEDSL", quant_attention_config=QuantAttentionConfig(ignored_steps=[0])
    )
    with pytest.raises(ValueError, match="per-component attention state"):
        create_attention(
            backend="CUTEDSL", layer_idx=0, num_heads=1, head_dim=128, attention_config=config
        )
    with pytest.raises(ValueError, match="retained BF16"):
        install_linear_quant_dispatcher(ToyDiT(quantized_attention=False), {})
