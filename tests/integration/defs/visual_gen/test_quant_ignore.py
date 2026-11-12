# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seven complete FLUX.2-klein-4B generations with observed quantization routes."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from PIL import Image
from test_common.llm_data import get_checkpoint
from test_common.visual_gen_quantization import (
    QuantizationTrace,
    assert_quantization_routes,
    observe_quantization_paths,
)

from defs.examples.visual_gen.visual_gen_test_utils import (
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
)
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.attention.attention import Attention
from tensorrt_llm._torch.visual_gen.attention.backends.utils import get_visual_gen_attention_backend
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._utils import get_sm_version

STEPS = 4
REQUEST = dict(
    prompt="A small red sailboat on a calm lake, mountains in the distance, morning light.",
    seed=42,
    height=512,
    width=512,
    num_inference_steps=STEPS,
    guidance_scale=1.0,
)

# Quality bounds for this fixed B200 fixture, with headroom for backend rounding.
LPIPS_LIMITS = {
    "linear_fp8_block_layer": 0.01,
    "linear_nvfp4_step": 0.03,
    "attention_flashinfer_step": 0.03,
    "attention_trtllm_sage_layer": 0.005,
    "attention_cutedsl_step": 0.02,
    "attention_cudnn_layer": 0.005,
}


@dataclass(frozen=True)
class QuantIgnoreCase:
    name: str
    component: str
    mode: str
    target: int
    backend: str = "VANILLA"
    quant_algo: str | None = None
    qk_dtype: str | None = None
    v_dtype: str | None = None


QUANT_IGNORE_CASES = (
    QuantIgnoreCase("linear_fp8_block_layer", "linear", "layer", 10, quant_algo="FP8_BLOCK_SCALES"),
    QuantIgnoreCase("linear_nvfp4_step", "linear", "step", 2, quant_algo="NVFP4"),
    QuantIgnoreCase(
        "attention_flashinfer_step",
        "attention",
        "step",
        1,
        "FLASHINFER",
        qk_dtype="nvfp4",
        v_dtype="fp8",
    ),
    QuantIgnoreCase(
        "attention_trtllm_sage_layer",
        "attention",
        "layer",
        20,
        "TRTLLM",
        qk_dtype="fp8",
        v_dtype="fp8",
    ),
    QuantIgnoreCase(
        "attention_cutedsl_step", "attention", "step", 1, "CUTEDSL", qk_dtype="nvfp4", v_dtype="fp8"
    ),
    QuantIgnoreCase(
        "attention_cudnn_layer",
        "attention",
        "layer",
        20,
        "CUDNN",
        qk_dtype="mxfp8",
        v_dtype="mxfp8",
    ),
)


@dataclass
class GenerationResult:
    image: torch.Tensor
    inventory: dict[str, list[str]]
    initial_noise_hash: str
    directory: Path


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_case_config(
    checkpoint: str, case: QuantIgnoreCase | None, inventory: dict[str, list[str]]
) -> dict:
    config = dict(
        model=checkpoint,
        torch_compile_config={"enable": False},
        cuda_graph_config={"enable": False},
        attention_config={"backend": "VANILLA"},
    )
    if case is None:
        return config
    ignore_config = (
        {"ignored_steps": [step for step in range(STEPS) if step != case.target]}
        if case.mode == "step"
        else {}
    )
    if case.mode == "layer":
        prefix = inventory["blocks"][case.target] + "."
        targets = [name for name in inventory[case.component] if name.startswith(prefix)]
        assert targets, f"No {case.component} modules in global block {case.target}: {prefix}"
        ignore_config = {
            "ignore": [name for name in inventory[case.component] if name not in targets]
        }
    if case.component == "linear":
        config["quant_config"] = dict(quant_algo=case.quant_algo, dynamic=True, **ignore_config)
    else:
        recipe = dict(qk_dtype=case.qk_dtype, v_dtype=case.v_dtype, **ignore_config)
        if case.backend == "TRTLLM":
            recipe.update(q_block_size=1, k_block_size=16, v_block_size=1)
        config["attention_config"] = dict(backend=case.backend, quant_attention_config=recipe)
    return config


def _module_inventory(model: torch.nn.Module) -> dict[str, list[str]]:
    blocks = [f"transformer_blocks.{index}" for index in range(len(model.transformer_blocks))]
    blocks += [
        f"single_transformer_blocks.{index}"
        for index in range(len(model.single_transformer_blocks))
    ]
    assert len(blocks) > 20, f"The fixture must contain target global block 20: {blocks}"
    return {
        "blocks": blocks,
        "linear": [name for name, module in model.named_modules() if isinstance(module, Linear)],
        "attention": [
            name for name, module in model.named_modules() if isinstance(module, Attention)
        ],
    }


def _expected_routes(case: QuantIgnoreCase | None, inventory: dict[str, list[str]]) -> dict:
    expected = {}
    for step in range(STEPS):
        for component in ("linear", "attention"):
            for name in inventory[component]:
                quantized = (
                    case is not None
                    and component == case.component
                    and (
                        step == case.target
                        if case.mode == "step"
                        else name.startswith(inventory["blocks"][case.target] + ".")
                    )
                )
                expected[step, name, component] = "quantized" if quantized else "bf16"
    return expected


def _save_case_artifacts(
    directory: Path,
    config: dict,
    image: torch.Tensor,
    trace: QuantizationTrace,
    initial_noise_hash: str,
) -> None:
    Image.fromarray(image[0].numpy()).save(directory / "image.png")
    (directory / "routes.json").write_text(json.dumps(trace.events, indent=2), encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps(
            {
                "request": REQUEST,
                "config": config,
                "initial_noise_sha256": initial_noise_hash,
                "method_entries": len(trace.events),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_generation(
    checkpoint: str,
    directory: Path,
    case: QuantIgnoreCase | None = None,
    inventory: dict[str, list[str]] | None = None,
) -> GenerationResult:
    directory.mkdir(parents=True, exist_ok=True)
    config = _make_case_config(checkpoint, case, inventory or {})
    yaml_path = directory / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    pipeline = PipelineLoader(VisualGenArgs.from_yaml(yaml_path)).load(skip_warmup=True)
    try:
        actual_inventory = _module_inventory(pipeline.transformer)
        if inventory is not None:
            assert actual_inventory == inventory
        if case is not None and case.component == "attention" and case.mode == "layer":
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, Attention):
                    # Match the physical module path, including the single-stream local index.
                    assert module.module_name in (name, "transformer." + name), (
                        name,
                        module.module_name,
                    )
        initial_noise_hashes = []
        prepare_latents = pipeline._prepare_latents
        publish_step = pipeline._publish_denoise_step
        with observe_quantization_paths(pipeline.transformer) as trace:

            def prepare(*args, **kwargs):
                latents, ids = prepare_latents(*args, **kwargs)
                initial_noise_hashes.append(
                    hashlib.sha256(
                        latents.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                )
                return latents, ids

            def publish(value):
                publish_step(value)
                trace.step = value[0] if value is not None else None

            with (
                patch.object(pipeline, "_prepare_latents", prepare),
                patch.object(pipeline, "_publish_denoise_step", publish),
            ):
                image = pipeline.forward(**REQUEST).image.detach().cpu()
        assert image.shape == (1, REQUEST["height"], REQUEST["width"], 3)
        assert image.dtype == torch.uint8 and image.float().std() > 0
        assert len(initial_noise_hashes) == 1
        _save_case_artifacts(directory, config, image, trace, initial_noise_hashes[0])
        assert_quantization_routes(trace, _expected_routes(case, actual_inventory))
        backend_name = case.backend if case is not None else "VANILLA"
        expected_backend = get_visual_gen_attention_backend(backend_name).__name__
        for event in trace.events:
            if event["component"] == "attention":
                assert event["backend"] == expected_backend, event
        state = pipeline.transformer.model_config.attention_metadata_state
        if state is not None:
            assert state["denoise_step"] is None
            assert not state["linear_quant_ignored"] and not state["attention_quant_ignored"]
        return GenerationResult(image, actual_inventory, initial_noise_hashes[0], directory)
    finally:
        pipeline.cleanup()
        del pipeline
        _cleanup_cuda()


@pytest.fixture(scope="module")
def bf16_reference(tmp_path_factory: pytest.TempPathFactory) -> GenerationResult:
    if get_sm_version() not in (100, 103):
        pytest.skip("This NVFP4/MXFP8 quantization ignore matrix targets B200/SM103")
    _disable_inductor_compile_worker_quiesce()
    checkpoint = get_checkpoint("FLUX.2-klein-4B")
    directory = tmp_path_factory.mktemp("flux2_klein_quant_ignore") / "bf16_reference"
    return _run_generation(checkpoint, directory)


def test_flux2_klein_bf16_reference(bf16_reference: GenerationResult) -> None:
    assert (bf16_reference.directory / "image.png").is_file()


@pytest.mark.parametrize("case", QUANT_IGNORE_CASES, ids=lambda case: case.name)
def test_flux2_klein_quant_ignore(case: QuantIgnoreCase, bf16_reference: GenerationResult) -> None:
    import lpips

    checkpoint = get_checkpoint("FLUX.2-klein-4B")
    result = _run_generation(
        checkpoint, bf16_reference.directory.parent / case.name, case, bf16_reference.inventory
    )
    assert result.initial_noise_hash == bf16_reference.initial_noise_hash
    metric = lpips.LPIPS(net="alex").eval()

    def normalize(image):
        return image.permute(0, 3, 1, 2).float() / 127.5 - 1

    with torch.inference_mode():
        score = metric(normalize(result.image), normalize(bf16_reference.image)).mean().item()
    assert torch.isfinite(torch.tensor(score)), score
    report_path = result.directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["lpips_against_bf16"] = score
    report["lpips_limit"] = LPIPS_LIMITS[case.name]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"case": case.name, "lpips_against_bf16": score}), flush=True)
    assert score <= LPIPS_LIMITS[case.name], (
        f"{case.name}: LPIPS {score:.6f} exceeds {LPIPS_LIMITS[case.name]:.6f} "
        "against the BF16 reference"
    )
