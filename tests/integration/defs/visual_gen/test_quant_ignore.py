# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete FLUX.2-dev generations with observed quantization routes and shared goldens."""

import hashlib
import json
import time
from contextlib import ExitStack
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
    VISUAL_GEN_LPIPS_GOLDEN_DIR,
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _golden_media_path,
    _lpips_deterministic_algorithms,
)
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.attention.attention import Attention
from tensorrt_llm._torch.visual_gen.attention.backends.utils import get_visual_gen_attention_backend
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._utils import get_sm_version

GOLDEN = json.loads(
    (Path(VISUAL_GEN_LPIPS_GOLDEN_DIR) / "flux2_lpips_golden.json").read_text(encoding="utf-8")
)
REQUEST = {
    key: GOLDEN[key]
    for key in ("prompt", "seed", "height", "width", "num_inference_steps", "guidance_scale")
}
STEPS = REQUEST["num_inference_steps"]

# Quality bound for this fixed B200 fixture, with headroom for backend rounding.
LPIPS_LIMIT = 0.03


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
    directory: Path


def _tensor_storage_bytes(model: torch.nn.Module) -> int:
    storages = {
        (tensor.device, tensor.untyped_storage().data_ptr()): tensor.untyped_storage().nbytes()
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type != "meta"
    }
    return sum(storages.values())


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_case_config(checkpoint: str, case: QuantIgnoreCase | None) -> dict:
    config = dict(
        model=checkpoint,
        torch_compile_config={"enable": False},
        cuda_graph_config={"enable": False},
        attention_config={"backend": "VANILLA"},
    )
    if case is None:
        return config
    ignore_config = (
        {"ignore_steps": [step for step in range(STEPS) if step != case.target]}
        if case.mode == "step"
        else {}
    )
    if case.mode == "layer":
        transformer_config = json.loads(
            (Path(checkpoint) / "transformer" / "config.json").read_text(encoding="utf-8")
        )
        blocks = [f"transformer_blocks.{i}" for i in range(transformer_config["num_layers"])]
        blocks += [
            f"single_transformer_blocks.{i}" for i in range(transformer_config["num_single_layers"])
        ]
        ignored = [block + ".*" for index, block in enumerate(blocks) if index != case.target]
        if case.component == "linear":
            # Modulation and embedding Linears also live outside the transformer blocks.
            ignored += [
                "x_embedder",
                "context_embedder",
                "double_stream_modulation_*",
                "single_stream_modulation",
                "norm_out",
                "proj_out",
            ]
        ignore_config = {"ignore": ignored}
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
        "linear": [
            name.replace("._orig_mod.", ".")
            for name, module in model.named_modules()
            if isinstance(module, Linear)
        ],
        "attention": [
            name.replace("._orig_mod.", ".")
            for name, module in model.named_modules()
            if isinstance(module, Attention)
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
    *,
    compiled: bool = False,
) -> GenerationResult:
    directory.mkdir(parents=True, exist_ok=True)
    config = _make_case_config(checkpoint, case)
    if compiled:
        config.update(
            torch_compile_config={"enable": True},
            cuda_graph_config={"enable": True},
            compilation_config={
                "resolutions": [[REQUEST["height"], REQUEST["width"]]],
                "num_frames": [1],
            },
        )
    yaml_path = directory / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipeline = PipelineLoader(VisualGenArgs.from_yaml(yaml_path)).load(skip_warmup=True)
    torch.cuda.synchronize()
    measurements = {
        "load_seconds": time.perf_counter() - started,
        "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "resident_tensor_storage_bytes": _tensor_storage_bytes(pipeline),
    }
    try:
        actual_inventory = _module_inventory(pipeline.transformer)
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
        observed = pipeline._transformer_quant_dispatchers
        observed = observed["transformer"] if "transformer" in observed else pipeline.transformer
        with observe_quantization_paths(observed, device_only=compiled) as trace:
            if compiled:
                started = time.perf_counter()
                pipeline.warmup()
                torch.cuda.synchronize()
                measurements["warmup_seconds"] = time.perf_counter() - started
                assert pipeline._cuda_graph_runners
                assert all(runner.graphs for runner in pipeline._cuda_graph_runners.values())
                graph_counts = [
                    len(runner.graphs) for runner in pipeline._cuda_graph_runners.values()
                ]
                trace.reset()
            device_routes = []
            previous_step = None

            def prepare(*args, **kwargs):
                latents, ids = prepare_latents(*args, **kwargs)
                initial_noise_hashes.append(
                    hashlib.sha256(
                        latents.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                )
                return latents, ids

            def publish(value):
                nonlocal previous_step
                if compiled and previous_step is not None:
                    device_routes.append((previous_step, trace.device_counts.cpu().tolist()))
                    trace.reset()
                publish_step(value)
                trace.step = value[0] if value is not None else None
                previous_step = trace.step

            with (
                ExitStack() as replay_observers,
                patch.object(pipeline, "_prepare_latents", prepare),
                patch.object(pipeline, "_publish_denoise_step", publish),
            ):
                replays = (
                    [
                        replay_observers.enter_context(
                            patch.object(runner, "replay", wraps=runner.replay)
                        )
                        for runner in pipeline._cuda_graph_runners.values()
                    ]
                    if compiled
                    else []
                )
                image = pipeline.forward(**REQUEST).image.detach().cpu()
                if compiled:
                    assert all(replay.call_count > 0 for replay in replays)
        assert image.shape == (1, REQUEST["height"], REQUEST["width"], 3)
        assert image.dtype == torch.uint8 and image.float().std() > 0
        assert len(initial_noise_hashes) == 1
        _save_case_artifacts(directory, config, image, trace, initial_noise_hashes[0])
        if compiled:
            assert len(device_routes) == STEPS, device_routes
            for step, counts in device_routes:
                quantized_linear = case.component == "linear" and step == case.target
                assert counts[int(quantized_linear)] > 0, (step, counts)
                assert counts[int(not quantized_linear)] == 0, (step, counts)
                assert counts[2] > 0 and counts[3] == 0, (step, counts)
            assert graph_counts == [
                len(runner.graphs) for runner in pipeline._cuda_graph_runners.values()
            ]
            measurements["device_routes"] = device_routes
        else:
            assert_quantization_routes(trace, _expected_routes(case, actual_inventory))
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["measurements"] = measurements
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        backend_name = case.backend if case is not None else "VANILLA"
        expected_backend = get_visual_gen_attention_backend(backend_name).__name__
        for event in trace.events:
            if event["component"] == "attention":
                assert event["backend"] == expected_backend, event
        for _, model in pipeline.transformer_quant_models():
            state = model.model_config.attention_metadata_state
            if state is not None:
                assert state["denoise_step"] is None
        return GenerationResult(image, directory)
    finally:
        pipeline.cleanup()
        del pipeline
        _cleanup_cuda()


@pytest.fixture(scope="module")
def bf16_reference(tmp_path_factory: pytest.TempPathFactory) -> torch.Tensor:
    if get_sm_version() not in (100, 103):
        pytest.skip("This NVFP4/MXFP8 quantization ignore matrix targets B200/SM103")
    _disable_inductor_compile_worker_quiesce()
    directory = tmp_path_factory.mktemp("flux2_quant_ignore_reference")
    image_path = _golden_media_path(directory, GOLDEN["image"], "FLUX.2 BF16 golden image")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        assert image.size == (REQUEST["width"], REQUEST["height"])
        return torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
            1, REQUEST["height"], REQUEST["width"], 3
        )


def _assert_image_quality(
    case: QuantIgnoreCase, result: GenerationResult, bf16_reference: torch.Tensor
) -> None:
    import lpips

    metric = lpips.LPIPS(net="alex").eval()

    def normalize(image):
        return image.permute(0, 3, 1, 2).float() / 127.5 - 1

    with torch.inference_mode():
        score = metric(normalize(result.image), normalize(bf16_reference)).mean().item()
    assert torch.isfinite(torch.tensor(score)), score
    report_path = result.directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["reference_image"] = GOLDEN["image"]
    report["lpips_against_bf16"] = score
    report["lpips_limit"] = LPIPS_LIMIT
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"case": case.name, "lpips_against_bf16": score}), flush=True)
    assert score <= LPIPS_LIMIT, (
        f"{case.name}: LPIPS {score:.6f} exceeds {LPIPS_LIMIT:.6f} against the BF16 reference"
    )


@pytest.mark.parametrize("case", QUANT_IGNORE_CASES, ids=lambda case: case.name)
def test_flux2_quant_ignore(
    case: QuantIgnoreCase,
    bf16_reference: torch.Tensor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PyTorch's deterministic empty-memory fill does not support packed FP4 tensors.
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", False)
    with _lpips_deterministic_algorithms():
        result = _run_generation(get_checkpoint(GOLDEN["model"]), tmp_path / case.name, case)
    _assert_image_quality(case, result, bf16_reference)


def test_flux2_quant_ignore_compile_cuda_graph(
    bf16_reference: torch.Tensor, tmp_path: Path
) -> None:
    case = QUANT_IGNORE_CASES[1]
    with _lpips_deterministic_algorithms():
        result = _run_generation(
            get_checkpoint(GOLDEN["model"]),
            tmp_path / "linear_nvfp4_step_compile_cuda_graph",
            case,
            compiled=True,
        )
    _assert_image_quality(case, result, bf16_reference)
