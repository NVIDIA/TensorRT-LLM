# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev golden checks and LTX-2 two-stage BF16 parity with observed routes."""

import json
import time
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from PIL import Image
from test_common.llm_data import get_checkpoint, llm_models_root
from test_common.visual_gen_quantization import (
    QuantizationTrace,
    assert_quantization_routes,
    observe_quantization_paths,
)

from defs.examples.visual_gen.conftest import _visual_gen_deps as _visual_gen_deps
from defs.examples.visual_gen.visual_gen_test_utils import (
    VISUAL_GEN_LPIPS_GOLDEN_DIR,
    ReusableLPIPSScorer,
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _run_reusable_video_lpips_eval,
    _save_lpips_video_mp4,
)
from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
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
LTX2_LPIPS_LIMIT = 0.05


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


def _make_case_config(checkpoint: str, case: QuantIgnoreCase) -> dict:
    config = dict(
        model=checkpoint,
        torch_compile_config={"enable": False},
        cuda_graph_config={"enable": False},
        attention_config={"backend": "VANILLA"},
    )
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


def _expected_routes(case: QuantIgnoreCase, inventory: dict[str, list[str]]) -> dict:
    expected = {}
    for step in range(STEPS):
        for component in ("linear", "attention"):
            for name in inventory[component]:
                quantized = component == case.component and (
                    step == case.target
                    if case.mode == "step"
                    else name.startswith(inventory["blocks"][case.target] + ".")
                )
                expected[step, name, component] = "quantized" if quantized else "bf16"
    return expected


def _save_case_artifacts(
    directory: Path,
    config: dict,
    image: torch.Tensor,
    trace: QuantizationTrace,
) -> None:
    Image.fromarray(image[0].numpy()).save(directory / "image.png")
    (directory / "routes.json").write_text(json.dumps(trace.events, indent=2), encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps(
            {
                "request": REQUEST,
                "config": config,
                "method_entries": len(trace.events),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_generation(
    checkpoint: str,
    directory: Path,
    case: QuantIgnoreCase,
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
        if case.component == "attention" and case.mode == "layer":
            for name, module in pipeline.transformer.named_modules():
                if isinstance(module, Attention):
                    # Match the physical module path, including the single-stream local index.
                    assert module.module_name in (name, "transformer." + name), (
                        name,
                        module.module_name,
                    )
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
        _save_case_artifacts(directory, config, image, trace)
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
        expected_backend = get_visual_gen_attention_backend(case.backend).__name__
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


def _run_ltx2_two_stage_generation(directory: Path, settings: dict) -> None:
    from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2_two_stages import (
        STAGE_2_DISTILLED_SIGMA_VALUES,
        LTX2TwoStagesPipeline,
    )

    request = {
        key: settings[key]
        for key in (
            "prompt",
            "negative_prompt",
            "height",
            "width",
            "num_frames",
            "num_inference_steps",
            "guidance_scale",
            "seed",
        )
    }
    stage1_steps = request["num_inference_steps"]
    stage2_steps = len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1
    quantized_steps = {7}
    model_root = llm_models_root(check=True)
    assert model_root is not None
    text_encoder = model_root / "gemma-3-12b-it"
    if not text_encoder.is_dir():
        text_encoder = Path(get_checkpoint("gemma/gemma-3-12b-it"))
    config = dict(
        model=get_checkpoint("LTX-2/ltx-2-19b-dev.safetensors"),
        pipeline_config=dict(
            text_encoder_path=str(text_encoder),
            spatial_upsampler_path=get_checkpoint(
                "LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
            ),
            distilled_lora_path=get_checkpoint("LTX-2/ltx-2-19b-distilled-lora-384.safetensors"),
        ),
        quant_config=dict(
            quant_algo="FP8_BLOCK_SCALES",
            dynamic=True,
            ignore_steps=sorted(set(range(stage1_steps + stage2_steps)) - quantized_steps),
        ),
        attention_config=dict(backend="VANILLA"),
        torch_compile_config=dict(enable=True),
        cuda_graph_config=dict(enable=True),
        compilation_config=dict(
            resolutions=[[request["height"], request["width"]]],
            num_frames=[request["num_frames"]],
        ),
    )
    yaml_path = directory / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    reference_config = dict(config, quant_config=None)
    reference_pipeline = PipelineLoader(VisualGenArgs(**reference_config)).load(skip_warmup=True)
    try:
        assert not reference_pipeline._transformer_quant_dispatchers
        with torch.no_grad():
            reference_video = reference_pipeline.forward(**request).video.detach().cpu()
    finally:
        reference_pipeline.cleanup()
        del reference_pipeline
        _cleanup_cuda()
    _save_lpips_video_mp4(
        reference_video, directory / "bf16.mp4", frame_rate=settings["frame_rate"]
    )
    pipeline = PipelineLoader(VisualGenArgs.from_yaml(yaml_path)).load(skip_warmup=True)
    try:
        assert isinstance(pipeline, LTX2TwoStagesPipeline)
        assert pipeline._distilled_lora_weight_cache.applied_count > 0
        dispatcher = pipeline._transformer_quant_dispatchers["transformer"]
        routes, current = [], None
        publish_step = pipeline._publish_denoise_step
        with torch.no_grad(), observe_quantization_paths(dispatcher, device_only=True) as trace:
            pipeline.warmup()
            graph_counts = {name: len(r.graphs) for name, r in pipeline._cuda_graph_runners.items()}
            assert graph_counts and all(graph_counts.values())
            trace.reset()

            def publish(value):
                nonlocal current
                if current is not None:
                    routes.append({**current, "counts": trace.device_counts.cpu().tolist()})
                trace.reset()
                publish_step(value)
                current = None
                if value is not None:
                    stage, step = dispatcher.stage, value[0]
                    global_step = step + (stage1_steps if stage == "refinement" else 0)
                    projection = pipeline.transformer.get_submodule(
                        "transformer_blocks.0.attn1.qkv_proj"
                    )
                    quantized = global_step in quantized_steps
                    assert (type(projection.quant_method) is UnquantizedLinearMethod) != quantized
                    assert projection.weight.dtype == (
                        torch.float8_e4m3fn if quantized else torch.bfloat16
                    )
                    current = dict(
                        stage=stage,
                        step=step,
                        global_step=global_step,
                        variant="transformer_"
                        + dispatcher.variant_name(stage, dispatcher.schedule.key(step)),
                    )

            with ExitStack() as stack:
                replays = {
                    name: stack.enter_context(patch.object(r, "replay", wraps=r.replay))
                    for name, r in pipeline._cuda_graph_runners.items()
                }
                stack.enter_context(patch.object(pipeline, "_publish_denoise_step", publish))
                video = pipeline.forward(**request).video.detach().cpu()
                assert {name for name, replay in replays.items() if replay.call_count} == {
                    route["variant"] for route in routes
                }
            assert graph_counts == {
                name: len(r.graphs) for name, r in pipeline._cuda_graph_runners.items()
            }
        assert [(r["stage"], r["step"]) for r in routes] == [
            *(("base", step) for step in range(stage1_steps)),
            *(("refinement", step) for step in range(stage2_steps)),
        ]
        for route in routes:
            bf16, quantized, attention_bf16, attention_quantized = route["counts"]
            if route["global_step"] in quantized_steps:
                assert quantized > 0, route
            else:
                assert bf16 > 0 and quantized == 0, route
            assert attention_bf16 > 0 and attention_quantized == 0, route
        assert dispatcher.stage == "base" and dispatcher.step is None
        assert pipeline._distilled_lora_weight_cache._bound_state == "original"
        assert (
            video.shape
            == reference_video.shape
            == (1, request["num_frames"], request["height"], request["width"], 3)
        )
        assert video.dtype == reference_video.dtype == torch.uint8 and video.float().std() > 0
        video_path = directory / "video.mp4"
        _save_lpips_video_mp4(video, video_path, frame_rate=settings["frame_rate"])
        (directory / "report.json").write_text(
            json.dumps(
                dict(
                    request=request,
                    config=config,
                    reference_config=reference_config,
                    routes=routes,
                    graphs=graph_counts,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        pipeline.cleanup()


@pytest.mark.usefixtures("_visual_gen_deps")
def test_ltx2_two_stage_linear_step_ignore_compile_cuda_graph(tmp_path: Path) -> None:
    if get_sm_version() not in (100, 103):
        pytest.skip("This FP8 two-stage fixture targets B200/SM103")
    _disable_inductor_compile_worker_quiesce()
    settings = json.loads(
        (Path(VISUAL_GEN_LPIPS_GOLDEN_DIR) / "ltx2_lpips_golden_video.json").read_text(
            encoding="utf-8"
        )
    )
    with _lpips_deterministic_algorithms():
        try:
            _run_ltx2_two_stage_generation(tmp_path, settings)
        finally:
            _cleanup_cuda()
        with closing(ReusableLPIPSScorer()) as scorer:
            score = _run_reusable_video_lpips_eval(
                "ltx2-two-stage-step-ignore", tmp_path / "bf16.mp4", tmp_path / "video.mp4", scorer
            )
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(lpips_against_bf16=score, lpips_limit=LTX2_LPIPS_LIMIT)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert score < LTX2_LPIPS_LIMIT, (
        f"LTX-2 two-stage: LPIPS {score:.6f} exceeds {LTX2_LPIPS_LIMIT:.6f}"
    )
