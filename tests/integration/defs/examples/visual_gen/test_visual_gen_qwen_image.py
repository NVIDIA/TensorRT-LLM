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

"""Single-GPU integration and accuracy tests for Qwen-Image."""

import math
import os
import shutil
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    FeatureConfigState,
    _assert_feature_quantization_installed,
    _assert_lpips_below_threshold,
    _assert_resolved_single_device_feature_config,
    _assert_single_device_feature_executed,
    _build_single_device_feature_args,
    _cleanup_cuda,
    _cleanup_single_device_feature_pipeline,
    _disable_inductor_compile_worker_quiesce,
    _fixed_nvfp4_quantization_backend,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_image_lpips_eval,
    _run_single_device_feature_generator,
    _skip_if_missing,
    _validate_single_feature_config,
)

# QwenImage (text-to-image) — default-setting LPIPS golden.
# Params mirror the QwenImage 20B reference defaults (pipeline_qwen_image.py).
# NOTE: QwenImage's forward CFG knob is ``negative_prompt_cfg_scale`` (not
# ``guidance_scale``), and negative-prompt CFG only engages when it is > 1.0.
QWEN_IMAGE_MODEL_SUBPATH = "qwen-image"
QWENIMAGE_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
QWENIMAGE_LPIPS_NEGATIVE_PROMPT = ""
QWENIMAGE_LPIPS_HEIGHT = 1328
QWENIMAGE_LPIPS_WIDTH = 1328
QWENIMAGE_LPIPS_NUM_INFERENCE_STEPS = 50
QWENIMAGE_LPIPS_NEGATIVE_PROMPT_CFG_SCALE = 4.0
QWENIMAGE_LPIPS_SEED = 42
QWENIMAGE_LPIPS_THRESHOLD = 0.05

QWEN_IMAGE_EDIT_MODEL_SUBPATH = "Qwen-Image-Edit-2511"
QWEN_IMAGE_LAYERED_MODEL_SUBPATH = "qwen-image-layered"
QWEN_IMAGE_LAYERED_LPIPS_PROMPT = ""
QWEN_IMAGE_LAYERED_LPIPS_NEGATIVE_PROMPT = " "
QWEN_IMAGE_LAYERED_LPIPS_NUM_INFERENCE_STEPS = 50
QWEN_IMAGE_LAYERED_LPIPS_TRUE_CFG_SCALE = 4.0
QWEN_IMAGE_LAYERED_LPIPS_LAYERS = 4
QWEN_IMAGE_LAYERED_LPIPS_RESOLUTION = 640
QWEN_IMAGE_LAYERED_LPIPS_SEED = 777
QWEN_IMAGE_LAYERED_LPIPS_THRESHOLD = 0.05
QWENIMAGE_FEATURE_LPIPS_THRESHOLD = 0.05
QWENIMAGE_SUPPORTED_FEATURES = frozenset({"fp8-blockwise", "nvfp4", "cuda-graph"})


@dataclass(frozen=True)
class QwenImageAccuracyCase:
    id: str
    golden_file: str
    features: FeatureConfigState
    lpips_threshold: float


QWENIMAGE_FEATURE_PROFILES = (
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("cuda-graph", FeatureConfigState(cuda_graph=True)),
)


def _build_qwenimage_accuracy_cases():
    cases = []
    for profile_id, features in QWENIMAGE_FEATURE_PROFILES:
        _validate_single_feature_config(
            features,
            QWENIMAGE_SUPPORTED_FEATURES,
            "Qwen-Image",
        )
        cases.append(
            pytest.param(
                QwenImageAccuracyCase(
                    id=profile_id,
                    golden_file=(f"qwenimage_{profile_id.replace('-', '_')}_lpips_golden.png"),
                    features=features,
                    lpips_threshold=QWENIMAGE_FEATURE_LPIPS_THRESHOLD,
                ),
                id=profile_id,
            )
        )
    return cases


QWENIMAGE_ACCURACY_CASES = _build_qwenimage_accuracy_cases()


def _generate_qwenimage_lpips_image(model_path, output_path, *, enable_cuda_graph=False):
    """Generate the QwenImage text-to-image LPIPS sample."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import CudaGraphConfig, TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "QwenImage checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    args = VisualGenArgs(
        model=model_path,
        torch_compile_config=TorchCompileConfig(enable=False),
        cuda_graph_config=CudaGraphConfig(enable=enable_cuda_graph),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    try:
        with torch.no_grad():
            result = pipeline.forward(
                prompt=QWENIMAGE_LPIPS_PROMPT,
                negative_prompt=QWENIMAGE_LPIPS_NEGATIVE_PROMPT,
                height=QWENIMAGE_LPIPS_HEIGHT,
                width=QWENIMAGE_LPIPS_WIDTH,
                num_inference_steps=QWENIMAGE_LPIPS_NUM_INFERENCE_STEPS,
                negative_prompt_cfg_scale=QWENIMAGE_LPIPS_NEGATIVE_PROMPT_CFG_SCALE,
                seed=QWENIMAGE_LPIPS_SEED,
            )
        generated_image = result.image[0].detach().cpu()
    finally:
        del pipeline
        _cleanup_cuda()

    save_image(generated_image, output_path)


def _copy_qwen_image_layered_lpips_input(tmp_path, input_path):
    source = _golden_media_path(
        tmp_path,
        "qwen_image_layered_lpips_input.png",
        "Qwen-Image-Layered LPIPS input image",
    )
    shutil.copyfile(source, input_path)


def _qwen_image_layered_golden_layer_paths(tmp_path):
    golden_dir = _golden_media_path(
        tmp_path,
        "qwen_image_layered_lpips_golden",
        "Qwen-Image-Layered LPIPS golden layer directory",
    )
    layer_paths = sorted(
        golden_dir.glob("layer_*.png"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    assert layer_paths, f"Qwen-Image-Layered golden layer directory is empty: {golden_dir}"
    return layer_paths


def _write_qwen_image_layered_lpips_golden_grid(tmp_path, output_path):
    from PIL import Image

    layer_paths = _qwen_image_layered_golden_layer_paths(tmp_path)
    layers = []
    for path in layer_paths:
        with Image.open(path) as image:
            layers.append(image.convert("RGBA").copy())

    width, height = layers[0].size
    assert all(layer.size == (width, height) for layer in layers), (
        "Qwen-Image-Layered golden layers must have identical sizes, got "
        f"{[layer.size for layer in layers]}"
    )
    grid_cols = math.ceil(math.sqrt(len(layers)))
    grid_rows = math.ceil(len(layers) / grid_cols)
    grid = Image.new("RGBA", (grid_cols * width, grid_rows * height), (0, 0, 0, 0))
    for index, layer in enumerate(layers):
        row, col = divmod(index, grid_cols)
        grid.alpha_composite(layer, dest=(col * width, row * height))
    grid.save(output_path)


def _flatten_qwen_image_layered_lpips_image(input_path, output_path):
    from PIL import Image

    with Image.open(input_path) as image:
        rgba_image = image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        background.alpha_composite(rgba_image)
        background.convert("RGB").save(output_path)


def _generate_qwen_image_layered_lpips_image(model_path, input_path, output_path):
    """Generate the Qwen-Image-Layered LPIPS sample (default setting, compile-off)."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "Qwen-Image-Layered checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    args = VisualGenArgs(
        model=model_path,
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    try:
        with torch.no_grad():
            result = pipeline.forward(
                image=str(input_path),
                prompt=QWEN_IMAGE_LAYERED_LPIPS_PROMPT,
                negative_prompt=QWEN_IMAGE_LAYERED_LPIPS_NEGATIVE_PROMPT,
                num_inference_steps=QWEN_IMAGE_LAYERED_LPIPS_NUM_INFERENCE_STEPS,
                true_cfg_scale=QWEN_IMAGE_LAYERED_LPIPS_TRUE_CFG_SCALE,
                layers=QWEN_IMAGE_LAYERED_LPIPS_LAYERS,
                resolution=QWEN_IMAGE_LAYERED_LPIPS_RESOLUTION,
                cfg_normalize=True,
                use_en_prompt=True,
                seed=QWEN_IMAGE_LAYERED_LPIPS_SEED,
            )
        generated_image = result.image[0].detach().cpu()
    finally:
        del pipeline
        _cleanup_cuda()

    save_image(generated_image, output_path)


def _generate_qwenimage_feature_image(case, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image

    model_path = _lpips_model_path(QWEN_IMAGE_MODEL_SUBPATH)
    _skip_if_missing(model_path, "QwenImage checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    pipeline = None
    with _lpips_deterministic_algorithms(), _fixed_nvfp4_quantization_backend(case.features):
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(QWENIMAGE_LPIPS_HEIGHT, QWENIMAGE_LPIPS_WIDTH),
            num_frames=1,
        )
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=False)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(QWENIMAGE_LPIPS_HEIGHT, QWENIMAGE_LPIPS_WIDTH),
                num_frames=1,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=QWENIMAGE_LPIPS_PROMPT,
                negative_prompt=QWENIMAGE_LPIPS_NEGATIVE_PROMPT,
                height=QWENIMAGE_LPIPS_HEIGHT,
                width=QWENIMAGE_LPIPS_WIDTH,
                num_inference_steps=QWENIMAGE_LPIPS_NUM_INFERENCE_STEPS,
                negative_prompt_cfg_scale=QWENIMAGE_LPIPS_NEGATIVE_PROMPT_CFG_SCALE,
                seed=QWENIMAGE_LPIPS_SEED,
            )
            _assert_single_device_feature_executed(pipeline, case.features)
            generated_image = result.image[0].detach().cpu()
        finally:
            try:
                if pipeline is not None:
                    _cleanup_single_device_feature_pipeline(pipeline)
                    del pipeline
            finally:
                _cleanup_cuda()

    save_image(generated_image, output_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", QWENIMAGE_ACCURACY_CASES)
def test_qwenimage_feature_accuracy_against_golden(
    request,
    tmp_path,
    case,
    _visual_gen_lpips_scorer,
):
    generated_path = tmp_path / f"qwenimage_{case.id}_generated.png"
    golden_path = _golden_media_path(
        tmp_path,
        case.golden_file,
        f"QwenImage {case.id} LPIPS golden image",
    )
    _run_single_device_feature_generator(
        case.features, _generate_qwenimage_feature_image, case, generated_path
    )
    score = _run_reusable_image_lpips_eval(
        f"qwenimage-{case.id}",
        golden_path,
        generated_path,
        _visual_gen_lpips_scorer,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        case.lpips_threshold,
        generated_path,
        f"qwenimage_{case.id}_generated.png",
    )
    _assert_lpips_below_threshold(score, case.lpips_threshold)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwenimage_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "qwenimage_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "qwenimage_lpips_golden.png", "QwenImage LPIPS golden image"
    )
    _generate_qwenimage_lpips_image(_lpips_model_path(QWEN_IMAGE_MODEL_SUBPATH), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "qwenimage",
        "image",
        QWENIMAGE_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, QWENIMAGE_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwen_image_layered_lpips_against_golden(tmp_path):
    input_path = tmp_path / "qwen_image_layered_input.png"
    generated_path = tmp_path / "qwen_image_layered_generated.png"
    golden_path = tmp_path / "qwen_image_layered_golden_grid.png"
    generated_lpips_path = tmp_path / "qwen_image_layered_generated_lpips.png"
    golden_lpips_path = tmp_path / "qwen_image_layered_golden_grid_lpips.png"
    _copy_qwen_image_layered_lpips_input(tmp_path, input_path)
    _write_qwen_image_layered_lpips_golden_grid(tmp_path, golden_path)
    _generate_qwen_image_layered_lpips_image(
        _lpips_model_path(QWEN_IMAGE_LAYERED_MODEL_SUBPATH),
        input_path,
        generated_path,
    )
    # Ignore invisible RGB values under transparent pixels while preserving
    # partially transparent layer edges.
    _flatten_qwen_image_layered_lpips_image(generated_path, generated_lpips_path)
    _flatten_qwen_image_layered_lpips_image(golden_path, golden_lpips_path)
    score = _run_lpips_eval(
        tmp_path,
        "qwen_image_layered",
        "image",
        QWEN_IMAGE_LAYERED_LPIPS_PROMPT,
        golden_lpips_path,
        generated_lpips_path,
    )
    _assert_lpips_below_threshold(score, QWEN_IMAGE_LAYERED_LPIPS_THRESHOLD)


def test_qwen_image_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/qwen_image.py with FP8 config end-to-end.

    Validates that the Qwen-Image example script and
    ``configs/qwen-image-fp8-1gpu.yaml`` work together as documented. Uses the
    local Qwen-Image checkpoint and the shared FP8 blockwise dynamic-quant config.
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, QWEN_IMAGE_MODEL_SUBPATH)
    _skip_if_missing(model_path, "Qwen-Image checkpoint", is_dir=True)
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(model_index_path):
        pytest.skip(
            f"Qwen-Image checkpoint is incomplete: {model_path} (missing {model_index_path})"
        )

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "qwen_image_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "qwen_image_output.png")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "qwen_image.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "qwen-image-fp8-1gpu.yaml"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"
    assert os.path.isfile(config_path), f"Config not found: {config_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"


def test_qwen_image_layered_example(_visual_gen_deps, tmp_path, llm_root, llm_venv):
    """Run examples/visual_gen/models/qwen_image_layered.py end-to-end."""
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, QWEN_IMAGE_LAYERED_MODEL_SUBPATH)
    _skip_if_missing(model_path, "Qwen-Image-Layered checkpoint", is_dir=True)
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(model_index_path):
        pytest.skip(
            f"Qwen-Image-Layered checkpoint is incomplete: {model_path} "
            f"(missing {model_index_path})"
        )

    input_path = tmp_path / "qwen_image_layered_input.png"
    _copy_qwen_image_layered_lpips_input(tmp_path, input_path)

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "qwen_image_layered_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "qwen_image_layered_output.png")

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "qwen_image_layered.py"
    )
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "qwen-image-layered-1gpu.yaml"
    )
    assert os.path.isfile(config_path), f"Config not found: {config_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--image",
            str(input_path),
            "--prompt",
            QWEN_IMAGE_LAYERED_LPIPS_PROMPT,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"


def test_qwen_image_edit_example(_visual_gen_deps: Any, llm_root: str, llm_venv: Any) -> None:
    """Run examples/visual_gen/models/qwen_image_edit.py end-to-end.

    Validates that the Qwen-Image-Edit example script and
    ``configs/qwen-image-edit-2511-fp8-1gpu.yaml`` work together as documented.
    """
    model_path = os.environ.get("QWEN_IMAGE_EDIT_MODEL_PATH") or os.path.join(
        conftest.llm_models_root(), QWEN_IMAGE_EDIT_MODEL_SUBPATH
    )
    _skip_if_missing(model_path, "Qwen-Image-Edit-2511 checkpoint", is_dir=True)
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(model_index_path):
        pytest.skip(
            f"Qwen-Image-Edit-2511 checkpoint is incomplete: {model_path} "
            f"(missing {model_index_path})"
        )

    out_dir = os.path.join(
        llm_venv.get_working_directory(), "visual_gen_output", "qwen_image_edit_example"
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "qwen_image_edit_output.png")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "qwen_image_edit.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "qwen-image-edit-2511-fp8-1gpu.yaml"
    )
    image_path = os.path.join(llm_root, "examples", "visual_gen", "cat_piano.png")
    assert os.path.isfile(script_path), f"Example script not found: {script_path}"
    assert os.path.isfile(config_path), f"Config not found: {config_path}"
    assert os.path.isfile(image_path), f"Input image not found: {image_path}"

    venv_check_call(
        llm_venv,
        [
            script_path,
            "--model",
            model_path,
            "--visual_gen_args",
            config_path,
            "--image",
            image_path,
            "--prompt",
            "Add a small red wizard hat to the cat while preserving the source image.",
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
