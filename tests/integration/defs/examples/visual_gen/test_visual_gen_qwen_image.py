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

import os
from dataclasses import dataclass

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.examples.visual_gen.visual_gen_test_utils import (
    FeatureConfigState,
    _assert_feature_quantization_installed,
    _assert_feature_torch_compile_installed,
    _assert_lpips_below_threshold,
    _assert_resolved_single_device_feature_config,
    _assert_single_device_feature_executed,
    _build_single_device_feature_args,
    _cleanup_cuda,
    _cleanup_single_device_feature_pipeline,
    _disable_inductor_compile_worker_quiesce,
    _golden_media_path,
    _lpips_deterministic_algorithms,
    _lpips_model_path,
    _preserve_lpips_candidate_on_failure,
    _run_lpips_eval,
    _run_reusable_image_lpips_eval,
    _skip_if_missing,
    _validate_single_feature_config,
)

# QwenImage (text-to-image) — default-setting LPIPS golden.
# Params mirror the QwenImage 20B reference defaults (pipeline_qwen_image.py).
# NOTE: QwenImage's forward CFG knob is ``true_cfg_scale`` (not ``guidance_scale``),
# and real-CFG only engages when a negative prompt is supplied.
QWEN_IMAGE_MODEL_SUBPATH = "qwen-image"
QWENIMAGE_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
QWENIMAGE_LPIPS_NEGATIVE_PROMPT = ""
QWENIMAGE_LPIPS_HEIGHT = 1328
QWENIMAGE_LPIPS_WIDTH = 1328
QWENIMAGE_LPIPS_NUM_INFERENCE_STEPS = 50
QWENIMAGE_LPIPS_TRUE_CFG_SCALE = 4.0
QWENIMAGE_LPIPS_SEED = 42
QWENIMAGE_LPIPS_THRESHOLD = 0.05

QWENIMAGE_FEATURE_HEIGHT = 256
QWENIMAGE_FEATURE_WIDTH = 256
QWENIMAGE_FEATURE_NUM_INFERENCE_STEPS = 4
QWENIMAGE_FEATURE_GOLDEN_FILE = "qwenimage_feature_eager_lpips_golden.png"
# Rounded-up regression envelopes from the initial B200 calibration. These
# preserve current feature behavior; they are not absolute image-quality bars.
QWENIMAGE_FEATURE_THRESHOLDS = {
    "baseline": 0.05,
    "fp8-blockwise": 0.30,
    "nvfp4": 0.50,
    "cuda-graph": 0.05,
    "torch-compile": 0.10,
}
QWENIMAGE_SUPPORTED_FEATURES = frozenset(QWENIMAGE_FEATURE_THRESHOLDS).difference({"baseline"})


@dataclass(frozen=True)
class QwenImageAccuracyCase:
    id: str
    features: FeatureConfigState
    lpips_threshold: float


QWENIMAGE_FEATURE_PROFILES = (
    ("baseline", FeatureConfigState()),
    ("fp8-blockwise", FeatureConfigState(quantization="FP8_BLOCK_SCALES")),
    ("nvfp4", FeatureConfigState(quantization="NVFP4")),
    ("cuda-graph", FeatureConfigState(cuda_graph=True)),
    ("torch-compile", FeatureConfigState(torch_compile=True)),
)


def _build_qwenimage_accuracy_cases():
    cases = []
    for profile_id, features in QWENIMAGE_FEATURE_PROFILES:
        enabled_count = _validate_single_feature_config(
            features,
            QWENIMAGE_SUPPORTED_FEATURES,
            "Qwen-Image",
        )
        expected_count = 0 if profile_id == "baseline" else 1
        if enabled_count != expected_count:
            raise ValueError(
                f"Qwen-Image profile {profile_id!r} must enable "
                f"{expected_count} feature(s), got {enabled_count}"
            )
        cases.append(
            pytest.param(
                QwenImageAccuracyCase(
                    id=profile_id,
                    features=features,
                    lpips_threshold=QWENIMAGE_FEATURE_THRESHOLDS[profile_id],
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
                true_cfg_scale=QWENIMAGE_LPIPS_TRUE_CFG_SCALE,
                seed=QWENIMAGE_LPIPS_SEED,
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
    with _lpips_deterministic_algorithms():
        args = _build_single_device_feature_args(
            model_path,
            case.features,
            resolution=(QWENIMAGE_FEATURE_HEIGHT, QWENIMAGE_FEATURE_WIDTH),
            num_frames=1,
        )
        skip_warmup = not (case.features.cuda_graph or case.features.torch_compile)
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=skip_warmup)
            _assert_resolved_single_device_feature_config(
                pipeline,
                case.features,
                resolution=(QWENIMAGE_FEATURE_HEIGHT, QWENIMAGE_FEATURE_WIDTH),
                num_frames=1,
            )
            _assert_feature_quantization_installed(pipeline, case.features)
            _assert_feature_torch_compile_installed(pipeline, case.features)
            result = pipeline.forward(
                prompt=QWENIMAGE_LPIPS_PROMPT,
                negative_prompt=QWENIMAGE_LPIPS_NEGATIVE_PROMPT,
                height=QWENIMAGE_FEATURE_HEIGHT,
                width=QWENIMAGE_FEATURE_WIDTH,
                num_inference_steps=QWENIMAGE_FEATURE_NUM_INFERENCE_STEPS,
                true_cfg_scale=QWENIMAGE_LPIPS_TRUE_CFG_SCALE,
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
        QWENIMAGE_FEATURE_GOLDEN_FILE,
        "QwenImage compact eager LPIPS golden image",
    )
    _generate_qwenimage_feature_image(case, generated_path)
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
def test_qwenimage_cuda_graph_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "qwenimage_cuda_graph_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "qwenimage_lpips_golden.png", "QwenImage LPIPS golden image"
    )
    _generate_qwenimage_lpips_image(
        _lpips_model_path(QWEN_IMAGE_MODEL_SUBPATH),
        generated_path,
        enable_cuda_graph=True,
    )
    score = _run_lpips_eval(
        tmp_path,
        "qwenimage_cuda_graph",
        "image",
        QWENIMAGE_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, QWENIMAGE_LPIPS_THRESHOLD)


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
