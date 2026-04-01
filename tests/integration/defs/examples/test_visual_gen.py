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
"""Integration tests: VBench dimension scores for WAN and LTX-2 (TRT-LLM vs diffusers reference)."""

import glob
import json
import os
import random
import time
import urllib.request

import pytest
import torch
from defs import conftest
from defs.common import venv_check_call
from defs.trt_test_alternative import check_call

WAN_T2V_MODEL_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
WAN22_A14B_FP8_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-FP8"
WAN22_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-NVFP4"
VISUAL_GEN_OUTPUT_VIDEO = "trtllm_output.mp4"
DIFFUSERS_REFERENCE_VIDEO = "diffusers_reference.mp4"
WAN_T2V_PROMPT = "A cute cat playing piano"
WAN_T2V_HEIGHT = 480
WAN_T2V_WIDTH = 832
WAN_T2V_NUM_FRAMES = 165

# LTX-2 configuration
LTX2_MODEL_CHECKPOINT_PATH = "LTX-2/ltx-2-19b-dev.safetensors"
LTX2_TEXT_ENCODER_SUBPATH = "gemma-3-12b-it"
LTX2_T2V_PROMPT = (
    "A woman with long brown hair and light skin smiles at the camera while "
    "standing in a sunlit park, her hair gently blowing in the breeze as she "
    "tilts her head slightly to the side."
)
LTX2_T2V_HEIGHT = 512
LTX2_T2V_WIDTH = 768
LTX2_T2V_NUM_FRAMES = 121
LTX2_T2V_STEPS = 40
LTX2_T2V_GUIDANCE_SCALE = 4.0
LTX2_T2V_MAX_SEQ_LEN = 1024
LTX2_T2V_FRAME_RATE = 24.0
LTX2_T2V_SEED = 42
LTX2_T2V_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

# Dimensions to evaluate
VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# Golden VBench scores from HF reference video (WAN 2.1 1.3B); TRT-LLM is compared against these.
VBENCH_WAN_GOLDEN_SCORES = {
    "subject_consistency": 0.9381,
    "background_consistency": 0.9535,
    "motion_smoothness": 0.9923,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5033,
    "imaging_quality": 0.3033,
}


# TODO: Reference scores from bf16 baseline runs
VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES = {
    "subject_consistency": 0.9173,
    "background_consistency": 0.9717,
    "motion_smoothness": 0.9865,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5465,
    "imaging_quality": 0.7142,
}
VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES = {
    "subject_consistency": 0.9173,
    "background_consistency": 0.9717,
    "motion_smoothness": 0.9865,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5465,
    "imaging_quality": 0.7142,
}

VBENCH_LTX2_BF16_GOLDEN_SCORES = {
    "subject_consistency": 0.9683,
    "background_consistency": 0.9469,
    "motion_smoothness": 0.9941,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5097,
    "imaging_quality": 0.7309,
}

VBENCH_LTX2_FP8_GOLDEN_SCORES = {
    "subject_consistency": 0.9817,
    "background_consistency": 0.9704,
    "motion_smoothness": 0.9918,
    "dynamic_degree": 0.0000,
    "aesthetic_quality": 0.6062,
    "imaging_quality": 0.6546,
}

# LTX-2 Two-Stage configuration
LTX2_UPSAMPLER_SUBPATH = "LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_DISTILLED_LORA_SUBPATH = "LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
LTX2_TWO_STAGE_HEIGHT = 512
LTX2_TWO_STAGE_WIDTH = 768
LTX2_TWO_STAGE_NUM_FRAMES = 121
LTX2_TWO_STAGE_STEPS = 40
LTX2_TWO_STAGE_GUIDANCE_SCALE = 4.0

# Golden VBench scores for two-stage pipeline variants.
# Initially None — first CI run is a baseline that prints scores for capture.
VBENCH_LTX2_TWO_STAGE_BF16_GOLDEN_SCORES = None

VBENCH_LTX2_TWO_STAGE_FP8_GOLDEN_SCORES = None

VBENCH_REPO = "https://github.com/Vchitect/VBench.git"
# Pin to a fixed commit for reproducible shallow-fetch
VBENCH_COMMIT = "98b19513678e99c80d8377fda25ba53b81a491a6"

DINO_REPO = "https://github.com/facebookresearch/dino.git"
DINO_HUB_DIR_NAME = "facebookresearch_dino_main"

AESTHETIC_PREDICTOR_URL = (
    "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"
)
AESTHETIC_PREDICTOR_FILENAME = "sa_0_4_vit_l_14_linear.pth"
AESTHETIC_PREDICTOR_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "emb_reader")


@pytest.fixture(scope="session")
def _visual_gen_deps(llm_venv):
    """Install av + diffusers + ffmpeg once per session (shared by all video-gen fixtures)."""
    llm_venv.run_cmd(["-m", "pip", "install", "av"])
    llm_venv.run_cmd(["-m", "pip", "install", "git+https://github.com/huggingface/diffusers.git"])
    # Install ffmpeg system package required by MediaStorage.save_video for MP4 encoding
    check_call(["apt-get", "update", "-y"], shell=False)
    check_call(["apt-get", "install", "-y", "ffmpeg"], shell=False)


@pytest.fixture(scope="session")
def vbench_repo_root(llm_venv):
    """Clone VBench repo into workspace and install; return repo root path."""
    workspace = llm_venv.get_working_directory()
    repo_path = os.path.join(workspace, "VBench_repo")
    _precache_dino_for_torch_hub()
    _precache_aesthetic_predictor()
    if os.path.exists(repo_path):
        return repo_path
    # Shallow-fetch only the pinned commit to avoid downloading full history (~350 MB)
    os.makedirs(repo_path, exist_ok=True)
    check_call(["git", "init", repo_path], shell=False)
    check_call(
        ["git", "-C", repo_path, "remote", "add", "origin", VBENCH_REPO],
        shell=False,
    )
    check_call(
        ["git", "-C", repo_path, "fetch", "--depth", "1", "origin", VBENCH_COMMIT],
        shell=False,
    )
    check_call(["git", "-C", repo_path, "checkout", "FETCH_HEAD"], shell=False)
    llm_venv.run_cmd(
        [
            "-m",
            "pip",
            "install",
            "tqdm>=4.60.0",
            "openai-clip>=1.0",
            "easydict",
            "decord>=0.6.0",
            "imageio",
        ]
    )
    llm_venv.run_cmd(
        [
            "-m",
            "pip",
            "install",
            "--no-deps",
            "pyiqa>=0.1.0",
        ]
    )
    return repo_path


def _precache_dino_for_torch_hub():
    """Pre-clone facebookresearch/dino into torch.hub cache to avoid GitHub API rate limits.

    VBench's subject_consistency dimension calls
    torch.hub.load('facebookresearch/dino:main', ...) which validates the repo
    via the GitHub API and fails with HTTP 403 when rate-limited. Pre-cloning
    the repo makes torch.hub skip the API validation for cached repos.
    """
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)

    dino_cache = os.path.join(hub_dir, DINO_HUB_DIR_NAME)
    if not os.path.isdir(dino_cache):
        check_call(
            ["git", "clone", "--depth", "1", "-b", "main", DINO_REPO, dino_cache],
            shell=False,
        )


def _precache_aesthetic_predictor():
    """Pre-download LAION aesthetic predictor weights to avoid GitHub rate limits.

    VBench's aesthetic_quality dimension downloads sa_0_4_vit_l_14_linear.pth
    from GitHub via wget at evaluation time.  GitHub often returns HTTP 429
    (Too Many Requests) in CI environments.  Pre-downloading with retries
    and proper headers ensures the file is cached before VBench needs it.
    """
    os.makedirs(AESTHETIC_PREDICTOR_CACHE_DIR, exist_ok=True)
    cached_path = os.path.join(AESTHETIC_PREDICTOR_CACHE_DIR, AESTHETIC_PREDICTOR_FILENAME)
    if os.path.isfile(cached_path):
        return

    max_retries = 8
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                AESTHETIC_PREDICTOR_URL,
                headers={"User-Agent": "TensorRT-LLM-CI/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            tmp_path = cached_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.replace(tmp_path, cached_path)
            return
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = min(10 * 2**attempt, 120) + random.uniform(0, 5)
                print(
                    f"[precache] Aesthetic predictor download attempt {attempt + 1}/{max_retries} "
                    f"failed ({exc}), retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download aesthetic predictor after {max_retries} attempts: {exc}"
                ) from exc


@pytest.fixture(scope="session")
def wan_trtllm_video_path(_visual_gen_deps, llm_venv, llm_root):
    """Generate input video via visual_gen_wan_t2v.py and return path to trtllm_output.mp4."""
    return _generate_wan_video(llm_venv, llm_root, WAN_T2V_MODEL_SUBPATH, "wan")


def _generate_wan_video(llm_venv, llm_root, model_subpath, output_subdir):
    """Generate a video with visual_gen_wan_t2v.py for a given model checkpoint.

    Returns the path to the generated .mp4, or calls pytest.skip if the model
    is not found under LLM_MODELS_ROOT.
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, model_subpath)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {model_subpath} under scratch)"
        )
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)
    if os.path.isfile(output_path):
        return output_path
    script_path = os.path.join(llm_root, "examples", "visual_gen", "visual_gen_wan_t2v.py")
    assert os.path.isfile(script_path), f"Visual gen script not found: {script_path}"
    cmd = [
        script_path,
        "--height",
        str(WAN_T2V_HEIGHT),
        "--width",
        str(WAN_T2V_WIDTH),
        "--num_frames",
        str(WAN_T2V_NUM_FRAMES),
        "--model_path",
        model_path,
        "--prompt",
        WAN_T2V_PROMPT,
        "--output_path",
        output_path,
    ]
    if torch.cuda.device_count() >= 2:
        cmd.extend(["--cfg_size", "2"])
    venv_check_call(llm_venv, cmd)
    assert os.path.isfile(output_path), f"Visual gen did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def wan22_a14b_fp8_video_path(_visual_gen_deps, llm_venv, llm_root):
    """Generate video with Wan 2.2 A14B FP8 checkpoint."""
    return _generate_wan_video(llm_venv, llm_root, WAN22_A14B_FP8_MODEL_SUBPATH, "wan22_fp8")


@pytest.fixture(scope="session")
def wan22_a14b_nvfp4_video_path(_visual_gen_deps, llm_venv, llm_root):
    """Generate video with Wan 2.2 A14B NVFP4 checkpoint."""
    return _generate_wan_video(llm_venv, llm_root, WAN22_A14B_NVFP4_MODEL_SUBPATH, "wan22_nvfp4")


def _linear_type_to_quant_config(linear_type):
    """Map linear_type shortcut to quant_config dict for VisualGenArgs."""
    mapping = {
        "trtllm-fp8-per-tensor": {"quant_algo": "FP8", "dynamic": True},
        "trtllm-fp8-blockwise": {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        "trtllm-nvfp4": {"quant_algo": "NVFP4", "dynamic": True},
    }
    return mapping.get(linear_type)


def _generate_ltx2_video(llm_venv, output_subdir, linear_type="default"):
    """Generate a video using the LTX-2 Python API directly.

    Calls VisualGen / VisualGenArgs / VisualGenParams instead of shelling out
    to examples/visual_gen/visual_gen_ltx2.py (which may be removed).

    Returns the path to the generated .mp4, or calls pytest.skip if the model
    or text encoder is not found under LLM_MODELS_ROOT.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm.serve.media_storage import MediaStorage

    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, LTX2_MODEL_CHECKPOINT_PATH)
    text_encoder_path = os.path.join(scratch_space, LTX2_TEXT_ENCODER_SUBPATH)
    if not os.path.isfile(model_path):
        pytest.skip(
            f"LTX-2 checkpoint not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {LTX2_MODEL_CHECKPOINT_PATH} under models root)"
        )
    if not os.path.isdir(text_encoder_path):
        pytest.skip(
            f"LTX-2 text encoder not found: {text_encoder_path} "
            f"(set LLM_MODELS_ROOT or place {LTX2_TEXT_ENCODER_SUBPATH} under scratch)"
        )
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)
    if os.path.isfile(output_path):
        return output_path

    vg_kwargs = dict(text_encoder_path=text_encoder_path)
    quant_config = _linear_type_to_quant_config(linear_type)
    if quant_config is not None:
        vg_kwargs["quant_config"] = quant_config
    if torch.cuda.device_count() >= 2:
        vg_kwargs["parallel"] = {"dit_cfg_size": 2}

    diffusion_args = VisualGenArgs(**vg_kwargs)
    visual_gen = VisualGen(model_path=model_path, diffusion_args=diffusion_args)

    try:
        params = VisualGenParams(
            height=LTX2_T2V_HEIGHT,
            width=LTX2_T2V_WIDTH,
            num_frames=LTX2_T2V_NUM_FRAMES,
            num_inference_steps=LTX2_T2V_STEPS,
            guidance_scale=LTX2_T2V_GUIDANCE_SCALE,
            max_sequence_length=LTX2_T2V_MAX_SEQ_LEN,
            seed=LTX2_T2V_SEED,
            frame_rate=LTX2_T2V_FRAME_RATE,
        )
        output = visual_gen.generate(
            inputs={
                "prompt": LTX2_T2V_PROMPT,
                "negative_prompt": LTX2_T2V_NEGATIVE_PROMPT,
            },
            params=params,
        )
        MediaStorage.save_video(
            output.video,
            output_path,
            audio=output.audio,
            frame_rate=LTX2_T2V_FRAME_RATE,
        )
    finally:
        visual_gen.shutdown()

    assert os.path.isfile(output_path), f"LTX-2 visual gen did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def ltx2_bf16_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 BF16 T2V video and return path."""
    return _generate_ltx2_video(llm_venv, "ltx2_bf16")


@pytest.fixture(scope="session")
def ltx2_fp8_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 FP8 T2V video and return path."""
    return _generate_ltx2_video(llm_venv, "ltx2_fp8", linear_type="trtllm-fp8-per-tensor")


def _generate_ltx2_two_stage_video(llm_venv, output_subdir, linear_type="default"):
    """Generate a two-stage LTX-2 video using the Python API.

    Requires the main checkpoint, text encoder, spatial upsampler, and
    distilled LoRA.  Returns the path to the generated .mp4, or calls
    pytest.skip if any asset is missing.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm.serve.media_storage import MediaStorage

    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, LTX2_MODEL_CHECKPOINT_PATH)
    text_encoder_path = os.path.join(scratch_space, LTX2_TEXT_ENCODER_SUBPATH)
    upsampler_path = os.path.join(scratch_space, LTX2_UPSAMPLER_SUBPATH)
    lora_path = os.path.join(scratch_space, LTX2_DISTILLED_LORA_SUBPATH)

    for label, path, is_file in [
        ("LTX-2 checkpoint", model_path, True),
        ("text encoder", text_encoder_path, False),
        ("spatial upsampler", upsampler_path, True),
        ("distilled LoRA", lora_path, True),
    ]:
        exists = os.path.isfile(path) if is_file else os.path.isdir(path)
        if not exists:
            pytest.skip(f"Two-stage {label} not found: {path}")

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)
    if os.path.isfile(output_path):
        return output_path

    vg_kwargs = dict(
        text_encoder_path=text_encoder_path,
        spatial_upsampler_path=upsampler_path,
        distilled_lora_path=lora_path,
    )
    quant_config = _linear_type_to_quant_config(linear_type)
    if quant_config is not None:
        vg_kwargs["quant_config"] = quant_config
    if torch.cuda.device_count() >= 2:
        vg_kwargs["parallel"] = {"dit_cfg_size": 2}

    diffusion_args = VisualGenArgs(**vg_kwargs)
    visual_gen = VisualGen(model_path=model_path, diffusion_args=diffusion_args)

    try:
        params = VisualGenParams(
            height=LTX2_TWO_STAGE_HEIGHT,
            width=LTX2_TWO_STAGE_WIDTH,
            num_frames=LTX2_TWO_STAGE_NUM_FRAMES,
            num_inference_steps=LTX2_TWO_STAGE_STEPS,
            guidance_scale=LTX2_TWO_STAGE_GUIDANCE_SCALE,
            max_sequence_length=LTX2_T2V_MAX_SEQ_LEN,
            seed=LTX2_T2V_SEED,
            frame_rate=LTX2_T2V_FRAME_RATE,
        )
        output = visual_gen.generate(
            inputs={
                "prompt": LTX2_T2V_PROMPT,
                "negative_prompt": LTX2_T2V_NEGATIVE_PROMPT,
            },
            params=params,
        )
        MediaStorage.save_video(
            output.video,
            output_path,
            audio=output.audio,
            frame_rate=LTX2_T2V_FRAME_RATE,
        )
    finally:
        visual_gen.shutdown()

    assert os.path.isfile(output_path), f"LTX-2 two-stage did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def ltx2_two_stage_bf16_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 two-stage BF16 T2V video and return path."""
    return _generate_ltx2_two_stage_video(llm_venv, "ltx2_two_stage_bf16")


@pytest.fixture(scope="session")
def ltx2_two_stage_fp8_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 two-stage FP8 T2V video and return path."""
    return _generate_ltx2_two_stage_video(
        llm_venv, "ltx2_two_stage_fp8", linear_type="trtllm-fp8-per-tensor"
    )


def _normalize_score(val):
    """Normalize to 0-1 scale (e.g. imaging_quality can be 0-100)."""
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)) and val > 1.5:
        return val / 100.0
    return float(val)


def _get_per_video_scores(results, video_path_substr):
    """From VBench results, get per-dimension score for the video whose path contains video_path_substr."""
    scores = {}
    for dim in VBENCH_DIMENSIONS:
        dim_result = results[dim]
        assert isinstance(dim_result, list) and len(dim_result) >= 2, (
            f"Dimension '{dim}' result must be [overall_score, video_results]; got {type(dim_result)}"
        )
        video_results = dim_result[1]
        for entry in video_results:
            if video_path_substr in entry.get("video_path", ""):
                raw = entry.get("video_results")
                scores[dim] = _normalize_score(raw)
                break
        else:
            raise AssertionError(
                f"No video matching '{video_path_substr}' in dimension '{dim}'; "
                f"paths: {[e.get('video_path') for e in video_results]}"
            )
    return scores


def test_vbench_dimension_score_wan(vbench_repo_root, wan_trtllm_video_path, llm_venv):
    """Run VBench on WAN TRT-LLM video; compare to golden HF reference scores (diff < 0.05 or TRT-LLM >= golden)."""
    videos_dir = os.path.dirname(wan_trtllm_video_path)
    assert os.path.isfile(wan_trtllm_video_path), "TRT-LLM video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN",
        golden_scores=VBENCH_WAN_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def _run_vbench_and_report(
    vbench_repo_root,
    videos_dir,
    trtllm_filename,
    llm_venv,
    title,
    golden_scores=None,
    max_score_diff=0.10,
):
    """Run VBench, print scores, and optionally assert against golden values.

    When *golden_scores* is None the test is a baseline run: scores are printed
    for the operator to capture but no assertion is made.  Once golden values
    are populated, the comparison kicks in automatically.
    """
    output_path = os.path.join(
        llm_venv.get_working_directory(), "vbench_eval_output", title.replace(" ", "_").lower()
    )
    os.makedirs(output_path, exist_ok=True)
    evaluate_script = os.path.join(vbench_repo_root, "evaluate.py")
    cmd = [
        evaluate_script,
        "--videos_path",
        videos_dir,
        "--output_path",
        output_path,
        "--mode",
        "custom_input",
    ]
    cmd.extend(["--dimension"] + VBENCH_DIMENSIONS)
    venv_check_call(llm_venv, cmd)

    pattern = os.path.join(output_path, "*_eval_results.json")
    result_files = glob.glob(pattern)
    assert result_files, (
        f"No eval results found matching {pattern}; output dir: {os.listdir(output_path)}"
    )
    with open(result_files[0], "r") as f:
        results = json.load(f)
    for dim in VBENCH_DIMENSIONS:
        assert dim in results, (
            f"Expected dimension '{dim}' in results; keys: {list(results.keys())}"
        )

    scores_trtllm = _get_per_video_scores(results, trtllm_filename)
    max_len = max(len(d) for d in VBENCH_DIMENSIONS)

    if golden_scores is not None:
        header = f"{'Dimension':<{max_len}}  |  {'TRT-LLM':>10}  |  {'Golden':>10}  |  {'Diff':>8}"
    else:
        header = f"{'Dimension':<{max_len}}  |  {'TRT-LLM':>10}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(f"VBench dimension scores ({title})")
    print("=" * len(header))
    print(header)
    print(sep)

    for dim in VBENCH_DIMENSIONS:
        t = scores_trtllm[dim]
        if golden_scores is not None:
            r = golden_scores[dim]
            print(f"{dim:<{max_len}}  |  {t:>10.4f}  |  {r:>10.4f}  |  {abs(t - r):>8.4f}")
        else:
            print(f"{dim:<{max_len}}  |  {t:>10.4f}")
    print(sep)

    if golden_scores is None:
        print("\n** Baseline run — no golden scores to compare against. **")
        print("** Copy the values above into the GOLDEN_SCORES dict.  **\n")
        return scores_trtllm

    max_diff_val = max(abs(scores_trtllm[d] - golden_scores[d]) for d in VBENCH_DIMENSIONS)
    print(f"max_diff={max_diff_val:.4f}  (threshold={max_score_diff})")
    print("=" * len(header) + "\n")
    for dim in VBENCH_DIMENSIONS:
        diff = abs(scores_trtllm[dim] - golden_scores[dim])
        assert diff < max_score_diff or scores_trtllm[dim] >= golden_scores[dim], (
            f"Dimension '{dim}' score difference {diff:.4f} >= {max_score_diff} "
            f"(TRT-LLM={scores_trtllm[dim]:.4f}, golden={golden_scores[dim]:.4f})"
        )
    return scores_trtllm


def test_vbench_dimension_score_wan22_a14b_fp8(
    vbench_repo_root, wan22_a14b_fp8_video_path, llm_venv
):
    """VBench accuracy for Wan 2.2 A14B FP8 — full generate + evaluate."""
    videos_dir = os.path.dirname(wan22_a14b_fp8_video_path)
    assert os.path.isfile(wan22_a14b_fp8_video_path), "FP8 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B FP8",
        golden_scores=VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_wan22_a14b_nvfp4(
    vbench_repo_root, wan22_a14b_nvfp4_video_path, llm_venv
):
    """VBench accuracy for Wan 2.2 A14B NVFP4 — full generate + evaluate."""
    videos_dir = os.path.dirname(wan22_a14b_nvfp4_video_path)
    assert os.path.isfile(wan22_a14b_nvfp4_video_path), "NVFP4 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B NVFP4",
        golden_scores=VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_ltx2_bf16(vbench_repo_root, ltx2_bf16_video_path, llm_venv):
    """VBench accuracy for LTX-2 BF16 T2V — baseline run (golden scores TBD)."""
    videos_dir = os.path.dirname(ltx2_bf16_video_path)
    assert os.path.isfile(ltx2_bf16_video_path), "LTX-2 BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="LTX-2 BF16",
        golden_scores=VBENCH_LTX2_BF16_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_ltx2_fp8(vbench_repo_root, ltx2_fp8_video_path, llm_venv):
    """VBench accuracy for LTX-2 FP8 T2V — baseline run (golden scores TBD)."""
    videos_dir = os.path.dirname(ltx2_fp8_video_path)
    assert os.path.isfile(ltx2_fp8_video_path), "LTX-2 FP8 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="LTX-2 FP8",
        golden_scores=VBENCH_LTX2_FP8_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_ltx2_two_stage_bf16(
    vbench_repo_root, ltx2_two_stage_bf16_video_path, llm_venv
):
    """VBench accuracy for LTX-2 two-stage BF16 T2V."""
    videos_dir = os.path.dirname(ltx2_two_stage_bf16_video_path)
    assert os.path.isfile(ltx2_two_stage_bf16_video_path), "LTX-2 two-stage BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="LTX-2 Two-Stage BF16",
        golden_scores=VBENCH_LTX2_TWO_STAGE_BF16_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_vbench_dimension_score_ltx2_two_stage_fp8(
    vbench_repo_root, ltx2_two_stage_fp8_video_path, llm_venv
):
    """VBench accuracy for LTX-2 two-stage FP8 T2V."""
    videos_dir = os.path.dirname(ltx2_two_stage_fp8_video_path)
    assert os.path.isfile(ltx2_two_stage_fp8_video_path), "LTX-2 two-stage FP8 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="LTX-2 Two-Stage FP8",
        golden_scores=VBENCH_LTX2_TWO_STAGE_FP8_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


def test_visual_gen_quickstart(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/quickstart_example.py end-to-end."""
    scratch_space = conftest.llm_models_root()
    model_src = os.path.join(scratch_space, WAN_T2V_MODEL_SUBPATH)
    if not os.path.isdir(model_src):
        pytest.skip(
            f"Model not found: {model_src} "
            f"(set LLM_MODELS_ROOT or place {WAN_T2V_MODEL_SUBPATH} under scratch)"
        )

    model_dst = os.path.join(llm_venv.get_working_directory(), "Wan-AI", WAN_T2V_MODEL_SUBPATH)
    if not os.path.islink(model_dst):
        os.makedirs(os.path.dirname(model_dst), exist_ok=True)
        os.symlink(model_src, model_dst, target_is_directory=True)

    script_path = os.path.join(llm_root, "examples", "visual_gen", "quickstart_example.py")
    venv_check_call(llm_venv, [script_path])

    output_path = os.path.join(llm_venv.get_working_directory(), "output.avi")
    assert os.path.isfile(output_path), f"Quickstart did not produce output.avi at {output_path}"
