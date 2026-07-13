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
"""Integration tests for VisualGen examples and visual quality checks."""

import base64
import contextlib
import gc
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

import pytest
import torch
import torch._inductor.config as inductor_config
from defs import conftest
from defs.common import venv_check_call
from defs.trt_test_alternative import check_call
from torch._inductor.async_compile import shutdown_compile_workers

WAN_T2V_MODEL_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
WAN22_A14B_FP8_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-FP8"
WAN22_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-T2V-A14B-Diffusers-NVFP4"
WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH = "Wan2.2-I2V-A14B-Diffusers-NVFP4"
QWEN_IMAGE_MODEL_SUBPATH = "qwen-image"
VISUAL_GEN_OUTPUT_VIDEO = "trtllm_output.mp4"
DIFFUSERS_REFERENCE_VIDEO = "diffusers_reference.mp4"
WAN_T2V_PROMPT = "A cute cat playing piano"
WAN_T2V_HEIGHT = 480
WAN_T2V_WIDTH = 832
WAN_T2V_NUM_FRAMES = 165

# NB: this test file lives at tests/integration/defs/examples/visual_gen/, so the repo
# root is five levels up (the LPIPS eval script is referenced from <repo>/scripts/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
VISUAL_GEN_LPIPS_EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "visualgen_eval", "visual_gen_lpips_score_eval.py"
)
VISUAL_GEN_LPIPS_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden", "visual_gen_lpips")
VISUAL_GEN_LPIPS_GOLDEN_MEDIA_ZIP = os.path.join(
    VISUAL_GEN_LPIPS_GOLDEN_DIR, "visual_gen_lpips_golden_media.zip"
)

FLUX_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
FLUX_LPIPS_HEIGHT = 256
FLUX_LPIPS_WIDTH = 256
FLUX_LPIPS_NUM_INFERENCE_STEPS = 4
FLUX_LPIPS_GUIDANCE_SCALE = 3.5
FLUX_LPIPS_SEED = 42
FLUX_LPIPS_THRESHOLD = 0.05

LTX2_LPIPS_NUM_FRAMES = 49
LTX2_LPIPS_NUM_INFERENCE_STEPS = 8
LTX2_LPIPS_THRESHOLD = 0.05
LTX2_CUDA_GRAPH_LPIPS_THRESHOLD = 0.01

WAN21_LPIPS_PROMPT = "A cat sitting on a windowsill"
WAN21_LPIPS_NEGATIVE_PROMPT = None
WAN21_LPIPS_HEIGHT = 256
WAN21_LPIPS_WIDTH = 256
WAN21_LPIPS_NUM_FRAMES = 5
WAN21_LPIPS_NUM_INFERENCE_STEPS = 1
WAN21_LPIPS_GUIDANCE_SCALE = 5.0
WAN21_LPIPS_SEED = 42
WAN_LPIPS_FRAME_RATE = 16.0
WAN_LPIPS_THRESHOLD = 0.05

WAN22_LPIPS_PROMPT = "A cat sitting on a sunny windowsill watching birds outside."
WAN22_LPIPS_NEGATIVE_PROMPT = ""
WAN22_LPIPS_HEIGHT = 480
WAN22_LPIPS_WIDTH = 832
WAN22_LPIPS_NUM_FRAMES = 9
WAN22_LPIPS_NUM_INFERENCE_STEPS = 4
WAN22_LPIPS_GUIDANCE_SCALE = 4.0
WAN22_LPIPS_SEED = 42
WAN22_LPIPS_FRAME_RATE = 16.0

# QwenImage (text-to-image) — default-setting LPIPS golden.
# Params mirror the QwenImage 20B reference defaults (pipeline_qwen_image.py).
# NOTE: QwenImage's forward CFG knob is ``true_cfg_scale`` (not ``guidance_scale``),
# and real-CFG only engages when a negative prompt is supplied.
QWENIMAGE_MODEL_SUBPATH = "qwen-image"
QWENIMAGE_LPIPS_PROMPT = "a tiny astronaut hatching from an egg on the moon"
QWENIMAGE_LPIPS_NEGATIVE_PROMPT = ""
QWENIMAGE_LPIPS_HEIGHT = 1328
QWENIMAGE_LPIPS_WIDTH = 1328
QWENIMAGE_LPIPS_NUM_INFERENCE_STEPS = 50
QWENIMAGE_LPIPS_TRUE_CFG_SCALE = 4.0
QWENIMAGE_LPIPS_SEED = 42
QWENIMAGE_LPIPS_THRESHOLD = 0.05

# Cosmos3-Nano (text-to-video + text-to-image) — default-setting LPIPS golden.
# Params are the Cosmos3 720P defaults (cosmos3/defaults.py:COSMOS3_720P_PARAMS).
# Cosmos3 requires VANILLA attention and guardrails disabled in CI.
COSMOS3_NANO_MODEL_SUBPATH = "Cosmos3-Nano"
COSMOS3_LPIPS_PROMPT = "A serene mountain landscape with snow-capped peaks and a flowing river"
COSMOS3_LPIPS_HEIGHT = 720
COSMOS3_LPIPS_WIDTH = 1280
COSMOS3_LPIPS_T2V_NUM_FRAMES = 189
COSMOS3_LPIPS_T2I_NUM_FRAMES = 1
COSMOS3_LPIPS_NUM_INFERENCE_STEPS = 35
COSMOS3_LPIPS_GUIDANCE_SCALE = 6.0
COSMOS3_LPIPS_SEED = 42
COSMOS3_LPIPS_FRAME_RATE = 24.0
COSMOS3_LPIPS_THRESHOLD = 0.05

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
    "subject_consistency": 0.8907,
    "background_consistency": 0.9274,
    "motion_smoothness": 0.9818,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.2928,
    "imaging_quality": 0.3812,
}


# TODO: Reference scores from bf16 baseline runs
VBENCH_WAN22_BF16_GOLDEN_SCORES = {
    "subject_consistency": 0.9103,
    "background_consistency": 0.9516,
    "motion_smoothness": 0.9693,
    "dynamic_degree": 0.0000,
    "aesthetic_quality": 0.6821,
    "imaging_quality": 0.3993,
}
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

# LTX-2 Two-Stage configuration
LTX2_UPSAMPLER_SUBPATH = "LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_DISTILLED_LORA_SUBPATH = "LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
LTX2_TWO_STAGE_HEIGHT = 1024
LTX2_TWO_STAGE_WIDTH = 1536
LTX2_TWO_STAGE_NUM_FRAMES = 121
LTX2_TWO_STAGE_STEPS = 40
LTX2_TWO_STAGE_GUIDANCE_SCALE = 4.0

# Golden VBench scores for two-stage pipeline variants.
# Initially None — first CI run is a baseline that prints scores for capture.
VBENCH_LTX2_TWO_STAGE_BF16_GOLDEN_SCORES = {
    "subject_consistency": 0.9877,
    "background_consistency": 0.9601,
    "motion_smoothness": 0.9952,
    "dynamic_degree": 0.0,
    "aesthetic_quality": 0.5839,
    "imaging_quality": 0.5404,
}

VBENCH_LTX2_TWO_STAGE_FP8_GOLDEN_SCORES = {
    "subject_consistency": 0.9820,
    "background_consistency": 0.9617,
    "motion_smoothness": 0.9885,
    "dynamic_degree": 1.0,
    "aesthetic_quality": 0.6017,
    "imaging_quality": 0.7136,
}

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
    llm_venv.run_cmd(["-m", "pip", "install", "diffusers>=0.37.0"])
    # Install ffmpeg system package required by save_video() for MP4 encoding
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


def _lpips_model_path(*parts):
    return os.path.join(conftest.llm_models_root(), *parts)


def _skip_if_missing(path, label, is_dir=False):
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    if not exists:
        pytest.skip(f"{label} not found: {path}")


def _visual_gen_output_path(llm_venv, output_subdir):
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)


def _extract_visual_gen_lpips_golden_media(tmp_path):
    _skip_if_missing(VISUAL_GEN_LPIPS_GOLDEN_MEDIA_ZIP, "VisualGen LPIPS golden media zip")
    extract_dir = tmp_path / "visual_gen_lpips_golden_media"
    if extract_dir.exists():
        return extract_dir

    with zipfile.ZipFile(VISUAL_GEN_LPIPS_GOLDEN_MEDIA_ZIP) as archive:
        for member in archive.namelist():
            if os.path.isabs(member) or ".." in member.split("/"):
                raise ValueError(f"Unsafe golden media zip member: {member}")
        archive.extractall(extract_dir)
    return extract_dir


def _golden_media_path(tmp_path, media_name, label):
    path = _extract_visual_gen_lpips_golden_media(tmp_path) / media_name
    _skip_if_missing(path, label)
    return path


def _ltx2_lpips_text_encoder_path():
    scratch_space = conftest.llm_models_root()
    candidates = [
        os.path.join(scratch_space, LTX2_TEXT_ENCODER_SUBPATH),
        os.path.join(scratch_space, "gemma", LTX2_TEXT_ENCODER_SUBPATH),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _disable_inductor_compile_worker_quiesce():
    # The quiesce timer thread can outlive shutdown_compile_workers() long
    # enough for pytest-threadleak to report it as a leaked thread.
    if hasattr(inductor_config, "quiesce_async_compile_pool"):
        inductor_config.quiesce_async_compile_pool = False


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # torch.compile / unconditional @torch.compile decorators (e.g. on the TRT-LLM
    # attention backend's _concat_qkv) lazily spawn an InductorSubproc worker
    # pool the first time a compiled function runs. The pool's daemon threads
    # outlive the test and trip pytest-threadleak. Tear them down explicitly.
    shutdown_compile_workers()


@contextlib.contextmanager
def _lpips_deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _save_lpips_video_mp4(video, output_path, frame_rate):
    from tensorrt_llm.media.encoding import save_video

    try:
        save_video(video, output_path, frame_rate=frame_rate)
    except RuntimeError as err:
        if "MP4 format requires ffmpeg" not in str(err):
            raise
        # Never fall back to another codec here: the goldens are H.264/x264 and
        # LPIPS compares decoded pixels, so a silent cv2/mp4v fallback compares
        # codec artifacts, not model output (this produced a spurious 0.059 on
        # wan22 while the generated frames were identical to the golden).
        pytest.fail(
            "ffmpeg is unavailable for LPIPS video encoding; refusing to fall back "
            "to another codec because the golden comparison would measure codec "
            f"artifacts instead of model output: {err}"
        )
    assert os.path.isfile(output_path), f"Visual gen did not produce {output_path}"


def _run_lpips_eval(tmp_path, sample_id, media_type, prompt, reference_path, generated_path):
    reference_key = "reference_video_path" if media_type == "video" else "reference_image_path"
    generated_key = "generated_video_path" if media_type == "video" else "generated_image_path"
    dataset_path = tmp_path / f"{sample_id}_dataset.json"
    output_json = tmp_path / f"{sample_id}_lpips_results.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": sample_id,
                        "media_type": media_type,
                        "prompt": prompt,
                        reference_key: str(reference_path),
                        generated_key: str(generated_path),
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else REPO_ROOT
    )
    with _lpips_deterministic_algorithms():
        result = subprocess.run(
            [
                sys.executable,
                VISUAL_GEN_LPIPS_EVAL_SCRIPT,
                "--dataset",
                str(dataset_path),
                "--output-json",
                str(output_json),
                "--json",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        pytest.fail(f"LPIPS eval script failed for {sample_id}:\n{result.stdout}")

    scores = json.loads(output_json.read_text(encoding="utf-8"))
    score = float(scores["mean_lpips_score"])
    print(f"\n[E2E {sample_id} LPIPS] score: {score:.6f}")
    return score


def _assert_lpips_below_threshold(score, threshold):
    assert score < threshold, f"LPIPS too high: {score:.6f} (expected < {threshold:.6f})"


def _preserve_lpips_candidate_on_failure(request, score, threshold, candidate_path, artifact_name):
    """Copy the generated candidate into pytest's --output-dir when the LPIPS gate fails.

    CI archives the output dir per stage, so a threshold failure leaves behind the
    exact CI-generated media needed to refresh the golden without guessing at
    machine-to-machine kernel-stack drift (measured ~0.04 LPIPS across B200 hosts
    on the same container for 1-step Wan2.1).
    """
    if score < threshold:
        return
    output_dir = request.config.getoption("--output-dir", default=None)
    if not output_dir:
        return
    dest_dir = os.path.join(str(output_dir), "lpips_failure_artifacts")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, artifact_name)
    shutil.copy2(str(candidate_path), dest)
    print(f"[LPIPS] candidate preserved for golden refresh: {dest}")
    # CI's per-stage results tarball only collects flat files (results.xml etc.), not
    # this subdirectory — and junit stdout IS collected. Small candidates therefore
    # also go into stdout as base64 so a threshold failure is diagnosable from the
    # archived results.xml alone (this is how wan22's spurious 0.059 was traced to a
    # silent mpeg4 codec fallback rather than a generation difference).
    size = os.path.getsize(dest)
    if size <= 8 * 1024 * 1024:
        with open(dest, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        print(f"[LPIPS-B64-BEGIN {artifact_name} {size}]")
        for i in range(0, len(encoded), 3072):
            print(encoded[i : i + 3072])
        print(f"[LPIPS-B64-END {artifact_name}]")


def _generate_flux_lpips_image(model_path, output_path):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "FLUX checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    with _lpips_deterministic_algorithms():
        args = VisualGenArgs(
            model=model_path,
            torch_compile_config=TorchCompileConfig(enable=False),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            result = pipeline.forward(
                prompt=FLUX_LPIPS_PROMPT,
                height=FLUX_LPIPS_HEIGHT,
                width=FLUX_LPIPS_WIDTH,
                num_inference_steps=FLUX_LPIPS_NUM_INFERENCE_STEPS,
                guidance_scale=FLUX_LPIPS_GUIDANCE_SCALE,
                seed=FLUX_LPIPS_SEED,
            )
            generated_image = result.image[0].detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()

    save_image(generated_image, output_path)


def _generate_ltx2_lpips_video(output_path, *, enable_cuda_graph=False):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import CudaGraphConfig, TorchCompileConfig, VisualGenArgs

    checkpoint_path = _lpips_model_path("LTX-2", "ltx-2-19b-dev.safetensors")
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    spatial_upsampler_path = _lpips_model_path("LTX-2", "ltx-2-spatial-upscaler-x2-1.0.safetensors")
    distilled_lora_path = _lpips_model_path("LTX-2", "ltx-2-19b-distilled-lora-384.safetensors")
    _skip_if_missing(checkpoint_path, "LTX-2 checkpoint")
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder", is_dir=True)
    _skip_if_missing(spatial_upsampler_path, "LTX-2 spatial upsampler")
    _skip_if_missing(distilled_lora_path, "LTX-2 distilled LoRA")
    _disable_inductor_compile_worker_quiesce()

    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    # Wrapped here (not in the fixture) so the golden fixture and both sides of
    # test_ltx2_cuda_graph_lpips_matches_eager run the same eager numerics.
    with _lpips_deterministic_algorithms(), torch.compiler.set_stance("force_eager"):
        args = VisualGenArgs(
            model=checkpoint_path,
            pipeline_config={
                "text_encoder_path": text_encoder_path,
                "spatial_upsampler_path": spatial_upsampler_path,
                "distilled_lora_path": distilled_lora_path,
            },
            torch_compile_config=TorchCompileConfig(enable=False),
            cuda_graph_config=CudaGraphConfig(enable=enable_cuda_graph),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=LTX2_T2V_PROMPT,
                    negative_prompt=LTX2_T2V_NEGATIVE_PROMPT,
                    height=LTX2_T2V_HEIGHT,
                    width=LTX2_T2V_WIDTH,
                    num_frames=LTX2_LPIPS_NUM_FRAMES,
                    num_inference_steps=LTX2_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=LTX2_T2V_GUIDANCE_SCALE,
                    seed=LTX2_T2V_SEED,
                )
            generated_video = result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()

    _save_lpips_video_mp4(generated_video, output_path, frame_rate=LTX2_T2V_FRAME_RATE)


def _run_wan_lpips_pipeline(
    model_path,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    seed,
    attention_backend="VANILLA",
    parallel=None,
):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import AttentionConfig, TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "Wan checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    args_kwargs = dict(
        model=model_path,
        attention_config=AttentionConfig(backend=attention_backend),
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    if parallel is not None:
        args_kwargs["parallel_config"] = parallel
    with _lpips_deterministic_algorithms():
        args = VisualGenArgs(**args_kwargs)
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            if result is None or result.video is None:
                return None
            return result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()


def _generate_wan_lpips_video(
    model_path,
    output_path,
    prompt,
    negative_prompt,
    height,
    width,
    num_frames,
    num_inference_steps,
    guidance_scale,
    seed,
    frame_rate,
    parallel=None,
):
    generated_video = _run_wan_lpips_pipeline(
        model_path,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        seed,
        parallel=parallel,
    )
    assert generated_video is not None, "Single-GPU Wan LPIPS run produced no video"
    _save_lpips_video_mp4(generated_video, output_path, frame_rate=frame_rate)


@pytest.fixture(scope="session")
def wan21_bf16_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "wan21_bf16")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path("Wan2.1-T2V-1.3B-Diffusers"),
            output_path,
            WAN21_LPIPS_PROMPT,
            WAN21_LPIPS_NEGATIVE_PROMPT,
            WAN21_LPIPS_HEIGHT,
            WAN21_LPIPS_WIDTH,
            WAN21_LPIPS_NUM_FRAMES,
            WAN21_LPIPS_NUM_INFERENCE_STEPS,
            WAN21_LPIPS_GUIDANCE_SCALE,
            WAN21_LPIPS_SEED,
            WAN_LPIPS_FRAME_RATE,
        )
    return output_path


@pytest.fixture(scope="session")
def wan22_bf16_video_path(_visual_gen_deps, llm_venv):
    output_path = _visual_gen_output_path(llm_venv, "wan22_bf16")
    if os.path.isfile(output_path):
        return output_path
    # TorchCompileConfig(enable=False) does not suppress nested @torch.compile decorators.
    with torch.compiler.set_stance("force_eager"):
        _generate_wan_lpips_video(
            _lpips_model_path("Wan2.2-T2V-A14B-Diffusers"),
            output_path,
            WAN22_LPIPS_PROMPT,
            WAN22_LPIPS_NEGATIVE_PROMPT,
            WAN22_LPIPS_HEIGHT,
            WAN22_LPIPS_WIDTH,
            WAN22_LPIPS_NUM_FRAMES,
            WAN22_LPIPS_NUM_INFERENCE_STEPS,
            WAN22_LPIPS_GUIDANCE_SCALE,
            WAN22_LPIPS_SEED,
            WAN22_LPIPS_FRAME_RATE,
        )
    return output_path


def _generate_qwenimage_lpips_image(model_path, output_path):
    """Generate the QwenImage text-to-image LPIPS sample (default setting, compile-off)."""
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image
    from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs

    _skip_if_missing(model_path, "QwenImage checkpoint", is_dir=True)
    _disable_inductor_compile_worker_quiesce()
    args = VisualGenArgs(
        model=model_path,
        torch_compile_config=TorchCompileConfig(enable=False),
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


def _run_cosmos3_lpips_pipeline(num_frames):
    """Run the Cosmos3-Nano pipeline (default setting, VANILLA attn, compile-off).

    Returns the generated video tensor ``(B, T, H, W, C)`` (T == ``num_frames``),
    or ``None`` if generation produced no video.  ``num_frames=1`` yields the
    single-frame text-to-image path.
    """
    # Cosmos3 re-reads the guardrail flag in __init__; set it before the pipeline loads.
    guardrails_env_key = "TRTLLM_DISABLE_COSMOS3_GUARDRAILS"
    previous_guardrails_env = os.environ.get(guardrails_env_key)
    os.environ[guardrails_env_key] = "1"
    try:
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
        from tensorrt_llm.visual_gen.args import (
            AttentionConfig,
            CompilationConfig,
            TorchCompileConfig,
            VisualGenArgs,
        )

        model_path = _lpips_model_path(COSMOS3_NANO_MODEL_SUBPATH)
        _skip_if_missing(model_path, "Cosmos3-Nano checkpoint", is_dir=True)
        _disable_inductor_compile_worker_quiesce()
        args = VisualGenArgs(
            model=model_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=False),
            attention_config=AttentionConfig(backend="VANILLA"),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt=COSMOS3_LPIPS_PROMPT,
                    seed=COSMOS3_LPIPS_SEED,
                    height=COSMOS3_LPIPS_HEIGHT,
                    width=COSMOS3_LPIPS_WIDTH,
                    num_frames=num_frames,
                    num_inference_steps=COSMOS3_LPIPS_NUM_INFERENCE_STEPS,
                    guidance_scale=COSMOS3_LPIPS_GUIDANCE_SCALE,
                    frame_rate=COSMOS3_LPIPS_FRAME_RATE,
                    use_guardrails=False,
                )
            if result is None or result.video is None:
                return None
            return result.video.detach().cpu()
        finally:
            del pipeline
            _cleanup_cuda()
    finally:
        if previous_guardrails_env is None:
            os.environ.pop(guardrails_env_key, None)
        else:
            os.environ[guardrails_env_key] = previous_guardrails_env


def _generate_cosmos3_lpips_video(output_path):
    """Generate the Cosmos3-Nano text-to-video LPIPS sample."""
    video = _run_cosmos3_lpips_pipeline(COSMOS3_LPIPS_T2V_NUM_FRAMES)
    assert video is not None, "Cosmos3-Nano T2V LPIPS run produced no video"
    _save_lpips_video_mp4(video, output_path, frame_rate=COSMOS3_LPIPS_FRAME_RATE)


def _generate_cosmos3_lpips_image(output_path):
    """Generate the Cosmos3-Nano text-to-image LPIPS sample (single frame)."""
    from tensorrt_llm.media.encoding import save_image

    video = _run_cosmos3_lpips_pipeline(COSMOS3_LPIPS_T2I_NUM_FRAMES)
    assert video is not None, "Cosmos3-Nano T2I LPIPS run produced no frame"
    # video is (B, T, H, W, C); take the single frame -> (H, W, C) for save_image.
    save_image(video[0, 0], output_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flux1_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "flux1_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "flux1_lpips_golden.png", "FLUX.1 LPIPS golden image"
    )
    _generate_flux_lpips_image(_lpips_model_path("FLUX.1-dev"), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "flux1",
        "image",
        FLUX_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, FLUX_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flux2_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "flux2_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "flux2_lpips_golden.png", "FLUX.2 LPIPS golden image"
    )
    _generate_flux_lpips_image(_lpips_model_path("FLUX.2-dev"), generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "flux2",
        "image",
        FLUX_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, FLUX_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx2_lpips_against_golden(request, tmp_path, ltx2_two_stage_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "ltx2_lpips_golden_video.mp4", "LTX-2 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "ltx2",
        "video",
        LTX2_T2V_PROMPT,
        golden_path,
        ltx2_two_stage_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        LTX2_LPIPS_THRESHOLD,
        ltx2_two_stage_bf16_video_path,
        "ltx2_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, LTX2_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ltx2_cuda_graph_lpips_matches_eager(_visual_gen_deps, tmp_path):
    eager_path = tmp_path / "ltx2_eager_generated.mp4"
    cuda_graph_path = tmp_path / "ltx2_cuda_graph_generated.mp4"

    _generate_ltx2_lpips_video(eager_path, enable_cuda_graph=False)
    _generate_ltx2_lpips_video(cuda_graph_path, enable_cuda_graph=True)
    score = _run_lpips_eval(
        tmp_path,
        "ltx2_cuda_graph",
        "video",
        LTX2_T2V_PROMPT,
        eager_path,
        cuda_graph_path,
    )
    _assert_lpips_below_threshold(score, LTX2_CUDA_GRAPH_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wan21_t2v_lpips_against_golden(request, tmp_path, wan21_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "wan21_t2v_lpips_golden_video.mp4", "Wan 2.1 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "wan21_t2v",
        "video",
        WAN21_LPIPS_PROMPT,
        golden_path,
        wan21_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        WAN_LPIPS_THRESHOLD,
        wan21_bf16_video_path,
        "wan21_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, WAN_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_wan22_t2v_lpips_against_golden(request, tmp_path, wan22_bf16_video_path):
    golden_path = _golden_media_path(
        tmp_path, "wan22_t2v_lpips_golden_video.mp4", "Wan 2.2 LPIPS golden video"
    )
    score = _run_lpips_eval(
        tmp_path,
        "wan22_t2v",
        "video",
        WAN22_LPIPS_PROMPT,
        golden_path,
        wan22_bf16_video_path,
    )
    _preserve_lpips_candidate_on_failure(
        request,
        score,
        WAN_LPIPS_THRESHOLD,
        wan22_bf16_video_path,
        "wan22_t2v_lpips_golden_video.mp4",
    )
    _assert_lpips_below_threshold(score, WAN_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_qwenimage_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "qwenimage_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "qwenimage_lpips_golden.png", "QwenImage LPIPS golden image"
    )
    _generate_qwenimage_lpips_image(_lpips_model_path(QWENIMAGE_MODEL_SUBPATH), generated_path)
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
def test_cosmos3_nano_t2v_lpips_against_golden(_visual_gen_deps, tmp_path):
    generated_path = tmp_path / "cosmos3_nano_t2v_generated.mp4"
    golden_path = _golden_media_path(
        tmp_path,
        "cosmos3_nano_t2v_lpips_golden_video.mp4",
        "Cosmos3-Nano T2V LPIPS golden video",
    )
    _generate_cosmos3_lpips_video(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_nano_t2v",
        "video",
        COSMOS3_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, COSMOS3_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosmos3_nano_t2i_lpips_against_golden(tmp_path):
    generated_path = tmp_path / "cosmos3_nano_t2i_generated.png"
    golden_path = _golden_media_path(
        tmp_path, "cosmos3_nano_t2i_lpips_golden.png", "Cosmos3-Nano T2I LPIPS golden image"
    )
    _generate_cosmos3_lpips_image(generated_path)
    score = _run_lpips_eval(
        tmp_path,
        "cosmos3_nano_t2i",
        "image",
        COSMOS3_LPIPS_PROMPT,
        golden_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, COSMOS3_LPIPS_THRESHOLD)


def _generate_wan_video(llm_venv, model_subpath, output_subdir):
    """Generate a WAN video for a given model checkpoint.

    Returns the path to the generated .mp4, or calls pytest.skip if the model
    is not found under LLM_MODELS_ROOT.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

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

    visual_gen_args = VisualGenArgs(
        attention_config={"backend": "VANILLA"},
        parallel_config={"cfg_size": 2 if torch.cuda.device_count() >= 2 else 1},
        torch_compile_config={"enable": False},
        compilation_config={"skip_warmup": True},
    )
    visual_gen = VisualGen(model=model_path, args=visual_gen_args)
    try:
        frame_rate = visual_gen.default_params.frame_rate
        output = visual_gen.generate(
            inputs=WAN_T2V_PROMPT,
            params=VisualGenParams(
                height=WAN_T2V_HEIGHT,
                width=WAN_T2V_WIDTH,
                seed=42,
                num_frames=WAN_T2V_NUM_FRAMES,
                frame_rate=frame_rate,
            ),
        )
        assert output.error is None, f"unexpected error on WAN run: {output.error}"
        assert output.video is not None
        _save_lpips_video_mp4(output.video, output_path, frame_rate=frame_rate)
    finally:
        visual_gen.shutdown()

    assert os.path.isfile(output_path), f"Visual gen did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def wan22_a14b_fp8_video_path(_visual_gen_deps, llm_venv):
    """Generate video with Wan 2.2 A14B FP8 checkpoint."""
    return _generate_wan_video(llm_venv, WAN22_A14B_FP8_MODEL_SUBPATH, "wan22_fp8")


@pytest.fixture(scope="session")
def wan22_a14b_nvfp4_video_path(_visual_gen_deps, llm_venv):
    """Generate video with Wan 2.2 A14B NVFP4 checkpoint."""
    return _generate_wan_video(llm_venv, WAN22_A14B_NVFP4_MODEL_SUBPATH, "wan22_nvfp4")


def _linear_type_to_quant_config(linear_type):
    """Map linear_type shortcut to quant_config dict for VisualGenArgs."""
    mapping = {
        "trtllm-fp8-per-tensor": {"quant_algo": "FP8", "dynamic": True},
        "trtllm-fp8-blockwise": {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        "trtllm-nvfp4": {"quant_algo": "NVFP4", "dynamic": True},
    }
    return mapping.get(linear_type)


def _generate_ltx2_two_stage_video(llm_venv, output_subdir, linear_type="default"):
    """Generate a two-stage LTX-2 video using the Python API.

    Requires the main checkpoint, text encoder, spatial upsampler, and
    distilled LoRA.  Returns the path to the generated .mp4, or calls
    pytest.skip if any asset is missing.
    """
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

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
        pipeline_config={
            "text_encoder_path": text_encoder_path,
            "spatial_upsampler_path": upsampler_path,
            "distilled_lora_path": lora_path,
        },
    )
    quant_config = _linear_type_to_quant_config(linear_type)
    if quant_config is not None:
        vg_kwargs["quant_config"] = quant_config
    if torch.cuda.device_count() >= 2:
        vg_kwargs["parallel_config"] = {"cfg_size": 2}

    visual_gen_args = VisualGenArgs(**vg_kwargs)
    visual_gen = VisualGen(model=model_path, args=visual_gen_args)

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
            negative_prompt=LTX2_T2V_NEGATIVE_PROMPT,
        )
        output = visual_gen.generate(inputs=LTX2_T2V_PROMPT, params=params)
        assert output.error is None
        assert output.video is not None
        assert output.frame_rate == LTX2_T2V_FRAME_RATE
        assert output.metrics is not None
        assert output.metrics.generation > 0
        assert output.metrics.denoise > 0
        assert output.metrics.post_denoise >= 0
        _save_lpips_video_mp4(output.video, output_path, frame_rate=LTX2_T2V_FRAME_RATE)
    finally:
        visual_gen.shutdown()

    assert os.path.isfile(output_path), f"LTX-2 two-stage did not produce {output_path}"
    return output_path


@pytest.fixture(scope="session")
def ltx2_two_stage_bf16_video_path(_visual_gen_deps, llm_venv):
    """Generate LTX-2 two-stage BF16 video with the LPIPS config and return path."""
    output_path = _visual_gen_output_path(llm_venv, "ltx2_two_stage_bf16")
    if os.path.isfile(output_path):
        return output_path
    _generate_ltx2_lpips_video(output_path)
    return output_path


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


def test_vbench_dimension_score_wan(vbench_repo_root, wan21_bf16_video_path, llm_venv):
    """Run VBench on WAN 2.1 BF16 video generated with the LPIPS config."""
    videos_dir = os.path.dirname(wan21_bf16_video_path)
    assert os.path.isfile(wan21_bf16_video_path), "WAN 2.1 BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.1 BF16",
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


def test_vbench_dimension_score_wan22_bf16(vbench_repo_root, wan22_bf16_video_path, llm_venv):
    """VBench accuracy for Wan 2.2 A14B BF16 generated with the LPIPS config."""
    videos_dir = os.path.dirname(wan22_bf16_video_path)
    assert os.path.isfile(wan22_bf16_video_path), "WAN 2.2 BF16 video must exist"
    _run_vbench_and_report(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        llm_venv,
        title="WAN 2.2 A14B BF16",
        golden_scores=VBENCH_WAN22_BF16_GOLDEN_SCORES,
        max_score_diff=0.05,
    )


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
        max_score_diff=0.06,
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


def test_visual_gen_api_walkthrough(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/api_walkthrough.py end-to-end."""
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

    script_path = os.path.join(llm_root, "examples", "visual_gen", "api_walkthrough.py")
    venv_check_call(llm_venv, [script_path])

    output_path = os.path.join(llm_venv.get_working_directory(), "api_walkthrough_output.avi")
    assert os.path.isfile(output_path), f"API walkthrough did not produce {output_path}"


# =============================================================================
# Core example tests — run per-model scripts from examples/visual_gen/models/
# with shared YAML configs from examples/visual_gen/configs/.
# =============================================================================


def test_wan_t2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/wan_t2v.py with NVFP4 config end-to-end.

    This is a core example test: it validates that the per-model example script
    and the shared YAML config work together as documented in the README.
    Uses the pre-quantized Wan 2.2 T2V A14B NVFP4 checkpoint and the shared
    ``configs/wan2.2-t2v-fp4-1gpu.yaml`` (NVFP4 dynamic quant). The closest
    overlapping test is ``test_vbench_dimension_score_wan22_a14b_nvfp4``,
    which runs the same script but with a no-quant YAML synthesized at
    runtime and additionally evaluates VBench scores.
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, WAN22_A14B_NVFP4_MODEL_SUBPATH)
    assert os.path.isdir(model_path), (
        f"Model not found: {model_path} "
        f"(set LLM_MODELS_ROOT or place {WAN22_A14B_NVFP4_MODEL_SUBPATH} under models root)"
    )

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "wan_t2v_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "wan_t2v_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "wan_t2v.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "wan2.2-t2v-fp4-1gpu.yaml"
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


def test_flux1_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/flux1.py with NVFP4 config end-to-end.

    Validates that the FLUX.1-dev example script and ``configs/flux1-dev-fp4-1gpu.yaml``
    work together as documented. Uses the local FLUX.1-dev checkpoint and the shared
    NVFP4 dynamic-quant config.
    """
    model_path = _lpips_model_path("FLUX.1-dev")
    _skip_if_missing(model_path, "FLUX.1-dev checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "flux1_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "flux1_output.png")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "flux1.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "flux1-dev-fp4-1gpu.yaml"
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


def test_flux2_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/flux2.py with NVFP4 config end-to-end.

    Validates that the FLUX.2-dev example script and ``configs/flux2-dev-fp4-1gpu.yaml``
    work together as documented. Uses the local FLUX.2-dev checkpoint and the shared
    NVFP4 dynamic-quant config.
    """
    model_path = _lpips_model_path("FLUX.2-dev")
    _skip_if_missing(model_path, "FLUX.2-dev checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "flux2_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "flux2_output.png")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "flux2.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "flux2-dev-fp4-1gpu.yaml"
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


def test_ltx2_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/ltx2.py with NVFP4 config end-to-end.

    Validates that the LTX-2 example script and ``configs/ltx2-t2v-fp4-1gpu.yaml``
    work together as documented. The Gemma3 text encoder is passed separately via
    ``--text_encoder_path`` because the shared YAML intentionally omits it to keep
    the config model-path-agnostic.
    """
    model_path = _lpips_model_path("LTX-2", "ltx-2-19b-dev.safetensors")
    _skip_if_missing(model_path, "LTX-2 checkpoint")
    text_encoder_path = _ltx2_lpips_text_encoder_path()
    _skip_if_missing(text_encoder_path, "LTX-2 text encoder (gemma-3-12b-it)", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "ltx2_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "ltx2_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "ltx2.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "ltx2-t2v-fp4-1gpu.yaml"
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
            "--text_encoder_path",
            text_encoder_path,
            "--output_path",
            output_path,
        ],
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"


def test_wan_i2v_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/wan_i2v.py with NVFP4 config end-to-end.

    Validates that the Wan I2V example script and ``configs/wan2.2-i2v-fp4-1gpu.yaml``
    work together as documented. Uses the pre-quantized Wan 2.2 I2V A14B NVFP4
    checkpoint and the default input image (cat_piano.png) bundled with the examples.
    """
    scratch_space = conftest.llm_models_root()
    model_path = os.path.join(scratch_space, WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {WAN22_I2V_A14B_NVFP4_MODEL_SUBPATH} under models root)"
        )

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "wan_i2v_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "wan_i2v_output.mp4")

    script_path = os.path.join(llm_root, "examples", "visual_gen", "models", "wan_i2v.py")
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "wan2.2-i2v-fp4-1gpu.yaml"
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


def test_cosmos3_example(_visual_gen_deps, llm_root, llm_venv):
    """Run examples/visual_gen/models/cosmos3/cosmos3.py with FP8 config end-to-end.

    Validates that the Cosmos3-Nano example script and ``configs/cosmos3-nano-1gpu.yaml``
    work together as documented. Uses the local Cosmos3-Nano checkpoint and
    the shared FP8 dynamic-quant config.
    """
    model_path = _lpips_model_path("Cosmos3-Nano")
    _skip_if_missing(model_path, "Cosmos3-Nano checkpoint", is_dir=True)

    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", "cosmos3_example")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "cosmos3_output.mp4")

    script_path = os.path.join(
        llm_root, "examples", "visual_gen", "models", "cosmos3", "cosmos3.py"
    )
    config_path = os.path.join(
        llm_root, "examples", "visual_gen", "configs", "cosmos3-nano-1gpu.yaml"
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
            "--prompt",
            "A serene mountain landscape with snow-capped peaks and a flowing river",
            "--output_path",
            output_path,
        ],
        env={"TRTLLM_DISABLE_COSMOS3_GUARDRAILS": "1"},
    )
    assert os.path.isfile(output_path), f"Example did not produce output at {output_path}"
