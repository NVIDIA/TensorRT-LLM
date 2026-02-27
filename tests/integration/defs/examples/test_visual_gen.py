# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from defs.common import venv_check_call
from defs.conftest import llm_models_root
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


# TODO: populate after first successful baseline run for each checkpoint.
VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES = None
VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES = None

VBENCH_REPO = "https://github.com/Vchitect/VBench.git"
# Pin to a fixed commit for reproducible shallow-fetch
VBENCH_COMMIT = "98b19513678e99c80d8377fda25ba53b81a491a6"

DINO_REPO = "https://github.com/facebookresearch/dino.git"
DINO_HUB_DIR_NAME = "facebookresearch_dino_main"


@pytest.fixture(scope="session")
def vbench_repo_root(llm_venv):
    """Clone VBench repo into workspace and install; return repo root path."""
    workspace = llm_venv.get_working_directory()
    repo_path = os.path.join(workspace, "VBench_repo")
    _precache_dino_for_torch_hub()
    _ensure_vbench_runtime_deps(llm_venv)
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


def _ensure_vbench_runtime_deps(llm_venv):
    """Ensure VBench runtime deps are installed (idempotent, runs every session)."""
    llm_venv.run_cmd(["-m", "pip", "install", "-q", "imageio"])


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


@pytest.fixture(scope="session")
def wan_trtllm_video_path(llm_venv, llm_root):
    """Generate input video via visual_gen_wan_t2v.py and return path to trtllm_output.mp4."""
    scratch_space = llm_models_root()
    model_path = os.path.join(scratch_space, WAN_T2V_MODEL_SUBPATH)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Wan T2V model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {WAN_T2V_MODEL_SUBPATH} under scratch)"
        )
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)
    if os.path.isfile(output_path):
        return output_path
    # Install av and diffusers from main branch
    llm_venv.run_cmd(["-m", "pip", "install", "av"])
    llm_venv.run_cmd(
        [
            "-m",
            "pip",
            "install",
            "git+https://github.com/huggingface/diffusers.git",
        ]
    )
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
def wan_reference_video_path(llm_venv, llm_root):
    """Generate reference video via diffusers (hf_wan.py) using the same model checkpoint."""
    scratch_space = llm_models_root()
    model_path = os.path.join(scratch_space, WAN_T2V_MODEL_SUBPATH)
    if not os.path.isdir(model_path):
        pytest.skip(
            f"Wan T2V model not found: {model_path} "
            f"(set LLM_MODELS_ROOT or place {WAN_T2V_MODEL_SUBPATH} under scratch)"
        )
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output")
    os.makedirs(out_dir, exist_ok=True)
    reference_path = os.path.join(out_dir, DIFFUSERS_REFERENCE_VIDEO)
    if os.path.isfile(reference_path):
        return reference_path
    hf_script = os.path.join(llm_root, "examples", "visual_gen", "hf_wan.py")
    assert os.path.isfile(hf_script), f"Diffusers script not found: {hf_script}"
    venv_check_call(
        llm_venv,
        [
            hf_script,
            "--model_path",
            model_path,
            "--prompt",
            WAN_T2V_PROMPT,
            "--output_path",
            reference_path,
            "--height",
            str(WAN_T2V_HEIGHT),
            "--width",
            str(WAN_T2V_WIDTH),
            "--num_frames",
            str(WAN_T2V_NUM_FRAMES),
        ],
    )
    assert os.path.isfile(reference_path), f"Diffusers did not produce {reference_path}"
    return reference_path


def _generate_wan_video(llm_venv, llm_root, model_subpath, output_subdir):
    """Generate a video with visual_gen_wan_t2v.py for a given model checkpoint.

    Returns the path to the generated .mp4, or calls pytest.skip if the model
    is not found under LLM_MODELS_ROOT.
    """
    scratch_space = llm_models_root()
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
    llm_venv.run_cmd(["-m", "pip", "install", "av"])
    llm_venv.run_cmd(["-m", "pip", "install", "git+https://github.com/huggingface/diffusers.git"])
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
def wan22_a14b_fp8_video_path(llm_venv, llm_root):
    """Generate video with Wan 2.2 A14B FP8 checkpoint."""
    return _generate_wan_video(llm_venv, llm_root, WAN22_A14B_FP8_MODEL_SUBPATH, "wan22_fp8")


@pytest.fixture(scope="session")
def wan22_a14b_nvfp4_video_path(llm_venv, llm_root):
    """Generate video with Wan 2.2 A14B NVFP4 checkpoint."""
    return _generate_wan_video(llm_venv, llm_root, WAN22_A14B_NVFP4_MODEL_SUBPATH, "wan22_nvfp4")


def _visual_gen_out_dir(llm_venv, subdir=""):
    """Output directory for generated media; subdir e.g. 'ltx2' for model-specific outputs."""
    base = os.path.join(llm_venv.get_working_directory(), "visual_gen_output")
    return os.path.join(base, subdir) if subdir else base


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


def _run_vbench_and_compare_to_golden(
    vbench_repo_root,
    videos_dir,
    trtllm_filename,
    golden_scores,
    llm_venv,
    title,
    max_score_diff=0.1,
):
    """Run VBench on videos_dir (TRT-LLM output only), compare to golden HF reference scores."""
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
    scores_ref = golden_scores
    max_len = max(len(d) for d in VBENCH_DIMENSIONS)
    header = f"{'Dimension':<{max_len}}  |  {'TRT-LLM':>10}  |  {'HF Ref':>10}  |  {'Diff':>8}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print(f"VBench dimension scores ({title}): TRT-LLM vs golden HF reference scores")
    print("=" * len(header))
    print(header)
    print(sep)
    max_diff_val = 0.0
    for dim in VBENCH_DIMENSIONS:
        t, r = scores_trtllm[dim], scores_ref[dim]
        diff = abs(t - r)
        max_diff_val = max(max_diff_val, diff)
        print(f"{dim:<{max_len}}  |  {t:>10.4f}  |  {r:>10.4f}  |  {diff:>8.4f}")
    print(sep)
    print(
        f"{' (all dimensions)':<{max_len}}  |  (TRT-LLM)   |  (golden)   |  max_diff={max_diff_val:.4f}"
    )
    print("=" * len(header) + "\n")
    for dim in VBENCH_DIMENSIONS:
        diff = abs(scores_trtllm[dim] - scores_ref[dim])
        assert diff < max_score_diff or scores_trtllm[dim] >= scores_ref[dim], (
            f"Dimension '{dim}' score difference {diff:.4f} >= {max_score_diff} "
            f"(TRT-LLM={scores_trtllm[dim]:.4f}, golden={scores_ref[dim]:.4f})"
        )


def test_vbench_dimension_score_wan(vbench_repo_root, wan_trtllm_video_path, llm_venv):
    """Run VBench on WAN TRT-LLM video; compare to golden HF reference scores (diff < 0.05 or TRT-LLM >= golden)."""
    videos_dir = os.path.dirname(wan_trtllm_video_path)
    assert os.path.isfile(wan_trtllm_video_path), "TRT-LLM video must exist"
    _run_vbench_and_compare_to_golden(
        vbench_repo_root,
        videos_dir,
        VISUAL_GEN_OUTPUT_VIDEO,
        VBENCH_WAN_GOLDEN_SCORES,
        llm_venv,
        title="WAN",
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
        max_score_diff=0.10,
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
        max_score_diff=0.10,
    )


# ---------------------------------------------------------------------------
# Score-only tests: skip video generation, evaluate pre-existing .mp4 files.
#
#   WAN22_FP8_VIDEO_PATH=/path/to/trtllm_fp8.mp4 \
#   WAN22_NVFP4_VIDEO_PATH=/path/to/trtllm_nvfp4.mp4 \
#   pytest ... -k "score_only"
# ---------------------------------------------------------------------------


def test_vbench_score_only_wan22_fp8(vbench_repo_root, llm_venv):
    """Evaluate a pre-existing FP8 video (set WAN22_FP8_VIDEO_PATH)."""
    video_path = os.environ.get("WAN22_FP8_VIDEO_PATH")
    if not video_path:
        pytest.skip("WAN22_FP8_VIDEO_PATH not set")
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    _run_vbench_and_report(
        vbench_repo_root,
        os.path.dirname(video_path),
        os.path.basename(video_path),
        llm_venv,
        title="WAN 2.2 A14B FP8 (score-only)",
        golden_scores=VBENCH_WAN22_A14B_FP8_GOLDEN_SCORES,
    )


def test_vbench_score_only_wan22_nvfp4(vbench_repo_root, llm_venv):
    """Evaluate a pre-existing NVFP4 video (set WAN22_NVFP4_VIDEO_PATH)."""
    video_path = os.environ.get("WAN22_NVFP4_VIDEO_PATH")
    if not video_path:
        pytest.skip("WAN22_NVFP4_VIDEO_PATH not set")
    assert os.path.isfile(video_path), f"Video not found: {video_path}"
    _run_vbench_and_report(
        vbench_repo_root,
        os.path.dirname(video_path),
        os.path.basename(video_path),
        llm_venv,
        title="WAN 2.2 A14B NVFP4 (score-only)",
        golden_scores=VBENCH_WAN22_A14B_NVFP4_GOLDEN_SCORES,
    )
