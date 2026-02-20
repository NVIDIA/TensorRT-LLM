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
from defs.common import venv_check_call
from defs.conftest import llm_models_root
from defs.trt_test_alternative import check_call

WAN_T2V_MODEL_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
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

# Golden VBench scores from HF reference video (WAN); TRT-LLM is compared against these.
VBENCH_WAN_GOLDEN_SCORES = {
    "subject_consistency": 0.9381,
    "background_consistency": 0.9535,
    "motion_smoothness": 0.9923,
    "dynamic_degree": 1.0000,
    "aesthetic_quality": 0.5033,
    "imaging_quality": 0.3033,
}

VBENCH_REPO = "https://github.com/Vchitect/VBench.git"
VBENCH_BRANCH = "master"
# Pin to a fixed commit for reproducible runs
VBENCH_COMMIT = "98b19513678e99c80d8377fda25ba53b81a491a6"


@pytest.fixture(scope="session")
def vbench_repo_root(llm_venv):
    """Clone VBench repo into workspace and install; return repo root path."""
    workspace = llm_venv.get_working_directory()
    repo_path = os.path.join(workspace, "VBench_repo")
    if os.path.exists(repo_path):
        return repo_path
    # Clone without --depth=1 so we can checkout a specific commit
    check_call(
        ["git", "clone", "--single-branch", "--branch", VBENCH_BRANCH, VBENCH_REPO, repo_path],
        shell=False,
    )
    check_call(["git", "-C", repo_path, "checkout", VBENCH_COMMIT], shell=False)
    # # Install VBench dependencies explicitly
    # llm_venv.run_cmd([
    #     "-m", "pip", "install",
    #     "tqdm>=4.60.0",
    #     "openai-clip>=1.0",
    #     "pyiqa>=0.1.0", # install this might also install transformers=4.37.2, which is incompatible
    #     "easydict",
    #     "decord>=0.6.0",
    # ])
    return repo_path


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
    venv_check_call(
        llm_venv,
        [
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
        ],
    )
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
