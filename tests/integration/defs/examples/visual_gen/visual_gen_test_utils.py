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
"""Shared helpers for VisualGen accuracy tests."""

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
from dataclasses import dataclass
from typing import Collection, Literal

import pytest
import torch
import torch._inductor.config as inductor_config
from defs.trt_test_alternative import check_call
from torch._inductor.async_compile import shutdown_compile_workers

# =============================================================================
# Shared paths and feature configuration
# =============================================================================

# This file lives at tests/integration/defs/examples/visual_gen/, so the repo
# root is five levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
VISUAL_GEN_LPIPS_EVAL_SCRIPT = os.path.join(
    REPO_ROOT, "scripts", "visualgen_eval", "visual_gen_lpips_score_eval.py"
)
VISUAL_GEN_LPIPS_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden", "visual_gen_lpips")
VISUAL_GEN_LPIPS_GOLDEN_MEDIA_ZIP = os.path.join(
    VISUAL_GEN_LPIPS_GOLDEN_DIR, "visual_gen_lpips_golden_media.zip"
)
VISUAL_GEN_OUTPUT_VIDEO = "trtllm_output.mp4"

QuantizationMode = Literal["none", "FP8_BLOCK_SCALES", "NVFP4"]
CacheBackend = Literal["off", "teacache", "cache_dit"]
SparseAttentionMode = Literal["off", "vsa"]


@dataclass(frozen=True)
class FeatureConfigState:
    """Normalized single-device feature state requested by an accuracy case.

    ``quantization="none"`` means the unquantized BF16 VisualGen path. Model
    test modules remain responsible for validating which combinations they
    support and for mapping this state to ``VisualGenArgs``. In particular,
    ``sparse_attention="vsa"`` is a high-level feature request; a supporting
    model adapter owns its required attention backend and sparsity recipe.
    """

    quantization: QuantizationMode = "none"
    cache_backend: CacheBackend = "off"
    sparse_attention: SparseAttentionMode = "off"
    cuda_graph: bool = False
    torch_compile: bool = False


def _enabled_feature_ids(features):
    """Return the normalized IDs for features enabled in ``features``."""
    enabled = []
    if features.quantization != "none":
        enabled.append(
            {
                "FP8_BLOCK_SCALES": "fp8-blockwise",
                "NVFP4": "nvfp4",
            }[features.quantization]
        )
    if features.cache_backend != "off":
        enabled.append(
            {
                "teacache": "teacache",
                "cache_dit": "cache-dit",
            }[features.cache_backend]
        )
    if features.sparse_attention != "off":
        enabled.append(features.sparse_attention)
    if features.cuda_graph:
        enabled.append("cuda-graph")
    if features.torch_compile:
        enabled.append("torch-compile")
    return tuple(enabled)


def _validate_single_feature_config(features, supported_features: Collection[str], model_name):
    """Validate a baseline or exactly-one-feature accuracy configuration."""
    enabled = _enabled_feature_ids(features)
    if len(enabled) > 1:
        raise ValueError(
            f"{model_name} accuracy cases may enable only one feature at a time: {enabled}"
        )
    unsupported = set(enabled).difference(supported_features)
    if unsupported:
        raise ValueError(f"{model_name} does not support feature(s): {sorted(unsupported)}")
    return len(enabled)


def _build_single_device_feature_args(
    model_path,
    features,
    *,
    resolution,
    num_frames,
    pipeline_config=None,
    quantization_kwargs=None,
    teacache_kwargs=None,
    cache_dit_kwargs=None,
    vsa_sparsity=0.9,
):
    """Map normalized feature state to a one-GPU ``VisualGenArgs`` object."""
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        CacheDiTConfig,
        CompilationConfig,
        CudaGraphConfig,
        ParallelConfig,
        TeaCacheConfig,
        TorchCompileConfig,
        VideoSparseAttentionConfig,
        VisualGenArgs,
    )

    if features.quantization == "none":
        quant_config = QuantConfig()
    else:
        quant_config = {
            "quant_algo": features.quantization,
            "dynamic": True,
            **(quantization_kwargs or {}),
        }
    if features.cache_backend == "teacache":
        cache_config = TeaCacheConfig(**(teacache_kwargs or {}))
    elif features.cache_backend == "cache_dit":
        cache_config = CacheDiTConfig(**(cache_dit_kwargs or {}))
    else:
        cache_config = None
    sparse_attention_config = (
        VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity)
        if features.sparse_attention == "vsa"
        else None
    )
    kwargs = dict(
        model=model_path,
        quant_config=quant_config,
        compilation_config=CompilationConfig(
            resolutions=[resolution],
            num_frames=[num_frames],
        ),
        attention_config=AttentionConfig(
            backend="CUTEDSL" if sparse_attention_config is not None else "VANILLA",
            sparse_attention_config=sparse_attention_config,
        ),
        parallel_config=ParallelConfig(
            parallel_vae_size=1,
            cfg_size=1,
            ulysses_size=1,
            async_ulysses=False,
            ring_size=1,
            attn2d_size=(1, 1),
            tp_size=1,
        ),
        cache_config=cache_config,
        cuda_graph_config=CudaGraphConfig(enable=features.cuda_graph),
        torch_compile_config=TorchCompileConfig(
            enable=features.torch_compile,
            enable_fullgraph=False,
            enable_autotune=False,
        ),
    )
    if pipeline_config is not None:
        kwargs["pipeline_config"] = pipeline_config
    return VisualGenArgs(**kwargs)


def _assert_resolved_single_device_feature_config(
    pipeline,
    features,
    *,
    resolution,
    num_frames,
    vsa_sparsity=0.9,
):
    """Assert that every normalized feature reached the resolved pipeline config."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    config = pipeline.pipeline_config
    expected_quant_algo = {
        "none": None,
        "FP8_BLOCK_SCALES": QuantAlgo.FP8_BLOCK_SCALES,
        "NVFP4": QuantAlgo.NVFP4,
    }[features.quantization]
    assert config.quant_config.quant_algo == expected_quant_algo
    assert config.dynamic_weight_quant is (expected_quant_algo is not None)
    if features.quantization == "NVFP4":
        assert config.force_dynamic_quantization is True

    expected_cache_backend = None if features.cache_backend == "off" else features.cache_backend
    assert config.cache_backend == expected_cache_backend
    if features.cache_backend == "off":
        assert pipeline.cache_accelerator is None
    else:
        expected_accelerator = {
            "teacache": "TeaCacheAccelerator",
            "cache_dit": "CacheDiTAccelerator",
        }[features.cache_backend]
        assert pipeline.cache_accelerator is not None
        assert type(pipeline.cache_accelerator).__name__ == expected_accelerator
        assert pipeline.cache_accelerator.is_enabled()

    expected_attention_backend = "CUTEDSL" if features.sparse_attention == "vsa" else "VANILLA"
    assert config.attention.backend == expected_attention_backend
    sparse_config = config.attention.sparse_attention_config
    if features.sparse_attention == "vsa":
        assert sparse_config is not None
        assert sparse_config.algorithm == "vsa"
        assert sparse_config.vsa_sparsity == pytest.approx(vsa_sparsity)
    else:
        assert sparse_config is None

    assert config.cuda_graph.enable is features.cuda_graph
    assert config.torch_compile.enable is features.torch_compile
    assert config.parallel.total_parallel_size == 1
    assert resolution in config.compilation.resolutions
    assert num_frames in config.compilation.num_frames


def _transformer_components(pipeline):
    components = []
    for component_name in pipeline.transformer_components:
        component = getattr(pipeline, component_name, None)
        if component is not None:
            components.append(component)
    assert components, "No VisualGen transformer components were found"
    return components


def _assert_feature_quantization_installed(pipeline, features):
    """Verify that requested dynamic quantization changed real Linear modules."""
    from tensorrt_llm._torch.modules.linear import Linear

    linears = [
        module
        for component in _transformer_components(pipeline)
        for module in component.modules()
        if isinstance(module, Linear) and getattr(module, "_weights_created", False)
    ]
    assert linears, "No initialized VisualGen Linear modules were found"
    if features.quantization == "none":
        assert not [module for module in linears if module.has_any_quant]
        return

    predicate = {
        "FP8_BLOCK_SCALES": lambda module: module.has_fp8_block_scales,
        "NVFP4": lambda module: module.has_nvfp4,
    }[features.quantization]
    quantized_linears = [module for module in linears if predicate(module)]
    assert quantized_linears, f"No {features.quantization} VisualGen Linear modules were created"
    sample = quantized_linears[0]
    assert getattr(sample, "weight", None) is not None
    assert getattr(sample, "weight_scale", None) is not None
    if features.quantization == "FP8_BLOCK_SCALES":
        assert sample.weight.dtype == torch.float8_e4m3fn
        assert sample.weight_scale.ndim == 2
    else:
        assert sample.weight.shape[-1] * 2 == sample.in_features
        assert sample.scaling_vector_size == 16
        assert getattr(sample, "weight_scale_2", None) is not None


def _assert_feature_torch_compile_installed(pipeline, features):
    """Verify block-wise or whole-component ``torch.compile`` installation."""
    compile_targets = []
    for component in _transformer_components(pipeline):
        block_names = pipeline._find_transformer_blocks(component)
        if block_names:
            for block_name in block_names:
                compile_targets.extend(getattr(component, block_name))
        else:
            compile_targets.append(component)

    compiled_targets = [target for target in compile_targets if hasattr(target, "_orig_mod")]
    if features.torch_compile:
        assert len(compiled_targets) == len(compile_targets), (
            f"torch.compile compiled only {len(compiled_targets)}/{len(compile_targets)} "
            "VisualGen targets"
        )
    else:
        assert not compiled_targets, (
            f"{len(compiled_targets)}/{len(compile_targets)} VisualGen targets were compiled "
            "with torch_compile=False"
        )


def _cached_step_count(value):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        direct = [
            int(value[key])
            for key in ("cached", "cached_steps", "accumulated_cached_steps")
            if key in value and isinstance(value[key], (int, float))
        ]
        if direct:
            # Backends may expose the same counter under compatibility aliases.
            return max(direct)
        return sum(_cached_step_count(entry) for entry in value.values())
    if isinstance(value, (list, tuple)):
        # Cache-DiT can report duplicate views of the same cache context.
        return max((_cached_step_count(entry) for entry in value), default=0)
    direct = [
        _cached_step_count(getattr(value, attr))
        for attr in ("cached", "cached_steps", "accumulated_cached_steps")
        if hasattr(value, attr)
    ]
    return max(direct, default=0)


def _assert_single_device_feature_executed(pipeline, features):
    """Assert that runtime-only features did real work during generation."""
    if features.cache_backend != "off":
        stats = pipeline.cache_accelerator.get_stats()
        assert _cached_step_count(stats) > 0, (
            f"{features.cache_backend} executed without a cached step: {stats}"
        )

    runners = pipeline._cuda_graph_runners
    if features.cuda_graph:
        assert runners, "CUDA graph was requested but no runner was installed"
        for name, runner in runners.items():
            assert runner.enabled, f"CUDA graph runner {name} is disabled"
            assert runner.graphs, f"CUDA graph runner {name} captured no graphs"
    else:
        assert not runners, f"CUDA graph runners exist with cuda_graph=False: {list(runners)}"


# =============================================================================
# LPIPS, golden-media, and runtime helpers
# =============================================================================


class ReusableLPIPSScorer:
    """Lazily load one LPIPS model and reuse it across media test cases."""

    def __init__(self, net="alex", device=None):
        self.net = net
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = None

    def _get_model(self):
        # Keep model construction aligned with _make_lpips_model() in
        # visual_gen_lpips_score_eval.py.
        if self._model is None:
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"LPIPS device is {self.device}, but CUDA is not available.")
            import lpips

            self._model = lpips.LPIPS(net=self.net, verbose=False).to(self.device).eval()
        return self._model

    def _to_lpips_tensor(self, image):
        # Keep this preprocessing aligned with _to_lpips_tensor() and
        # _score_images() in visual_gen_lpips_score_eval.py.
        import numpy as np

        tensor = torch.from_numpy(np.array(image.convert("RGB")))
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def score(self, reference_path, generated_path):
        from PIL import Image

        with Image.open(reference_path) as reference_image:
            reference_tensor = self._to_lpips_tensor(reference_image.convert("RGB"))
        with Image.open(generated_path) as generated_image:
            generated_tensor = self._to_lpips_tensor(generated_image.convert("RGB"))

        if generated_tensor.shape != reference_tensor.shape:
            raise ValueError(
                "Generated image and reference image must have the same LPIPS tensor shape: "
                f"{tuple(generated_tensor.shape)} vs {tuple(reference_tensor.shape)}."
            )
        with torch.no_grad():
            return float(
                self._get_model()(generated_tensor, reference_tensor).reshape(-1).mean().item()
            )

    def _decode_video_to_lpips_batch(self, video_path):
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for LPIPS comparison: {video_path}")
        frames = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"Decoded video contains no frames: {video_path}")
        batch = torch.from_numpy(np.stack(frames, axis=0).copy())
        batch = batch.permute(0, 3, 1, 2).to(device=self.device, dtype=torch.float32)
        return batch / 127.5 - 1.0

    def score_video(self, reference_path, generated_path):
        reference = self._decode_video_to_lpips_batch(reference_path)
        generated = self._decode_video_to_lpips_batch(generated_path)
        if generated.shape[1:] != reference.shape[1:]:
            raise ValueError(
                "Generated and reference video frames must have the same LPIPS tensor shape: "
                f"{tuple(generated.shape[1:])} vs {tuple(reference.shape[1:])}."
            )
        paired_frame_count = min(generated.shape[0], reference.shape[0])
        with torch.no_grad():
            scores = self._get_model()(
                generated[:paired_frame_count],
                reference[:paired_frame_count],
            ).flatten()
        return float(scores.mean().item())

    def close(self):
        model = self._model
        self._model = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _llm_models_root():
    # Keep imports that load TRT-LLM bindings out of module initialization.
    # torch.multiprocessing.spawn workers import this utility before the
    # installed-wheel path is placed ahead of the source checkout.
    from defs import conftest

    return conftest.llm_models_root()


def _venv_check_call(*args, **kwargs):
    # Deferred for the same spawn-worker reason as _llm_models_root().
    from defs.common import venv_check_call

    return venv_check_call(*args, **kwargs)


def _lpips_model_path(*parts):
    return os.path.join(_llm_models_root(), *parts)


def _skip_if_missing(path, label, is_dir=False):
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    if not exists:
        pytest.skip(f"{label} not found: {path}")


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


def _disable_inductor_compile_worker_quiesce():
    # The quiesce timer thread can outlive shutdown_compile_workers() long
    # enough for pytest-threadleak to report it as a leaked thread.
    if hasattr(inductor_config, "quiesce_async_compile_pool"):
        inductor_config.quiesce_async_compile_pool = False


def _cleanup_single_device_feature_pipeline(pipeline):
    """Release feature wrappers before dropping a single-device pipeline."""
    if pipeline is None:
        return
    try:
        cache_accelerator = getattr(pipeline, "cache_accelerator", None)
        if cache_accelerator is not None:
            cache_accelerator.unwrap()
    finally:
        pipeline.cleanup()


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # torch.compile and unconditional @torch.compile helpers can lazily spawn
    # an Inductor worker pool whose daemon threads otherwise outlive the test.
    shutdown_compile_workers()


@contextlib.contextmanager
def _lpips_deterministic_algorithms(*, fully_eager=False):
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        compiler_context = (
            torch.compiler.set_stance("force_eager") if fully_eager else contextlib.nullcontext()
        )
        with compiler_context:
            yield
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
        if previous_cublas_workspace_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas_workspace_config


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


def _run_reusable_image_lpips_eval(sample_id, reference_path, generated_path, scorer):
    try:
        with _lpips_deterministic_algorithms():
            score = scorer.score(reference_path, generated_path)
    except Exception as error:
        pytest.fail(f"LPIPS image eval failed for {sample_id}: {error}")
    print(f"\n[E2E {sample_id} LPIPS] score: {score:.6f}")
    return score


def _run_reusable_video_lpips_eval(sample_id, reference_path, generated_path, scorer):
    try:
        with _lpips_deterministic_algorithms():
            score = scorer.score_video(reference_path, generated_path)
    except Exception as error:
        pytest.fail(f"LPIPS video eval failed for {sample_id}: {error}")
    print(f"\n[E2E {sample_id} LPIPS] score: {score:.6f}")
    return score


def _assert_lpips_below_threshold(score, threshold):
    assert score < threshold, f"LPIPS too high: {score:.6f} (expected < {threshold:.6f})"


def _preserve_lpips_candidate_on_failure(request, score, threshold, candidate_path, artifact_name):
    """Copy a failed LPIPS candidate into pytest's archived output directory."""
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

    size = os.path.getsize(dest)
    if size <= 8 * 1024 * 1024:
        with open(dest, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        print(f"[LPIPS-B64-BEGIN {artifact_name} {size}]")
        for i in range(0, len(encoded), 3072):
            print(encoded[i : i + 3072])
        print(f"[LPIPS-B64-END {artifact_name}]")


def _visual_gen_output_path(llm_venv, output_subdir):
    out_dir = os.path.join(llm_venv.get_working_directory(), "visual_gen_output", output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, VISUAL_GEN_OUTPUT_VIDEO)


def _save_lpips_video_mp4(video, output_path, frame_rate):
    from tensorrt_llm.media.encoding import save_video

    try:
        save_video(video, output_path, frame_rate=frame_rate)
    except RuntimeError as err:
        if "MP4 format requires ffmpeg" not in str(err):
            raise
        # Never fall back to another codec here: the goldens are H.264/x264 and
        # LPIPS compares decoded pixels, so a silent cv2/mp4v fallback compares
        # codec artifacts, not model output.
        pytest.fail(
            "ffmpeg is unavailable for LPIPS video encoding; refusing to fall back "
            "to another codec because the golden comparison would measure codec "
            f"artifacts instead of model output: {err}"
        )
    assert os.path.isfile(output_path), f"Visual gen did not produce {output_path}"


# =============================================================================
# VBench setup and score helpers
# =============================================================================

VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

VBENCH_REPO = "https://github.com/Vchitect/VBench.git"
# Pin to a fixed commit for reproducible shallow-fetch.
VBENCH_COMMIT = "98b19513678e99c80d8377fda25ba53b81a491a6"

DINO_REPO = "https://github.com/facebookresearch/dino.git"
DINO_HUB_DIR_NAME = "facebookresearch_dino_main"

AESTHETIC_PREDICTOR_URL = (
    "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"
)
AESTHETIC_PREDICTOR_FILENAME = "sa_0_4_vit_l_14_linear.pth"
AESTHETIC_PREDICTOR_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "emb_reader")


def _prepare_vbench_repo(llm_venv):
    """Prepare the pinned VBench checkout and dependencies for this venv."""
    workspace = llm_venv.get_working_directory()
    repo_path = os.path.join(workspace, "VBench_repo")
    _precache_dino_for_torch_hub()
    _precache_aesthetic_predictor()
    if not os.path.exists(repo_path):
        # Shallow-fetch only the pinned commit to avoid downloading full history (~350 MB).
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

    # The checkout persists across sessions, while llm_venv may be new.
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
    llm_venv.run_cmd(["-m", "pip", "install", "--no-deps", "pyiqa>=0.1.0"])
    return repo_path


def _precache_dino_for_torch_hub():
    """Pre-clone DINO into the torch.hub cache to avoid GitHub API rate limits."""
    hub_dir = torch.hub.get_dir()
    os.makedirs(hub_dir, exist_ok=True)

    dino_cache = os.path.join(hub_dir, DINO_HUB_DIR_NAME)
    if not os.path.isdir(dino_cache):
        check_call(
            ["git", "clone", "--depth", "1", "-b", "main", DINO_REPO, dino_cache],
            shell=False,
        )


def _precache_aesthetic_predictor():
    """Pre-download LAION aesthetic predictor weights with retry handling."""
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
                    "[precache] Aesthetic predictor download attempt "
                    f"{attempt + 1}/{max_retries} failed ({exc}), retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download aesthetic predictor after {max_retries} attempts: {exc}"
                ) from exc


def _normalize_score(val):
    """Normalize a VBench score to a zero-to-one scale."""
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)) and val > 1.5:
        return val / 100.0
    return float(val)


def _get_per_video_scores(results, video_path_substr):
    """Get per-dimension scores for a matching video from VBench results."""
    scores = {}
    for dim in VBENCH_DIMENSIONS:
        dim_result = results[dim]
        assert isinstance(dim_result, list) and len(dim_result) >= 2, (
            f"Dimension '{dim}' result must be [overall_score, video_results]; "
            f"got {type(dim_result)}"
        )
        video_results = dim_result[1]
        for entry in video_results:
            if video_path_substr in entry.get("video_path", ""):
                scores[dim] = _normalize_score(entry.get("video_results"))
                break
        else:
            raise AssertionError(
                f"No video matching '{video_path_substr}' in dimension '{dim}'; "
                f"paths: {[e.get('video_path') for e in video_results]}"
            )
    return scores


def _run_vbench_and_report(
    vbench_repo_root,
    videos_dir,
    trtllm_filename,
    llm_venv,
    title,
    golden_scores=None,
    max_score_diff=0.10,
):
    """Run VBench, report scores, and optionally compare against golden values."""
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
    _venv_check_call(llm_venv, cmd)

    pattern = os.path.join(output_path, "*_eval_results.json")
    result_files = glob.glob(pattern)
    assert result_files, (
        f"No eval results found matching {pattern}; output dir: {os.listdir(output_path)}"
    )
    result_file = max(result_files, key=os.path.getmtime)
    with open(result_file, "r") as f:
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
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print(f"VBench dimension scores ({title})")
    print("=" * len(header))
    print(header)
    print(separator)

    for dim in VBENCH_DIMENSIONS:
        trtllm_score = scores_trtllm[dim]
        if golden_scores is not None:
            golden_score = golden_scores[dim]
            print(
                f"{dim:<{max_len}}  |  {trtllm_score:>10.4f}  |  {golden_score:>10.4f}  |  "
                f"{abs(trtllm_score - golden_score):>8.4f}"
            )
        else:
            print(f"{dim:<{max_len}}  |  {trtllm_score:>10.4f}")
    print(separator)

    if golden_scores is None:
        print("\n** Baseline run — no golden scores to compare against. **")
        print("** Copy the values above into the GOLDEN_SCORES dict.  **\n")
        return scores_trtllm

    max_diff_val = max(abs(scores_trtllm[dim] - golden_scores[dim]) for dim in VBENCH_DIMENSIONS)
    print(f"max_diff={max_diff_val:.4f}  (threshold={max_score_diff})")
    print("=" * len(header) + "\n")
    for dim in VBENCH_DIMENSIONS:
        diff = abs(scores_trtllm[dim] - golden_scores[dim])
        assert diff < max_score_diff or scores_trtllm[dim] >= golden_scores[dim], (
            f"Dimension '{dim}' score difference {diff:.4f} >= {max_score_diff} "
            f"(TRT-LLM={scores_trtllm[dim]:.4f}, golden={golden_scores[dim]:.4f})"
        )
    return scores_trtllm


# =============================================================================
# WAN LPIPS configuration and generation helpers
# =============================================================================

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
    fully_eager=False,
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
    with _lpips_deterministic_algorithms(fully_eager=fully_eager):
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
    attention_backend="VANILLA",
    parallel=None,
    fully_eager=False,
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
        attention_backend=attention_backend,
        parallel=parallel,
        fully_eager=fully_eager,
    )
    assert generated_video is not None, "Single-GPU Wan LPIPS run produced no video"
    _save_lpips_video_mp4(generated_video, output_path, frame_rate=frame_rate)
