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
"""Feature-matrix e2e tests for VisualGen model support."""

import os

import pytest
import torch
import torch.distributed as dist
from defs import conftest
from defs.examples.test_visual_gen import (
    _assert_lpips_below_threshold,
    _cleanup_cuda,
    _run_lpips_eval,
    _skip_if_missing,
)
from defs.examples.test_visual_gen_multi_gpu import (
    _skip_if_insufficient_gpus_for_parallel,
    run_test_in_distributed,
)

try:
    from tensorrt_llm.visual_gen.args import ParallelConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

QWEN_IMAGE_MODEL_SUBPATH = "Qwen-Image"
QWEN_IMAGE_PROMPT = "A small robot painting a flower"
QWEN_IMAGE_HEIGHT = 256
QWEN_IMAGE_WIDTH = 256
QWEN_IMAGE_NUM_INFERENCE_STEPS = 2
QWEN_IMAGE_TRUE_CFG_SCALE = 1.0
QWEN_IMAGE_SEED = 42
QWEN_IMAGE_LPIPS_THRESHOLD = 0.1
QWEN_IMAGE_ULYSSES_PARALLEL = {"ulysses_size": 2}

QWEN_IMAGE_PRECISION_VARIANTS = [
    ("bf16", None),
    ("dynamic_fp8", {"quant_algo": "FP8", "dynamic": True}),
    ("dynamic_fp4", {"quant_algo": "NVFP4", "dynamic": True}),
]
QWEN_IMAGE_ATTENTION_BACKENDS = ["FA4", "CUTEDSL"]


def _qwen_image_model_path():
    return os.environ.get(
        "QWEN_IMAGE_CKPT",
        os.path.join(conftest.llm_models_root(), QWEN_IMAGE_MODEL_SUBPATH),
    )


def _skip_if_qwen_attention_backend_unavailable(attention_backend):
    if attention_backend == "FA4":
        try:
            from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
                _flash_attn_fwd_import_error,
            )
        except ImportError as exc:
            pytest.skip(f"FA4 attention backend unavailable: {exc}")
        if _flash_attn_fwd_import_error is not None:
            pytest.skip(f"FA4 attention backend unavailable: {_flash_attn_fwd_import_error}")
    elif attention_backend == "CUTEDSL":
        try:
            from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import (
                _cute_dsl_import_error,
            )
        except ImportError as exc:
            pytest.skip(f"CUTEDSL attention backend unavailable: {exc}")
        if _cute_dsl_import_error is not None:
            pytest.skip(f"CUTEDSL attention backend unavailable: {_cute_dsl_import_error}")


def _make_qwen_image_args(model_path, attention_backend, quant_config, parallel=None):
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        CompilationConfig,
        ParallelConfig,
        TorchCompileConfig,
        VisualGenArgs,
    )

    kwargs = {
        "model": model_path,
        "attention_config": AttentionConfig(backend=attention_backend),
        "compilation_config": CompilationConfig(skip_warmup=True),
        "torch_compile_config": TorchCompileConfig(enable=False, enable_autotune=False),
    }
    if quant_config is not None:
        kwargs["quant_config"] = dict(quant_config)
    if parallel is not None:
        kwargs["parallel_config"] = ParallelConfig(**parallel)
    return VisualGenArgs(**kwargs)


def _generate_qwen_image(
    model_path,
    output_path,
    attention_backend,
    quant_config,
    parallel=None,
    expect_ulysses=False,
):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.media.encoding import save_image

    _skip_if_missing(model_path, "Qwen-Image checkpoint", is_dir=True)
    args = _make_qwen_image_args(model_path, attention_backend, quant_config, parallel=parallel)
    precision = "bf16" if quant_config is None else quant_config["quant_algo"]
    print(
        "\n[Qwen-Image] "
        f"precision={precision}, requested_backend={attention_backend}, parallel={parallel}"
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    try:
        with torch.no_grad():
            result = pipeline.forward(
                prompt=QWEN_IMAGE_PROMPT,
                height=QWEN_IMAGE_HEIGHT,
                width=QWEN_IMAGE_WIDTH,
                num_inference_steps=QWEN_IMAGE_NUM_INFERENCE_STEPS,
                true_cfg_scale=QWEN_IMAGE_TRUE_CFG_SCALE,
                seed=QWEN_IMAGE_SEED,
        )
        image = result.image[0].detach().cpu()
        selected_backend = pipeline.transformer.transformer_blocks[0].attn.attn_backend
        sharder = pipeline.transformer.sharder
        uses_ulysses = pipeline.transformer.transformer_blocks[0].attn._uses_ulysses_attention
        print(
            "\n[Qwen-Image] "
            f"sharder_active={sharder.is_active}, sharder_size={sharder.size}, "
            f"sharder_rank={sharder.rank}, uses_ulysses_attention={uses_ulysses}"
        )
        if expect_ulysses:
            assert sharder.is_active
            assert sharder.size == parallel["ulysses_size"]
            assert uses_ulysses
        else:
            assert not sharder.is_active
            assert not uses_ulysses
    finally:
        del pipeline
        _cleanup_cuda()

    assert tuple(image.shape[-3:]) == (QWEN_IMAGE_HEIGHT, QWEN_IMAGE_WIDTH, 3)
    assert torch.isfinite(image).all()
    assert image.float().std().item() > 0.0
    if output_path is not None:
        save_image(image, output_path)
    print(f"\n[Qwen-Image] selected_backend={selected_backend}, output={output_path}")
    return selected_backend


def _qwen_image_distributed_worker(rank: int, world_size: int, **kwargs) -> None:
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    parallel = kwargs["parallel"]
    ParallelConfig(**parallel).validate_world_size(world_size)

    output_path = kwargs["output_path"] if rank == 0 else None
    selected_backend = _generate_qwen_image(
        kwargs["model_path"],
        output_path,
        "VANILLA",
        kwargs["quant_config"],
        parallel=parallel,
        expect_ulysses=True,
    )
    assert selected_backend == "VANILLA"

    if dist.is_initialized():
        dist.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "precision,quant_config",
    QWEN_IMAGE_PRECISION_VARIANTS,
    ids=[name.replace("_", "-") for name, _ in QWEN_IMAGE_PRECISION_VARIANTS],
)
def test_qwen_image_vanilla_precision_lpips_same_precision_repeat(
    tmp_path, precision, quant_config
):
    model_path = _qwen_image_model_path()
    reference_path = tmp_path / f"qwen_image_{precision}_vanilla_reference.png"
    generated_path = tmp_path / f"qwen_image_{precision}_vanilla_repeat.png"

    reference_backend = _generate_qwen_image(
        model_path, reference_path, "VANILLA", quant_config
    )
    generated_backend = _generate_qwen_image(
        model_path, generated_path, "VANILLA", quant_config
    )

    assert reference_backend == "VANILLA"
    assert generated_backend == "VANILLA"

    score = _run_lpips_eval(
        tmp_path,
        f"qwen_image_{precision}_vanilla_repeat",
        "image",
        QWEN_IMAGE_PROMPT,
        reference_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, QWEN_IMAGE_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "attention_backend",
    QWEN_IMAGE_ATTENTION_BACKENDS,
    ids=[backend.lower() for backend in QWEN_IMAGE_ATTENTION_BACKENDS],
)
def test_qwen_image_attention_backend_bf16_lpips_same_backend_repeat(
    tmp_path, attention_backend
):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    _skip_if_qwen_attention_backend_unavailable(attention_backend)

    model_path = _qwen_image_model_path()
    reference_path = tmp_path / f"qwen_image_bf16_{attention_backend.lower()}_reference.png"
    generated_path = tmp_path / f"qwen_image_bf16_{attention_backend.lower()}_repeat.png"

    reference_backend = _generate_qwen_image(
        model_path, reference_path, attention_backend, quant_config=None
    )
    generated_backend = _generate_qwen_image(
        model_path, generated_path, attention_backend, quant_config=None
    )

    assert reference_backend == attention_backend
    assert generated_backend == attention_backend

    score = _run_lpips_eval(
        tmp_path,
        f"qwen_image_bf16_{attention_backend.lower()}_repeat",
        "image",
        QWEN_IMAGE_PROMPT,
        reference_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, QWEN_IMAGE_LPIPS_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "precision,quant_config",
    QWEN_IMAGE_PRECISION_VARIANTS,
    ids=[name.replace("_", "-") for name, _ in QWEN_IMAGE_PRECISION_VARIANTS],
)
def test_qwen_image_vanilla_ulysses2_lpips_against_single_gpu_same_precision(
    tmp_path, precision, quant_config
):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    _skip_if_insufficient_gpus_for_parallel(QWEN_IMAGE_ULYSSES_PARALLEL)

    model_path = _qwen_image_model_path()
    parallel_cfg = ParallelConfig(**QWEN_IMAGE_ULYSSES_PARALLEL)
    single_gpu_path = tmp_path / f"qwen_image_{precision}_single_gpu_vanilla.png"
    multi_gpu_path = tmp_path / f"qwen_image_{precision}_ulysses2_vanilla.png"

    single_gpu_backend = _generate_qwen_image(
        model_path, single_gpu_path, "VANILLA", quant_config
    )
    assert single_gpu_backend == "VANILLA"

    run_test_in_distributed(
        world_size=parallel_cfg.total_parallel_size,
        test_fn=_qwen_image_distributed_worker,
        model_path=model_path,
        output_path=str(multi_gpu_path),
        quant_config=quant_config,
        parallel=QWEN_IMAGE_ULYSSES_PARALLEL,
    )

    assert multi_gpu_path.is_file(), f"Distributed run did not produce {multi_gpu_path}"
    score = _run_lpips_eval(
        tmp_path,
        f"qwen_image_{precision}_ulysses2_vs_single_gpu",
        "image",
        QWEN_IMAGE_PROMPT,
        single_gpu_path,
        multi_gpu_path,
    )
    _assert_lpips_below_threshold(score, QWEN_IMAGE_LPIPS_THRESHOLD)
