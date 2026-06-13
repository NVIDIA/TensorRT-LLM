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
from collections.abc import Callable, Mapping
from dataclasses import dataclass

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
QWEN_IMAGE_ATTENTION_BACKEND_CLASSES = {
    "VANILLA": "VanillaAttention",
    "FA4": "FlashAttn4Attention",
    "CUTEDSL": "CuTeDSLAttention",
}


QuantConfig = dict[str, object] | None
ParallelDict = dict[str, object]


@dataclass(frozen=True)
class VisualGenFeatureSpec:
    id: str
    name: str
    media_type: str
    output_extension: str
    prompt: str
    lpips_threshold: float
    precision_variants: tuple[tuple[str, QuantConfig], ...]
    attention_backends: tuple[str, ...]
    attention_backend_classes: Mapping[str, str]
    parallel_configs: Mapping[str, ParallelDict]
    model_path: Callable[[], str]
    make_args: Callable[[str, str, QuantConfig, ParallelDict | None], object]
    forward_kwargs: Callable[[], dict[str, object]]
    extract_output: Callable[[object], torch.Tensor]
    save_output: Callable[[torch.Tensor, object], None]
    assert_output: Callable[[torch.Tensor], None]
    attention_backend_root: Callable[[object], object]
    selected_backend: Callable[[object], str]
    assert_parallel: Callable[[object, ParallelDict | None, str | None], None]
    skip_if_attention_backend_unavailable: Callable[[str], None]
    parallel_skip_reasons: Mapping[str, str] | None = None
    default_attention_backend: str = "VANILLA"
    disable_distributed_checkpoint_prefetch: bool = False


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


def _iter_attention_backend_stack(attn):
    seen = set()
    while attn is not None and id(attn) not in seen:
        seen.add(id(attn))
        yield attn
        attn = getattr(attn, "inner_backend", None) or getattr(attn, "inner", None)


def _make_qwen_image_args(
    model_path: str,
    attention_backend: str,
    quant_config: QuantConfig,
    parallel: ParallelDict | None = None,
):
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


def _qwen_image_forward_kwargs():
    return {
        "prompt": QWEN_IMAGE_PROMPT,
        "height": QWEN_IMAGE_HEIGHT,
        "width": QWEN_IMAGE_WIDTH,
        "num_inference_steps": QWEN_IMAGE_NUM_INFERENCE_STEPS,
        "true_cfg_scale": QWEN_IMAGE_TRUE_CFG_SCALE,
        "seed": QWEN_IMAGE_SEED,
    }


def _extract_qwen_image_output(result):
    return result.image[0].detach().cpu()


def _save_qwen_image_output(image, output_path):
    from tensorrt_llm.media.encoding import save_image

    save_image(image, output_path)


def _assert_qwen_image_output(image):
    assert tuple(image.shape[-3:]) == (QWEN_IMAGE_HEIGHT, QWEN_IMAGE_WIDTH, 3)
    assert torch.isfinite(image).all()
    assert image.float().std().item() > 0.0


def _qwen_image_attention_module(pipeline):
    return pipeline.transformer.transformer_blocks[0].attn


def _qwen_image_attention_backend_root(pipeline):
    return _qwen_image_attention_module(pipeline).attn


def _qwen_image_selected_backend(pipeline):
    return _qwen_image_attention_module(pipeline).attn_backend


def _assert_qwen_image_parallel(
    pipeline,
    parallel: ParallelDict | None,
    expected_parallel: str | None,
):
    attn_module = _qwen_image_attention_module(pipeline)
    sharder = pipeline.transformer.sharder
    uses_ulysses = type(attn_module.attn).__name__ == "UlyssesAttention"
    print(
        "\n[Qwen-Image] "
        f"sharder_active={sharder.is_active}, sharder_size={sharder.size}, "
        f"sharder_rank={sharder.rank}, uses_ulysses_attention={uses_ulysses}"
    )
    if expected_parallel == "ulysses2":
        assert parallel is not None
        assert sharder.is_active
        assert sharder.size == parallel["ulysses_size"]
        assert uses_ulysses
    else:
        assert not sharder.is_active
        assert not uses_ulysses


QWEN_IMAGE_SPEC = VisualGenFeatureSpec(
    id="qwen_image",
    name="Qwen-Image",
    media_type="image",
    output_extension="png",
    prompt=QWEN_IMAGE_PROMPT,
    lpips_threshold=QWEN_IMAGE_LPIPS_THRESHOLD,
    precision_variants=tuple(QWEN_IMAGE_PRECISION_VARIANTS),
    attention_backends=tuple(QWEN_IMAGE_ATTENTION_BACKENDS),
    attention_backend_classes=QWEN_IMAGE_ATTENTION_BACKEND_CLASSES,
    parallel_configs={"ulysses2": QWEN_IMAGE_ULYSSES_PARALLEL},
    model_path=_qwen_image_model_path,
    make_args=_make_qwen_image_args,
    forward_kwargs=_qwen_image_forward_kwargs,
    extract_output=_extract_qwen_image_output,
    save_output=_save_qwen_image_output,
    assert_output=_assert_qwen_image_output,
    attention_backend_root=_qwen_image_attention_backend_root,
    selected_backend=_qwen_image_selected_backend,
    assert_parallel=_assert_qwen_image_parallel,
    skip_if_attention_backend_unavailable=_skip_if_qwen_attention_backend_unavailable,
    parallel_skip_reasons={
        "ulysses2": (
            "Qwen-Image Ulysses currently hangs during distributed denoising; "
            "keep the case visible but skip until the runtime issue is fixed."
        ),
    },
    disable_distributed_checkpoint_prefetch=True,
)
VISUAL_GEN_FEATURE_SPECS = {QWEN_IMAGE_SPEC.id: QWEN_IMAGE_SPEC}


def _find_attention_backend_for_spec(spec, attn, attention_backend):
    expected_class = spec.attention_backend_classes[attention_backend]
    for backend in _iter_attention_backend_stack(attn):
        if type(backend).__name__ == expected_class:
            return backend
    return None


def _generate_visual_gen_output(
    spec: VisualGenFeatureSpec,
    output_path,
    attention_backend: str,
    quant_config: QuantConfig,
    parallel: ParallelDict | None = None,
    expected_parallel: str | None = None,
    model_path: str | None = None,
):
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

    model_path = model_path or spec.model_path()
    _skip_if_missing(model_path, f"{spec.name} checkpoint", is_dir=True)
    args = spec.make_args(model_path, attention_backend, quant_config, parallel)
    precision = "bf16" if quant_config is None else quant_config["quant_algo"]
    print(
        f"\n[{spec.name}] "
        f"precision={precision}, requested_backend={attention_backend}, parallel={parallel}"
    )
    pipeline = None
    backend_obj = None
    original_backend_forward = None
    try:
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        attention_root = spec.attention_backend_root(pipeline)
        backend_stack = [
            type(backend).__name__ for backend in _iter_attention_backend_stack(attention_root)
        ]
        backend_obj = _find_attention_backend_for_spec(spec, attention_root, attention_backend)
        assert backend_obj is not None, (
            f"Expected {attention_backend} backend in attention stack, got {backend_stack}"
        )
        backend_call_count = {"count": 0}
        original_backend_forward = backend_obj.forward

        def counted_backend_forward(*args, **kwargs):
            backend_call_count["count"] += 1
            return original_backend_forward(*args, **kwargs)

        backend_obj.forward = counted_backend_forward

        with torch.no_grad():
            result = pipeline.forward(**spec.forward_kwargs())
        output = spec.extract_output(result)
        assert backend_call_count["count"] > 0, (
            f"{attention_backend} backend forward was not called; stack={backend_stack}"
        )
        selected_backend = spec.selected_backend(pipeline)
        spec.assert_parallel(pipeline, parallel, expected_parallel)
        print(
            f"\n[{spec.name}] "
            f"backend_stack={backend_stack}, backend_calls={backend_call_count['count']}"
        )
    finally:
        if backend_obj is not None and original_backend_forward is not None:
            backend_obj.forward = original_backend_forward
        if pipeline is not None:
            del pipeline
        _cleanup_cuda()

    spec.assert_output(output)
    if output_path is not None:
        spec.save_output(output, output_path)
    print(f"\n[{spec.name}] selected_backend={selected_backend}, output={output_path}")
    return selected_backend


def _run_visual_gen_lpips_eval(
    spec: VisualGenFeatureSpec,
    tmp_path,
    sample_id: str,
    reference_path,
    generated_path,
):
    score = _run_lpips_eval(
        tmp_path,
        sample_id,
        spec.media_type,
        spec.prompt,
        reference_path,
        generated_path,
    )
    _assert_lpips_below_threshold(score, spec.lpips_threshold)


def _run_precision_repeat_lpips_test(
    spec: VisualGenFeatureSpec,
    tmp_path,
    precision: str,
    quant_config: QuantConfig,
):
    reference_path = tmp_path / (f"{spec.id}_{precision}_vanilla_reference.{spec.output_extension}")
    generated_path = tmp_path / (f"{spec.id}_{precision}_vanilla_repeat.{spec.output_extension}")

    reference_backend = _generate_visual_gen_output(
        spec, reference_path, spec.default_attention_backend, quant_config
    )
    generated_backend = _generate_visual_gen_output(
        spec, generated_path, spec.default_attention_backend, quant_config
    )

    assert reference_backend == spec.default_attention_backend
    assert generated_backend == spec.default_attention_backend

    _run_visual_gen_lpips_eval(
        spec,
        tmp_path,
        f"{spec.id}_{precision}_vanilla_repeat",
        reference_path,
        generated_path,
    )


def _run_attention_backend_repeat_lpips_test(
    spec: VisualGenFeatureSpec,
    tmp_path,
    attention_backend: str,
):
    spec.skip_if_attention_backend_unavailable(attention_backend)

    reference_path = tmp_path / (
        f"{spec.id}_bf16_{attention_backend.lower()}_reference.{spec.output_extension}"
    )
    generated_path = tmp_path / (
        f"{spec.id}_bf16_{attention_backend.lower()}_repeat.{spec.output_extension}"
    )

    reference_backend = _generate_visual_gen_output(
        spec, reference_path, attention_backend, quant_config=None
    )
    generated_backend = _generate_visual_gen_output(
        spec, generated_path, attention_backend, quant_config=None
    )

    assert reference_backend == attention_backend
    assert generated_backend == attention_backend

    _run_visual_gen_lpips_eval(
        spec,
        tmp_path,
        f"{spec.id}_bf16_{attention_backend.lower()}_repeat",
        reference_path,
        generated_path,
    )


def _visual_gen_distributed_worker(rank: int, world_size: int, **kwargs) -> None:
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    spec = VISUAL_GEN_FEATURE_SPECS[kwargs["spec_id"]]
    if kwargs.get("disable_checkpoint_prefetch", False):
        import tensorrt_llm._torch.visual_gen.checkpoints.weight_loader as weight_loader

        def _skip_checkpoint_prefetch(*args, **kwargs):
            return False

        weight_loader.prefetch_files_to_host_cache = _skip_checkpoint_prefetch

    parallel = kwargs["parallel"]
    ParallelConfig(**parallel).validate_world_size(world_size)

    output_path = kwargs["output_path"] if rank == 0 else None
    selected_backend = _generate_visual_gen_output(
        spec,
        output_path,
        kwargs["attention_backend"],
        kwargs["quant_config"],
        parallel=parallel,
        expected_parallel=kwargs["parallel_name"],
        model_path=kwargs["model_path"],
    )
    assert selected_backend == kwargs["attention_backend"]

    if dist.is_initialized():
        dist.barrier()


def _run_parallel_lpips_test(
    spec: VisualGenFeatureSpec,
    tmp_path,
    parallel_name: str,
    precision: str,
    quant_config: QuantConfig,
):
    if spec.parallel_skip_reasons and parallel_name in spec.parallel_skip_reasons:
        pytest.skip(spec.parallel_skip_reasons[parallel_name])

    parallel = spec.parallel_configs[parallel_name]
    _skip_if_insufficient_gpus_for_parallel(parallel)

    model_path = spec.model_path()
    parallel_cfg = ParallelConfig(**parallel)
    single_gpu_path = tmp_path / (
        f"{spec.id}_{precision}_single_gpu_vanilla.{spec.output_extension}"
    )
    multi_gpu_path = tmp_path / (
        f"{spec.id}_{precision}_{parallel_name}_vanilla.{spec.output_extension}"
    )

    single_gpu_backend = _generate_visual_gen_output(
        spec,
        single_gpu_path,
        spec.default_attention_backend,
        quant_config,
        model_path=model_path,
    )
    assert single_gpu_backend == spec.default_attention_backend

    run_test_in_distributed(
        world_size=parallel_cfg.total_parallel_size,
        test_fn=_visual_gen_distributed_worker,
        spec_id=spec.id,
        model_path=model_path,
        output_path=str(multi_gpu_path),
        quant_config=quant_config,
        parallel=parallel,
        parallel_name=parallel_name,
        attention_backend=spec.default_attention_backend,
        disable_checkpoint_prefetch=spec.disable_distributed_checkpoint_prefetch,
    )

    assert multi_gpu_path.is_file(), f"Distributed run did not produce {multi_gpu_path}"
    _run_visual_gen_lpips_eval(
        spec,
        tmp_path,
        f"{spec.id}_{precision}_{parallel_name}_vs_single_gpu",
        single_gpu_path,
        multi_gpu_path,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "precision,quant_config",
    QWEN_IMAGE_SPEC.precision_variants,
    ids=[name.replace("_", "-") for name, _ in QWEN_IMAGE_SPEC.precision_variants],
)
def test_qwen_image_vanilla_precision_lpips_same_precision_repeat(
    tmp_path, precision, quant_config
):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    _run_precision_repeat_lpips_test(QWEN_IMAGE_SPEC, tmp_path, precision, quant_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "attention_backend",
    QWEN_IMAGE_SPEC.attention_backends,
    ids=[backend.lower() for backend in QWEN_IMAGE_SPEC.attention_backends],
)
def test_qwen_image_attention_backend_bf16_lpips_same_backend_repeat(tmp_path, attention_backend):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    _run_attention_backend_repeat_lpips_test(QWEN_IMAGE_SPEC, tmp_path, attention_backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "precision,quant_config",
    QWEN_IMAGE_SPEC.precision_variants,
    ids=[name.replace("_", "-") for name, _ in QWEN_IMAGE_SPEC.precision_variants],
)
def test_qwen_image_vanilla_ulysses2_lpips_against_single_gpu_same_precision(
    tmp_path, precision, quant_config
):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    _run_parallel_lpips_test(QWEN_IMAGE_SPEC, tmp_path, "ulysses2", precision, quant_config)
