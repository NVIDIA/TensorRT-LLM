# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 pipelines loaded with Ulysses, TP or CFG parallelism versus the single-GPU output."""

import os
from collections import defaultdict
from typing import Any, Optional

os.environ["TLLM_DISABLE_MPI"] = "1"

import numpy as np
import PIL.Image
import pytest
import torch
import torch.distributed as dist

# flux3_test_utils is in tests/unittest/_torch/visual_gen (spawned workers inherit
# pytest's sys.path entry for it).
__extra_import_path__ = [".."]

from flux3_test_utils import (
    assert_parity,
    gpu_transformer_config,
    patch_text_encoder_load,
    quantize_variant,
    write_gpu_checkpoint,
    write_variant,
)

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import (
    AttentionConfig,
    ParallelConfig,
    TorchCompileConfig,
    VisualGenArgs,
)

from ._worker_pool import run_distributed

WORLD_SIZE = 2
# Head size 64 gives the NVFP4 GEMMs 128-wide inputs; the last single block stays BF16.
NVFP4_OVERRIDES = dict(attention_head_dim=64, axes_dims_rope=[16] * 4, num_single_layers=3)
# Collectives reorder BF16 sums, which NVFP4 activation quantization can amplify.
PARITY = {None: dict(relative_l2=2e-2, cosine=0.999), "nvfp4": dict(relative_l2=5e-2, cosine=0.995)}

requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 needs a Blackwell GPU",
)


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _reference_image() -> PIL.Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return PIL.Image.fromarray(pixels)


def _generate(
    checkpoint: str, device: str, variant: Optional[str] = None, **parallel: int
) -> list[Optional[torch.Tensor]]:
    """Text-to-image and one-reference edit images, or None on ranks that do not decode."""
    args = VisualGenArgs(
        model=checkpoint,
        attention_config=AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        parallel_config=ParallelConfig(**parallel),
        pipeline_config={"variant": variant},
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_text_encoder_load(monkeypatch)
        pipeline = PipelineLoader(args, device=device).load(skip_warmup=True)
    images = []
    for reference in (None, _reference_image()):
        output = pipeline.forward(
            "a red fox", height=64, width=64, num_inference_steps=2, seed=0, image=reference
        )
        images.append(None if output.image is None else output.image.cpu())
    return images


def _logic_flux3_parallel(
    rank: int,
    world_size: int,
    *,
    checkpoint: str,
    expected: list[torch.Tensor],
    parallel: dict[str, int],
    variant: Optional[str],
) -> None:
    images = _generate(checkpoint, f"cuda:{rank}", variant, **parallel)
    if rank == 0:
        for actual, reference in zip(images, expected):
            assert actual.shape == reference.shape == (1, 64, 64, 3)
            assert_parity(actual, reference, **PARITY[variant])
    else:
        assert images == [None, None]
    dist.barrier()


@pytest.mark.parametrize(
    "flow_map, parallel, variant",
    [
        pytest.param(True, {"ulysses_size": 2}, None, id="klein-ulysses2"),
        pytest.param(False, {"ulysses_size": 2}, None, id="base-ulysses2"),
        pytest.param(True, {"tp_size": 2}, None, id="klein-tp2"),
        pytest.param(False, {"tp_size": 2}, None, id="base-tp2"),
        pytest.param(False, {"tp_size": 2}, "nvfp4", id="base-tp2-nvfp4", marks=requires_blackwell),
        pytest.param(False, {"cfg_size": 2}, None, id="base-cfg2"),
    ],
)
def test_parallel_pipeline_matches_single_gpu(
    tmp_path, flow_map: bool, parallel: dict, variant: Optional[str]
) -> None:
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"Requires {WORLD_SIZE} CUDA GPUs, only {torch.cuda.device_count()} available")
    overrides: dict[str, Any] = NVFP4_OVERRIDES if variant else {}
    weights = write_gpu_checkpoint(tmp_path, flow_map, num_layers=2, **overrides)
    if variant:
        config = gpu_transformer_config(flow_map, num_layers=2, **overrides)
        # Every quantized input shares one static scale; both runs load the same tensors.
        tensors = quantize_variant(config, weights, variant, defaultdict(lambda: 8.0))
        write_variant(tmp_path, config, variant, tensors)
    expected = _generate(str(tmp_path), "cuda:0", variant)
    run_distributed(
        WORLD_SIZE,
        _logic_flux3_parallel,
        checkpoint=str(tmp_path),
        expected=expected,
        parallel=parallel,
        variant=variant,
        preload=("tensorrt_llm",),
    )
