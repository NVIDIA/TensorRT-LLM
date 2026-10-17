# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 packed denoising with production CFG/PAG collectives and sequence sharding."""

__extra_import_path__ = [".."]

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Spawned workers do not load pytest plugins themselves.
import test_common.magic_import  # noqa: F401
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flux3_test_utils import assert_parity, small_model_params, small_weights


def _worker(rank, world_size, port, test_fn, kwargs):
    os.environ.update(
        MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        test_fn(rank, world_size, **kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(world_size, test_fn, **kwargs):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    from ._visual_gen_dist_utils import spawn_with_retry

    with patch.dict(os.environ, {"TLLM_DISABLE_MPI": "1"}):
        spawn_with_retry(
            lambda port: mp.spawn(
                _worker, args=(world_size, port, test_fn, kwargs), nprocs=world_size, join=True
            )
        )


def checkpoint_weights(hidden_size=512, recipe=None):
    """Quantize once before TP loading so all candidates receive identical source tensors."""
    weights = small_weights(hidden_size)
    if recipe is None:
        return weights
    from tensorrt_llm._torch.visual_gen.models.flux3.fp8 import quantize_fp8r
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4

    for name, weight in list(weights.items()):
        if not name.endswith(".weight") or not name.startswith(
            ("content_mode_blocks.image.", "single_blocks.")
        ):
            continue
        prefix = name.removesuffix(".weight")
        if recipe == "fp8":
            data, scales = quantize_fp8r(weight.cuda())
            weights[name] = data.cpu()
            weights[prefix + ".weight_scale"] = scales.cpu()
        else:
            data, scales, global_scale = quantize_nvfp4(weight.cuda())
            weights[name] = data.cpu()
            weights[prefix + ".weight_scale"] = scales.cpu()
            weights[prefix + ".weight_scale_2"] = global_scale.cpu()
            weights[prefix + ".input_scale"] = torch.tensor(0.01)
    return weights


def make_pipeline(weights, *, hidden_size=512, tp_size=1, ulysses_size=1, cfg_size=1, recipe=None):
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        DiffusionPipelineConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import Flux3Pipeline
    from tensorrt_llm.models.modeling_utils import QuantConfig

    world_size = tp_size * ulysses_size * cfg_size
    mapping = VisualGenMapping(
        world_size=world_size,
        rank=dist.get_rank() if world_size > 1 else 0,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        cfg_size=cfg_size,
    )
    quant = QuantConfig()
    if recipe:
        excluded = [
            name.removesuffix(".weight")
            for name in weights
            if name.endswith(".weight")
            and name.removesuffix(".weight") + ".weight_scale" not in weights
        ]
        quant = QuantConfig(
            quant_algo="FP8_PER_CHANNEL_PER_TOKEN" if recipe == "fp8" else "NVFP4",
            exclude_modules=excluded,
        )
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**small_model_params(hidden_size)),
        quant_config=quant,
        visual_gen_mapping=mapping,
        mapping=mapping.to_llm_mapping(),
        attention=AttentionConfig(backend="VANILLA"),
    )
    pipeline_config = DiffusionPipelineConfig(
        model_configs={"transformer": config}, visual_gen_mapping=mapping
    )
    pipeline_config.extra_attrs["flux3_checkpoint"] = SimpleNamespace(
        config={"sampler": {"alpha": 6.93}}
    )
    pipeline = Flux3Pipeline(pipeline_config)
    pipeline.transformer.load_weights(weights)
    pipeline.transformer.post_load_weights()
    pipeline.transformer.cuda().eval()
    if recipe is None:
        pipeline.transformer.bfloat16()
    return pipeline


def denoise_inputs(with_reference=False):
    from tensorrt_llm._torch.visual_gen.models.flux3.pipeline_flux3 import sample_image_latents

    torch.manual_seed(42)
    image, ids, _ = sample_image_latents(height=64, width=64, seed=17, device="cuda")
    # Unequal lengths divisible by Ulysses=4 exercise packed sample boundaries.
    contexts = [torch.randn(1, n, 48, device="cuda", dtype=torch.bfloat16) for n in (8, 12)]
    cond = {}
    if with_reference:
        cond = dict(x_image_cond=image[:, :8].clone(), x_image_cond_ids=ids[:, :8].clone())
        cond["x_image_cond_ids"][..., 3] = -1
    return dict(
        x=image,
        x_ids=ids,
        ctx=contexts,
        num_inference_steps=2,
        cfg=3.5,
        pag_single_layers=None,
        pag_strength=0.0,
        cond_kwargs=cond,
    )


def _denoise_parity(
    rank, world_size, *, tp_size, ulysses_size, cfg_size, with_reference, pag_strength, guidance
):
    """L2 T1: distributed denoised latents versus TP/SP/CFG-off with identical inputs."""
    weights = checkpoint_weights()
    reference = make_pipeline(weights)
    candidate = make_pipeline(
        weights, tp_size=tp_size, ulysses_size=ulysses_size, cfg_size=cfg_size
    )
    inputs = denoise_inputs(with_reference)
    inputs.update(
        cfg=guidance, pag_strength=pag_strength, pag_single_layers=[0] if pag_strength else None
    )
    calls = []

    def record(module, args, kwargs):
        calls.append((dict(kwargs["layout"])["text"], kwargs["pag_samples"]))

    handle = candidate.transformer.register_forward_pre_hook(record, with_kwargs=True)
    try:
        with torch.inference_mode():
            expected = reference._run_denoise(**inputs)
            actual = candidate._run_denoise(**inputs)
        assert torch.isfinite(actual).all()
        assert actual.shape == expected.shape == (1, 16, 128)
        # BF16 partial-sum and attention ordering may differ; do not hide drift behind atol.
        assert_parity(actual, expected)
        mapping = candidate.pipeline_config.visual_gen_mapping
        if cfg_size > 1 and guidance > 1:
            expected_call = ((12,), ()) if mapping.is_cfg_conditional else ((8,), (0,))
        else:
            expected_call = ((8, 12), (0,))
        assert calls == [expected_call] * inputs["num_inference_steps"]
        assert candidate.transformer.sharder.size == ulysses_size
        if tp_size > 1:
            assert candidate.transformer.single_blocks[0].attn.tp_size == tp_size
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "tp_size,ulysses_size,cfg_size,with_reference,pag_strength,guidance",
    [
        (1, 2, 1, False, 0.0, 3.5),
        (1, 4, 1, True, 0.0, 3.5),
        (1, 1, 2, False, 0.0, 3.5),
        (1, 1, 2, True, 0.5, 3.5),
        (1, 1, 2, False, 0.0, 1.0),
        (2, 2, 2, True, 0.5, 3.5),
    ],
    ids=[
        "ulysses2-t2i",
        "ulysses4-i2i",
        "cfg2-t2i",
        "cfg2-i2i-pag",
        "cfg2-guidance-one",
        "cfg2-tp2-ulysses2-i2i-pag",
    ],
)
def test_distributed_denoise_matches_single_gpu(
    tp_size, ulysses_size, cfg_size, with_reference, pag_strength, guidance
):
    run_distributed(
        tp_size * ulysses_size * cfg_size,
        _denoise_parity,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        cfg_size=cfg_size,
        with_reference=with_reference,
        pag_strength=pag_strength,
        guidance=guidance,
    )
