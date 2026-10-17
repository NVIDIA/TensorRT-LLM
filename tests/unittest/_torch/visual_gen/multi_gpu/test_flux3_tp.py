# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 TP checkpoint loading and PrimsTS collective parity."""

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

from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping


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


def _checkpoint_tp_parity(rank, world_size, *, recipe):
    """L2 T1: production loader and TP DiT output versus TP1 from the same package tensors."""
    hidden_size = 1024
    weights = checkpoint_weights(hidden_size, recipe)
    reference = make_pipeline(weights, hidden_size=hidden_size, recipe=recipe)
    candidate = make_pipeline(weights, hidden_size=hidden_size, recipe=recipe, tp_size=world_size)
    inputs = denoise_inputs(with_reference=True)
    if recipe:
        from tensorrt_llm._torch.modules.linear import Linear

        for name, module in candidate.transformer.named_modules():
            if name.startswith("txt_mode_blocks.") and isinstance(module, Linear):
                assert module.quant_config.quant_algo is None
                assert module.weight.dtype == torch.bfloat16
    block = candidate.transformer.single_blocks[0]
    assert block.attn.tp_size == world_size
    assert block.attn.local_q_dim == hidden_size // world_size
    if recipe == "fp8":
        assert block.attn._primsts_fp8_qkv_rope
    elif recipe == "nvfp4":
        assert block.attn._flux3_primsts_nvfp4_qkv is not None
    with torch.inference_mode():
        expected = reference._run_denoise(**inputs)
        actual = candidate._run_denoise(**inputs)
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape == (1, 16, 128)
    assert_parity(actual, expected)


@pytest.mark.parametrize(
    "world_size,recipe",
    [(2, None), (4, None), (4, "fp8"), (4, "nvfp4")],
    ids=["tp2-bf16", "tp4-bf16", "tp4-fp8", "tp4-nvfp4"],
)
def test_checkpoint_loading_tp_parity(world_size, recipe):
    if recipe and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    ):
        pytest.skip("PrimsTS requires SM100/SM103")
    run_distributed(world_size, _checkpoint_tp_parity, recipe=recipe)


def _logic_prims_ts_tp2(rank, world_size):
    from test_flux3_primsts import (
        _assert_fp8_parity,
        _assert_nvfp4_parity,
        _make_fp8_block,
        _make_nvfp4_block,
    )

    from tensorrt_llm._torch.visual_gen.models.flux3 import primsts
    from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import (
        Flux3ModeBlock,
        Flux3SingleStreamBlock,
    )

    mapping = VisualGenMapping(
        world_size=world_size, rank=rank, tp_size=world_size
    ).to_llm_mapping()
    for block_type in (Flux3ModeBlock, Flux3SingleStreamBlock):
        for recipe, make, prepare, compare in (
            ("fp8", _make_fp8_block, primsts.prepare_fp8_block, _assert_fp8_parity),
            ("nvfp4", _make_nvfp4_block, primsts.prepare_nvfp4_block, _assert_nvfp4_parity),
        ):
            torch.manual_seed(42)
            block = make(block_type, mapping=mapping, hidden_size=1024)
            assert block._allreduce.mapping.tp_size == world_size
            x = torch.randn(1, 33, 1024, device="cuda", dtype=torch.bfloat16)
            mod = tuple(torch.randn_like(x) * 0.1 for _ in range(3))
            angles = torch.randn(1, 33, 1, 64, device="cuda").repeat_interleave(2, -1)
            freqs = angles.cos(), angles.sin()
            with torch.inference_mode():
                expected = block(x, freqs, mod)
                prepare(block)
                if recipe == "fp8":
                    assert block.attn._primsts_fp8_qkv_rope
                else:
                    assert block.attn._flux3_primsts_nvfp4_qkv is not None
                assert getattr(block.ff.gate_up_proj, f"_flux3_primsts_{recipe}").swiglu
                actual = block(x, freqs, mod)
                compare(actual, expected)
                rank_zero = actual.clone()
                dist.broadcast(rank_zero, src=0)
                torch.testing.assert_close(actual, rank_zero, rtol=0, atol=0)


def test_prims_ts_tp2_collective_parity():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("PrimsTS requires SM100/SM103")
    run_distributed(world_size=2, test_fn=_logic_prims_ts_tp2)
