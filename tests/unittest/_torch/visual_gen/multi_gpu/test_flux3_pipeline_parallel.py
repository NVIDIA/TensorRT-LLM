# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 pipelines loaded with Ulysses, TP or CFG parallelism versus the single-GPU output.

Also TP2 fused block GEMMs versus TRT-LLM's on the same ranks.
"""

import math
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

from tensorrt_llm._torch.modules.linear import FP8RowwiseLinearMethod, NVFP4LinearMethod
from tensorrt_llm._torch.visual_gen.attention.attention import apply_rotary_emb
from tensorrt_llm._torch.visual_gen.models.flux3 import fused_gemm_common
from tensorrt_llm._torch.visual_gen.models.flux3.transformer_flux3 import _Flux3Block
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
PRIMS_TS_ARCHES = ((10, 0), (10, 3), (10, 7))
# Head size 128 for the fused QK-norm + RoPE epilogue. Per TP2 rank, 4 FP8 heads put Q/K/V and
# gate/up on Prims-TS's N % 512 (1536, 3072); 3 NVFP4 heads give the 384-wide Q/K/V planes the
# released model has at tp_size 8.
FUSED_OVERRIDES = {
    variant: dict(
        attention_head_dim=128,
        num_attention_heads=heads,
        axes_dims_rope=[32] * 4,
        num_single_layers=3,
    )
    for variant, heads in (("fp8r", 8), ("nvfp4", 6))
}
# Fused versus TRT-LLM GEMMs on the same ranks: the single-GPU fp8r transformer bound, and the
# NVFP4 bound above.
FUSED_PARITY = {"fp8r": dict(relative_l2=0.05, cosine=0.998), "nvfp4": PARITY["nvfp4"]}

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


def _load_pipeline(checkpoint: str, device: str, variant: Optional[str] = None, **parallel: int):
    args = VisualGenArgs(
        model=checkpoint,
        attention_config=AttentionConfig(backend="VANILLA"),
        torch_compile_config=TorchCompileConfig(enable=False, enable_autotune=False),
        parallel_config=ParallelConfig(**parallel),
        pipeline_config={"variant": variant},
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_text_encoder_load(monkeypatch)
        return PipelineLoader(args, device=device).load(skip_warmup=True)


def _images(pipeline) -> list[Optional[torch.Tensor]]:
    """Text-to-image and one-reference edit images, or None on ranks that do not decode."""
    images = []
    for reference in (None, _reference_image()):
        output = pipeline.forward(
            "a red fox", height=64, width=64, num_inference_steps=2, seed=0, image=reference
        )
        images.append(None if output.image is None else output.image.cpu())
    return images


def _generate(
    checkpoint: str, device: str, variant: Optional[str] = None, **parallel: int
) -> list[Optional[torch.Tensor]]:
    return _images(_load_pipeline(checkpoint, device, variant, **parallel))


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


def _block_inputs(attn) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """A [1, 96, hidden] block input and its interleaved RoPE tables."""
    tokens = 96
    generator = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(
        1, tokens, attn.hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    angles = torch.rand(tokens, attn.head_dim // 2, device="cuda", generator=generator) * math.tau
    cos, sin = (
        f(angles).repeat_interleave(2, dim=-1).view(1, tokens, 1, attn.head_dim)
        for f in (torch.cos, torch.sin)
    )
    return hidden, (cos, sin)


def _logic_tp2_fused_gemms(rank: int, world_size: int, *, checkpoint: str, variant: str) -> None:
    pipelines = {}
    for fused in (True, False):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(fused_gemm_common, "FUSED_GEMMS", fused)
            pipelines[fused] = _load_pipeline(checkpoint, f"cuda:{rank}", variant, tp_size=2)
    method = FP8RowwiseLinearMethod if variant == "fp8r" else NVFP4LinearMethod
    blocks, trtllm_blocks = (
        [
            module
            for module in pipelines[fused].transformer.modules()
            if isinstance(module, _Flux3Block)
            and isinstance(module.ff.gate_up_proj.quant_method, method)
        ]
        for fused in (True, False)
    )
    assert blocks and len(blocks) == len(trtllm_blocks), "no fused blocks to compare"
    block, trtllm_block = blocks[0], trtllm_blocks[0]
    assert block.attn.tp_size == 2
    hidden, (cos, sin) = _block_inputs(block.attn)
    with torch.inference_mode():
        if variant == "fp8r":
            assert all(b.prims_ts_fp8 is not None for b in blocks)
            assert not any(b.prims_ts_fp8 is not None for b in trtllm_blocks)
            # This rank's attention-out + MLP partial sums, before the block all-reduce.
            actual = block.prims_ts_fp8(hidden, (cos, sin), None, None)
            attended = trtllm_block.attn.attend(hidden, (cos, sin))
            mlp = trtllm_block.ff(hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)
            expected = trtllm_block.attn.to_out[0](attended) + mlp
            assert_parity(actual, expected, relative_l2=0.03, cosine=0.999)
        else:
            assert all(b.attn.prims_ts_qkv is not None and b.share_nvfp4_quant for b in blocks)
            assert not any(b.attn.prims_ts_qkv is not None for b in trtllm_blocks)
            # This rank's heads: one fused GEMM against TRT-LLM's three and eager norm + RoPE.
            attn = block.attn
            planes = attn.prims_ts_qkv(hidden, cos, sin)
            shape = (*hidden.shape[:-1], attn.local_num_attention_heads, attn.head_dim)
            q, k, v = (t.view(shape) for t in attn.get_qkv(hidden))
            q, k = attn.apply_qk_norm(q, k)
            expected = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin), v)
            for plane, reference in zip(planes, expected):
                assert_parity(
                    plane, reference.flatten(2).to(plane.dtype), relative_l2=0.02, cosine=0.9995
                )
    images = {fused: _images(pipeline) for fused, pipeline in pipelines.items()}
    if rank == 0:
        for actual, reference in zip(images[True], images[False]):
            assert_parity(actual, reference, **FUSED_PARITY[variant])
    dist.barrier()


@pytest.mark.parametrize("variant", ["fp8r", "nvfp4"])
def test_tp2_fused_gemms_match_trtllm_gemms(tmp_path, variant: str) -> None:
    """Each rank runs the fused Q/K/V and gate/up GEMMs on its own column shards."""
    if torch.cuda.device_count() < WORLD_SIZE or any(
        torch.cuda.get_device_capability(i) not in PRIMS_TS_ARCHES for i in range(WORLD_SIZE)
    ):
        pytest.skip(f"Requires {WORLD_SIZE} SM100, SM103 or SM107 GPUs")
    overrides = FUSED_OVERRIDES[variant]
    weights = write_gpu_checkpoint(tmp_path, True, num_layers=2, **overrides)
    config = gpu_transformer_config(True, num_layers=2, **overrides)
    tensors = quantize_variant(config, weights, variant, defaultdict(lambda: 8.0))
    write_variant(tmp_path, config, variant, tensors)
    run_distributed(
        WORLD_SIZE,
        _logic_tp2_fused_gemms,
        checkpoint=str(tmp_path),
        variant=variant,
        preload=("tensorrt_llm",),
    )
