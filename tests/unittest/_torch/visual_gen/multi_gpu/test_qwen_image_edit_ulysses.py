# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for Qwen-Image-Edit Ulysses sequence parallelism."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.qwen_image.pipeline_qwen_image_edit import (
        QwenImageEditPlusPipeline,
    )
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        QwenJointAttention,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def init_distributed_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
    DeviceMeshTopologyImpl.device_mesh = None
    VisualGenMapping.seq_mesh = None


def _distributed_worker(rank: int, world_size: int, test_fn: Callable, port: int) -> None:
    try:
        init_distributed_worker(rank, world_size, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable) -> None:
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, test_fn, port),
            nprocs=world_size,
            join=True,
        )
    )


def _stabilize_attention_weights(module: torch.nn.Module) -> None:
    """Use small deterministic weights for stable BF16 synthetic forward."""
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.ndim >= 2:
                fan_in = parameter.shape[1]
                std = 0.02 / max(1.0, fan_in**0.5)
                parameter.data.uniform_(-std, std)
            else:
                parameter.data.uniform_(-0.01, 0.01)


def _test_qwen_image_edit_ulysses_attention(rank: int, world_size: int) -> None:
    torch.manual_seed(1234)
    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        cfg_size=1,
        tp_size=1,
        ring_size=1,
        ulysses_size=world_size,
    )
    config = DiffusionModelConfig(
        mapping=vgm.to_llm_mapping(),
        visual_gen_mapping=vgm,
        torch_dtype=torch.bfloat16,
    )

    attn = QwenJointAttention(
        dim=16,
        num_attention_heads=4,
        attention_head_dim=4,
        dtype=torch.bfloat16,
        config=config,
    ).cuda()
    _stabilize_attention_weights(attn)
    assert attn.attn.__class__.__name__ == "UlyssesAttention"

    hidden_states = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    encoder_hidden_states = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16) * 0.1
    image_out, text_out = attn(
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb=None,
        attention_mask=None,
        timestep=None,
    )

    assert image_out.shape == hidden_states.shape
    assert text_out.shape == encoder_hidden_states.shape
    assert torch.isfinite(image_out).all()
    assert torch.isfinite(text_out).all()

    QwenImageEditPlusPipeline._validate_ulysses_prompt_masks(world_size, None, None)
    with pytest.raises(ValueError, match="requires unmasked prompt conditioning"):
        QwenImageEditPlusPipeline._validate_ulysses_prompt_masks(
            world_size,
            torch.ones(1, 4, dtype=torch.bool, device="cuda"),
        )


def test_qwen_image_edit_ulysses_attention_2gpu():
    run_test_in_distributed(2, _test_qwen_image_edit_ulysses_attention)
