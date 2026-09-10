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
"""Synthetic multi-GPU topology tests for Qwen Image joint attention."""

import os
from typing import Callable

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
        Attention2DAttention,
        RingAttention,
        UlyssesAttention,
    )
    from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.qwen_image import QwenJointAttention
    from tensorrt_llm.visual_gen.args import ParallelConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        _flash_attn_fwd as _fa4_fwd,
    )
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
        _flash_attn_combine as _fa_combine,
    )

    _FLASH_ATTN4_AVAILABLE = _fa4_fwd is not None
    _ATTN2D_AVAILABLE = _fa4_fwd is not None and _fa_combine is not None
except (ImportError, OSError):
    _FLASH_ATTN4_AVAILABLE = False
    _ATTN2D_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _init_distributed_worker(
    rank: int, world_size: int, backend: str = "nccl", port: int = 29500
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port, kwargs):
    try:
        _init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, **kwargs)
    finally:
        _cleanup_distributed()


def _run_test_in_distributed(world_size: int, test_fn: Callable, **kwargs) -> None:
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, "nccl", test_fn, port, kwargs),
            nprocs=world_size,
            join=True,
        )
    )


def _test_qwen_image_attention_parallel_topology(
    rank: int,
    world_size: int,
    *,
    parallel: dict,
    backend: str,
    topology: str,
) -> None:
    parallel_config = ParallelConfig(**parallel)
    parallel_config.validate_world_size(world_size)
    attn2d_row_size, attn2d_col_size = parallel_config.attn2d_size
    visual_gen_mapping = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=parallel_config.tp_size,
        ring_size=parallel_config.ring_size,
        ulysses_size=parallel_config.ulysses_size,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
    )
    config = DiffusionModelConfig(
        mapping=visual_gen_mapping.to_llm_mapping(),
        visual_gen_mapping=visual_gen_mapping,
        attention=AttentionConfig(backend=backend),
        parallel=parallel_config,
    )
    attention = QwenJointAttention(
        dim=256,
        num_attention_heads=4,
        attention_head_dim=64,
        config=config,
    ).cuda()

    torch.manual_seed(11)
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0.0, std=0.02)

    if topology == "tp":
        assert not isinstance(attention.attn, UlyssesAttention)
    else:
        assert isinstance(attention.attn, UlyssesAttention)
        if topology == "ring":
            assert isinstance(attention.attn.inner_backend, RingAttention)
        else:
            assert isinstance(attention.attn.inner_backend, Attention2DAttention)

    hidden_states = torch.randn(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, 4, 256, device="cuda", dtype=torch.bfloat16)
    image_output, text_output = attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
    )

    assert image_output.shape == hidden_states.shape
    assert text_output.shape == encoder_hidden_states.shape
    assert torch.isfinite(image_output).all()
    assert torch.isfinite(text_output).all()
    dist.barrier()


@pytest.mark.parametrize(
    "world_size,parallel,backend,topology",
    [
        pytest.param(2, {"tp_size": 2}, "VANILLA", "tp", marks=pytest.mark.gpu2, id="tp2"),
        pytest.param(
            4,
            {"ring_size": 2, "ulysses_size": 2},
            "FA4",
            "ring",
            marks=pytest.mark.gpu4,
            id="ring2_ulysses2",
        ),
        pytest.param(
            4,
            {"attn2d_size": (2, 1), "ulysses_size": 2},
            "FA4",
            "attn2d",
            marks=pytest.mark.gpu4,
            id="attn2d_2x1_ulysses2",
        ),
    ],
)
def test_qwen_image_attention_parallel_topologies(world_size, parallel, backend, topology) -> None:
    if topology == "ring" and not _FLASH_ATTN4_AVAILABLE:
        pytest.skip("FlashAttn4 JIT kernels not available")
    if topology == "attn2d" and not _ATTN2D_AVAILABLE:
        pytest.skip("FA4 / flash_attn_combine JIT kernels not available")
    _run_test_in_distributed(
        world_size,
        _test_qwen_image_attention_parallel_topology,
        parallel=parallel,
        backend=backend,
        topology=topology,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
