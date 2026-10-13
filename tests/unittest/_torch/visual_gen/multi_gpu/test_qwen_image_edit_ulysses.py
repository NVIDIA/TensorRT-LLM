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

"""Multi-GPU coverage for Qwen-Image-Edit Ulysses attention."""

import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual_gen_dist_utils import spawn_with_retry

# The unit test creates its own torch.distributed NCCL process group. Disable the
# TRT-LLM MPI bootstrap path so importing tensorrt_llm does not initialize MPI.
_OLD_TLLM_DISABLE_MPI = os.environ.get("TLLM_DISABLE_MPI")
os.environ["TLLM_DISABLE_MPI"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import UlyssesAttention
    from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        QwenJointAttention,
    )
except ImportError as e:  # pragma: no cover - import guard for direct collection
    pytest.skip(f"TensorRT-LLM modules unavailable: {e}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _cleanup_distributed_env() -> Generator[None, None, None]:
    old_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    yield
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env() -> Generator[None, None, None]:
    yield
    if _OLD_TLLM_DISABLE_MPI is None:
        os.environ.pop("TLLM_DISABLE_MPI", None)
    else:
        os.environ["TLLM_DISABLE_MPI"] = _OLD_TLLM_DISABLE_MPI


def _distributed_worker(
    rank: int,
    world_size: int,
    backend: str,
    test_fn: Callable,
    port: int,
    kwargs: dict,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size, **kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_test_in_distributed(world_size: int, test_fn: Callable, **kwargs) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, "nccl", test_fn, port, kwargs),
            nprocs=world_size,
            join=True,
        )
    )


def _make_config(rank: int, world_size: int, ulysses_size: int) -> DiffusionModelConfig:
    mapping = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=1,
        cp_size=1,
        pp_size=1,
        cfg_size=1,
        sp_size=ulysses_size,
        ulysses_size=ulysses_size,
        device_mesh=DeviceMeshTopologyImpl.create().initialize(world_size),
    )
    return DiffusionModelConfig(
        mapping=mapping.to_llm_mapping(),
        visual_gen_mapping=mapping,
        attention=AttentionConfig(backend="VANILLA"),
        dtype="bfloat16",
    )


def _make_attention(config: DiffusionModelConfig) -> QwenJointAttention:
    return QwenJointAttention(
        dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        config=config,
    ).cuda()


def _rank_ordered_joint_mask(
    text_mask: torch.Tensor,
    image_seq_len: int,
    world_size: int,
) -> torch.Tensor:
    image_mask = torch.ones(
        (text_mask.shape[0], image_seq_len),
        device=text_mask.device,
        dtype=torch.bool,
    )
    text_chunks = text_mask.chunk(world_size, dim=1)
    image_chunks = image_mask.chunk(world_size, dim=1)
    return torch.cat(
        [chunk for pair in zip(text_chunks, image_chunks) for chunk in pair],
        dim=1,
    )


def _assert_ulysses_matches_reference(
    rank: int,
    world_size: int,
    ulysses_attn: QwenJointAttention,
    reference_attn: QwenJointAttention,
    full_hidden_states: torch.Tensor,
    full_encoder_hidden_states: torch.Tensor,
    text_mask: torch.Tensor | None,
) -> None:
    local_hidden_states = full_hidden_states.chunk(world_size, dim=1)[rank].contiguous()
    local_encoder_hidden_states = full_encoder_hidden_states.chunk(world_size, dim=1)[
        rank
    ].contiguous()

    reference_attention_mask = None
    ulysses_attention_mask = None
    if text_mask is not None:
        image_mask = torch.ones(
            (text_mask.shape[0], full_hidden_states.shape[1]),
            device=text_mask.device,
            dtype=torch.bool,
        )
        reference_attention_mask = torch.cat([text_mask, image_mask], dim=1)
        ulysses_attention_mask = _rank_ordered_joint_mask(
            text_mask,
            full_hidden_states.shape[1],
            world_size,
        )

    with torch.no_grad():
        image_out, text_out = ulysses_attn(
            hidden_states=local_hidden_states,
            encoder_hidden_states=local_encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=ulysses_attention_mask,
            timestep=None,
        )
        expected_image_out, expected_text_out = reference_attn(
            hidden_states=full_hidden_states,
            encoder_hidden_states=full_encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=reference_attention_mask,
            timestep=None,
        )

    expected_image_out = expected_image_out.chunk(world_size, dim=1)[rank].contiguous()
    expected_text_out = expected_text_out.chunk(world_size, dim=1)[rank].contiguous()

    torch.testing.assert_close(image_out, expected_image_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(text_out, expected_text_out, rtol=2e-2, atol=2e-2)


def _test_qwen_image_edit_ulysses_attention(rank: int, world_size: int) -> None:
    torch.manual_seed(2026)
    ulysses_attn = _make_attention(_make_config(rank, world_size, world_size))
    reference_attn = _make_attention(_make_config(rank=0, world_size=1, ulysses_size=1))

    with torch.no_grad():
        for name, parameter in ulysses_attn.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(mean=0.0, std=0.02)
    reference_attn.load_state_dict(ulysses_attn.state_dict())

    assert isinstance(ulysses_attn.attn, UlyssesAttention)

    full_hidden_states = torch.randn(
        1,
        4,
        16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    full_encoder_hidden_states = torch.randn(
        1,
        4,
        16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    masked_prompt = torch.tensor(
        [[True, True, True, False]],
        device="cuda",
        dtype=torch.bool,
    )

    for text_mask in (None, masked_prompt):
        _assert_ulysses_matches_reference(
            rank,
            world_size,
            ulysses_attn,
            reference_attn,
            full_hidden_states,
            full_encoder_hidden_states,
            text_mask,
        )

    dist.barrier()


@pytest.mark.gpu2
def test_qwen_image_edit_ulysses_attention_2gpu() -> None:
    run_test_in_distributed(2, _test_qwen_image_edit_ulysses_attention)
