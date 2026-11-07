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
"""TRTLLM per-tensor FP8 attention under Ulysses (FP8 Q/K/V all-to-all) and Attention2D."""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.parallel import (
    Attention2DAttention,
    UlyssesAttention,
)
from tensorrt_llm._torch.visual_gen.attention.backends.parallel import (
    _flash_attn_combine as flash_attn_combine,
)
from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.visual_gen.args import AttentionConfig, QuantAttentionConfig

from ._worker_pool import run_distributed

HEAD_DIM = 128
HEADS = 40
FP8_RECIPE = QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8")


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _run(world_size, fn, **kwargs):
    if not torch.cuda.is_available() or get_sm_version() != 107:
        pytest.skip("trtllm-gen dense FP8 attention serves the per-tensor FP8 recipe on SM107")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    run_distributed(world_size, fn, dist_backend="nccl", preload=("tensorrt_llm",), **kwargs)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _config(backend, vgm=None, quant_attention_config=FP8_RECIPE):
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=HEADS * HEAD_DIM,
            num_attention_heads=HEADS,
            attention_head_dim=HEAD_DIM,
            eps=1e-6,
        ),
        attention=AttentionConfig(backend=backend, quant_attention_config=quant_attention_config),
    )
    if vgm is not None:
        config.mapping = vgm.to_llm_mapping()
        config.visual_gen_mapping = vgm
    config.attention_metadata_state = create_attention_metadata_state()
    return config


def _wan_self_attention(config, enable_sequence_parallel, norm_gain=1.0):
    attn = Attention(
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        qkv_mode=QKVMode.FUSE_QKV,
        qk_norm=True,
        qk_norm_mode="full",
        fuse_qk_norm_rope=True,
        config=config,
        enable_sequence_parallel=enable_sequence_parallel,
    ).cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if "norm" in name:
                p.copy_(norm_gain * (1.0 + 0.1 * torch.randn(p.shape, device="cuda", generator=g)))
            else:
                p.copy_(0.02 * torch.randn(p.shape, device="cuda", generator=g))
    return attn


def _inputs(batch, seq):
    g = torch.Generator(device="cuda").manual_seed(11)
    hidden = torch.randn(batch, seq, HEADS * HEAD_DIM, device="cuda", generator=g).bfloat16()
    pos = torch.arange(seq, dtype=torch.float64)
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    angles = torch.outer(pos, inv)
    cos = angles.cos().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    sin = angles.sin().repeat_interleave(2, dim=-1).float().view(1, seq, 1, HEAD_DIM).cuda()
    return hidden, cos, sin


def _logic_ulysses_matches_single_gpu(rank, world_size, *, batch):
    seq = 1560 * world_size
    shard = seq // world_size
    rows = slice(rank * shard, (rank + 1) * shard)
    vgm = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
    attn = _wan_self_attention(_config("TRTLLM", vgm), enable_sequence_parallel=True)
    assert isinstance(attn.attn, UlyssesAttention)
    assert attn._fp8_qkv_producer

    hidden, cos, sin = _inputs(batch, seq)
    local, local_cos, local_sin = hidden[:, rows], cos[:, rows], sin[:, rows]
    with torch.no_grad():
        out = attn(local.contiguous(), freqs=(local_cos, local_sin))

        # One V scale on every rank: the group maximum.
        qkv = attn.qkv_proj(local.contiguous()).view(batch * shard, -1)
        amax = attn.attn.fp8_v_amax(qkv, HEADS, HEAD_DIM)
        gathered = [torch.empty_like(amax) for _ in range(world_size)]
        dist.all_gather(gathered, amax)
        for other in gathered:
            assert torch.equal(other, amax)

        single = _wan_self_attention(_config("TRTLLM"), enable_sequence_parallel=False)
        assert single._fp8_qkv_producer
        single.load_state_dict(attn.state_dict())
        ref = single(hidden, freqs=(cos, sin))[:, rows]
        full_qkv = single.qkv_proj(hidden).view(batch * seq, -1)
        assert torch.equal(amax, torch.ops.trtllm.dit_v_amax(full_qkv, HEADS, HEAD_DIM))

    assert out.shape == ref.shape
    assert _cosine(out, ref) > 0.9999, f"rank {rank}: cosine {_cosine(out, ref)}"


def _logic_ulysses_cuda_graph(rank, world_size):
    batch, seq = 1, 1560 * world_size
    shard = seq // world_size
    rows = slice(rank * shard, (rank + 1) * shard)
    vgm = VisualGenMapping(world_size=world_size, rank=rank, ulysses_size=world_size)
    attn = _wan_self_attention(_config("TRTLLM", vgm), enable_sequence_parallel=True)
    hidden, cos, sin = _inputs(batch, seq)
    local = hidden[:, rows].contiguous()
    freqs = (cos[:, rows].contiguous(), sin[:, rows].contiguous())
    with torch.no_grad():
        eager = attn(local, freqs=freqs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            attn(local, freqs=freqs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attn(local, freqs=freqs)
        for _ in range(10):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, eager), f"rank {rank}: replay differs from eager"


def _logic_attn2d_runs_on_cudnn_fp8(rank, world_size):
    batch, seq = 1, 1560 * world_size
    shard = seq // world_size
    rows = slice(rank * shard, (rank + 1) * shard)
    vgm = VisualGenMapping(world_size=world_size, rank=rank, attn2d_row_size=2, attn2d_col_size=2)
    # Peaked attention (QK norm gain 2): each rank's partial output differs from the others, so
    # the result depends on merging the partials with the right LSE.
    attn = _wan_self_attention(_config("TRTLLM", vgm), enable_sequence_parallel=True, norm_gain=2.0)
    assert isinstance(attn.attn, Attention2DAttention)
    assert isinstance(attn.attn.inner_backend, CuDNNAttention)
    assert not attn._fp8_qkv_producer

    hidden, cos, sin = _inputs(batch, seq)
    with torch.no_grad():
        out = attn(hidden[:, rows].contiguous(), freqs=(cos[:, rows], sin[:, rows]))
        single = _wan_self_attention(
            _config("CUDNN", quant_attention_config=None), enable_sequence_parallel=False
        )
        single.load_state_dict(attn.state_dict())
        ref = single(hidden, freqs=(cos, sin))[:, rows]
    # BF16 single-GPU reference: single-GPU cuDNN FP8 reaches ~0.995 here and a mis-merged LSE ~0.85.
    assert _cosine(out, ref) > 0.99, f"rank {rank}: cosine {_cosine(out, ref)}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("batch", [1, 2])
def test_ulysses_fp8_matches_single_gpu(world_size, batch):
    _run(world_size, _logic_ulysses_matches_single_gpu, batch=batch)


def test_ulysses_fp8_cuda_graph_replay():
    _run(2, _logic_ulysses_cuda_graph)


@pytest.mark.skipif(flash_attn_combine is None, reason="Attention2D needs flash_attn_combine.")
def test_attn2d_fp8_runs_on_cudnn():
    _run(4, _logic_attn2d_runs_on_cudnn_fp8)
