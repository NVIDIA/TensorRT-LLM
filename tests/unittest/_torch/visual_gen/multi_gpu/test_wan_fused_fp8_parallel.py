# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wan fused FP8 self-attention under Ulysses and Attention2D sequence parallelism."""

import math
import os
from types import SimpleNamespace

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.visual_gen.attention_backend import parallel as parallel_backend
from tensorrt_llm._torch.visual_gen.config import (
    AttentionConfig,
    DiffusionModelConfig,
    TorchCompileConfig,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel
from tensorrt_llm._torch.visual_gen.modules.wan_fused_fp8 import ops as fused_ops
from tensorrt_llm._torch.visual_gen.modules.wan_fused_fp8 import ulysses_overlap
from tensorrt_llm._torch.visual_gen.modules.wan_fused_fp8._common import v_scale_from_amax
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ._visual_gen_dist_utils import spawn_with_retry

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="Wan fused FP8 kernels target Rubin (SM107).",
)

_HEAD_DIM = 128
_WAN_CONFIG = dict(
    num_attention_heads=8,
    attention_head_dim=_HEAD_DIM,
    num_layers=2,
    in_channels=16,
    out_channels=16,
    text_dim=256,
    freq_dim=256,
    ffn_dim=512,
    patch_size=[1, 2, 2],
    eps=1e-6,
    cross_attn_norm=True,
)
# Patchified sequence 4 * 8 * 8 = 256 tokens.
_LATENT_SHAPE = (1, 16, 4, 16, 16)
_TEXT_SEQ = 32


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _worker(rank, world_size, test_fn, port, kwargs):
    os.environ.update(
        MASTER_ADDR="localhost", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size)
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size, **kwargs)
    finally:
        dist.destroy_process_group()


def _run_distributed(world_size, test_fn, **kwargs):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs.")
    spawn_with_retry(
        lambda port: mp.spawn(
            _worker, args=(world_size, test_fn, port, kwargs), nprocs=world_size, join=True
        )
    )


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _attention_inputs(batch, seq, heads, device, seed, unit_norm=False):
    """Packed qkv, QK norm weights and interleaved RoPE tables from a per-device generator."""
    dim = heads * _HEAD_DIM
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv = (torch.randn(batch, seq, 3 * dim, device=device, generator=gen) * 3).to(torch.bfloat16)
    if unit_norm:
        norm_q_w = torch.ones(dim, device=device, dtype=torch.bfloat16)
        norm_k_w = torch.ones(dim, device=device, dtype=torch.bfloat16)
    else:
        norm_q_w = (torch.rand(dim, device=device, generator=gen) + 0.5).to(torch.bfloat16)
        norm_k_w = (torch.rand(dim, device=device, generator=gen) + 0.5).to(torch.bfloat16)
    angle = torch.rand(seq, _HEAD_DIM // 2, device=device, generator=gen) * 2 * math.pi
    cos = torch.repeat_interleave(angle.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angle.sin(), 2, dim=-1)
    return qkv, norm_q_w, norm_k_w, cos, sin


def _ulysses_ops_logic(rank, world_size, batch, seq, heads):
    device = torch.device("cuda", rank)
    qkv, norm_q_w, norm_k_w, cos, sin = _attention_inputs(batch, seq, heads, device, seed=0)
    args = (heads, 1e-6, True)

    ref = torch.ops.wanfused.fp8_self_attention(qkv, norm_q_w, norm_k_w, cos, sin, *args)
    local = slice(rank * seq // world_size, (rank + 1) * seq // world_size)
    out = fused_ops.fp8_self_attention_ulysses(
        qkv[:, local].contiguous(),
        norm_q_w,
        norm_k_w,
        cos[local].contiguous(),
        sin[local].contiguous(),
        *args,
        dist.group.WORLD,
    )
    expected = ref[:, local]
    # Same kernels and scales; only head placement differs across ranks.
    assert _cosine(out, expected) > 0.99999
    assert (out.float() - expected.float()).abs().max().item() < 0.05


@pytest.mark.parametrize("batch", [1, 2])
def test_fused_fp8_attention_ulysses_matches_single_gpu(batch):
    _run_distributed(2, _ulysses_ops_logic, batch=batch, seq=4096, heads=40)


def _ulysses_overlap_logic(rank, world_size, batch, seq, heads, groups):
    device = torch.device("cuda", rank)
    qkv, norm_q_w, norm_k_w, cos, sin = _attention_inputs(
        batch, seq, heads, device, seed=rank, unit_norm=True
    )
    args = (qkv, norm_q_w, norm_k_w, cos, sin, heads, 1e-6, True)
    pg = dist.group.WORLD
    expected = fused_ops.fp8_self_attention_ulysses(*args, pg)

    # Lag rank 1 after the V amax all-reduce, which would otherwise absorb the lag,
    # so rank 0 reaches the peer copies first.
    quant_into = ulysses_overlap.norm_rope_quant_into

    def lagged_quant_into(*a, **kw):
        if rank == 1:
            torch.cuda._sleep(100_000_000)
        return quant_into(*a, **kw)

    ulysses_overlap.norm_rope_quant_into = lagged_quant_into
    try:
        # Two calls on new data: the second reuses buffers with a new flag epoch, so a
        # missing flag wait reads the first call's data and corrupts out.
        for i in range(2):
            step_args = _attention_inputs(
                batch, seq, heads, device, seed=rank + world_size * (i + 1), unit_norm=True
            )
            step_args = (*step_args, heads, 1e-6, True)
            step_expected = fused_ops.fp8_self_attention_ulysses(*step_args, pg)
            out = ulysses_overlap.fp8_self_attention_ulysses_overlap(
                *step_args, pg.group_name, groups
            )
            torch.cuda.synchronize()
            # Same kernels and scales; only data movement differs.
            assert torch.equal(out, step_expected)
    finally:
        ulysses_overlap.norm_rope_quant_into = quant_into
    # The group is on one node, so the calls above overlapped.
    assert ulysses_overlap._single_node_groups[pg.group_name]

    # CUDA graph capture takes the all-to-all path, which replays correctly.
    overlap_args = (*args, pg.group_name, groups)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ulysses_overlap.fp8_self_attention_ulysses_overlap(*overlap_args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ulysses_overlap.fp8_self_attention_ulysses_overlap(*overlap_args)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)

    # A group spanning nodes takes the all-to-all path.
    ulysses_overlap._single_node_groups[pg.group_name] = False
    out = ulysses_overlap.fp8_self_attention_ulysses_overlap(*overlap_args)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("batch", [1, 2])
def test_fused_fp8_attention_ulysses_overlap_matches_all_to_all(batch):
    _run_distributed(2, _ulysses_overlap_logic, batch=batch, seq=2048, heads=40, groups=2)


def _attn2d_groups(rank, world_size, parallel):
    vgm = VisualGenMapping(world_size=world_size, rank=rank, **parallel)
    ulysses = vgm.ulysses_group if parallel.get("ulysses_size", 1) > 1 else None
    return ulysses, vgm.attn2d_row_group, vgm.attn2d_col_group


def _exact_fp8_attention(qkv, norm_q_w, norm_k_w, cos, sin, heads, rows):
    """FP32 attention over the fused path's FP8 Q/K/V for query rows."""
    scale_v = v_scale_from_amax(fused_ops.v_amax(qkv, heads))
    q8, k8, v8 = fused_ops.norm_rope_quant_fp8(
        qkv, norm_q_w, norm_k_w, cos, sin, heads, 1e-6, True, 1.0 / scale_v
    )
    # Q/K scales fold the softmax scale, so q.k is in log2 units.
    logits = torch.einsum("bqhd,bkhd->bhqk", q8[:, rows].float(), k8.float()) * math.log(2.0)
    out = torch.einsum("bhqk,bkhd->bqhd", logits.softmax(-1), v8.float() * scale_v)
    return out.flatten(2)


def _attn2d_ops_logic(rank, world_size, parallel, batch, seq, heads):
    device = torch.device("cuda", rank)
    qkv, norm_q_w, norm_k_w, cos, sin = _attention_inputs(batch, seq, heads, device, seed=0)
    args = (heads, 1e-6, True)

    ref = torch.ops.wanfused.fp8_self_attention(qkv, norm_q_w, norm_k_w, cos, sin, *args)
    # Any shard placement works: each rank's rows see every K/V shard once.
    local = slice(rank * seq // world_size, (rank + 1) * seq // world_size)
    out = fused_ops.fp8_self_attention_attn2d(
        qkv[:, local].contiguous(),
        norm_q_w,
        norm_k_w,
        cos[local].contiguous(),
        sin[local].contiguous(),
        *args,
        *_attn2d_groups(rank, world_size, parallel),
    )
    expected = ref[:, local]
    exact = _exact_fp8_attention(qkv, norm_q_w, norm_k_w, cos, sin, heads, local)
    # Split KV re-rounds FP8 P per partial, so bitwise-close is not expected;
    # require single-GPU accuracy against exact attention on the same FP8 Q/K/V.
    assert 1 - _cosine(out, exact) <= 1.25 * (1 - _cosine(expected, exact)) + 1e-6
    assert _cosine(out, expected) > 0.999
    assert (out.float() - expected.float()).abs().max().item() < 0.05


_ATTN2D_PARALLEL = [
    dict(attn2d_row_size=1, attn2d_col_size=2),
    dict(attn2d_row_size=2, attn2d_col_size=2),
    dict(attn2d_row_size=1, attn2d_col_size=2, ulysses_size=2),
]
_ATTN2D_IDS = ["attn2d_1x2", "attn2d_2x2", "attn2d_1x2_ulysses2"]
_needs_combine = pytest.mark.skipif(
    parallel_backend._flash_attn_combine is None, reason="Attention2D needs flash_attn_combine."
)


@_needs_combine
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("parallel", _ATTN2D_PARALLEL, ids=_ATTN2D_IDS)
def test_fused_fp8_attention_attn2d_matches_single_gpu(parallel, batch):
    world_size = math.prod(parallel.values())
    _run_distributed(
        world_size, _attn2d_ops_logic, parallel=parallel, batch=batch, seq=4096, heads=40
    )


def _wan_model(device, **parallel):
    use_dist = any(v > 1 for v in parallel.values())
    vgm = VisualGenMapping(
        world_size=dist.get_world_size() if use_dist else 1,
        rank=dist.get_rank() if use_dist else 0,
        **parallel,
    )
    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(**_WAN_CONFIG),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(
            backend="CUDNN",
            quant_attention_config=QuantAttentionConfig(qk_dtype="fp8", v_dtype="fp8"),
        ),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=False,
        extra_attrs={"fused_fp8": "auto"},
    )
    config.mapping = vgm.to_llm_mapping()
    torch.manual_seed(42)
    model = WanTransformer3DModel(config).to(device).to(torch.bfloat16)
    # Small init keeps BF16 activations bounded across blocks.
    with torch.no_grad():
        for param in model.parameters():
            bound = 0.02 / max(1.0, param.shape[1] ** 0.5) if param.ndim >= 2 else 0.01
            param.uniform_(-bound, bound)
    # fused_fp8=auto with cuDNN FP8 attention on Rubin turns fused attention on.
    assert all(block.attn1.fused_self_attention is not None for block in model.blocks)
    return model


def _wan_forward(model, device):
    torch.manual_seed(100)
    latents = torch.randn(_LATENT_SHAPE, device=device, dtype=torch.bfloat16) * 0.1
    text = (
        torch.randn(1, _TEXT_SEQ, _WAN_CONFIG["text_dim"], device=device, dtype=torch.bfloat16)
        * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        return model(hidden_states=latents, timestep=timestep, encoder_hidden_states=text)


def _wan_ulysses_logic(rank, world_size):
    device = torch.device("cuda", rank)
    single = _wan_forward(_wan_model(device), device)
    model = _wan_model(device, ulysses_size=world_size)
    assert fused_ops.sp_mode(model.blocks[0].attn1)[0] == "ulysses"
    out = _wan_forward(model, device)
    assert _cosine(out, single) > 0.9999


def test_wan_transformer_fused_attention_ulysses():
    _run_distributed(2, _wan_ulysses_logic)


def _wan_attn2d_logic(rank, world_size, parallel):
    device = torch.device("cuda", rank)
    single = _wan_forward(_wan_model(device), device)
    model = _wan_model(device, **parallel)
    assert fused_ops.sp_mode(model.blocks[0].attn1)[0] == "attn2d"
    out = _wan_forward(model, device)
    assert _cosine(out, single) > 0.9999


@_needs_combine
@pytest.mark.parametrize("parallel", _ATTN2D_PARALLEL, ids=_ATTN2D_IDS)
def test_wan_transformer_fused_attention_attn2d(parallel):
    world_size = math.prod(parallel.values())
    _run_distributed(world_size, _wan_attn2d_logic, parallel=parallel)


def _wan_attn2d_compiled_logic(rank, world_size, parallel):
    device = torch.device("cuda", rank)
    single = _wan_forward(_wan_model(device), device)
    model = _wan_model(device, **parallel)
    # Per-block torch.compile, as VisualGenPipeline.torch_compile does.
    model.blocks = nn.ModuleList(
        torch.compile(block, mode="default", dynamic=None, fullgraph=False)
        for block in model.blocks
    )
    torch._dynamo.reset()
    out = _wan_forward(model, device)
    assert fused_ops.sp_mode(model.blocks[0].attn1)[0] == "attn2d"
    assert _cosine(out, single) > 0.9999
    # After the first call the SP spec is cached: no further graph breaks.
    breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    again = _wan_forward(model, device)
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == breaks
    assert torch.equal(again, out)


@_needs_combine
@pytest.mark.parametrize(
    "parallel",
    [_ATTN2D_PARALLEL[0], _ATTN2D_PARALLEL[2]],
    ids=[_ATTN2D_IDS[0], _ATTN2D_IDS[2]],
)
def test_wan_transformer_fused_attention_attn2d_compiled(parallel):
    world_size = math.prod(parallel.values())
    _run_distributed(world_size, _wan_attn2d_compiled_logic, parallel=parallel)
