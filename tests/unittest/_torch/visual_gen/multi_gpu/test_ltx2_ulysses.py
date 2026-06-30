# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU e2e tests for LTX-2 AudioVideo Ulysses sequence parallelism.

Runs ``LTXModel.forward`` end-to-end with ``ulysses_size=2`` and compares
to the single-GPU reference (same weights, same inputs). Covers self-attn,
v2a, and a2v cross-attn paths simultaneously.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ltx2_ulysses.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import (
        DiffusionModelConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType

    # Reuse the CI-aware free-port allocator from tests/integration so that
    # sequentially spawned distributed workers get disjoint MASTER_PORTs and
    # don't collide with ports still in TIME_WAIT (EADDRINUSE).
    _integration_dir = Path(__file__).resolve().parents[4] / "integration"
    if str(_integration_dir) not in sys.path:
        sys.path.insert(0, str(_integration_dir))
    from defs.common import get_free_port_in_ci

    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import AttentionConfig, ParallelConfig, TorchCompileConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_flux_ulysses.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port, fn_args):
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, *fn_args)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, *fn_args):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    port = get_free_port_in_ci()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port, fn_args),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# Model config
# =============================================================================

# Small AudioVideo config. ulysses_size=2 needs:
# - num_attention_heads % 2 == 0 (video heads sharded by Ulysses)
# - audio_num_attention_heads % 2 == 0 (audio heads sharded by Ulysses)
# - head_dim ∈ {64, 128} so LTX2Attention.forward() takes the fused-rope
#   branch (otherwise eager fallback masks the cos/num_tokens check that
#   exposes the audio-pad seq_dim regression).
# cross_attention_dim must equal inner_dim (=num_heads * head_dim) because
# caption_projection projects caption_channels → inner_dim and the result is
# fed straight into block.attn2.to_k whose input dim = cross_attention_dim.
# Same constraint applies to audio_*.
_AV_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=64,
    in_channels=16,
    out_channels=16,
    num_layers=2,
    cross_attention_dim=256,  # = num_attention_heads * attention_head_dim
    caption_channels=64,
    norm_eps=1e-6,
    positional_embedding_max_pos=[4, 32, 32],
    timestep_scale_multiplier=1000,
    use_middle_indices_grid=True,
    audio_num_attention_heads=4,
    audio_attention_head_dim=64,
    audio_in_channels=16,
    audio_out_channels=16,
    audio_cross_attention_dim=256,  # = audio_num_attention_heads * audio_attention_head_dim
    audio_positional_embedding_max_pos=[64],
    av_ca_timestep_scale_multiplier=1,
    rope_type=LTXRopeType.SPLIT,
)


def _make_model_config(
    ulysses_size: int = 1,
    backend: str = "VANILLA",
) -> "DiffusionModelConfig":
    """Create DiffusionModelConfig for LTX-2 tests."""
    if ulysses_size > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        ws = ulysses_size
        rk = 0
    vgm = VisualGenMapping(world_size=ws, rank=rk, ulysses_size=ulysses_size)

    config = DiffusionModelConfig(
        pretrained_config=SimpleNamespace(),
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        cache=None,
        attention_metadata_state=(
            create_attention_metadata_state() if backend.upper() == "TRTLLM" else None
        ),
        parallel=ParallelConfig(ulysses_size=ulysses_size),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _init_all_weights(model: torch.nn.Module, std: float = 0.02):
    """Init weights to small values — TRT-LLM Linear uses empty() (uninit mem)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and "weight" in name:
                p.fill_(1.0)
            elif p.numel() > 0:
                torch.nn.init.normal_(p, mean=0.0, std=std)


def _make_video_positions(
    batch: int, n_patches: int, n_frames: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    positions = torch.zeros(batch, 3, n_patches, 2, device=device)
    idx = 0
    for f in range(n_frames):
        for h in range(grid_h):
            for w in range(grid_w):
                positions[:, 0, idx, :] = torch.tensor([f, f + 1], dtype=torch.float32)
                positions[:, 1, idx, :] = torch.tensor([h, h + 1], dtype=torch.float32)
                positions[:, 2, idx, :] = torch.tensor([w, w + 1], dtype=torch.float32)
                idx += 1
    return positions


def _make_audio_positions(batch: int, n_patches: int, device: torch.device) -> torch.Tensor:
    positions = torch.zeros(batch, 1, n_patches, 2, device=device)
    for i in range(n_patches):
        positions[:, 0, i, :] = torch.tensor([i, i + 1], dtype=torch.float32)
    return positions


# =============================================================================
# Test logic
# =============================================================================


def _build_inputs(batch, v_patches, v_dims, a_patches, dtype, device, seed=456):
    """Construct identical inputs across all ranks via shared seed."""
    g = torch.Generator(device=device).manual_seed(seed)
    v_frames, v_h, v_w = v_dims
    in_channels = _AV_CONFIG["in_channels"]
    audio_in_channels = _AV_CONFIG["audio_in_channels"]
    caption_channels = _AV_CONFIG["caption_channels"]
    text_len = 8

    v_context = (
        torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype, generator=g)
        * 0.02
    )
    a_context = (
        torch.randn(batch, text_len, caption_channels, device=device, dtype=dtype, generator=g)
        * 0.02
    )
    v_positions = _make_video_positions(batch, v_patches, v_frames, v_h, v_w, device)
    a_positions = _make_audio_positions(batch, a_patches, device)

    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality

    video = Modality(
        latent=torch.randn(batch, v_patches, in_channels, device=device, dtype=dtype, generator=g)
        * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=v_positions,
        context=v_context,
    )
    audio = Modality(
        latent=torch.randn(
            batch, a_patches, audio_in_channels, device=device, dtype=dtype, generator=g
        )
        * 0.02,
        timesteps=torch.tensor([0.5], device=device),
        positions=a_positions,
        context=a_context,
    )
    return video, audio, v_context, a_context, v_positions, a_positions


def _logic_ltx2_av_ulysses_vs_single_gpu(rank, world_size, backend, audio_seq_len):
    """LTX-2 AudioVideo Ulysses (ws=2) vs single-GPU reference parity.

    Builds the same LTXModel weights on both ranks (shared seed), runs:
      1. Reference: ulysses_size=1, no collectives, computed locally.
      2. Ulysses:   ulysses_size=2, collective forward across ranks.
    Compares (video_out, audio_out). Drift comes from BF16 accumulation order.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    batch = 1
    v_dims = (1, 4, 4)  # n_frames, grid_h, grid_w
    v_patches = v_dims[0] * v_dims[1] * v_dims[2]  # 16, divisible by ws=2

    # ── Reference: single-GPU (no Ulysses) ──────────────────────────────────
    torch.manual_seed(123)
    ref_cfg = _make_model_config(ulysses_size=1, backend=backend)
    ref_model = (
        LTXModel(model_type=LTXModelType.AudioVideo, model_config=ref_cfg, **_AV_CONFIG)
        .to(device, dtype=dtype)
        .eval()
    )
    _init_all_weights(ref_model)
    ref_model.configure_audio_ulysses(audio_seq_len)
    ref_state = ref_model.state_dict()

    # ── Ulysses model: same weights ─────────────────────────────────────────
    torch.manual_seed(123)
    u_cfg = _make_model_config(ulysses_size=world_size, backend=backend)
    u_model = (
        LTXModel(model_type=LTXModelType.AudioVideo, model_config=u_cfg, **_AV_CONFIG)
        .to(device, dtype=dtype)
        .eval()
    )
    u_model.load_state_dict(ref_state)
    u_model.configure_audio_ulysses(audio_seq_len)

    # ── Inputs (identical across ranks) ─────────────────────────────────────
    video, audio, v_ctx, a_ctx, v_pos, a_pos = _build_inputs(
        batch, v_patches, v_dims, audio_seq_len, dtype, device
    )

    ref_cache = ref_model.prepare_text_cache(
        video_context=v_ctx,
        video_positions=v_pos,
        audio_context=a_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )
    u_cache = u_model.prepare_text_cache(
        video_context=v_ctx,
        video_positions=v_pos,
        audio_context=a_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )

    with torch.no_grad():
        ref_v, ref_a = ref_model(video=video, audio=audio, text_cache=ref_cache)
        u_v, u_a = u_model(video=video, audio=audio, text_cache=u_cache)

    # Output shapes match (audio padded tail is stripped inside forward()).
    assert ref_v.shape == u_v.shape, f"Rank {rank}: video shape mismatch"
    assert ref_a.shape == u_a.shape, f"Rank {rank}: audio shape mismatch"

    # BF16 drift through 2 transformer layers + Ulysses collectives lands in
    # the ~1-3% range with these small-scale weights; 5e-2 leaves headroom
    # without being so loose that real numerical regressions slip through.
    torch.testing.assert_close(
        u_v,
        ref_v,
        rtol=5e-2,
        atol=5e-2,
        msg=f"Rank {rank}: LTX-2 AV Ulysses video output differs from single-GPU ref",
    )
    torch.testing.assert_close(
        u_a,
        ref_a,
        rtol=5e-2,
        atol=5e-2,
        msg=f"Rank {rank}: LTX-2 AV Ulysses audio output differs from single-GPU ref",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestLTX2AVUlysses:
    """End-to-end Ulysses parity for LTX-2 AudioVideo model."""

    @pytest.mark.parametrize(
        "backend",
        ["VANILLA", "FA4"],
    )
    def test_av_ulysses_no_audio_pad(self, backend):
        """ws=2, audio_seq_len % 2 == 0 — pure sharding, no padding mask."""
        run_test_in_distributed(2, _logic_ltx2_av_ulysses_vs_single_gpu, backend, 16)

    @pytest.mark.parametrize(
        "backend",
        ["VANILLA", "FA4"],
    )
    def test_av_ulysses_audio_pad(self, backend):
        """ws=2, audio_seq_len % 2 != 0 — audio padding + key_padding_mask."""
        run_test_in_distributed(2, _logic_ltx2_av_ulysses_vs_single_gpu, backend, 15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
