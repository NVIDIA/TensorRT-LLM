# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for LTX-2 Tensor Parallelism (TP).

Tests that the LTX-2 (VideoOnly) transformer produces correct outputs when
using TP (sharding weights across GPUs), combined TP + Ulysses, and the
two-stage pipeline pattern (Stage 1: multi-GPU TP, Stage 2: Ulysses disabled,
rank-0 only refinement).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ltx_tp.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_flux_tp.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    """Initialize distributed environment for a worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    """Worker function that runs in each process. Module-level for pickling."""
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    """Run a test function in a distributed environment."""
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")

    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()

    mp.spawn(
        _distributed_worker, args=(world_size, backend, test_fn, port), nprocs=world_size, join=True
    )


# =============================================================================
# Model config helpers
# =============================================================================

# Small LTX-2 VideoOnly config for testing (reduced dims: 4 heads, 32 head_dim = 128 inner_dim)
_LTX_VIDEO_NUM_HEADS = 4
_LTX_VIDEO_HEAD_DIM = 32
_LTX_VIDEO_INNER_DIM = _LTX_VIDEO_NUM_HEADS * _LTX_VIDEO_HEAD_DIM  # 128
_LTX_IN_CHANNELS = 16
_LTX_OUT_CHANNELS = 16
_LTX_CROSS_ATTENTION_DIM = 64
_LTX_CAPTION_CHANNELS = 48
_LTX_NUM_LAYERS = 2


def _make_model_config(tp_size=1, ulysses_size=1, backend="VANILLA"):
    """Create DiffusionModelConfig for testing with TP and/or Ulysses."""
    ws = tp_size * ulysses_size
    if ws > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        rk = 0
    vgm = VisualGenMapping(world_size=ws, rank=rk, tp_size=tp_size, ulysses_size=ulysses_size)

    config = DiffusionModelConfig(
        pretrained_config=None,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        cache=None,
        attention_metadata_state=(
            create_attention_metadata_state() if backend.upper() == "TRTLLM" else None
        ),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _make_ltx_model(model_config):
    """Instantiate a small LTX-2 VideoOnly model for testing."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    return LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_attention_heads=_LTX_VIDEO_NUM_HEADS,
        attention_head_dim=_LTX_VIDEO_HEAD_DIM,
        in_channels=_LTX_IN_CHANNELS,
        out_channels=_LTX_OUT_CHANNELS,
        num_layers=_LTX_NUM_LAYERS,
        cross_attention_dim=_LTX_CROSS_ATTENTION_DIM,
        norm_eps=1e-6,
        caption_channels=_LTX_CAPTION_CHANNELS,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4, 8, 8],
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        apply_gated_attention=False,
        model_config=model_config,
    )


def _stabilize_model_weights(model):
    """Reinitialize model weights for stable BF16 forward pass."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                fan_in = p.shape[1] if p.ndim >= 2 else p.shape[0]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


def _make_test_inputs(device, compute_dtype, video_seq=16, seed=456):
    """Create deterministic test inputs for LTX-2 VideoOnly forward pass."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.modality import Modality

    batch = 1
    txt_seq = 8

    torch.manual_seed(seed)
    # Modality inputs
    latent = (
        torch.randn(batch, video_seq, _LTX_IN_CHANNELS, device=device, dtype=compute_dtype) * 0.1
    )
    timesteps = torch.tensor([0.5], device=device, dtype=compute_dtype)
    # positions: (B, n_dims, T) — 3 dims for video (T, H, W)
    positions = (
        torch.arange(video_seq, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch, 3, -1)
        .float()
    )
    # Text context
    context = (
        torch.randn(batch, txt_seq, _LTX_CAPTION_CHANNELS, device=device, dtype=compute_dtype) * 0.1
    )

    video = Modality(
        latent=latent,
        timesteps=timesteps,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
    )
    return video


# =============================================================================
# TP weight sharding helpers
# =============================================================================


def _shard_dim0(tensor, tp_rank, tp_size):
    """Shard a tensor along dim 0."""
    chunk = tensor.shape[0] // tp_size
    return tensor[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_dim1(tensor, tp_rank, tp_size):
    """Shard a tensor along dim 1."""
    chunk = tensor.shape[1] // tp_size
    return tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_fused_qkv(tensor, tp_rank, tp_size, q_dim, kv_dim):
    """Shard a fused QKV weight [q_dim + 2*kv_dim, ...] preserving Q/K/V structure."""
    q, k, v = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
    return torch.cat(
        [
            _shard_dim0(q, tp_rank, tp_size),
            _shard_dim0(k, tp_rank, tp_size),
            _shard_dim0(v, tp_rank, tp_size),
        ],
        dim=0,
    )


def _shard_fused_gate_up(tensor, tp_rank, tp_size):
    """Shard a fused gate_up weight [2*intermediate, ...] preserving gate/up structure."""
    half = tensor.shape[0] // 2
    gate, up = tensor.split([half, half], dim=0)
    return torch.cat(
        [
            _shard_dim0(gate, tp_rank, tp_size),
            _shard_dim0(up, tp_rank, tp_size),
        ],
        dim=0,
    )


def _copy_ref_weights_to_tp(ref_model, tp_model, tp_rank, tp_size):
    """Copy weights from a TP=1 reference model into a TP model with correct sharding."""
    ref_params = dict(ref_model.named_parameters())

    with torch.no_grad():
        for tp_name, tp_param in tp_model.named_parameters():
            if tp_name not in ref_params:
                continue

            ref_param = ref_params[tp_name]

            if tp_param.shape == ref_param.shape:
                tp_param.data.copy_(ref_param.data)
            elif tp_param.ndim >= 2 and tp_param.shape[1] == ref_param.shape[1]:
                # Column parallel: dim 0 is smaller
                if "qkv_proj" in tp_name:
                    q_dim = ref_param.shape[0] // 3
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, q_dim)
                    )
                elif "gate_up_proj" in tp_name or "up_proj" in tp_name:
                    if ref_param.shape[0] == tp_param.shape[0] * tp_size * 2 // (tp_size):
                        # Not actually fused gate_up — just column parallel
                        tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
                    else:
                        tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim >= 2 and tp_param.shape[0] == ref_param.shape[0]:
                # Row parallel: dim 1 is smaller
                tp_param.data.copy_(_shard_dim1(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim == 1 and tp_param.shape[0] < ref_param.shape[0]:
                # 1D bias for column parallel
                if "qkv_proj" in tp_name:
                    q_dim = ref_param.shape[0] // 3
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, q_dim)
                    )
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            else:
                raise ValueError(
                    f"Cannot shard {tp_name}: ref={ref_param.shape}, tp={tp_param.shape}"
                )


# =============================================================================
# LTX-2 test logic functions (module-level for pickling)
# =============================================================================


def _logic_ltx_tp_forward(rank, world_size):
    """LTX-2 VideoOnly transformer forward with TP — verify shape and no NaN/Inf."""

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16
    torch.manual_seed(42)

    model_config = _make_model_config(tp_size=world_size)
    model = _make_ltx_model(model_config).to(device).to(compute_dtype)
    _stabilize_model_weights(model)

    video = _make_test_inputs(device, compute_dtype)

    # Prepare text cache (pre-computes text projections and RoPE)
    with torch.no_grad():
        text_cache = model.prepare_text_cache(
            video_context=video.context,
            video_context_mask=video.context_mask,
            video_positions=video.positions,
            dtype=compute_dtype,
        )

        video_out, audio_out = model(
            video=video,
            audio=None,
            text_cache=text_cache,
        )

    assert audio_out is None, f"Rank {rank}: Expected no audio output for VideoOnly model"
    assert video_out is not None, f"Rank {rank}: Expected video output"
    assert video_out.shape == video.latent.shape, (
        f"Rank {rank}: Expected shape {video.latent.shape}, got {video_out.shape}"
    )
    assert not torch.isnan(video_out).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(video_out).any(), f"Rank {rank}: Inf in output"


def _logic_ltx_tp_vs_single_gpu(rank, world_size):
    """LTX-2 VideoOnly: TP 2-GPU output matches single-GPU reference."""

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(tp_size=1)
    ref_model = _make_ltx_model(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create TP model and copy sharded weights from ref
    torch.manual_seed(123)
    tp_config = _make_model_config(tp_size=world_size)
    tp_model = _make_ltx_model(tp_config).to(device).to(compute_dtype)
    _copy_ref_weights_to_tp(ref_model, tp_model, rank, world_size)

    video = _make_test_inputs(device, compute_dtype)

    with torch.no_grad():
        ref_text_cache = ref_model.prepare_text_cache(
            video_context=video.context,
            video_context_mask=video.context_mask,
            video_positions=video.positions,
            dtype=compute_dtype,
        )
        ref_out, _ = ref_model(video=video, audio=None, text_cache=ref_text_cache)

        tp_text_cache = tp_model.prepare_text_cache(
            video_context=video.context,
            video_context_mask=video.context_mask,
            video_positions=video.positions,
            dtype=compute_dtype,
        )
        tp_out, _ = tp_model(video=video, audio=None, text_cache=tp_text_cache)

    torch.testing.assert_close(
        tp_out,
        ref_out,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: LTX-2 TP output differs from single-GPU reference",
    )


def _logic_ltx_tp_ulysses_vs_single_gpu(rank, world_size):
    """LTX-2 VideoOnly: TP=2 + Ulysses=2 (4 GPUs) output matches single-GPU reference."""

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16
    tp_size = 2
    ulysses_size = 2

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(tp_size=1, ulysses_size=1)
    ref_model = _make_ltx_model(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create combined TP + Ulysses model and copy sharded weights
    torch.manual_seed(123)
    combined_config = _make_model_config(tp_size=tp_size, ulysses_size=ulysses_size)
    combined_model = _make_ltx_model(combined_config).to(device).to(compute_dtype)
    vgm = combined_config.visual_gen_mapping
    _copy_ref_weights_to_tp(ref_model, combined_model, vgm.tp_rank, tp_size)

    video = _make_test_inputs(device, compute_dtype)

    with torch.no_grad():
        ref_text_cache = ref_model.prepare_text_cache(
            video_context=video.context,
            video_context_mask=video.context_mask,
            video_positions=video.positions,
            dtype=compute_dtype,
        )
        ref_out, _ = ref_model(video=video, audio=None, text_cache=ref_text_cache)

        combined_text_cache = combined_model.prepare_text_cache(
            video_context=video.context,
            video_context_mask=video.context_mask,
            video_positions=video.positions,
            dtype=compute_dtype,
        )
        combined_out, _ = combined_model(video=video, audio=None, text_cache=combined_text_cache)

    torch.testing.assert_close(
        combined_out,
        ref_out,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: LTX-2 TP+Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# Two-stage pipeline test logic (module-level for pickling)
#
# Simulates the LTX2TwoStagesPipeline pattern:
#   Stage 1: TP + Ulysses forward at lower resolution
#   Stage 2: set_ulysses_enabled(False), rank-0 only refinement forward
# =============================================================================


def _logic_ltx_two_stage_tp_ulysses(rank, world_size):
    """LTX-2 two-stage: Stage 1 with TP+Ulysses, Stage 2 with Ulysses disabled on rank 0.

    Verifies that:
    1. Stage 1 TP+Ulysses forward produces valid output matching single-GPU ref.
    2. After set_ulysses_enabled(False), rank 0 can run a second forward pass
       (simulating refinement denoising on upsampled latents) that matches a
       single-GPU reference.
    """

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16
    tp_size = 2
    ulysses_size = 2

    # --- Build single-GPU reference ---
    torch.manual_seed(123)
    ref_config = _make_model_config(tp_size=1, ulysses_size=1)
    ref_model = _make_ltx_model(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # --- Build TP + Ulysses model ---
    torch.manual_seed(123)
    combined_config = _make_model_config(tp_size=tp_size, ulysses_size=ulysses_size)
    combined_model = _make_ltx_model(combined_config).to(device).to(compute_dtype)
    vgm = combined_config.visual_gen_mapping
    _copy_ref_weights_to_tp(ref_model, combined_model, vgm.tp_rank, tp_size)

    # ====================================================================
    # Stage 1: TP + Ulysses forward (lower resolution)
    # ====================================================================
    video_s1 = _make_test_inputs(device, compute_dtype, video_seq=16, seed=456)

    with torch.no_grad():
        ref_tc_s1 = ref_model.prepare_text_cache(
            video_context=video_s1.context,
            video_context_mask=video_s1.context_mask,
            video_positions=video_s1.positions,
            dtype=compute_dtype,
        )
        ref_out_s1, _ = ref_model(video=video_s1, audio=None, text_cache=ref_tc_s1)

        combined_tc_s1 = combined_model.prepare_text_cache(
            video_context=video_s1.context,
            video_context_mask=video_s1.context_mask,
            video_positions=video_s1.positions,
            dtype=compute_dtype,
        )
        combined_out_s1, _ = combined_model(video=video_s1, audio=None, text_cache=combined_tc_s1)

    torch.testing.assert_close(
        combined_out_s1,
        ref_out_s1,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: Stage 1 TP+Ulysses output differs from single-GPU reference",
    )

    # ====================================================================
    # Stage 2: Disable Ulysses, rank 0 only refinement
    # (Mirrors pipeline_ltx2_two_stages.py: set_ulysses_enabled(False))
    # ====================================================================
    combined_model.set_ulysses_enabled(False)

    if rank == 0:
        # Stage 2 uses different (upsampled) latents — simulate with larger seq
        video_s2 = _make_test_inputs(device, compute_dtype, video_seq=32, seed=789)

        with torch.no_grad():
            ref_tc_s2 = ref_model.prepare_text_cache(
                video_context=video_s2.context,
                video_context_mask=video_s2.context_mask,
                video_positions=video_s2.positions,
                dtype=compute_dtype,
            )
            ref_out_s2, _ = ref_model(video=video_s2, audio=None, text_cache=ref_tc_s2)

            combined_tc_s2 = combined_model.prepare_text_cache(
                video_context=video_s2.context,
                video_context_mask=video_s2.context_mask,
                video_positions=video_s2.positions,
                dtype=compute_dtype,
            )
            combined_out_s2, _ = combined_model(
                video=video_s2, audio=None, text_cache=combined_tc_s2
            )

        assert combined_out_s2.shape == video_s2.latent.shape, (
            f"Stage 2: Expected shape {video_s2.latent.shape}, got {combined_out_s2.shape}"
        )
        assert not torch.isnan(combined_out_s2).any(), "Stage 2: NaN in output"
        assert not torch.isinf(combined_out_s2).any(), "Stage 2: Inf in output"

        torch.testing.assert_close(
            combined_out_s2,
            ref_out_s2,
            rtol=1e-2,
            atol=1e-2,
            msg="Stage 2: rank-0 output after disabling Ulysses differs from single-GPU ref",
        )

    # Restore Ulysses (mirrors pipeline finally block)
    combined_model.set_ulysses_enabled(True)
    dist.barrier()


# =============================================================================
# Test classes
# =============================================================================


class TestLTXSingleStageTP:
    """Single-stage TP tests for LTX-2 VideoOnly transformer."""

    def test_ltx_tp_forward(self):
        """LTX-2 TP forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ltx_tp_forward)

    def test_ltx_tp_vs_single_gpu(self):
        """LTX-2 TP 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ltx_tp_vs_single_gpu)


class TestLTXSingleStageTPUlyssesCombined:
    """Single-stage combined TP + Ulysses tests for LTX-2 VideoOnly transformer."""

    def test_ltx_tp_ulysses_vs_single_gpu(self):
        """LTX-2 TP=2 + Ulysses=2 (4 GPUs) matches single-GPU reference."""
        run_test_in_distributed(world_size=4, test_fn=_logic_ltx_tp_ulysses_vs_single_gpu)


class TestLTXTwoStageTP:
    """Two-stage pipeline pattern tests for LTX-2 VideoOnly transformer.

    Simulates the LTX2TwoStagesPipeline behavior:
      Stage 1: TP + Ulysses forward at lower resolution
      Stage 2: Ulysses disabled, rank-0 only refinement forward
    """

    def test_ltx_two_stage_tp_ulysses(self):
        """LTX-2 two-stage: Stage 1 TP+Ulysses, Stage 2 Ulysses-off rank-0 refinement."""
        run_test_in_distributed(world_size=4, test_fn=_logic_ltx_two_stage_tp_ulysses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
