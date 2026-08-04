# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for LTX-2 async-Ulysses parity and perf.

Compares ``LTXModel.forward`` outputs with ``async_ulysses=True`` (V/Q/K
rolling A2A via ``Attention.forward_async``) vs ``async_ulysses=False``
(standard sync Ulysses) at ``ulysses_size=2``. Both paths wrap the inner
backend with ``UlyssesAttention`` and produce mathematically equivalent
results — drift comes only from per-kernel accumulation order under BF16.

Mirrors ``test_ltx2_ulysses.py`` (PR 14044) but specifically exercises the
async-Ulysses code paths added on this branch:
  - ``LTX2Attention.forward`` async self-attn dispatch
  - ``LTX2Attention.forward_async`` (fused split kernel or naive fallback)
  - ``UlyssesAttention.forward_async`` (V/Q/K rolling side-stream pipeline)

Uses ``attention_head_dim=64`` so ``LTX2Attention.forward_async`` takes
the fused split kernel path (``apply_split_norm_rope``) — the prod code
path. Requires LTX-2 C++ extensions to be built.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ltx2_async_ulysses.py -v
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
    from tensorrt_llm._torch.visual_gen.config import (
        DiffusionModelConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port
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
# Distributed helpers (same pattern as test_ltx2_ulysses.py)
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
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port, fn_args),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# Model config
# =============================================================================

# Small AudioVideo config. head_dim=64 matches the fused split kernel's
# {64, 128} template — exercises ``LTX2Attention.forward_async`` fused
# branch (apply_split_norm_rope) rather than the naive fallback.
# cross_attention_dim must equal num_heads * head_dim (= 4 * 64 = 256) so
# caption_projection's output matches block.attn2.to_k's input dim. Same
# constraint for audio_*.
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
)


def _make_model_config(
    ulysses_size: int = 1,
    backend: str = "VANILLA",
    async_ulysses: bool = False,
) -> "DiffusionModelConfig":
    """Create DiffusionModelConfig for LTX-2 tests.

    The async_ulysses flag toggles the V/Q/K rolling A2A pipeline (this PR's
    feature). ulysses_size > 1 is required for async_ulysses to fire.
    """
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
        parallel=ParallelConfig(
            ulysses_size=ulysses_size,
            async_ulysses=async_ulysses,
        ),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _init_all_weights(model: torch.nn.Module, std: float = 0.02):
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


def _build_inputs(batch, v_patches, v_dims, a_patches, dtype, device, seed=456):
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


def _pack_async_state_for_sync(async_state, sync_state_keys):
    """Translate a SEPARATE_QKV (async) state_dict to FUSE_QKV (sync) layout:
    concatenates to_q/to_k/to_v.{weight,bias} into packed qkv_proj.{weight,bias}.
    Non-self-attn weights pass through unchanged.

    Used to load identical effective weights into both async and sync LTX
    models despite their different attn1 layer structure.
    """
    out = {}
    for k in sync_state_keys:
        if k in async_state:
            out[k] = async_state[k]
        elif k.endswith(".qkv_proj.weight"):
            prefix = k[: -len(".qkv_proj.weight")]
            out[k] = torch.cat(
                [
                    async_state[f"{prefix}.to_q.weight"],
                    async_state[f"{prefix}.to_k.weight"],
                    async_state[f"{prefix}.to_v.weight"],
                ],
                dim=0,
            )
        elif k.endswith(".qkv_proj.bias"):
            prefix = k[: -len(".qkv_proj.bias")]
            out[k] = torch.cat(
                [
                    async_state[f"{prefix}.to_q.bias"],
                    async_state[f"{prefix}.to_k.bias"],
                    async_state[f"{prefix}.to_v.bias"],
                ],
                dim=0,
            )
        else:
            raise KeyError(f"sync key {k!r} not present in async state and not a qkv_proj fuse")
    return out


def _build_av_model(
    world_size: int,
    backend: str,
    audio_seq_len: int,
    async_ulysses: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 123,
):
    """Build LTXModel (AudioVideo) with deterministic weights via shared seed.

    ``configure_audio_ulysses(audio_seq_len)`` gates audio_attn1's Ulysses
    activity by divisibility: not divisible → ``set_ulysses_active(False)``
    swaps the audio backend to plain (no ``forward_async``), forcing async
    self-attn to fall through the ``hasattr`` guard in ``LTX2Attention.forward``.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    torch.manual_seed(seed)
    cfg = _make_model_config(
        ulysses_size=world_size,
        backend=backend,
        async_ulysses=async_ulysses,
    )
    model = (
        LTXModel(model_type=LTXModelType.AudioVideo, model_config=cfg, **_AV_CONFIG)
        .to(device, dtype=dtype)
        .eval()
    )
    _init_all_weights(model)
    model.configure_audio_ulysses(audio_seq_len)
    return model


# =============================================================================
# Test logic
# =============================================================================


def _logic_async_vs_sync_parity(rank, world_size, backend, audio_seq_len):
    """LTX-2 AV at ws=2: async-Ulysses output matches sync-Ulysses output.

    Both models use ``UlyssesAttention`` (forced by ``ulysses_size=2``); the
    only difference is whether self-attn dispatches through
    ``Attention.forward_async`` (V/Q/K closures + side-stream A2A) or
    ``Attention.forward`` (precomputed Q/K/V + sync A2A). Math is equivalent;
    BF16 accumulation order drift lands well under the 5e-2 tolerance.
    """
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    batch = 1
    v_dims = (1, 4, 4)
    v_patches = v_dims[0] * v_dims[1] * v_dims[2]  # 16, divisible by ws=2

    # Build async (SEPARATE_QKV self-attn → to_q/k/v) first as the canonical
    # source, then mirror its weights into a sync (FUSE_QKV → qkv_proj) model
    # by concatenating to_q/k/v into the packed layout.
    async_model = _build_av_model(world_size, backend, audio_seq_len, True, device, dtype)
    async_state = async_model.state_dict()
    sync_model = _build_av_model(world_size, backend, audio_seq_len, False, device, dtype)
    sync_model.load_state_dict(
        _pack_async_state_for_sync(async_state, sync_model.state_dict().keys())
    )

    # Sanity: confirm sync vs async actually take different code paths.
    # transformer_blocks may be wrapped by LTX2CacheDiTPattern0BlockWrapper;
    # unwrap via .inner if present.
    def _block_attn1(model):
        b = model.transformer_blocks[0]
        b = b.inner if hasattr(b, "inner") else b
        return b.attn1

    assert _block_attn1(async_model)._use_async_ulysses is True, (
        "async model didn't enable async path"
    )
    assert _block_attn1(sync_model)._use_async_ulysses is False, (
        "sync model unexpectedly enabled async path"
    )
    assert _block_attn1(async_model).qkv_mode != _block_attn1(sync_model).qkv_mode, (
        "test bug: sync and async models have identical qkv_mode "
        f"(both {_block_attn1(sync_model).qkv_mode})"
    )

    video, audio, v_ctx, a_ctx, v_pos, a_pos = _build_inputs(
        batch, v_patches, v_dims, audio_seq_len, dtype, device
    )

    sync_cache = sync_model.prepare_text_cache(
        video_context=v_ctx,
        video_positions=v_pos,
        audio_context=a_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )
    async_cache = async_model.prepare_text_cache(
        video_context=v_ctx,
        video_positions=v_pos,
        audio_context=a_ctx,
        audio_positions=a_pos,
        dtype=dtype,
    )

    with torch.no_grad():
        sync_v, sync_a = sync_model(video=video, audio=audio, text_cache=sync_cache)
        async_v, async_a = async_model(video=video, audio=audio, text_cache=async_cache)

    assert sync_v.shape == async_v.shape, f"Rank {rank}: video shape mismatch"
    assert sync_a.shape == async_a.shape, f"Rank {rank}: audio shape mismatch"

    # Diagnostic: actual BF16 drift between sync (packed kernel + sync a2a) and
    # async (split kernel + side-stream a2a) — both at ws=2, same collectives.
    for name, s, a in [("video", sync_v, async_v), ("audio", sync_a, async_a)]:
        diff = (a.float() - s.float()).abs()
        ref = s.float().abs()
        print(
            f"\n[LTX-2 rank={rank} backend={backend} {name}] "
            f"max_abs_diff={diff.max().item():.3e} "
            f"max_rel_diff={(diff / ref.clamp(min=1e-6)).max().item():.3e} "
            f"sync_abs_max={ref.max().item():.3e}"
        )

    torch.testing.assert_close(
        async_v,
        sync_v,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: LTX-2 AV async-Ulysses video differs from sync-Ulysses",
    )
    torch.testing.assert_close(
        async_a,
        sync_a,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: LTX-2 AV async-Ulysses audio differs from sync-Ulysses",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestLTX2AsyncUlysses:
    """async_ulysses=True/False parity for LTX-2 AudioVideo at ws=2."""

    @pytest.mark.parametrize("backend", ["VANILLA", "FA4"])
    def test_av_async_vs_sync_parity(self, backend):
        """ws=2, audio_seq_len % 2 == 0 — audio_attn1 uses Ulysses on both
        sync and async paths."""
        run_test_in_distributed(2, _logic_async_vs_sync_parity, backend, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
