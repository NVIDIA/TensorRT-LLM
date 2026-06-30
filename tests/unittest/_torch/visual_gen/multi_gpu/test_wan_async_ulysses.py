# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for WAN async-Ulysses parity and perf.

Compares ``WanTransformer3DModel.forward`` outputs with
``async_ulysses=True`` (V/Q/K rolling A2A via ``Attention.forward_async``,
fused split RMSNorm+RoPE kernel) vs ``async_ulysses=False`` (standard sync
Ulysses with packed FUSE_QKV fused kernel) at ``ulysses_size=2``. Math is
equivalent (unify PR ties split vs packed kernel families); BF16 drift
through the 2-layer model is small.

Exercises:
  - ``WanTransformerBlock.forward`` block-level dispatch to ``forward_async``
  - ``Attention.forward_async`` fused split kernel path (head_dim=64)
  - ``UlyssesAttention.forward_async`` V/Q/K rolling side-stream pipeline

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_async_ulysses.py -v
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

    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

    # Spawn distributed workers via a helper that retries with a fresh master
    # port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.visual_gen.args import (
        AttentionConfig,
        ParallelConfig,
        TeaCacheConfig,
        TorchCompileConfig,
    )

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers
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
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, "nccl", test_fn, port, fn_args),
            nprocs=world_size,
            join=True,
        )
    )


# =============================================================================
# Model config
# =============================================================================

# Small WAN config. head_dim=64 matches the fused split kernel template; with
# num_attention_heads=4 and ulysses_size=2, each rank holds 2 heads after the
# Ulysses head-shard. Video patches yield 8 tokens, divisible by ws=2.
_WAN_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=64,
    num_layers=2,
    in_channels=4,
    out_channels=4,
    patch_size=[1, 2, 2],
    text_dim=64,
    freq_dim=32,
)
_VIDEO_SHAPE = (1, 4, 2, 4, 4)  # [B, C, T, H, W]
_TEXT_SEQ = 4


def _make_model_config(
    ulysses_size: int = 1,
    backend: str = "FA4",
    async_ulysses: bool = False,
) -> "DiffusionModelConfig":
    """Create DiffusionModelConfig for WAN tests."""
    if ulysses_size > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        ws = ulysses_size
        rk = 0
    vgm = VisualGenMapping(world_size=ws, rank=rk, ulysses_size=ulysses_size)

    pretrained_config = SimpleNamespace(**_WAN_CONFIG)
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        cache=TeaCacheConfig(),
        parallel=ParallelConfig(
            ulysses_size=ulysses_size,
            async_ulysses=async_ulysses,
        ),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _stabilize_model_weights(model):
    """Reinit weights to small values — prevents BF16 overflow through layers."""
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                fan_in = p.shape[1]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


def _build_wan_model(
    world_size: int,
    backend: str,
    async_ulysses: bool,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
):
    """Build WanTransformer3DModel with deterministic weights via shared seed."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    torch.manual_seed(seed)
    cfg = _make_model_config(ulysses_size=world_size, backend=backend, async_ulysses=async_ulysses)
    model = WanTransformer3DModel(cfg).to(device).to(dtype)
    _stabilize_model_weights(model)
    return model


def _pack_async_state_for_sync(async_state, sync_state_keys):
    """Translate a SEPARATE_QKV (async) state_dict to FUSE_QKV (sync) layout:
    concatenates to_q/to_k/to_v.{weight,bias} into packed qkv_proj.{weight,bias}.
    Non-self-attn weights pass through unchanged.

    Used to load identical effective weights into both async and sync models
    despite their different attn1 layer structure.
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


def _build_inputs(device, dtype, seed=100):
    """Identical inputs across all ranks via shared seed."""
    B, C, T, H, W = _VIDEO_SHAPE
    text_dim = _WAN_CONFIG["text_dim"]
    torch.manual_seed(seed)
    hidden_states = torch.randn(_VIDEO_SHAPE, device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([0.5], device=device, dtype=dtype)
    return hidden_states, encoder_hidden_states, timestep


# =============================================================================
# Test logic
# =============================================================================


def _logic_async_vs_sync_parity(rank, world_size, backend):
    """WAN at ws=2: async-Ulysses output matches sync-Ulysses output.

    Sync path: block calls ``self.attn1(...)`` → ``Attention.forward`` →
    FUSE_QKV packed kernel → sync ``UlyssesAttention.forward``.
    Async path: block calls ``self.attn1.forward_async(...)`` → split
    kernel via closures → ``UlyssesAttention.forward_async`` rolling A2A.
    Both should match within BF16 accumulation drift.
    """
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    # Build async (SEPARATE_QKV self-attn → to_q/k/v) first as the canonical
    # source, then mirror its weights into a sync (FUSE_QKV → qkv_proj) model
    # by concatenating to_q/k/v into the packed layout.
    async_model = _build_wan_model(world_size, backend, True, device, dtype)
    async_state = async_model.state_dict()
    sync_model = _build_wan_model(world_size, backend, False, device, dtype)
    sync_model.load_state_dict(
        _pack_async_state_for_sync(async_state, sync_model.state_dict().keys())
    )

    # Sanity: confirm sync vs async actually take different code paths.
    assert async_model.blocks[0]._use_async_ulysses is True, "async model didn't enable async path"
    assert sync_model.blocks[0]._use_async_ulysses is False, (
        "sync model unexpectedly enabled async path"
    )
    assert async_model.blocks[0].attn1.qkv_mode != sync_model.blocks[0].attn1.qkv_mode, (
        "test bug: sync and async models have identical qkv_mode "
        f"(both {sync_model.blocks[0].attn1.qkv_mode})"
    )

    hidden_states, encoder_hidden_states, timestep = _build_inputs(device, dtype)

    with torch.no_grad():
        sync_out = sync_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        async_out = async_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    assert sync_out.shape == async_out.shape, f"Rank {rank}: output shape mismatch"
    assert not torch.isnan(async_out).any(), f"Rank {rank}: NaN in async output"
    assert not torch.isinf(async_out).any(), f"Rank {rank}: Inf in async output"

    # Diagnostic: actual BF16 drift between sync (packed kernel + sync a2a) and
    # async (split kernel + side-stream a2a) — both at ws=2, same collectives.
    diff = (async_out.float() - sync_out.float()).abs()
    ref = sync_out.float().abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.clamp(min=1e-6)).max().item()
    print(
        f"\n[WAN rank={rank} backend={backend}] "
        f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} "
        f"sync_abs_max={ref.max().item():.3e}"
    )

    torch.testing.assert_close(
        async_out,
        sync_out,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: WAN async-Ulysses output differs from sync-Ulysses",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestWanAsyncUlysses:
    """async_ulysses=True/False parity for WAN at ws=2."""

    @pytest.mark.parametrize("backend", ["VANILLA", "FA4"])
    def test_async_vs_sync_parity(self, backend):
        """ws=2: async path (block-level inline-replaced by forward_async)
        matches sync path (Attention.forward FUSE_QKV packed kernel)."""
        run_test_in_distributed(2, _logic_async_vs_sync_parity, backend)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
