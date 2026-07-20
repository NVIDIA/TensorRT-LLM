# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Cosmos3VFMTransformer parallelism (TP, Ulysses, CFG).

Synthetic config + stabilized random weights; compares distributed forwards against
a single-GPU reference on the same rank (same pattern as test_wan_tp / test_flux_tp).
No checkpoint loading.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_cosmos3_transformer_parallel.py -v -s
"""

import os
from types import SimpleNamespace
from typing import Callable, Tuple

os.environ["TLLM_DISABLE_MPI"] = "1"
os.environ["TRTLLM_DISABLE_COSMOS3_GUARDRAILS"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
        Cosmos3VFMTransformer,
    )

    # Spawn distributed workers via a helper that retries with a fresh master
    # port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Attention2D (attn2d) wraps the compute backend in Attention2DAttention, which
# requires (a) an LSE-capable inner backend — only FA4, VANILLA does not support
# LSE — and (b) the ``flash_attn_combine`` JIT kernel.  Detect both up front so the
# attn2d tests skip cleanly when the kernels are not built (e.g. non-Blackwell CI).
try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        _flash_attn_fwd as _fa4_fwd,
    )
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
        _flash_attn_combine as _fa_combine,
    )

    _ATTN2D_AVAILABLE = MODULES_AVAILABLE and _fa4_fwd is not None and _fa_combine is not None
except ImportError:
    _ATTN2D_AVAILABLE = False

pytestmark = pytest.mark.cosmos3

# Small Cosmos3 config: 8 Q / 4 KV (GQA), hidden_size = 8 * 64 = 512.
# Divisible by TP=2 and TP×Ulysses=4 for both Q and KV head counts.
_COSMOS3_TEST_CONFIG = dict(
    hidden_size=512,
    intermediate_size=512,
    num_hidden_layers=4,
    latent_patch_size=2,
    latent_channel=4,
    position_embedding_type="unified_3d_mrope",
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=64,
    rope_scaling={"rope_type": "default", "mrope_section": [16, 12, 12]},
    rms_norm_eps=1e-6,
    vocab_size=1024,
    rope_theta=1_000_000.0,
    max_position_embeddings=4096,
    timestep_scale=1.0,
    base_fps=24.0,
    unified_3d_mrope_temporal_modality_margin=100,
    enable_fps_modulation=True,
)

# attn2d needs an LSE-capable backend (FA4); FA4's CUTE kernels run with head_dim=128.
# Same architecture as _COSMOS3_TEST_CONFIG otherwise (heads still divisible by
# Ulysses=2 for the attn2d+ulysses case). mrope_section is unchanged: the interleave
# slices clip to head_dim//2 (=64 here), so [16, 12, 12] stays valid.
_COSMOS3_FA4_CONFIG = dict(
    _COSMOS3_TEST_CONFIG,
    head_dim=128,
    hidden_size=8 * 128,
    intermediate_size=1024,
)

# Video: [B, C, T, H, W]. patch_size=2 → seq_len = T * (H/2) * (W/2).
# T=2, H=W=4 → seq_len=8 (divisible by Ulysses=2). T>1 exercises fps modulation.
_LATENT_T = 2
_LATENT_H = 4
_LATENT_W = 4
_TEXT_LEN = 8
_MAX_TEXT_LEN = 16
_TIMESTEP = 500.0
_NUM_TRAIN_TIMESTEPS = 1000.0
_FPS = 24.0

# Audio (sound) modality: audio tokens are appended to the gen sequence, so the
# combined seq becomes video_tokens (8) + T_audio. Keep T_audio even so the
# combined length stays divisible by Ulysses=2 (the sharder also pads, but this
# keeps the parity comparison free of padding artifacts).
_AUDIO_DIM = 16
_T_AUDIO = 4
_SOUND_LATENT_FPS = 24.0

# Same architecture as _COSMOS3_TEST_CONFIG, with the audio modality enabled.
# The transformer reads audio attributes via the legacy ``sound_*`` keys.
_COSMOS3_AUDIO_CONFIG = dict(
    **_COSMOS3_TEST_CONFIG,
    sound_gen=True,
    sound_dim=_AUDIO_DIM,
    sound_latent_fps=_SOUND_LATENT_FPS,
    temporal_compression_factor_sound=1,
)

SEED_WEIGHTS = 123
SEED_INPUT = 456
SEED_COND_TEXT = 42
SEED_UNCOND_TEXT = 123

RTOL = 1e-2
ATOL = 1e-2


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_wan_tp.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    try:
        from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

        DeviceMeshTopologyImpl.device_mesh = None
        DeviceMeshTopologyImpl.tp_mesh = None
        VisualGenMapping.seq_mesh = None
    except ImportError:
        pass
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}", flush=True)
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, backend, test_fn, port),
            nprocs=world_size,
            join=True,
        )
    )


# =============================================================================
# Model config + weight helpers
# =============================================================================


def _make_model_config(
    pretrained_dict,
    *,
    cfg_size=1,
    tp_size=1,
    ulysses_size=1,
    attn2d_row_size=1,
    attn2d_col_size=1,
    backend="VANILLA",
):
    pretrained_config = SimpleNamespace(**pretrained_dict)
    ws = cfg_size * tp_size * ulysses_size * attn2d_row_size * attn2d_col_size
    if ws > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        rk = 0
    vgm = VisualGenMapping(
        world_size=ws,
        rank=rk,
        cfg_size=cfg_size,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _stabilize_model_weights(model: Cosmos3VFMTransformer) -> None:
    """Small uniform init so BF16 forwards stay bounded through both pathways."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and name.endswith(".weight"):
                p.fill_(1.0)
            elif p.ndim >= 2:
                fan_in = p.shape[1]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


def _shard_dim0(tensor, tp_rank, tp_size):
    chunk = tensor.shape[0] // tp_size
    return tensor[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_dim1(tensor, tp_rank, tp_size):
    chunk = tensor.shape[1] // tp_size
    return tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_fused_qkv(tensor, tp_rank, tp_size, q_dim, kv_dim):
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
    half = tensor.shape[0] // 2
    gate, up = tensor.split([half, half], dim=0)
    return torch.cat(
        [
            _shard_dim0(gate, tp_rank, tp_size),
            _shard_dim0(up, tp_rank, tp_size),
        ],
        dim=0,
    )


def _qkv_output_dims() -> Tuple[int, int]:
    """Full (q_dim, kv_dim) for fused gen-path QKV (GQA-aware)."""
    q_dim = _COSMOS3_TEST_CONFIG["num_attention_heads"] * _COSMOS3_TEST_CONFIG["head_dim"]
    kv_dim = _COSMOS3_TEST_CONFIG["num_key_value_heads"] * _COSMOS3_TEST_CONFIG["head_dim"]
    return q_dim, kv_dim


def _fused_qkv_dims(ref_param: torch.Tensor) -> Tuple[int, int]:
    """Return (q_dim, kv_dim) for a fused QKV weight/bias stacked as [Q|K|V] on dim 0."""
    q_dim, kv_dim = _qkv_output_dims()
    expected = q_dim + 2 * kv_dim
    if ref_param.shape[0] == expected:
        return q_dim, kv_dim
    raise ValueError(
        f"Cannot infer fused QKV dims from shape {ref_param.shape}; "
        f"expected dim0={expected} for GQA config"
    )


def _copy_ref_weights_to_tp(ref_model, tp_model, tp_rank, tp_size):
    """Copy TP=1 reference weights into a TP-sharded model (WAN/Flux pattern)."""
    ref_params = dict(ref_model.named_parameters())

    with torch.no_grad():
        for tp_name, tp_param in tp_model.named_parameters():
            if tp_name not in ref_params:
                continue

            ref_param = ref_params[tp_name]

            if tp_param.shape == ref_param.shape:
                tp_param.data.copy_(ref_param.data)
            elif tp_param.ndim >= 2 and tp_param.shape[1] == ref_param.shape[1]:
                if "qkv_proj" in tp_name:
                    q_dim, kv_dim = _fused_qkv_dims(ref_param)
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, kv_dim)
                    )
                elif "gate_up_proj" in tp_name:
                    tp_param.data.copy_(_shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim >= 2 and tp_param.shape[0] == ref_param.shape[0]:
                tp_param.data.copy_(_shard_dim1(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim == 1 and tp_param.shape[0] < ref_param.shape[0]:
                if "qkv_proj" in tp_name:
                    q_dim, kv_dim = _fused_qkv_dims(ref_param)
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, kv_dim)
                    )
                elif "gate_up_proj" in tp_name:
                    tp_param.data.copy_(_shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            else:
                raise ValueError(
                    f"Cannot shard {tp_name}: ref={ref_param.shape}, tp={tp_param.shape}"
                )


def _cosmos3_inputs(
    device: torch.device,
    *,
    channels: int,
    text_seed: int,
    batch: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    torch.manual_seed(SEED_INPUT)
    hidden_states = (
        torch.randn(
            batch,
            channels,
            _LATENT_T,
            _LATENT_H,
            _LATENT_W,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    timestep = torch.full((batch,), _TIMESTEP, device=device, dtype=torch.float32)
    torch.manual_seed(text_seed)
    text_ids = torch.randint(1, 1000, (batch, _MAX_TEXT_LEN), device=device, dtype=torch.long)
    text_mask = torch.zeros(batch, _MAX_TEXT_LEN, device=device, dtype=torch.long)
    text_mask[:, :_TEXT_LEN] = 1
    return hidden_states, timestep, text_ids, text_mask, (_LATENT_T, _LATENT_H, _LATENT_W)


def _forward(model: Cosmos3VFMTransformer, device: torch.device, text_seed: int) -> torch.Tensor:
    channels = _COSMOS3_TEST_CONFIG["latent_channel"]
    hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
        device, channels=channels, text_seed=text_seed
    )
    model.reset_cache()
    with torch.inference_mode():
        return model(
            hidden_states=hs,
            timestep=ts / _NUM_TRAIN_TIMESTEPS,
            raw_timestep=ts,
            text_ids=text_ids,
            text_mask=text_mask,
            video_shape=video_shape,
            fps=_FPS,
        ).video


def _forward_with_audio(
    model: Cosmos3VFMTransformer, device: torch.device, text_seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward an audio-enabled model; returns (video_velocity, audio_velocity)."""
    channels = _COSMOS3_TEST_CONFIG["latent_channel"]
    hs, ts, text_ids, text_mask, video_shape = _cosmos3_inputs(
        device, channels=channels, text_seed=text_seed
    )
    # Deterministic audio noise, independent of the (seed-controlled) video/text
    # inputs, so ref and parallel models see identical audio_latents.
    torch.manual_seed(SEED_INPUT + 1)
    audio_latents = (
        torch.randn(hs.shape[0], _AUDIO_DIM, _T_AUDIO, device=device, dtype=hs.dtype) * 0.1
    )
    model.reset_cache()
    with torch.inference_mode():
        out = model(
            hidden_states=hs,
            timestep=ts / _NUM_TRAIN_TIMESTEPS,
            raw_timestep=ts,
            text_ids=text_ids,
            text_mask=text_mask,
            video_shape=video_shape,
            fps=_FPS,
            audio_latents=audio_latents,
        )
    return out.video, out.audio


def _build_ref_and_parallel(
    *,
    tp_size: int = 1,
    ulysses_size: int = 1,
    cfg_size: int = 1,
    attn2d_row_size: int = 1,
    attn2d_col_size: int = 1,
    backend: str = "VANILLA",
    pretrained_dict: dict = None,
) -> Tuple[Cosmos3VFMTransformer, Cosmos3VFMTransformer, VisualGenMapping, torch.device]:
    pretrained_dict = pretrained_dict if pretrained_dict is not None else _COSMOS3_TEST_CONFIG
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")

    torch.manual_seed(SEED_WEIGHTS)
    # Reference is the unsharded model on the same compute backend, so parity isolates
    # the parallelism (e.g. FA4-full vs FA4+attn2d) rather than comparing backends.
    ref_config = _make_model_config(pretrained_dict, tp_size=1, ulysses_size=1, backend=backend)
    # Do not .to(bfloat16) the whole module — RoPE inv_freq must stay fp32 (see
    # Qwen3VLTextRotaryEmbedding.forward). post_load_weights() sets Linear/MLP to bf16.
    ref_model = Cosmos3VFMTransformer(ref_config).to(device).eval()
    _stabilize_model_weights(ref_model)
    ref_model.post_load_weights()

    torch.manual_seed(SEED_WEIGHTS)
    parallel_config = _make_model_config(
        pretrained_dict,
        cfg_size=cfg_size,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
        backend=backend,
    )
    vgm = parallel_config.visual_gen_mapping
    parallel_model = Cosmos3VFMTransformer(parallel_config).to(device).eval()
    if tp_size > 1:
        _copy_ref_weights_to_tp(ref_model, parallel_model, vgm.tp_rank, tp_size)
    else:
        parallel_model.load_state_dict(ref_model.state_dict())
    parallel_model.post_load_weights()

    return ref_model, parallel_model, vgm, device


def _cfg_text_seed(rank: int, *, tp_size: int, ulysses_size: int, cfg_size: int) -> int:
    if cfg_size > 1:
        cfg_rank = rank // (tp_size * ulysses_size)
        return SEED_COND_TEXT if cfg_rank == 0 else SEED_UNCOND_TEXT
    return SEED_COND_TEXT


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, *, msg: str) -> None:
    assert actual.shape == expected.shape, f"{msg}: shape {actual.shape} vs {expected.shape}"
    actual_f = actual.float()
    expected_f = expected.float()
    assert not torch.isnan(actual_f).any(), msg
    assert not torch.isinf(actual_f).any(), msg
    torch.testing.assert_close(actual_f, expected_f, rtol=RTOL, atol=ATOL, msg=msg)


# =============================================================================
# Test logic (module-level for mp.spawn pickling)
# =============================================================================


def _logic_cosmos3_tp_vs_single_gpu(rank, world_size):
    ref_model, tp_model, _, device = _build_ref_and_parallel(tp_size=world_size)
    text_seed = _cfg_text_seed(rank, tp_size=world_size, ulysses_size=1, cfg_size=1)

    ref_out = _forward(ref_model, device, text_seed)
    tp_out = _forward(tp_model, device, text_seed)

    if rank == 0:
        diff = (tp_out.float() - ref_out.float()).abs()
        print(
            f"[tp={world_size}] max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(tp_out, ref_out, msg=f"Rank {rank}: TP output differs from single-GPU reference")


def _logic_cosmos3_ulysses_vs_single_gpu(rank, world_size):
    ref_model, ulysses_model, _, device = _build_ref_and_parallel(ulysses_size=world_size)
    text_seed = _cfg_text_seed(rank, tp_size=1, ulysses_size=world_size, cfg_size=1)

    ref_out = _forward(ref_model, device, text_seed)
    ulysses_out = _forward(ulysses_model, device, text_seed)

    if rank == 0:
        diff = (ulysses_out.float() - ref_out.float()).abs()
        print(
            f"[ulysses={world_size}] max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(
        ulysses_out,
        ref_out,
        msg=f"Rank {rank}: Ulysses output differs from single-GPU reference",
    )


def _logic_cosmos3_ulysses_audio_vs_single_gpu(rank, world_size):
    ref_model, ulysses_model, _, device = _build_ref_and_parallel(
        ulysses_size=world_size, pretrained_dict=_COSMOS3_AUDIO_CONFIG
    )
    text_seed = _cfg_text_seed(rank, tp_size=1, ulysses_size=world_size, cfg_size=1)

    ref_video, ref_audio = _forward_with_audio(ref_model, device, text_seed)
    ulysses_video, ulysses_audio = _forward_with_audio(ulysses_model, device, text_seed)

    if rank == 0:
        vdiff = (ulysses_video.float() - ref_video.float()).abs()
        adiff = (ulysses_audio.float() - ref_audio.float()).abs()
        print(
            f"[ulysses={world_size}+audio] "
            f"video max_abs_diff={vdiff.max().item():.6e}, "
            f"audio max_abs_diff={adiff.max().item():.6e}",
            flush=True,
        )

    _assert_parity(
        ulysses_video,
        ref_video,
        msg=f"Rank {rank}: Ulysses+audio VIDEO differs from single-GPU reference",
    )
    _assert_parity(
        ulysses_audio,
        ref_audio,
        msg=f"Rank {rank}: Ulysses+audio AUDIO differs from single-GPU reference",
    )


def _logic_cosmos3_tp_ulysses_vs_single_gpu(rank, world_size):
    tp_size = 2
    ulysses_size = 2
    ref_model, combined_model, _, device = _build_ref_and_parallel(
        tp_size=tp_size, ulysses_size=ulysses_size
    )
    text_seed = _cfg_text_seed(rank, tp_size=tp_size, ulysses_size=ulysses_size, cfg_size=1)

    ref_out = _forward(ref_model, device, text_seed)
    combined_out = _forward(combined_model, device, text_seed)

    if rank == 0:
        diff = (combined_out.float() - ref_out.float()).abs()
        print(
            f"[tp={tp_size},ulysses={ulysses_size}] max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(
        combined_out,
        ref_out,
        msg=f"Rank {rank}: TP+Ulysses output differs from single-GPU reference",
    )


def _logic_cosmos3_cfg_ulysses_vs_single_gpu(rank, world_size):
    cfg_size = 2
    ulysses_size = 2
    ref_model, parallel_model, _, device = _build_ref_and_parallel(
        cfg_size=cfg_size, ulysses_size=ulysses_size
    )
    text_seed = _cfg_text_seed(rank, tp_size=1, ulysses_size=ulysses_size, cfg_size=cfg_size)

    ref_out = _forward(ref_model, device, text_seed)
    parallel_out = _forward(parallel_model, device, text_seed)

    cfg_rank = rank // ulysses_size
    if rank == 0:
        diff = (parallel_out.float() - ref_out.float()).abs()
        print(
            f"[cfg={cfg_size},ulysses={ulysses_size},stream={cfg_rank}] "
            f"max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(
        parallel_out,
        ref_out,
        msg=f"Rank {rank}: CFG+Ulysses output differs from single-GPU reference",
    )


def _logic_cosmos3_attn2d_vs_single_gpu(rank, world_size):
    # 2x1 attn2d mesh (Q gathered across rows, K/V kept local). FA4 backend since
    # Attention2DAttention requires an LSE-capable inner backend.
    try:
        ref_model, attn2d_model, _, device = _build_ref_and_parallel(
            attn2d_row_size=2,
            attn2d_col_size=1,
            backend="FA4",
            pretrained_dict=_COSMOS3_FA4_CONFIG,
        )
    except ImportError:
        pytest.skip("FA4 / flash_attn_combine JIT kernels not available")

    text_seed = _cfg_text_seed(rank, tp_size=1, ulysses_size=1, cfg_size=1)

    ref_out = _forward(ref_model, device, text_seed)
    attn2d_out = _forward(attn2d_model, device, text_seed)

    if rank == 0:
        diff = (attn2d_out.float() - ref_out.float()).abs()
        print(
            f"[attn2d=2x1] max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(
        attn2d_out,
        ref_out,
        msg=f"Rank {rank}: Attention2D (2x1) output differs from single-GPU reference",
    )


def _logic_cosmos3_attn2d_ulysses_vs_single_gpu(rank, world_size):
    # 2x1 attn2d mesh composed with Ulysses=2 (seq_size = attn2d * ulysses = 4).
    attn2d_row_size = 2
    ulysses_size = 2
    try:
        ref_model, parallel_model, _, device = _build_ref_and_parallel(
            ulysses_size=ulysses_size,
            attn2d_row_size=attn2d_row_size,
            attn2d_col_size=1,
            backend="FA4",
            pretrained_dict=_COSMOS3_FA4_CONFIG,
        )
    except ImportError:
        pytest.skip("FA4 / flash_attn_combine JIT kernels not available")

    text_seed = _cfg_text_seed(rank, tp_size=1, ulysses_size=ulysses_size, cfg_size=1)

    ref_out = _forward(ref_model, device, text_seed)
    parallel_out = _forward(parallel_model, device, text_seed)

    if rank == 0:
        diff = (parallel_out.float() - ref_out.float()).abs()
        print(
            f"[attn2d=2x1,ulysses={ulysses_size}] max_abs_diff={diff.max().item():.6e}, "
            f"mean_abs_diff={diff.mean().item():.6e}",
            flush=True,
        )

    _assert_parity(
        parallel_out,
        ref_out,
        msg=f"Rank {rank}: Attention2D+Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.gpu2
class TestCosmos3TransformerParallel:
    """Cosmos3 TP / Ulysses / CFG parity vs single-GPU (synthetic weights, no checkpoint)."""

    def _skip_if_unavailable(self):
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")

    def test_tp2_vs_single_gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(world_size=2, test_fn=_logic_cosmos3_tp_vs_single_gpu)

    def test_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(world_size=2, test_fn=_logic_cosmos3_ulysses_vs_single_gpu)

    def test_ulysses2_audio_vs_single_gpu(self):
        """Ulysses parity with the audio modality on: video + audio tokens are
        sharded together across the sequence dimension."""
        self._skip_if_unavailable()
        run_test_in_distributed(world_size=2, test_fn=_logic_cosmos3_ulysses_audio_vs_single_gpu)

    @pytest.mark.gpu4
    def test_tp2_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(world_size=4, test_fn=_logic_cosmos3_tp_ulysses_vs_single_gpu)

    @pytest.mark.gpu4
    def test_cfg2_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(world_size=4, test_fn=_logic_cosmos3_cfg_ulysses_vs_single_gpu)

    def test_attn2d_2x1_vs_single_gpu(self):
        self._skip_if_unavailable()
        if not _ATTN2D_AVAILABLE:
            pytest.skip("FA4 / flash_attn_combine JIT kernels not available")
        run_test_in_distributed(world_size=2, test_fn=_logic_cosmos3_attn2d_vs_single_gpu)

    @pytest.mark.gpu4
    def test_attn2d_2x1_ulysses2_vs_single_gpu(self):
        self._skip_if_unavailable()
        if not _ATTN2D_AVAILABLE:
            pytest.skip("FA4 / flash_attn_combine JIT kernels not available")
        run_test_in_distributed(world_size=4, test_fn=_logic_cosmos3_attn2d_ulysses_vs_single_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
