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

    # Spawn distributed workers via a helper that retries with a fresh master
    # port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

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


def _make_model_config_cfg(
    cfg_size: int,
    ulysses_size: int,
    backend: str = "VANILLA",
) -> "DiffusionModelConfig":
    """DiffusionModelConfig with CFG x Ulysses parallelism (dist must be up)."""
    ws = dist.get_world_size()
    rk = dist.get_rank()
    vgm = VisualGenMapping(world_size=ws, rank=rk, cfg_size=cfg_size, ulysses_size=ulysses_size)
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
        parallel=ParallelConfig(cfg_size=cfg_size, ulysses_size=ulysses_size),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _build_stage2_groups_for_test(vgm):
    """Mirror of LTX2TwoStagesPipeline._stage2_transformer_groups."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import Stage2Groups

    rank = dist.get_rank()
    fold = vgm.flatten_cfg_ranks()
    uly_group = None
    for ranks in fold:
        g = dist.new_group(ranks, use_local_synchronization=False)
        if rank in ranks:
            uly_group = g
    fibers = vgm.flatten_cfg_seq_ranks()
    if len(fold) == len(fibers):
        my_fiber = next(ranks for ranks in fold if rank in ranks)
        seq_group, gather_index = uly_group, None
    else:
        seq_group = my_fiber = None
        for ranks in fibers:
            g = dist.new_group(sorted(ranks), use_local_synchronization=False)
            if rank in ranks:
                seq_group, my_fiber = g, ranks
        sorted_fiber = sorted(my_fiber)
        gather_index = [sorted_fiber.index(r) for r in my_fiber]
    return Stage2Groups(
        ulysses_group=uly_group,
        seq_group=seq_group,
        seq_rank=my_fiber.index(rank),
        seq_size=len(my_fiber),
        gather_index=gather_index,
    )


def _logic_ltx2_dual_topology(rank, world_size, backend, audio_seq_len):
    """{default, stage2} switch: stack alternation, shard/gather round-trip,
    and full-forward numerical equivalence vs the single-GPU reference.

    cfg2 x u(world/2): the default topology shards each cfg branch's sequence
    over its own fiber; stage2 folds cfg into one world-spanning ulysses group.
    Stage 2 has no CFG, so a forward on identical inputs must match the
    reference in BOTH topologies, across repeated back-and-forth switches.
    """
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    batch = 1
    v_dims = (1, 4, 4)
    v_patches = v_dims[0] * v_dims[1] * v_dims[2]  # 16: divisible by u2 and u4

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

    torch.manual_seed(123)
    d_cfg = _make_model_config_cfg(cfg_size=2, ulysses_size=world_size // 2, backend=backend)
    s2 = _build_stage2_groups_for_test(d_cfg.visual_gen_mapping)
    d_model = (
        LTXModel(
            model_type=LTXModelType.AudioVideo,
            model_config=d_cfg,
            stage2_groups=s2,
            **_AV_CONFIG,
        )
        .to(device, dtype=dtype)
        .eval()
    )
    d_model.load_state_dict(ref_state)
    d_model.configure_audio_ulysses(audio_seq_len)

    assert d_model._has_stage2
    assert d_model._sharder.size == world_size // 2
    assert d_model._sharder_s2.size == world_size

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
    with torch.no_grad():
        ref_v, ref_a = ref_model(video=video, audio=audio, text_cache=ref_cache)

    x = torch.randn(1, 32, 8, device=device, dtype=dtype)
    for is_stage2 in (False, True, False, True):
        d_model.set_ulysses_topology(is_stage2=is_stage2)

        blk = d_model.transformer_blocks[0]
        expect = blk.attn1._attn_stage2 if is_stage2 else blk.attn1._attn_default
        assert blk.attn1.attn is expect, f"Rank {rank}: attn stack not switched"
        # is_ulysses must track the ACTIVE stack (the pair can differ in type).
        from tensorrt_llm._torch.visual_gen.attention_backend.parallel import UlyssesAttention

        for name in ("attn1", "video_to_audio_attn", "audio_attn1"):
            mod = getattr(blk, name, None)
            if mod is not None:
                assert mod.is_ulysses == isinstance(mod.attn, UlyssesAttention), (
                    f"Rank {rank}: {name}.is_ulysses stale (is_stage2={is_stage2})"
                )
        sh = d_model._active_sharder
        assert sh.size == (world_size if is_stage2 else world_size // 2)
        assert torch.equal(sh.gather(sh.shard(x, dim=1), dim=1), x), (
            f"Rank {rank}: shard/gather round-trip broken (is_stage2={is_stage2})"
        )

        # prepare_text_cache is topology-dependent — always AFTER the switch.
        d_cache = d_model.prepare_text_cache(
            video_context=v_ctx,
            video_positions=v_pos,
            audio_context=a_ctx,
            audio_positions=a_pos,
            dtype=dtype,
        )
        video_i, audio_i, *_ = _build_inputs(batch, v_patches, v_dims, audio_seq_len, dtype, device)
        with torch.no_grad():
            d_v, d_a = d_model(video=video_i, audio=audio_i, text_cache=d_cache)

        torch.testing.assert_close(
            d_v,
            ref_v,
            rtol=5e-2,
            atol=5e-2,
            msg=f"Rank {rank}: video mismatch (is_stage2={is_stage2})",
        )
        torch.testing.assert_close(
            d_a,
            ref_a,
            rtol=5e-2,
            atol=5e-2,
            msg=f"Rank {rank}: audio mismatch (is_stage2={is_stage2})",
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


def _logic_ltx2_full_audio_construction(rank, world_size, backend, audio_seq_len):
    """Legacy FULL mode at ulysses>1: audio_attn1 is CONSTRUCTED as ulysses (env
    constant is the only selector) and, given stage-2 groups under cfg2, gets
    its own distinct {default, stage2} stack pair like attn1/v2a."""
    import tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 as tl
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    orig = tl._LTX2_AUDIO_CONDITIONAL_SHARD
    tl._LTX2_AUDIO_CONDITIONAL_SHARD = False
    try:
        torch.manual_seed(123)
        d_cfg = _make_model_config_cfg(cfg_size=2, ulysses_size=world_size // 2, backend=backend)
        s2 = _build_stage2_groups_for_test(d_cfg.visual_gen_mapping)
        model = (
            LTXModel(
                model_type=LTXModelType.AudioVideo,
                model_config=d_cfg,
                stage2_groups=s2,
                **_AV_CONFIG,
            )
            .to(device, dtype=dtype)
            .eval()
        )
        _init_all_weights(model)
        model.configure_audio_ulysses(audio_seq_len)
        blk = model.transformer_blocks[0]
        assert blk.audio_attn1.is_ulysses, "FULL mode must construct ulysses audio_attn1"
        assert blk.audio_attn1._attn_stage2 is not blk.audio_attn1._attn_default, (
            "FULL audio_attn1 must get a distinct stage-2 stack under cfg2"
        )
        # Both topologies forward without error (audio sharded across the active group).
        video, audio, v_ctx, a_ctx, v_pos, a_pos = _build_inputs(
            1, 16, (1, 4, 4), audio_seq_len, dtype, device
        )
        for is_stage2 in (False, True, False):
            model.set_ulysses_topology(is_stage2=is_stage2)
            cache = model.prepare_text_cache(
                video_context=v_ctx,
                video_positions=v_pos,
                audio_context=a_ctx,
                audio_positions=a_pos,
                dtype=dtype,
            )
            video_i, audio_i, *_ = _build_inputs(1, 16, (1, 4, 4), audio_seq_len, dtype, device)
            with torch.no_grad():
                v, a = model(video=video_i, audio=audio_i, text_cache=cache)
            assert v.shape[1] == 16 and a.shape[1] == audio_seq_len
    finally:
        tl._LTX2_AUDIO_CONDITIONAL_SHARD = orig


def _logic_ltx2_stage2_head_divisibility_raises(rank, world_size, backend, _unused):
    """Construction fails fast when the head count divides the stage-1 ulysses
    size (u = ws/2) but not the stage-2 group size (cfg*u = ws)."""
    from tensorrt_llm._torch.visual_gen.models.ltx2.transformer_ltx2 import LTXModel, LTXModelType

    d_cfg = _make_model_config_cfg(cfg_size=2, ulysses_size=world_size // 2, backend=backend)
    s2 = _build_stage2_groups_for_test(d_cfg.visual_gen_mapping)
    bad = dict(
        _AV_CONFIG,
        num_attention_heads=6,
        cross_attention_dim=6 * _AV_CONFIG["attention_head_dim"],
    )
    assert 6 % (world_size // 2) == 0 and 6 % world_size != 0
    try:
        LTXModel(
            model_type=LTXModelType.AudioVideo,
            model_config=d_cfg,
            stage2_groups=s2,
            **bad,
        )
        raise AssertionError(f"Rank {rank}: indivisible stage-2 head count was not rejected")
    except ValueError as e:
        assert "stage-2 ulysses requires" in str(e), f"Rank {rank}: {e}"


class TestLTX2TopologySwitch:
    """{default, stage2} dual-topology switch on 4 GPUs (cfg2 x u2 -> u4):
    pointer alternation, shard/gather round-trip, and whole-dataflow numerical
    equivalence vs the single-GPU reference in both topologies."""

    @pytest.mark.parametrize("audio_seq_len", [64, 62], ids=["no_pad", "pad2"])
    def test_dual_topology_alternation_matches_reference(self, audio_seq_len):
        run_test_in_distributed(4, _logic_ltx2_dual_topology, "VANILLA", audio_seq_len)

    def test_full_audio_mode_construction_and_switch(self):
        run_test_in_distributed(4, _logic_ltx2_full_audio_construction, "VANILLA", 64)

    def test_stage2_head_divisibility_raises(self):
        run_test_in_distributed(4, _logic_ltx2_stage2_head_divisibility_raises, "VANILLA", 0)
