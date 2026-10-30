# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real NCCL coverage of H3 TP, Ulysses padding and distributed spatial VAE tiles."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3

from tensorrt_llm._torch.visual_gen.config import (
    DiffusionModelConfig,
    create_attention_metadata_state,
)
from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
from tensorrt_llm._torch.visual_gen.models.minimax_h3.parallel_vae import (
    TiledAutoencoderKLMiniMaxH3,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.visual_gen.args import AttentionConfig, ParallelConfig

from ._worker_pool import run_distributed


def _make_model_config(
    backend: str = "VANILLA", async_ulysses: bool = False
) -> DiffusionModelConfig:
    parallel = (
        ParallelConfig(ulysses_size=2, async_ulysses=True) if async_ulysses else ParallelConfig()
    )
    return DiffusionModelConfig(
        pretrained_config=SimpleNamespace(
            num_attention_heads=2,
            attention_head_dim=128 if backend == "FA4" else 16,
            hidden_size=256 if backend == "FA4" else 32,
            num_layers=2,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=3,
            patch_size=(1, 1, 1),
            text_dim=5,
            freq_dim=8,
            time_embed_hidden_dim=12,
            time_embed_dim=6,
            rope_freq_dim=1,
            rope_theta=10000.0,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        ),
        quant_config=QuantConfig(),
        mapping=Mapping(),
        attention=AttentionConfig(backend=backend),
        attention_metadata_state=create_attention_metadata_state(),
        parallel=parallel,
    )


def _split_fused_qkv(state: dict[str, torch.Tensor], keys) -> dict[str, torch.Tensor]:
    """Translate a fused-QKV state dict to the separate to_q / to_k / to_v layout of async Ulysses."""
    out = {}
    for key in keys:
        if key in state:
            out[key] = state[key]
            continue
        prefix, name = key.rsplit(".", 2)[0], key.rsplit(".", 2)[1]
        q, k, v = state[f"{prefix}.qkv_proj.weight"].chunk(3, dim=0)
        out[key] = {"to_q": q, "to_k": k, "to_v": v}[name]
    return out


class _TileDecoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tile-dependent values expose wrong ordering and overlap blending.
        # Per batch entry, like the ViT decoder: the single-rank path batches tiles.
        mean = x.mean(dim=tuple(range(1, x.ndim)), keepdim=True)
        return (x + mean).repeat_interleave(2, -2).repeat_interleave(2, -1)


def _make_vae() -> TiledAutoencoderKLMiniMaxH3:
    vae = TiledAutoencoderKLMiniMaxH3.__new__(TiledAutoencoderKLMiniMaxH3)
    torch.nn.Module.__init__(vae)
    vae.spatial_compression_ratio = 2
    vae.use_tiling = True
    vae.tile_sample_min_height = vae.tile_sample_min_width = 8
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 2
    vae.post_quant_conv = torch.nn.Conv3d(1, 1, 1)
    vae.decoder = _TileDecoder()
    return vae


def _logic_h3_ulysses_and_parallel_vae(
    rank: int,
    world_size: int,
    backend: str,
    async_ulysses: bool = False,
    tp_size: int = 1,
    ulysses_size: int = 2,
) -> None:
    assert world_size == tp_size * ulysses_size
    device = torch.device("cuda", rank)
    torch.manual_seed(42)
    config = _make_model_config(backend)
    config.pretrained_config.num_attention_heads = 4
    reference = MiniMaxH3Transformer3DModel(config).to(device).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.normal_(0, 0.02)
    config = _make_model_config(backend, async_ulysses)
    config.pretrained_config.num_attention_heads = 4
    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        parallel_vae_size=2,
    )
    config.visual_gen_mapping = vgm
    config.mapping = vgm.to_llm_mapping()
    candidate = MiniMaxH3Transformer3DModel(config).to(device).eval()
    if async_ulysses:
        candidate.load_state_dict(_split_fused_qkv(reference.state_dict(), candidate.state_dict()))
    else:
        checkpoint = {}
        for name, parameter in reference.state_dict().items():
            if ".qkv_proj." in name:
                prefix, suffix = name.split("qkv_proj.", maxsplit=1)
                for projection, value in zip(("to_q", "to_k", "to_v"), parameter.chunk(3, dim=0)):
                    checkpoint[f"{prefix}{projection}.{suffix}"] = value
            elif ".gate_up_proj." in name:
                prefix, suffix = name.split("gate_up_proj.", maxsplit=1)
                gate, up = parameter.chunk(2, dim=0)
                checkpoint[f"{prefix}net.0.proj.{suffix}"] = torch.cat((up, gate), dim=0)
            else:
                checkpoint[name.replace(".ff.down_proj.", ".ff.net.2.")] = parameter
        candidate.load_weights(checkpoint)
    assert candidate.transformer_blocks[0].attn.local_num_attention_heads == 4 // tp_size
    assert candidate.transformer_blocks[0].adaln_proj.linear.weight.shape == (
        reference.transformer_blocks[0].adaln_proj.linear.weight.shape
    )
    if tp_size > 1:
        assert sum(p.numel() for p in candidate.parameters()) < sum(
            p.numel() for p in reference.parameters()
        )
    # Five packed rows and one text row both require padding for two ranks.
    inputs = {
        "hidden_states": torch.randn(1, 3, 2, device=device),
        "audio_hidden_states": torch.randn(1, 1, 3, device=device),
        "encoder_hidden_states": torch.randn(1, 1, 5, device=device),
        "timestep": torch.tensor([1.0], device=device),
        "conditioning_timesteps": torch.tensor([0.0, 0.75], device=device),
        "token_tags": torch.tensor([1, 0, 2, 0, 0], device=device),
        "timestep_indices": torch.tensor([0, 1, 1, 0, 1], device=device),
        "position_ids": torch.tensor(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 0, 1], [2, 1, 1]], device=device
        ),
        "video_indices": torch.tensor([1, 3, 4], device=device),
        "audio_indices": torch.tensor([2], device=device),
        "text_indices": torch.tensor([0], device=device),
    }
    with torch.inference_mode():
        expected = reference(**inputs)
        actual = candidate(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=2e-3, atol=2e-3)
    # Exercise the same VAE group and real gathers, including an idle tail rank
    # (3x3 tiles) and the fewer-tiles-than-ranks fallback.
    torch.manual_seed(43)
    vae = _make_vae().to(device).eval()
    for shape in ((2, 2), (10, 10), (7, 10)):
        z = torch.randn(1, 1, 3, *shape, device=device)
        with torch.inference_mode():
            expected_video = AutoencoderKLMiniMaxH3._decode_clip(vae, z)
            vae.tile_parallel_group = vgm.vae_group
            actual_video = vae._decode_clip(z)
        torch.testing.assert_close(actual_video, expected_video, rtol=0, atol=0)
    dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
@pytest.mark.parametrize("async_ulysses", [False, True], ids=["nccl", "async"])
@pytest.mark.parametrize("backend", ["VANILLA", "FA4"])
def test_h3_ulysses_and_parallel_vae(backend: str, async_ulysses: bool) -> None:
    if backend == "FA4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FA4 requires SM100 or newer")
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Test requires 2 GPUs, only {torch.cuda.device_count()} available")
    run_distributed(
        2,
        _logic_h3_ulysses_and_parallel_vae,
        backend,
        async_ulysses,
        preload=("tensorrt_llm",),
    )


def _logic_h3_partial_vae_groups(rank: int, world_size: int) -> None:
    device = torch.device("cuda", rank)
    for size in (1, 2, 3, 4):
        vgm = VisualGenMapping(world_size=4, rank=rank, ulysses_size=4, parallel_vae_size=size)
        torch.manual_seed(43)
        pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.pipeline_config = SimpleNamespace(
            parallel=SimpleNamespace(parallel_vae_size=size),
            visual_gen_mapping=vgm,
        )
        pipeline.vae = _make_vae().to(device).eval()
        # Two tiles per decoder call: several gathers per clip, the last one padded.
        pipeline.vae.max_tiles_per_decoder_call = 2
        pipeline.setup_parallel_vae()
        assert pipeline._parallel_vae_enabled == (size > 1)
        assert (pipeline.vae.tile_parallel_group is not None) == (size > 1 and rank < size)
        for shape in ((2, 2), (10, 10), (7, 10)):
            z = torch.randn(1, 1, 3, *shape, device=device)
            calls = []

            def decode_video(latents):
                calls.append("video")
                return pipeline.vae._decode_clip(latents)

            def decode_audio(latents):
                calls.append("audio")
                return latents + 1

            with torch.inference_mode():
                expected = AutoencoderKLMiniMaxH3._decode_clip(pipeline.vae, z)
                actual, audio = pipeline.decode_latents(
                    z,
                    decode_video,
                    extra_latents={"audio": (z, decode_audio)},
                )
            if rank < size:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(audio, z + 1, rtol=0, atol=0)
                assert calls == ["video", "audio"]
            else:
                assert actual is None and audio is None
                assert not calls
            # Non-VAE ranks can finish early; all ranks must remain alive
            # until the VAE group's collectives finish.
            dist.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
def test_h3_partial_vae_groups() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip(f"Test requires 4 GPUs, only {torch.cuda.device_count()} available")
    run_distributed(4, _logic_h3_partial_vae_groups, preload=("tensorrt_llm",))


@pytest.mark.parametrize("tp_size,ulysses_size", [(2, 1), (4, 1), (2, 2)])
@pytest.mark.parametrize("backend", ["VANILLA", "FA4"])
def test_h3_tensor_parallel(tp_size: int, ulysses_size: int, backend: str) -> None:
    world_size = tp_size * ulysses_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    if backend == "FA4" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FA4 requires SM100 or newer")
    run_distributed(
        world_size,
        _logic_h3_ulysses_and_parallel_vae,
        backend,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
        preload=("tensorrt_llm",),
    )
