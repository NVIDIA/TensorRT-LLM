# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from defs.examples.visual_gen.visual_gen_test_utils import _multi_resolution_log_stft_distance
from defs.visual_gen.minimax_h3.quality_utils import MINIMAX_H3_NUM_INFERENCE_STEPS

MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD = 0.30


@pytest.mark.parametrize("world_size", [2, 4])
def test_minimax_h3_text_encoder_tp_lpips(tmp_path, monkeypatch, world_size):
    """Compare encoder TP with HF encoding while keeping DiT parallelism fixed."""
    from defs.visual_gen.minimax_h3.quality_utils import (
        MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD,
        _mean_lpips_distance,
        _minimax_h3_checkpoint_path,
    )
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    checkpoint = _minimax_h3_checkpoint_path()
    videos, audios = [], []
    for encoder_tp in (1, 2):
        args = VisualGenArgs(
            model=checkpoint,
            parallel_config={"ulysses_size": world_size, "text_encoder_tp_size": encoder_tp},
            attention_config={"backend": "VANILLA"},
            torch_compile_config={"enable": False},
            cuda_graph_config={"enable": False},
            compilation_config={"skip_warmup": True},
        )
        engine = VisualGen(model=checkpoint, args=args)
        try:
            output = engine.generate(
                inputs="A woman smiles and waves in a sunlit park, with birds chirping.",
                params=VisualGenParams(
                    height=128, width=128, num_frames=124, num_inference_steps=4, seed=42
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            audio = torch.as_tensor(output.audio).cpu()
            assert video.shape == (1, 124, 128, 128, 3)
            assert video.float().std() > 1
            assert torch.isfinite(audio).all()
            videos.append(video)
            audios.append(audio)
            torch.save(
                {"video": video, "audio": audio},
                tmp_path / f"h3_world{world_size}_encoder_tp{encoder_tp}.pt",
            )
        finally:
            engine.shutdown()
    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    score = _mean_lpips_distance(videos[1], reference)
    audio_score = _multi_resolution_log_stft_distance(audios[1], audios[0])
    print(f"H3 encoder TP=2/{world_size}: LPIPS={score:.6f}, audio log-STFT={audio_score:.6f}")
    assert score < 0.05, f"Encoder TP changed video: LPIPS={score:.6f}"
    assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD


@pytest.mark.parametrize(
    "ulysses_size,vae_size,backend",
    [(2, 2, "VANILLA"), (4, 3, "VANILLA"), (2, 2, "FA4"), (2, 2, "TRTLLM"), (2, 2, "FLASHINFER")],
)
def test_minimax_h3_ulysses_parallel_vae_lpips(tmp_path, ulysses_size, vae_size, backend):
    """Compare real Ulysses + tiled VAE collectives with a fresh one-GPU baseline."""
    from defs.visual_gen.minimax_h3.quality_utils import (
        MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD,
        _mean_lpips_distance,
        _minimax_h3_checkpoint_path,
    )
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    if torch.cuda.device_count() < ulysses_size:
        pytest.skip(f"Requires {ulysses_size} CUDA GPUs")
    prompt = "A woman smiles and waves in a sunlit park, with birds chirping."
    videos, audios = [], []
    for size, decode_size in ((1, 1), (ulysses_size, 1), (ulysses_size, vae_size)):
        args = VisualGenArgs(
            model=_minimax_h3_checkpoint_path(),
            parallel_config={"ulysses_size": size, "parallel_vae_size": decode_size},
            attention_config={"backend": backend},
            torch_compile_config={"enable": True},
            cuda_graph_config={"enable": False},
            compilation_config={"skip_warmup": True},
        )
        engine = VisualGen(model=args.model, args=args)
        try:
            output = engine.generate(
                inputs=prompt,
                params=VisualGenParams(
                    height=384,
                    width=384,
                    num_frames=124,
                    num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
                    seed=42,
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            assert video.shape == (1, 124, 384, 384, 3)
            assert video.float().std() > 1
            assert torch.isfinite(torch.as_tensor(output.audio)).all()
            torch.save(
                {"video": video, "audio": output.audio},
                tmp_path / f"h3_ulysses{size}_vae{decode_size}.pt",
            )
            videos.append(video)
            audios.append(torch.as_tensor(output.audio).cpu())
        finally:
            engine.shutdown()
    # The tiled decoder batches a rank's tiles into one call, so the two runs decode with
    # different batch shapes; compiled kernels may then round a pixel to the adjacent uint8
    # level. Audio takes no tile batching and stays exact.
    torch.testing.assert_close(videos[2], videos[1], atol=1, rtol=0)
    torch.testing.assert_close(audios[2], audios[1])
    for decode_size, audio in zip((1, vae_size), audios[1:]):
        audio_score = _multi_resolution_log_stft_distance(audio, audios[0])
        print(
            f"H3 Ulysses={ulysses_size} + parallel_vae_size={decode_size} vs single GPU: "
            f"audio log-STFT={audio_score:.6f}"
        )
        assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD, (
            f"H3 parallelism changed audio: log-STFT={audio_score:.6f} "
            f">= {MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD}"
        )
    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    score = _mean_lpips_distance(videos[2], reference)
    print(
        f"H3 Ulysses={ulysses_size} + parallel_vae_size={vae_size} vs single GPU: mean LPIPS={score:.6f}"
    )
    assert score < MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD, (
        f"H3 parallelism changed output: mean LPIPS={score:.6f} "
        f">= {MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD}"
    )


def test_minimax_h3_tp2_example_lpips(tmp_path):
    """Check the shipped TP2 configuration against a same-build TP1 pipeline."""
    from defs.visual_gen.minimax_h3.quality_utils import (
        MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD,
        MINIMAX_H3_LPIPS_THRESHOLD,
        _mean_lpips_distance,
        _minimax_h3_checkpoint_path,
    )
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    config_path = (
        Path(__file__).resolve().parents[6] / "examples/visual_gen/configs/minimax-h3-bf16-tp2.yaml"
    )
    model = _minimax_h3_checkpoint_path()
    prompt = "A woman smiles and waves in a sunlit park, with birds chirping."
    videos, audios = [], []
    for tp_size in (1, 2):
        args = VisualGenArgs.from_yaml(config_path, model=model)
        assert args.parallel_config.tp_size == 2
        assert args.attention_config.backend == "VANILLA"
        assert args.torch_compile_config.enable
        assert not args.cuda_graph_config.enable
        # Keep the reference identical to the example except for TP degree.
        args.parallel_config.tp_size = tp_size
        engine = VisualGen(model=model, args=args)
        try:
            output = engine.generate(
                inputs=prompt,
                params=VisualGenParams(
                    height=128,
                    width=128,
                    num_frames=124,
                    num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
                    seed=42,
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            audio = torch.as_tensor(output.audio).cpu()
            assert video.shape == (1, 124, 128, 128, 3)
            assert video.float().std() > 1
            assert torch.isfinite(audio).all()
            assert audio.float().std() > 0
            torch.save({"video": video, "audio": audio}, tmp_path / f"h3_tp{tp_size}.pt")
            videos.append(video)
            audios.append(audio)
        finally:
            engine.shutdown()

    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    video_score = _mean_lpips_distance(videos[1], reference)
    audio_score = _multi_resolution_log_stft_distance(audios[1], audios[0])
    print(f"H3 TP2 example vs TP1: LPIPS={video_score:.6f}, audio log-STFT={audio_score:.6f}")
    assert video_score < MINIMAX_H3_LPIPS_THRESHOLD, (
        f"H3 TP2 changed video: LPIPS={video_score:.6f} >= {MINIMAX_H3_LPIPS_THRESHOLD}"
    )
    assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD, (
        f"H3 TP2 changed audio: log-STFT={audio_score:.6f} >= {MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD}"
    )
