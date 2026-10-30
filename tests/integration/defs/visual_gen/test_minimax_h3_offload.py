# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-checkpoint H3 offload parity and peak-memory regression test."""

import gc
import json
import os
import time
from pathlib import Path
from threading import Event, Thread

import pytest
import torch
from test_common.llm_data import get_checkpoint

from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_minimax_h3_offload_memory_and_parity(tmp_path: Path) -> None:
    if torch.cuda.get_device_properties(0).total_memory < 140 * 1024**3:
        pytest.skip("The resident H3 baseline requires at least 140 GiB of GPU memory")
    checkpoint = os.environ.get("MINIMAX_H3_CHECKPOINT") or get_checkpoint("MiniMax-H3")
    assert Path(checkpoint).is_dir(), f"MiniMax-H3 checkpoint not found: {checkpoint}"
    request = dict(
        prompt="A spacecraft passes an icy moon, with a deep engine rumble.",
        seed=42,
        height=128,
        width=128,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )
    measurements = {}
    results = {}
    for offload in (False, True):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # VAE calibration resets PyTorch peak counters while loading. Sample
        # allocation independently so loading before calibration is included.
        stop_sampling = Event()
        sampled_peak = [0]

        def sample_memory() -> None:
            while not stop_sampling.is_set():
                sampled_peak[0] = max(sampled_peak[0], torch.cuda.memory_allocated(0))
                stop_sampling.wait(0.01)

        sampler = Thread(target=sample_memory, daemon=True)
        sampler.start()
        start = time.perf_counter()
        args = VisualGenArgs(
            model=checkpoint,
            cpu_offload_config={"enable": offload},
            cuda_graph_config={"enable": False},
            attention_config={"backend": "VANILLA"},
        )
        pipeline = None
        try:
            pipeline = PipelineLoader(args).load(skip_warmup=True)
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - start
            load_peak = max(sampled_peak[0], torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = pipeline.forward(**request)
            torch.cuda.synchronize()
            name = "offload" if offload else "resident"
            measurements[name] = {
                "load_seconds": load_seconds,
                "inference_seconds": time.perf_counter() - start,
                "load_sampled_peak_allocated_bytes": load_peak,
                "inference_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            video, audio = output.video.cpu(), output.audio.cpu()
            assert video.shape == (1, 124, 128, 128, 3)
            assert video.float().std() > 0
            assert audio.shape[:2] == (1, 2)
            assert torch.isfinite(audio).all() and audio.abs().max() > 0
            results[name] = (video, audio)
            torch.save({"video": video, "audio": audio}, tmp_path / f"{name}.pt")
            del output
            if offload:
                repeated = pipeline.forward(**request)
                torch.testing.assert_close(repeated.video.cpu(), video, rtol=0, atol=0)
                torch.testing.assert_close(repeated.audio.cpu(), audio, rtol=0, atol=0)
                del repeated
        finally:
            stop_sampling.set()
            sampler.join()
            if pipeline is not None:
                pipeline.offloader.cleanup()
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

    import lpips

    metric = lpips.LPIPS(net="alex").eval()
    resident_video, resident_audio = results["resident"]
    offload_video, offload_audio = results["offload"]
    scores = []
    with torch.inference_mode():
        for start in range(0, resident_video.shape[1], 8):
            reference = resident_video[0, start : start + 8].permute(0, 3, 1, 2).float()
            candidate = offload_video[0, start : start + 8].permute(0, 3, 1, 2).float()
            scores.append(metric(reference / 127.5 - 1, candidate / 127.5 - 1).flatten())
    measurements["lpips"] = torch.cat(scores).mean().item()
    measurements["request"] = request
    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
    print(json.dumps(measurements, indent=2), flush=True)
    assert measurements["lpips"] < 0.01
    torch.testing.assert_close(offload_audio, resident_audio, rtol=1e-4, atol=1e-5)
    for phase in ("load_sampled_peak_allocated_bytes", "inference_peak_allocated_bytes"):
        assert measurements["offload"][phase] < measurements["resident"][phase] - 1024**3
