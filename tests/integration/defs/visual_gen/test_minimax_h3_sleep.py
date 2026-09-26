# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local/QA single-GPU H3 sleep test; requires the released model and host RAM."""

import gc
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

import psutil
import pytest
import torch
from test_common.llm_data import get_checkpoint

from tensorrt_llm import VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("release_cpu_backup", [False, True])
def test_minimax_h3_sleep_preserves_video_and_audio(release_cpu_backup: bool) -> None:
    if torch.cuda.get_device_properties(0).total_memory < 140 * 1024**3:
        pytest.skip("H3 FP8 sleep test requires a GPU with at least 140 GiB")
    checkpoint = os.environ.get("MINIMAX_H3_CHECKPOINT") or get_checkpoint("MiniMax-H3")
    if not Path(checkpoint).is_dir():
        pytest.fail(f"MiniMax-H3 checkpoint not found: {checkpoint}")
    config = VisualGenArgs(
        model=checkpoint,
        quant_config={"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True},
        cuda_graph_config={"enable": False},
        attention_config={"backend": "VANILLA"},
    )
    pipeline = PipelineLoader(config).load(
        skip_warmup=True,
        sleep_restore_mode="PINNED",
        sleep_release_cpu_backup=release_cpu_backup,
    )
    process = psutil.Process()
    request = dict(
        prompt="A spacecraft passes an icy moon, with a deep engine rumble.",
        seed=42,
        height=128,
        width=128,
        num_frames=124,
        frame_rate=24.0,
        num_inference_steps=4,
    )
    output = pipeline.forward(**request)
    reference_video, reference_audio = output.video.cpu(), output.audio.cpu()
    assert reference_video.shape == (1, 124, 128, 128, 3)
    assert reference_audio.shape[:2] == (1, 2)
    assert reference_video.float().std() > 0
    assert torch.isfinite(reference_audio).all()
    assert reference_audio.abs().max() > 0
    del output
    pointers = {name: tensor.data_ptr() for name, tensor in pipeline.named_parameters()}
    try:
        for cycle in range(2):
            time.sleep(10)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_before = torch.cuda.mem_get_info()[0]
            entered, finish, draining = Event(), Event(), Event()
            wait_for = pipeline._sleep_manager._condition.wait_for
            timings = {}

            def wait_for_generation(predicate: Callable[[], bool]) -> bool:
                timings["drain_start"] = time.perf_counter()
                draining.set()
                result = wait_for(predicate)
                timings["drain_end"] = time.perf_counter()
                return result

            def before_transformer(_module, _args) -> None:
                entered.set()
                assert finish.wait(120), "Sleep did not close admission"

            hook = pipeline.transformer.register_forward_pre_hook(before_transformer)
            with ThreadPoolExecutor(max_workers=2) as executor:
                generation = executor.submit(pipeline.forward, **request)
                try:
                    assert entered.wait(120), "Generation did not reach the transformer"
                    with patch.object(
                        pipeline._sleep_manager._condition, "wait_for", wait_for_generation
                    ):
                        sleeping = executor.submit(pipeline.sleep)
                        assert draining.wait(10), "Sleep did not start draining generation"
                        assert not sleeping.done()
                        with pytest.raises(RuntimeError, match="sleeping"):
                            pipeline.forward(**request)
                        finish.set()
                        output = generation.result(timeout=120)
                        sleeping.result(timeout=120)
                        timings["sleep_end"] = time.perf_counter()
                finally:
                    finish.set()
                    hook.remove()
            torch.testing.assert_close(output.video.cpu(), reference_video, rtol=0, atol=0)
            torch.testing.assert_close(output.audio.cpu(), reference_audio, rtol=0, atol=0)
            del output
            assert pipeline.is_sleeping
            released_gib = (torch.cuda.mem_get_info()[0] - free_before) / 1024**3
            assert released_gib > 50
            with pytest.raises(RuntimeError, match="asleep"):
                pipeline.forward(**request)
            asleep_rss = process.memory_info().rss
            wake_start = time.perf_counter()
            pipeline.wake_up()
            wake_s = time.perf_counter() - wake_start
            freed_host_gib = (asleep_rss - process.memory_info().rss) / 1024**3
            if release_cpu_backup:
                assert freed_host_gib > 50
            else:
                assert abs(freed_host_gib) < 1
            output = pipeline.forward(**request)
            torch.testing.assert_close(output.video.cpu(), reference_video, rtol=0, atol=0)
            torch.testing.assert_close(output.audio.cpu(), reference_audio, rtol=0, atol=0)
            del output
            print(
                {
                    "cycle": cycle + 1,
                    "drain_s": timings["drain_end"] - timings["drain_start"],
                    "offload_s": timings["sleep_end"] - timings["drain_end"],
                    "wake_s": wake_s,
                    "freed_host_gib": freed_host_gib,
                    "released_gib": released_gib,
                }
            )
            assert not pipeline.is_sleeping
            assert pointers == {
                name: tensor.data_ptr() for name, tensor in pipeline.named_parameters()
            }
        pipeline.sleep()
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
