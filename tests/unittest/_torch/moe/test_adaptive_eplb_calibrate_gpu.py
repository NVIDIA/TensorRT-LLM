# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of adaptive EPLB's time model calibration at start-up: DeepSeek-V4-Pro NVFP4 on MegaMoE
CuTe DSL, DEP4 on one node. The first start calibrates in the CUDA graph warmup and caches the
calibration; the second start of the same run takes it from the cache and calibrates nothing. Every
rank logs at info, so the run's log holds each decode bucket's tile rows and tactic source, the
cache's hit or miss and any reason adaptive EPLB stays off."""

import json
import tempfile
import time
from pathlib import Path

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig
from tensorrt_llm.logger import logger


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="DEP4 on four GPUs")
def test_the_first_start_calibrates_and_the_second_reuses_the_cache(
    request, tmp_path, record_property
):
    from utils.llm_data import llm_models_root

    root = llm_models_root()
    if root is None or not (root / "DeepSeek-V4-Pro-nvfp4").exists():
        pytest.skip("no DeepSeek-V4-Pro NVFP4 checkpoint")
    model = root / "DeepSeek-V4-Pro-nvfp4"
    experts = json.loads((model / "config.json").read_text())["n_routed_experts"]
    junit = request.config.option.xmlpath  # a new empty cache, kept beside the run's results
    cache = (
        Path(tempfile.mkdtemp(".calibration", Path(junit).stem, Path(junit).parent))
        if junit
        else tmp_path
    )
    balancer = MoeAdaptiveLoadBalancerConfig(
        num_slots=experts + 4 * 4, calibration_cache_dir=str(cache)
    )
    logger.set_level("info")  # the leader worker's level; the others read TLLM_LOG_LEVEL
    starts, cached = [], []
    for start_index in range(2):
        start = time.monotonic()
        with LLM(
            str(model),
            tensor_parallel_size=4,
            moe_expert_parallel_size=4,
            enable_attention_dp=True,
            moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL", load_balancer=balancer),
            max_batch_size=16,
            cuda_graph_config=CudaGraphConfig(max_batch_size=16),
            max_seq_len=1024,
            max_num_tokens=1024,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
            custom_tokenizer="deepseek_v4",
            enable_iter_perf_stats=True,
            max_stats_len=-1,
            env_overrides={"TLLM_LOG_LEVEL": "INFO"},
        ) as llm:
            starts.append(time.monotonic() - start)
            out = llm.generate(["The capital of France is"] * 16, SamplingParams(max_tokens=32))
            stats = llm.get_stats(timeout=2)
        files = sorted(p.name for p in cache.iterdir())
        record_property(f"start{start_index}_cache_files", json.dumps(files))
        record_property(f"start{start_index}_stats_rows", len(stats))
        assert all(o.outputs[0].text for o in out)
        # The runtime runs (it reports its counters), on a time model of the cache.
        assert any("moeLoadBalancerStats" in s for s in stats), (start_index, files, stats[-1:])
        (path,) = (p for p in cache.glob("*.json") if not p.name.endswith(".times.json"))
        cached.append((path.stat().st_mtime_ns, path.read_text()))
    assert cached[0] == cached[1]  # the second start only read it
    doc = json.loads(cached[0][1])
    record_property("start_s", [round(s, 1) for s in starts])
    record_property("calibration_s", doc["measurement"]["seconds"])
    record_property("host_us_per_launch", doc["measurement"]["host_us_per_launch"])
    # Every decode bucket resolved its tactic in the warmup's first pass: none is refused unlaunched.
    assert all(doc["identity"]["tile_rows"].values()), doc["identity"]["tile_rows"]
    keys = ("held", "scale", "se", "holdout_z", "rmse_us", "median_us", "designs")
    # Every bucket's state: held, or refused with its z (no model there, so no decision there).
    for bucket in doc["buckets"]:
        fit = {k: bucket[k] for k in keys + ("per_active_slot_us",)}
        record_property(f"bucket_{bucket['tokens']}", json.dumps(fit))
    record_property("cache_dir", str(cache))
