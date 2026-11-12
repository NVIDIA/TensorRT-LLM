# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end check of the asynchronous TRTLLM-Gen FMHA JIT warmup.

With ``TRTLLM_GEN_FMHA_ASYNC_WARMUP=1`` the warmup grid compiles on a
background thread, and the engine's pre-capture barrier waits for it and
re-verifies every sweep before CUDA graphs are captured. This test boots a
small model with CUDA graphs enabled and checks, from the engine process:

* capture succeeds (the LLM reaches readiness),
* the warmup exercised the NVRTC path at all (the count of distinct kernel
  configurations requested from the FMHA export library grew during warmup),
* the first requests after readiness compile nothing: that count is unchanged
  across them. The export library compiles a configuration the first time it
  is requested, so this is a direct measure of compilation activity, not a
  latency heuristic.

The flag is read once per process, so each arm runs in a fresh subprocess
with the worker in that process (``TLLM_WORKER_USE_SINGLE_PROCESS=1``), where
the counters are observable. The flag-off arm is the control: the same
invariant must hold for the synchronous warmup.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root
from utils.util import skip_pre_blackwell

_CHILD = textwrap.dedent("""
    import json
    import sys

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    model_dir = sys.argv[1]
    misses = torch.ops.trtllm.trtllm_gen_fmha_jit_num_cache_misses
    unknown = torch.ops.trtllm.trtllm_gen_fmha_jit_num_unknown_cache_results
    result = {"misses_before": misses()}
    llm = LLM(
        model=model_dir,
        attn_backend="TRTLLM",
        tensor_parallel_size=1,
        max_batch_size=8,
        max_num_tokens=512,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8]),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
    )
    with llm:
        result["async_enabled"] = bool(
            torch.ops.trtllm.trtllm_gen_fmha_async_jit_warmup_enabled())
        result["misses_ready"] = misses()
        prompts = [
            "The capital of France is",
            "Write one sentence about the ocean.",
            "List three prime numbers:",
            "Explain why the sky is blue in a few words.",
        ]
        outputs = llm.generate(prompts, SamplingParams(max_tokens=48))
        result["num_outputs"] = len(outputs)
        result["all_generated"] = all(
            len(o.outputs[0].token_ids) > 0 for o in outputs)
        result["misses_after_first_requests"] = misses()
        result["unknown_cache_results"] = unknown()
    print("RESULT " + json.dumps(result))
""")


def _run_arm(model_dir: Path, async_warmup: bool) -> dict:
    env = dict(os.environ)
    env["TRTLLM_GEN_FMHA_ASYNC_WARMUP"] = "1" if async_warmup else "0"
    # Run the executor worker in the child's own process so the kernel-key
    # counter reflects the engine that served the requests.
    env["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(model_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert proc.returncode == 0, (
        f"engine subprocess failed (async_warmup={async_warmup}):\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"no RESULT line in child stdout:\n{proc.stdout[-4000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def model_dir() -> Path:
    return llm_models_root() / "Qwen3/Qwen3-0.6B"


@skip_pre_blackwell
@pytest.mark.parametrize("async_warmup", [True, False], ids=["async_warmup", "sync_warmup"])
def test_first_requests_after_readiness_do_not_jit_compile(
    model_dir: Path, async_warmup: bool
) -> None:
    result = _run_arm(model_dir, async_warmup)

    assert result["async_enabled"] is async_warmup
    assert result["num_outputs"] == 4 and result["all_generated"]

    # Every compile request must have come back with a reported cache result;
    # otherwise the miss count is not a complete measure and the invariants
    # below could hold while compiles went uncounted.
    assert result["unknown_cache_results"] == 0, (
        f"{result['unknown_cache_results']} compile request(s) returned an "
        "unknown cache status; the miss count cannot be trusted"
    )

    # The warmup must have gone through the NVRTC path, otherwise the
    # invariant below would hold vacuously.
    assert result["misses_ready"] > result["misses_before"], (
        "warmup reported no kernel-cache misses; this model/config does not "
        "exercise the TRTLLM-Gen JIT path"
    )

    # No cache miss after readiness, i.e. the first requests compiled nothing:
    # every kernel they needed was compiled (and, for the async arm, verified)
    # before capture. A key evicted and recompiled would count as a miss too.
    assert result["misses_after_first_requests"] == result["misses_ready"], (
        f"{result['misses_after_first_requests'] - result['misses_ready']} "
        "kernel(s) were compiled by the first requests after readiness"
    )
