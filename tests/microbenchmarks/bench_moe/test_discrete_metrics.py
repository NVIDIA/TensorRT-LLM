# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discrete-metric perf-regression tripwires for MoE.

Unlike e2e throughput benchmarks, these assert *deterministic, discrete*
properties of the MoE path, so they are zero-/low-flake and can gate pre-merge:

  Test 1 ``test_moe_backend_selection`` -- the AUTO backend-routing logic
      (``ModelConfig.resolve_moe_backend``) resolves to the intended backend for
      a given (architecture, GPU SM, quant) combination. Pure CPU, deterministic.
      Guards the "AUTO routes to the wrong MoE backend on a new GPU" regression
      class -- e.g. PR#13399 (SM100 + FP8_BLOCK_SCALES must pick TRTLLM, not the
      silent CUTLASS fallback). Catalogued as PERF_FIX_HISTORY Cat 3.4 / Cat 13.2.

  Test 2 ``test_moe_forward_launch_count`` -- a CUDA-graph-captured MoE forward
      issues exactly the expected number of GPU kernels, and the requested
      backend is not silently downgraded. Guards the "fusion silently broke" /
      "CUDA graph coverage dropped" classes (PERF_FIX_HISTORY Cat 2 + Cat 11),
      where kernels-per-step jumps. Needs a GPU + the optional ``cupti`` package.

Why discrete metrics (vs timing): the expected value is a reviewed constant
asserted with ``==``, so there is no hardware-noise threshold to tune and the
test never flakes. An *intended* change (new fast kernel, new fused path) makes
the assertion fail on purpose -- update the golden value in the same PR so the
perf-behavior change is explicit and reviewed.
"""

from __future__ import annotations

import pytest
import torch

# --------------------------------------------------------------------------- #
# Test 1: backend selection (pure CPU, always runnable, gates pre-merge)
# --------------------------------------------------------------------------- #
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# (moe_backend, architecture, quant_algo, sm_version, expected_backend)
# sm_version is mocked so the test is hardware-independent and runs on CPU.
_BACKEND_SELECTION_CASES = [
    # Non-AUTO is always passed through unchanged.
    pytest.param("CUTLASS", "LlamaForCausalLM", None, 90, "CUTLASS", id="non_auto_passthrough"),
    # GptOss picks the best backend per SM family.
    pytest.param("AUTO", "GptOssForCausalLM", None, 100, "TRTLLM", id="gptoss_blackwell_sm100"),
    pytest.param("AUTO", "GptOssForCausalLM", None, 90, "TRITON", id="gptoss_hopper_sm90"),
    pytest.param("AUTO", "GptOssForCausalLM", None, 120, "CUTLASS", id="gptoss_sm120_fallback"),
    # PERF_FIX_HISTORY Cat 3.4 / PR#13399: AUTO + FP8 block-scales on SM100
    # MUST resolve to TRTLLM (CUTLASS FP8 block-scale path JIT-checks SM90 only).
    pytest.param(
        "AUTO",
        "DeepseekV3ForCausalLM",
        QuantAlgo.FP8_BLOCK_SCALES,
        100,
        "TRTLLM",
        id="dsv3_fp8block_sm100_PR13399",
    ),
    # Same config on Hopper stays on CUTLASS.
    pytest.param(
        "AUTO",
        "DeepseekV3ForCausalLM",
        QuantAlgo.FP8_BLOCK_SCALES,
        90,
        "CUTLASS",
        id="dsv3_fp8block_sm90",
    ),
    # No quant -> default CUTLASS.
    pytest.param("AUTO", "DeepseekV3ForCausalLM", None, 100, "CUTLASS", id="dsv3_no_quant_default"),
]


@pytest.mark.parametrize(
    "moe_backend, architecture, quant_algo, sm_version, expected_backend",
    _BACKEND_SELECTION_CASES,
)
def test_moe_backend_selection(
    monkeypatch, moe_backend, architecture, quant_algo, sm_version, expected_backend
):
    """AUTO MoE-backend routing resolves to the intended backend per (arch, SM, quant)."""
    import tensorrt_llm._torch.model_config as mc_mod

    # Mock the SM-version probes so the routing logic is exercised on any host.
    monkeypatch.setattr(mc_mod, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(mc_mod, "is_sm_100f", lambda: 100 <= sm_version < 110)

    quant_config = QuantConfig(quant_algo=quant_algo) if quant_algo is not None else None

    resolved = mc_mod.ModelConfig.resolve_moe_backend(moe_backend, architecture, quant_config)

    assert resolved == expected_backend, (
        f"AUTO routing regression: arch={architecture} sm={sm_version} "
        f"quant={quant_algo} resolved to {resolved!r}, expected {expected_backend!r}. "
        f"A silent fallback here is the PERF_FIX_HISTORY Cat 3.4 regression class."
    )


# --------------------------------------------------------------------------- #
# Test 2: MoE forward launch count (GPU + optional cupti; golden-pinned)
# --------------------------------------------------------------------------- #

# Golden kernel counts, captured ONCE on a GPU and pinned here. Keyed by
# (case_id, sm_arch) -- the count is arch-dependent (empirically SM120=9, SM90=10
# for the same case), so a flat per-case golden would false-fire cross-GPU. Leave a
# key absent to have the test print the observed count and skip (bootstrap a new
# arch). Updating a value is an explicit, reviewed perf-behavior change.
#   build 2774 / wheel 1.3.0rc18:
#   - sm120 (RTX Pro 6000): 9 kernels, verified 5/5 zero-variance (local docker).
#   - sm90  (H100 viking, computelab slurm): 10 kernels, verified 3/3 zero-variance.
#   - sm103 (B300-pcie, computelab slurm): 10 kernels, single bootstrap obs (TODO K-verify).
_EXPECTED_LAUNCH_COUNT: dict[str, dict[str, int]] = {
    "cutlass_e8_k2_h256_bf16_t8": {"sm120": 9, "sm90": 10, "sm103": 10},
}


def _current_sm() -> str:
    """e.g. 'sm90' (H100), 'sm100' (B200), 'sm103' (B300), 'sm120' (RTX Pro 6000)."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def _build_tiny_moe(backend: str, dtype: torch.dtype):
    """Build a small bf16 MoE via the existing bench_moe builder (single GPU)."""
    from tensorrt_llm.mapping import Mapping

    from .build import _build_moe_module
    from .specs import ConfigSpec, ModelSpec

    model = ModelSpec(
        name="custom",
        num_experts=8,
        top_k=2,
        hidden_size=256,
        intermediate_size=256,
        quant_algo=None,  # bf16: avoids quant weight-prep branches
        routing_method="RENORMALIZE",
    )
    config = ConfigSpec(
        backend=backend,
        parallel_mode="DEP",
        moe_ep_size=1,
        moe_tp_size=1,
        enable_attention_dp=False,
        comm_method="AUTO",
        cuda_graph=True,
    )
    mapping = Mapping(world_size=1, rank=0, moe_ep_size=1, moe_tp_size=1)
    device = torch.device("cuda")

    moe, routing_logits_dtype = _build_moe_module(
        model=model,
        config=config,
        mapping=mapping,
        moe_backend=backend,
        use_cuda_graph=True,
        max_num_tokens=8,
        use_low_precision_moe_combine=False,
        enable_perfect_router=False,
        dtype=dtype,
        routing_logits_dtype=torch.float32,
        device=device,
    )
    return moe, model, routing_logits_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_moe_backend_not_silently_downgraded():
    """The requested MoE backend is honored, not silently downgraded (Cat 3.4).

    Split out from ``test_moe_forward_launch_count`` so this fallback-regression
    guard runs on ANY GPU host, not only ones with the optional ``cupti`` package
    installed (the launch-count test skips early when cupti is unavailable).
    """
    from .backend import MoeBackendType
    from .build import _backend_name_from_module

    backend = MoeBackendType.CUTLASS.value
    moe, _model, _routing_logits_dtype = _build_tiny_moe(backend, torch.bfloat16)

    actual_backend = _backend_name_from_module(moe)
    assert actual_backend == backend, (
        f"requested backend {backend!r} was silently downgraded to "
        f"{actual_backend!r} for this config (silent fallback regression)."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_moe_forward_launch_count():
    """A CUDA-graph MoE forward issues a fixed, expected number of kernels.

    Asserts two discrete properties:
      1. the requested backend is not silently downgraded (no fallback), and
      2. the kernel count equals the pinned golden (fusion / graph coverage intact).
    """
    from .backend import MoeBackendType
    from .build import _backend_name_from_module
    from .timing.cuda_graph import _time_moe_forward_cuda_graph
    from .timing.cupti import _try_init_cupti

    cupti_ctx = _try_init_cupti()
    if not cupti_ctx.ok:
        pytest.skip("cupti package unavailable; kernel-count breakdown not possible")

    backend = MoeBackendType.CUTLASS.value
    dtype = torch.bfloat16
    num_tokens = 8
    case_id = f"cutlass_e8_k2_h256_bf16_t{num_tokens}"

    moe, model, routing_logits_dtype = _build_tiny_moe(backend, dtype)

    # ---- Assertion 1: no silent backend fallback (PERF_FIX_HISTORY Cat 3.4) ----
    actual_backend = _backend_name_from_module(moe)
    assert actual_backend == backend, (
        f"requested backend {backend!r} was silently downgraded to "
        f"{actual_backend!r} for this config (silent fallback regression)."
    )

    # ---- Run the CUDA-graph forward and count kernels via CUPTI ----
    x = torch.randn(num_tokens, model.hidden_size, dtype=dtype, device="cuda")
    router_logits = torch.randn(
        num_tokens, model.num_experts, dtype=routing_logits_dtype, device="cuda"
    )

    _times_ms, detailed_stats = _time_moe_forward_cuda_graph(
        moe,
        x,
        router_logits,
        all_rank_num_tokens=[num_tokens],
        warmup=3,
        iters=1,
        cupti_ctx=cupti_ctx,
        flush_l2=True,
    )
    observed = len(detailed_stats.get("moe_forward_kernels") or [])

    # ---- Assertion 2: kernel count == pinned golden (per arch) ----
    sm = _current_sm()
    expected = _EXPECTED_LAUNCH_COUNT.get(case_id, {}).get(sm)
    if expected is None:
        pytest.skip(
            f"No golden pinned for {case_id!r} on {sm}. Observed moe_forward kernel "
            f"count = {observed}. Pin _EXPECTED_LAUNCH_COUNT[{case_id!r}][{sm!r}] to gate."
        )

    assert observed == expected, (
        f"MoE forward launch count changed for {case_id} on {sm}: observed {observed}, "
        f"golden {expected}. If intentional (new fusion / kernel), update the "
        f"golden in this PR; otherwise this is a Cat 2/11 fusion-or-graph regression."
    )
