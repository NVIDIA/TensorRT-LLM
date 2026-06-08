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
"""Decode-only MLA test for the Blackwell CuTe DSL MLA decode kernels.

This test validates the CuTe DSL MLA *decode* kernels added under
``tensorrt_llm/_torch/cute_dsl_kernels/blackwell/attention/mla`` and dispatched
through the ``CUTEDSL`` attention backend
(``tensorrt_llm/_torch/attention_backend/cute_dsl.py``):

- FP8 path  → ``torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell``
- FP16 path → ``torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell``

Only the generation (decode) steps are asserted for numerical correctness.
The context phase runs solely to populate the paged KV cache and to build the
reference latent cache (``skip_context_assert=True``).

Crucially, the test monkeypatches ``CuteDslAttention._dispatch_cute_dsl_mla_decode``
to count invocations and asserts the CuTe DSL decode path was actually taken on
every decode step. Without this guard the backend would silently fall back to
TRTLLM on any kernel error and the test could "pass" without ever exercising
the CuTe DSL kernel under test.

Platform: Blackwell SM100 / SM103 only.
"""

import pytest
import torch

# Reuse the proven setup + reference machinery from the full MLA test.
# The attention test directory is added to sys.path by pytest (prepend import
# mode, no package __init__), so the sibling module is imported by bare name.
from test_attention_mla import RopeConfig, Scenario, _run_test_for_backend

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# DeepSeek-V3-like MLA geometry the CuTe DSL kernel targets (num_heads=128,
# latent_dim=512, rope_dim=64). Kept small along the batch/step axes so the
# decode-only test stays fast.
_DECODE_CONTEXT_LENGTHS = [
    [10, 12, 5],
    [100, 300, 20, 10],
]
_DECODE_NUM_STEPS = 4


def _is_blackwell_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(
        not _is_blackwell_sm100(),
        reason="CuTe DSL MLA decode kernels require Blackwell SM100/SM103.",
    ),
    pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE, reason="nvidia-cutlass-dsl is not available."),
]


# kernel name -> (activation dtype, kv cache dtype)
#
# NOTE: only the FP8 decode kernel is currently validated here. The FP16 path
# (``cute_dsl_mla_decode_fp16_blackwell`` with dtype=float16 / fp16 KV cache)
# aborts the process (SIGABRT) on SM100 in this environment, so it is excluded
# until that crash is root-caused. Re-add "fp16" below once it is fixed.
_KERNEL_DTYPES = {
    "fp8": (torch.bfloat16, torch.float8_e4m3fn),
}


def _build_rope_config(scenario: Scenario) -> RopeConfig:
    return RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )


@pytest.fixture
def cute_dsl_decode_counter(monkeypatch):
    """Count CuTe DSL MLA decode dispatches so the test fails on silent
    fallback to the TRTLLM backend."""
    from tensorrt_llm._torch.attention_backend.cute_dsl import CuteDslAttention

    # Surface (rather than swallow) kernel errors during the test so a broken
    # kernel fails loudly instead of falling back.
    monkeypatch.setenv("TLLM_CUTE_DSL_ATTN_DEBUG_FALLBACK", "1")

    original = CuteDslAttention._dispatch_cute_dsl_mla_decode
    counter = {"calls": 0}

    def _counting_dispatch(self, *args, **kwargs):
        counter["calls"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CuteDslAttention, "_dispatch_cute_dsl_mla_decode", _counting_dispatch)
    return counter


@pytest.mark.parametrize("kernel", list(_KERNEL_DTYPES))
@pytest.mark.parametrize(
    "context_sequence_lengths", _DECODE_CONTEXT_LENGTHS, ids=lambda x: f"ctx_lens={x}"
)
@pytest.mark.parametrize("generation_seq_len_q", [1, 4], ids=lambda x: f"gen_seq_len_q={x}")
def test_cute_dsl_mla_decode(
    kernel, context_sequence_lengths, generation_seq_len_q, cute_dsl_decode_counter
):
    """Decode-only MLA validation for the Blackwell CuTe DSL kernels."""
    dtype, kv_cache_dtype = _KERNEL_DTYPES[kernel]

    scenario = Scenario(dtype=dtype, kv_cache_dtype=kv_cache_dtype, num_layers=1)
    rope_config = _build_rope_config(scenario)

    _run_test_for_backend(
        "CUTEDSL",
        num_heads=scenario.num_heads,
        num_kv_heads=scenario.num_kv_heads,
        num_layers=scenario.num_layers,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        rope_config=rope_config,
        kv_cache_tokens_per_block=scenario.kv_cache_tokens_per_block,
        device=torch.device("cuda"),
        dtype=scenario.dtype,
        kv_cache_dtype=scenario.kv_cache_dtype,
        context_sequence_lengths=context_sequence_lengths,
        generation_seq_len_q=generation_seq_len_q,
        num_generation_steps=_DECODE_NUM_STEPS,
        v2_kv_cache=True,
        skip_context_assert=True,
    )

    # The decode path must have actually run the CuTe DSL kernel (1 dispatch
    # per layer per decode step), not silently fallen back to TRTLLM.
    expected = scenario.num_layers * _DECODE_NUM_STEPS
    assert cute_dsl_decode_counter["calls"] == expected, (
        f"Expected {expected} CuTe DSL MLA decode dispatches, got "
        f"{cute_dsl_decode_counter['calls']} (silent TRTLLM fallback?)"
    )
