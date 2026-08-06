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

import pytest
import torch
from backend_case import BackendCase, generate_inputs, run_backend, run_case
from utils.util import isSM100Family

pytestmark = pytest.mark.skipif(
    not isSM100Family(),
    reason="PrimsTS attention kernels require SM100 or SM103",
)


_QWEN2_7B = {
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "dtype": "bfloat16",
    "kv_layout": "HND",
    "page_size": 32,
    "rope": {
        "dim": 128,
        "theta": 1_000_000.0,
        "max_positions": 8192,
        "is_neox": True,
    },
    "fused_rope": True,
}

_DEEPSEEK_V3_LITE_MLA = {
    "num_heads": 32,
    "num_kv_heads": 1,
    "head_dim": 192,
    "dtype": "bfloat16",
    "kv_layout": "HND",
    "page_size": 32,
    "is_mla": True,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize(
    "phase_args",
    [
        pytest.param(
            {
                "seq_lens": [65, 37],
                "num_cached_tokens": [0, 0],
                "num_contexts": 2,
            },
            id="context",
        ),
        pytest.param(
            {
                "seq_lens": [1, 1],
                "num_cached_tokens": [64, 96],
                "num_contexts": 0,
            },
            id="generation",
        ),
        pytest.param(
            {
                "seq_lens": [41, 1],
                "num_cached_tokens": [0, 63],
                "num_contexts": 1,
            },
            id="mixed",
        ),
    ],
)
def test_prims_ts_qwen2_gqa(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
    phase_args: dict,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_QWEN2_7B,
        **phase_args,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    run_case(case)


def test_prims_ts_fp16_dense_context_with_alternate_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        seq_lens=[67, 33],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="float16",
        causal=False,
        kv_layout="HND",
        page_size=64,
        use_kv_cache_manager_v2=True,
    )

    results = run_case(case)

    assert "TRTLLM" in results


def test_prims_ts_cuda_graph_replay_refreshes_qkv_and_page_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_QWEN2_7B,
        seq_lens=[1, 1],
        num_cached_tokens=[64, 96],
        num_contexts=0,
        use_kv_cache_manager_v2=True,
    )
    inputs = generate_inputs(case, seed=0)
    golden = run_backend(
        case,
        "VANILLA",
        inputs,
        kv_dtype=case.compute_dtype,
        kv_layout="NHD",
    )

    def swap_requests_before_replay(metadata, static_inputs) -> None:
        qkv = static_inputs["q"]
        qkv.copy_(qkv.flip(0).clone())

        block_offsets = metadata.kv_cache_block_offsets
        block_offsets.copy_(block_offsets.flip(1).clone())

        kv_lens = metadata.kv_lens_cuda_runtime
        kv_lens.copy_(kv_lens.flip(0).clone())

    actual = run_backend(
        case,
        "TRTLLM",
        inputs,
        kv_dtype=case.compute_dtype,
        fuse_rope=True,
        cuda_graph=True,
        kv_layout="HND",
        before_cuda_graph_replay=swap_requests_before_replay,
    )

    torch.testing.assert_close(
        actual,
        golden.flip(0),
        atol=3e-2,
        rtol=3e-3,
    )


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_prims_ts_deepseek_v3_lite_mla_generation(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_DEEPSEEK_V3_LITE_MLA,
        seq_lens=[1, 1],
        num_cached_tokens=[64, 96],
        num_contexts=0,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    run_case(case)


def test_prims_ts_unsupported_context_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha

    calls = {"fallback": 0, "prims_context": 0}
    fallback_forward = FallbackFmha.forward
    prims_context = PrimsTSFmha.run_context

    def counted_fallback(self, *args, **kwargs):
        calls["fallback"] += 1
        return fallback_forward(self, *args, **kwargs)

    def counted_prims_context(self, *args, **kwargs):
        calls["prims_context"] += 1
        return prims_context(self, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", counted_fallback)
    monkeypatch.setattr(PrimsTSFmha, "run_context", counted_prims_context)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts,fallback")
    case = BackendCase(
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[65, 37],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="bfloat16",
        kv_layout="HND",
        page_size=32,
    )

    run_case(case)

    assert calls["fallback"] > 0
    assert calls["prims_context"] == 0
