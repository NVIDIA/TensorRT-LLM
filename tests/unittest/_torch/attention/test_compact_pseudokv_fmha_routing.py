# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FMHA_ROOT = REPO_ROOT / "tensorrt_llm" / "_torch" / "attention_backend" / "fmha"


def test_flashinfer_trtllm_gen_rejects_compact_pseudokv_carrier_before_dispatch() -> None:
    source = (FMHA_ROOT / "flashinfer_trtllm_gen.py").read_text(encoding="utf-8")

    guard = 'if fwd.compact_pseudokv is not None:'
    reason = 'return False, "trtllm-gen does not support compact pseudo-KV."'

    assert guard in source
    assert reason in source
    assert source.index(guard) < source.index("if meta.kv_cache_block_offsets is None:")


def test_fallback_fmha_forwards_compact_pseudokv_carrier_to_thop_attention() -> None:
    source = (FMHA_ROOT / "fallback.py").read_text(encoding="utf-8")

    required_kwargs = [
        "compact_pseudokv_key=forward_args.compact_pseudokv_key",
        "compact_pseudokv_value=forward_args.compact_pseudokv_value",
        "compact_pseudokv_positions=forward_args.compact_pseudokv_positions",
        "compact_pseudokv_causal_mask=forward_args.compact_pseudokv_causal_mask",
        "compact_pseudokv_source_seq_len=forward_args.compact_pseudokv_source_seq_len",
    ]

    for kwarg in required_kwargs:
        assert kwarg in source
