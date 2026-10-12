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

"""Shared DS-V4.1 baseline settings and deterministic completion prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.tokenizer import TokenizerBase

MAX_BATCH_SIZE = 8
MAX_NUM_TOKENS = 2048
MAX_SEQ_LEN = 8192
MAX_OUTPUT_TOKENS = 32


@dataclass(frozen=True)
class SmokeCase:
    """A ready-to-submit completion prompt, not an unformatted chat message."""

    name: str
    prompt: str
    expected: str
    input_tokens: int


def model_kwargs(tp_size: int = 4, *, attention_dp: bool = False) -> dict[str, object]:
    """Keep aggregate and disaggregate workers on the same non-speculative baseline."""
    return {
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": tp_size,
        "enable_attention_dp": attention_dp,
        "attn_backend": "TRTLLM",
        "moe_config": {"backend": "CUTLASS"},
        "allreduce_strategy": "NCCL",
        "max_batch_size": MAX_BATCH_SIZE,
        "max_num_tokens": MAX_NUM_TOKENS,
        "max_seq_len": MAX_SEQ_LEN,
        "enable_chunked_prefill": True,
        "trust_remote_code": True,
        "custom_tokenizer": "deepseek_v41",
        "speculative_config": None,
        "sparse_attention_config": {
            "algorithm": "csa2",
            "use_fp8_staging": True,
        },
        "kv_cache_config": {
            # CSA2 owns NVFP4 main, MXFP4 index and FP8 SWA formats independently of this dtype.
            "dtype": "fp8_ds_mla",
            "tokens_per_block": 128,
            # Leave headroom for CSA2 pool partitioning.
            "max_tokens": 2 * MAX_BATCH_SIZE * MAX_SEQ_LEN,
            "free_gpu_memory_fraction": 0.6,
            "enable_block_reuse": False,
            "enable_partial_reuse": False,
            "enable_swa_scratch_reuse": False,
        },
        "cuda_graph_config": {"enable_padding": True, "batch_sizes": [1, 2, 4, 8]},
    }


def baseline_env(environ: dict[str, str]) -> dict[str, str]:
    """Exclude inherited experimental V4.1 switches from the serving baseline."""
    return {name: value for name, value in environ.items() if not name.startswith("TRTLLM_V41_")}


def build_smoke_cases(tokenizer: TokenizerBase) -> list[SmokeCase]:
    """Build chat-formatted smoke prompts, including one chunked recall request."""
    recall = (
        "Ledger 47 was filed under archive reference number 60931. Keep that number in mind; "
        "you will be asked for it at the end of this text.\n\n"
    )
    filler = (
        "The archivist checks paper condition, documents provenance, and stores records in "
        "labelled boxes. Researchers consult a finding aid before requesting material. "
        "Preservation and access both matter when maintaining a collection.\n"
    )
    question = (
        "\nQuestion: according to the text above, what is the archive reference number for "
        "Ledger 47? Reply with the digits only.\nAnswer:"
    )
    while len(tokenizer.encode(recall + question, add_special_tokens=False)) <= 2 * MAX_NUM_TOKENS:
        recall += filler
    prompts = []
    for name, message, expected in (
        ("long_recall", recall + question, "60931"),
        (
            "arithmetic",
            "What is 17 multiplied by 23? Reply exactly in the format Product: NUMBER.",
            "Product: 391",
        ),
        (
            "code_reasoning",
            "What does Python print for sum([2, 3, 5])? "
            "Reply exactly in the format Result: NUMBER.",
            "Result: 10",
        ),
    ):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        prompts.append((name, prompt, expected))
    cases = []
    for name, prompt, expected in prompts:
        input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        assert input_tokens + MAX_OUTPUT_TOKENS <= MAX_SEQ_LEN, name
        # One-token answers can finish on context without transferring any KV.
        assert len(tokenizer.encode(expected, add_special_tokens=False)) > 1, name
        cases.append(SmokeCase(name, prompt, expected, input_tokens))
    return cases
