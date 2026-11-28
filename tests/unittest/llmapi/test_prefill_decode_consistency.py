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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only
_SOURCE = Path(__file__).resolve().parents[3] / "examples/llm-api/llm_prefill_decode_consistency.py"
_SPEC = importlib.util.spec_from_file_location("prefill_decode_consistency", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_common_normalization_handles_small_probabilities_and_masked_padding() -> None:
    logits = torch.tensor([[0.0, -100.0, float("-inf")], [3.0, 3.0, float("-inf")]])
    assert probe._logprobs_from_logits(logits, [1, 0]) == pytest.approx([-100.0, -0.69314718])


def test_common_normalization_upcasts_bf16_logits() -> None:
    logits = torch.tensor([[10.0, 9.0]], dtype=torch.bfloat16)
    assert probe._logprobs_from_logits(logits, [1]) == pytest.approx([-1.31326169])


@pytest.mark.parametrize("extra_generation_row", [False, True])
def test_response_scores_use_the_preceding_position(extra_generation_row: bool) -> None:
    rows = [
        {token: SimpleNamespace(logprob=value)}
        for token, value in (
            (20, -9.0),
            (30, -0.1),
            (40, -0.2),
            (50, -0.3),
        )
    ]
    if extra_generation_row:
        rows.append({60: SimpleNamespace(logprob=-7.0)})
    assert probe._score_response(rows, [10, 20], [30, 40, 50]) == [-0.1, -0.2, -0.3]


def test_response_scores_with_one_token_prompt() -> None:
    rows = [{20: SimpleNamespace(logprob=-0.1)}, {30: SimpleNamespace(logprob=-0.2)}]
    assert probe._score_response(rows, [10], [20, 30]) == [-0.1, -0.2]


@pytest.mark.parametrize("rows", [[], [{99: SimpleNamespace(logprob=-0.1)}]])
def test_missing_scores_are_not_silently_dropped(rows: list) -> None:
    with pytest.raises(ValueError):
        probe._score_response(rows, [10], [20])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_scores_fail(value: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        probe._extract_logprobs([{20: SimpleNamespace(logprob=value)}], [20])


def test_prefill_boundary_does_not_inflate_decode_error() -> None:
    result = probe._summarize(
        [
            {
                "generation_logprobs": [-5.0, -0.2, -0.3],
                "prefill_logprobs": [[-1.0, -0.2, -0.4], [-1.0, -0.2, -0.4]],
            }
        ]
    )
    assert result["decode_vs_prefill"]["token_count"] == 2
    assert result["decode_vs_prefill"]["max_abs_delta"] == pytest.approx(0.1)
    assert result["prefill_boundary_vs_full_prefill"]["max_abs_delta"] == 4.0
    assert result["prefill_repeatability"]["max_abs_delta"] == 0.0


@pytest.mark.parametrize("reference,observed", [([], []), ([-1.0], [-1.0, -2.0])])
def test_comparison_rejects_unaligned_sequences(reference: list, observed: list) -> None:
    with pytest.raises(ValueError):
        probe._compare(reference, observed)


@pytest.mark.parametrize(
    "config",
    [
        {"max_batch_size": 2},
        {"enable_chunked_prefill": True},
        {"disable_overlap_scheduler": False},
        {"speculative_config": {"decoding_type": "MTP"}},
        {"kv_cache_config": {"enable_block_reuse": True}},
        {"max_num_tokens": 10},
    ],
)
def test_probe_rejects_confounding_execution_options(config: dict) -> None:
    with pytest.raises(ValueError):
        probe._engine_options(config, 256, 1)
