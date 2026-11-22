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

"""CPU regression for request admission and the scheduler's speculative verify width."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


def _find_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "tensorrt_llm").is_dir():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def _load_host_module(relative_path: str):
    # Import the host policy without initializing the TensorRT-LLM CUDA bindings.
    path = _find_repo_root() / "tensorrt_llm" / relative_path
    spec = spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget = _load_host_module("_torch/speculative/draft_budget.py")
admission = _load_host_module("executor/sequence_limits.py")


@pytest.fixture
def request_factory():
    def make_request(prompt_len, generated, max_tokens):
        return SimpleNamespace(
            is_dummy=False,
            total_input_len_cp=prompt_len,
            py_orig_prompt_len=prompt_len,
            get_num_tokens=Mock(return_value=prompt_len + generated),
            max_new_tokens=max_tokens,
            py_max_new_tokens=max_tokens,
            py_draft_tokens=list(range(1, 8)),
            py_needs_onehot_draft_probs=False,
        )

    return make_request


@pytest.fixture
def schedule():
    mode = SimpleNamespace(is_pard=lambda: False, support_dynamic_draft_len=lambda: True)
    config = SimpleNamespace(spec_dec_mode=mode, draft_len_schedule=None, use_rejection_sampling=True)

    def prepare(requests, max_seq_len):
        step_limit = budget.batch_draft_budget(requests, 7, max_seq_len, 1, is_pard=False)
        batch = SimpleNamespace(batch_size=len(requests), generation_requests=requests)
        return budget.resolve_draft_len(
            config, batch, max_draft_len=7, static_draft_len=7, draft_budget=step_limit
        )

    return prepare


@pytest.mark.parametrize("accepted_drafts", [0, 1, 2])
def test_verify_prefix_and_finish_at_context_limit(request_factory, schedule, accepted_drafts):
    max_seq_len = 32
    req = request_factory(prompt_len=24, generated=5, max_tokens=64)
    tokens = [None] * max_seq_len
    cursor = req.get_num_tokens(0)
    width = schedule([req], max_seq_len)
    assert width == 2
    assert req.py_draft_tokens == [1, 2]

    for token in req.py_draft_tokens[:accepted_drafts] + [99]:
        tokens[cursor] = token
        cursor += 1

    # Reject subsequent proposals so every possible first acceptance reaches
    # the same endpoint, including a target-only final step when needed.
    while cursor < max_seq_len:
        req.get_num_tokens.return_value = cursor
        width = schedule([req], max_seq_len)
        assert width == max_seq_len - cursor - 1
        if cursor == max_seq_len - 1:
            assert width == 0
            assert req.py_draft_tokens == []
        tokens[cursor] = 99
        cursor += 1
    assert cursor == max_seq_len
    assert tokens[-1] == 99


def test_output_budget_limits_the_whole_verify_batch(request_factory, schedule):
    short = request_factory(prompt_len=8, generated=2, max_tokens=3)
    long = request_factory(prompt_len=8, generated=2, max_tokens=24)
    assert schedule([short, long], 64) == 0
    assert short.py_draft_tokens == long.py_draft_tokens == []
    # Static decoding preserves empty proposals when bounded:
    assert schedule([long], 64) == 7
    assert long.py_draft_tokens == []
    # Fresh proposals get the full draft budget:
    long.py_draft_tokens = list(range(1, 8))
    assert schedule([long], 64) == 7
    assert long.py_draft_tokens == list(range(1, 8))


@pytest.mark.parametrize("requested, expected", [(65536, 158), (158, 158), (1, 1)])
def test_admission_limits_native_and_python_requests(request_factory, requested, expected):
    req = request_factory(1048418, 0, requested)
    admission.clamp_generation_request(req, 1048576, 1)
    assert req.max_new_tokens == req.py_max_new_tokens == expected


@pytest.mark.parametrize("prompt_len", [32, 33])
def test_admission_rejects_prompt_without_a_target_slot(request_factory, prompt_len):
    req = request_factory(prompt_len, 0, 5)
    with pytest.raises(ValueError, match="maximum context length is 32 tokens"):
        admission.clamp_generation_request(req, 32, 1)


def test_static_step_preserves_empty_proposals_when_bounded(request_factory):
    # Static decoding preserves proposals: a request intentionally entering
    # with no draft tokens must remain empty even when a draft_budget is set.
    mode = SimpleNamespace(is_pard=lambda: False, support_dynamic_draft_len=lambda: False)
    config = SimpleNamespace(spec_dec_mode=mode, draft_len_schedule=None, use_rejection_sampling=True)
    req_with_draft = request_factory(prompt_len=8, generated=2, max_tokens=24)
    req_without_draft = request_factory(prompt_len=8, generated=2, max_tokens=24)
    req_without_draft.py_draft_tokens = []

    batch = SimpleNamespace(batch_size=2, generation_requests=[req_with_draft, req_without_draft])
    width = budget.resolve_draft_len(
        config, batch, max_draft_len=7, static_draft_len=7, draft_budget=4
    )
    assert width == 4
    assert len(req_with_draft.py_draft_tokens) == 4
    assert req_without_draft.py_draft_tokens == []


def test_helix_child_requests_share_total_input_len_cp_capacity():
    # Helix ranks slice prompts across CP ranks. Both the parent request
    # and all child requests receive total_input_len_cp and must clamp to
    # the exact same capacity.
    parent = SimpleNamespace(
        is_dummy=False,
        total_input_len_cp=100,
        max_new_tokens=200,
        py_max_new_tokens=200,
    )
    child = SimpleNamespace(
        is_dummy=False,
        total_input_len_cp=100,
        max_new_tokens=200,
        py_max_new_tokens=200,
    )
    admission.clamp_generation_request(parent, max_seq_len=120, cp_size=2)
    admission.clamp_generation_request(child, max_seq_len=120, cp_size=2)
    assert parent.max_new_tokens == parent.py_max_new_tokens == 70
    assert child.max_new_tokens == child.py_max_new_tokens == 70


def test_remaining_output_tokens_detects_speculative_tail(request_factory):
    max_seq_len = 64
    max_total_draft_tokens = 7
    threshold = 2 * (max_total_draft_tokens + 1)

    near_limit = request_factory(prompt_len=40, generated=10, max_tokens=50)
    assert budget.remaining_output_tokens(near_limit, max_seq_len, cp_size=1) < threshold

    far_from_limit = request_factory(prompt_len=20, generated=10, max_tokens=50)
    assert budget.remaining_output_tokens(far_from_limit, max_seq_len, cp_size=1) >= threshold

