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
import torch

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
    config = SimpleNamespace(
        spec_dec_mode=mode, draft_len_schedule=None, use_rejection_sampling=True
    )

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


def test_short_output_preserves_full_width_for_other_requests(request_factory, schedule):
    short = request_factory(prompt_len=8, generated=2, max_tokens=3)
    long = request_factory(prompt_len=8, generated=2, max_tokens=24)
    assert schedule([short, long], 64) == 7
    assert short.py_draft_tokens == long.py_draft_tokens == list(range(1, 8))


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
    config = SimpleNamespace(
        spec_dec_mode=mode, draft_len_schedule=None, use_rejection_sampling=True
    )
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


@pytest.mark.parametrize(
    "linear, dynamic_tree, fixed_target, shared, overlap, reserve",
    [
        (True, False, False, False, True, 0),
        (True, False, True, False, False, 7),
        (True, False, False, True, True, 15),
        (True, True, False, False, False, 7),
        (False, False, False, False, True, 21),
    ],
)
def test_fixed_verify_layout_fits_at_admission(
    request_factory, linear, dynamic_tree, fixed_target, shared, overlap, reserve
):
    physical_width = 7 if linear else 13
    cfg = SimpleNamespace(
        max_draft_len=7,
        tokens_per_gen_step=physical_width + 1,
        is_linear_tree=linear,
        use_dynamic_tree=dynamic_tree,
    )
    assert (
        budget.fixed_draft_reserve(
            cfg,
            SimpleNamespace(modeling_v2_target=object if fixed_target else None),
            SimpleNamespace(pp_size=1, tp_size=2, enable_attention_dp=shared),
            disaggregated=False,
            overlap=overlap,
        )
        == reserve
    )
    req = request_factory(32, 0, 100)
    admission.clamp_generation_request(req, 64, 1, draft_reserve=reserve)
    assert req.py_max_new_tokens == req.max_new_tokens == 32 - reserve
    if reserve:
        with pytest.raises(
            ValueError, match=rf"maximum context length is 64 tokens.*reserves {reserve}"
        ):
            admission.clamp_generation_request(
                request_factory(64 - reserve, 0, 1), 64, 1, draft_reserve=reserve
            )


def test_pard_requires_room_for_mask_rows(request_factory):
    req = request_factory(prompt_len=24, generated=3, max_tokens=100)
    assert budget.request_draft_budget(req, 7, 32, 1, is_pard=True) == 2


@pytest.mark.parametrize("mapped_ids", [[3, 1], [3, 0], [0, 3]])
def test_reduced_vocab_padding_uses_target_zero_and_clears_reused_rows(mapped_ids):
    expand = _load_host_module("_torch/speculative/ragged_helpers.py").expand_mapped_draft_probs
    indices = torch.tensor(mapped_ids)
    out = torch.zeros(1, 2, 4)
    proposals = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    # First row is synthetic target zero. The second really sampled draft ID zero.
    expand(out, proposals, indices, torch.tensor([[0, mapped_ids[0]]]))
    assert out[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert out[0, 1, mapped_ids[0]] == 1
    # A narrower step reuses the first scratch row, then a wider step reuses both.
    real_probs = torch.tensor([[[0.25, 0.75], [0.25, 0.75]]])
    real_tokens = torch.full((1, 2), mapped_ids[1])
    for width in (1, 2):
        expand(out[:, :width], real_probs[:, :width], indices, real_tokens[:, :width])
        torch.testing.assert_close(out[:, :width, indices], real_probs[:, :width])
        torch.testing.assert_close(out[:, :width].sum(-1), torch.ones(1, width))
