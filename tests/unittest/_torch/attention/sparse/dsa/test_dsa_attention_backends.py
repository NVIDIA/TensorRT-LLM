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

"""Differential DSA backend tests using VanillaAttention as the golden."""

import pytest
from utils.util import skip_pre_blackwell

from .test_dsa_sparse_mla import _assert_matches_vanilla, _test_sparse_attention_mla, scenarios

BACKENDS_UNDER_TEST = ("TRTLLM",)

DSA_CASES = {
    "bf16-paged-context-generation": dict(
        scenario=scenarios[0],
        context_sequence_lengths=[160],
        generation_seq_len_q=1,
        num_generation_steps=2,
        sparse_topk=128,
        seed=123,
        topk_seed=456,
    ),
}


@skip_pre_blackwell
@pytest.mark.parametrize("name", list(DSA_CASES), ids=lambda name: name)
def test_dsa_attention_backend(name: str) -> None:
    """Run Vanilla first, then compare every production DSA phase to it."""
    case = DSA_CASES[name]
    golden = _test_sparse_attention_mla("VANILLA", **case)

    for backend in BACKENDS_UNDER_TEST:
        actual = _test_sparse_attention_mla(backend, **case)
        _assert_matches_vanilla(
            actual,
            golden,
            case["scenario"].kv_cache_dtype,
            backend,
        )
