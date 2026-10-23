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
"""Locality domain split arithmetic of the decoder runner, without a model."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.locality_domain import (
    balanced_generation_dispatch,
    locality_domain_row_splits,
    sort_generation_requests_by_locality_domain,
)

pytestmark = pytest.mark.cpu_only


def _req(request_id, domain, dummy=False, batch_idx=0):
    return SimpleNamespace(
        py_request_id=request_id,
        py_locality_domain_id=domain,
        is_cuda_graph_dummy=dummy,
        is_dummy=dummy,
        py_batch_idx=None if dummy else batch_idx,
    )


@pytest.mark.parametrize("num_rows", range(0, 9))
def test_balanced_dispatch_depends_only_on_row_count(num_rows):
    dispatch = balanced_generation_dispatch(num_rows, 2)
    assert len(dispatch) == num_rows
    assert dispatch == sorted(dispatch)
    assert dispatch.count(0) == (num_rows + 1) // 2
    assert dispatch.count(1) == num_rows // 2


def test_balanced_dispatch_spreads_remainder_over_leading_domains():
    assert balanced_generation_dispatch(7, 3) == [0, 0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("num_real", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 5, 6])
def test_repeated_padding_dummies_count_per_occurrence(num_real, padding):
    dummy = _req(-1, 0, dummy=True)
    packed = [_req(i, i % 2) for i in range(num_real)] + [dummy] * padding
    reqs, tokens = locality_domain_row_splits(packed, [1] * len(packed), 2, positional=True)
    total = num_real + padding
    assert reqs == [(total + 1) // 2, total // 2]
    assert tokens == reqs


def test_same_padded_size_gives_same_split_for_any_composition():
    dummy = _req(-1, 1, dummy=True)
    a = [_req(0, 0), _req(1, 1)] + [dummy] * 6
    b = [_req(i, 1) for i in range(5)] + [dummy] * 3
    assert locality_domain_row_splits(a, [1] * 8, 2, positional=True) == (
        locality_domain_row_splits(b, [1] * 8, 2, positional=True)
    )


def test_ownership_split_counts_tokens_per_domain():
    packed = [_req(0, 0), _req(1, 0), _req(2, 1)]
    reqs, tokens = locality_domain_row_splits(packed, [3, 1, 2], 2, positional=False)
    assert reqs == [2, 1]
    assert tokens == [4, 2]


def test_ownership_split_falls_back_to_positional_when_interleaved():
    packed = [_req(0, 0), _req(1, 1), _req(2, 0)]
    assert locality_domain_row_splits(packed, [1, 1, 1], 2, positional=False) == (
        locality_domain_row_splits(packed, [1, 1, 1], 2, positional=True)
    )


def test_ownership_split_falls_back_to_positional_when_domains_descend():
    # Contiguous but out of domain order (an extend row before a generation
    # row): ownership counts [1, 4] would hand domain 0 the first token of the
    # domain-1 request, so the split must be positional.
    packed = [_req(0, 1), _req(1, 0)]
    assert locality_domain_row_splits(packed, [4, 1], 2, positional=False) == ([1, 1], [4, 1])


def test_sort_keeps_padding_dummies_last():
    dummy = _req(-1, 0, dummy=True)
    reqs = [dummy, _req(0, 1), _req(1, 0), dummy]
    ordered = sort_generation_requests_by_locality_domain(reqs)
    assert [r.py_request_id for r in ordered] == [1, 0, -1, -1]


def test_sort_keeps_host_token_rows_ahead_of_previous_step_rows():
    # The overlap scheduler packs host tokens before the previous step's device
    # tokens; a domain-only sort would hand row 2's host token to row 0.
    reqs = [
        _req(0, 0),
        _req(1, 1, batch_idx=None),
        _req(2, 0, batch_idx=None),
        _req(3, 1),
        _req(4, 0),
        _req(-1, 0, dummy=True),
    ]
    ordered = sort_generation_requests_by_locality_domain(reqs, frozenset({4}))
    assert [r.py_request_id for r in ordered] == [2, 4, 1, 0, 3, -1]
