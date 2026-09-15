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
"""Hidden-state capture buffer sharing across CUDA-graph metadata copies.

``create_cuda_graph_metadata`` shallow-copies the spec metadata once per
CUDA-graph bucket and re-runs ``__post_init__`` on the copy. The DFlash /
DSpark ``captured_hidden_states`` buffer is sized off the full engine
``max_num_tokens`` (e.g. [8192, hidden * num_capture_layers], 640 MiB at
hidden 8192 with 5 capture layers), so re-allocating it per bucket multiplied
that footprint by the number of graph buckets and OOM'd warmup at large
max_batch_size. These tests pin the sharing: every graph copy must reuse the
parent's buffer, and creating graph metadata must not allocate new capture
buffers.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

pytestmark = pytest.mark.cpu_only

HIDDEN_SIZE = 64
MAX_NUM_TOKENS = 512
MAX_NUM_REQUESTS = 32
MAX_DRAFT_LEN = 7
LAYERS = [3, 19, 31]


@pytest.fixture
def cpu_alloc(monkeypatch):
    """Route ``torch.empty(device='cuda')`` to CPU and count allocations.

    ``__post_init__`` allocates on the CUDA device; these tests only check
    allocation identity/shape, so CPU tensors stand in. Returns a list of the
    shapes allocated, so tests can assert how many capture-buffer-sized
    allocations happened.
    """
    real_empty = torch.empty
    shapes = []

    def counting_cpu_empty(*args, **kwargs):
        kwargs.pop("device", None)
        if args and isinstance(args[0], (tuple, list, torch.Size)):
            shapes.append(tuple(args[0]))
        else:
            shapes.append(tuple(args))
        return real_empty(*args, device="cpu", **kwargs)

    monkeypatch.setattr(torch, "empty", counting_cpu_empty)
    return shapes


def _make_metadata(cls, mode):
    return cls(
        max_num_requests=MAX_NUM_REQUESTS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=mode,
        layers_to_capture=list(LAYERS),
        hidden_size=HIDDEN_SIZE,
        max_num_tokens=MAX_NUM_TOKENS,
        dtype=torch.bfloat16,
    )


CASES = [
    (DFlashSpecMetadata, SpeculativeDecodingMode.DFLASH),
    (DSparkSpecMetadata, SpeculativeDecodingMode.DSPARK),
]
CASE_IDS = ["dflash", "dspark"]

CAPTURE_SHAPE = (MAX_NUM_TOKENS, HIDDEN_SIZE * len(LAYERS))


@pytest.mark.parametrize("cls,mode", CASES, ids=CASE_IDS)
def test_capture_buffer_allocated_once_at_full_token_budget(cls, mode, cpu_alloc):
    md = _make_metadata(cls, mode)
    assert md.captured_hidden_states is not None
    assert tuple(md.captured_hidden_states.shape) == CAPTURE_SHAPE
    assert cpu_alloc.count(CAPTURE_SHAPE) == 1


@pytest.mark.parametrize("cls,mode", CASES, ids=CASE_IDS)
def test_cuda_graph_copies_share_the_parent_capture_buffer(cls, mode, cpu_alloc):
    md = _make_metadata(cls, mode)
    parent_buffer = md.captured_hidden_states

    # One graph copy per bucket, the way _capture_generation_cuda_graphs
    # iterates batch sizes.
    for bucket in (32, 16, 8, 4, 2, 1):
        graph_md = md.create_cuda_graph_metadata(bucket)
        assert graph_md.is_cuda_graph
        assert graph_md.max_num_requests == bucket
        assert graph_md.captured_hidden_states is parent_buffer, (
            f"graph bucket {bucket} allocated its own capture buffer instead "
            "of sharing the parent's"
        )

    # No additional capture-buffer-sized allocation beyond the parent's one.
    assert cpu_alloc.count(CAPTURE_SHAPE) == 1


@pytest.mark.parametrize("cls,mode", CASES, ids=CASE_IDS)
def test_shared_buffer_covers_any_graph_bucket_rows(cls, mode, cpu_alloc):
    # The config-time token budget guard enforces
    # max_batch_size * (1 + max_draft_len) <= max_num_tokens, so the shared
    # buffer's rows bound what any generation-only graph bucket writes.
    md = _make_metadata(cls, mode)
    graph_md = md.create_cuda_graph_metadata(MAX_NUM_REQUESTS)
    rows_needed = MAX_NUM_REQUESTS * (1 + MAX_DRAFT_LEN)
    assert rows_needed <= graph_md.captured_hidden_states.shape[0]
    hs = graph_md.get_hidden_states(rows_needed)
    assert hs.shape == (rows_needed, HIDDEN_SIZE * len(LAYERS))


@pytest.mark.parametrize("cls,mode", CASES, ids=CASE_IDS)
def test_no_capture_layers_skips_the_buffer(cls, mode, cpu_alloc):
    md = cls(
        max_num_requests=MAX_NUM_REQUESTS,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
        spec_dec_mode=mode,
        layers_to_capture=None,
        hidden_size=HIDDEN_SIZE,
        max_num_tokens=MAX_NUM_TOKENS,
        dtype=torch.bfloat16,
    )
    assert md.captured_hidden_states is None
    assert CAPTURE_SHAPE not in cpu_alloc
