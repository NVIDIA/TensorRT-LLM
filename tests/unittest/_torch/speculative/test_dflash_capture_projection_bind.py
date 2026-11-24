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
"""The engine DFlash worker binds a context-KV-only drafter's capture projection.

With speculative_config.context_kv_only the spec metadata folds the drafter's fc
into the target's capture hooks (project_captures). The worker binds the
drafter's fc blocks before every forward, and the prefill store finishes the
projection (finish_capture_projection) instead of projecting the captures.
"""

import math
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

HIDDEN = 16  # target hidden size
LAYERS = (3, 7, 11)  # captured target layers
OUT = 8  # drafter hidden size: fc maps 3 * HIDDEN -> OUT
LATENT = 4  # drafter context latent per layer


def _metadata(*, project_captures=True, max_num_tokens=32):
    """A DFlashSpecMetadata without __post_init__, which allocates on the GPU."""
    metadata = object.__new__(DFlashSpecMetadata)
    metadata.project_captures = project_captures
    metadata.layers_to_capture = list(LAYERS)
    metadata.num_capture_layers = len(LAYERS)
    metadata._layer_to_idx = {layer: i for i, layer in enumerate(LAYERS)}
    metadata._capture_layer_set = frozenset(LAYERS)
    metadata.hidden_size = HIDDEN
    metadata.max_num_tokens = max_num_tokens
    metadata.dtype = torch.float32
    metadata.captured_hidden_states = None
    metadata.projected_captures = None
    metadata._capture_projection = None
    metadata._projected_captures_by_width = {}
    metadata._warned_capture_dtype = False
    metadata.batch_indices_cuda = torch.empty(1, dtype=torch.int)
    metadata.host_position_ids = None
    metadata.request_ids = None
    return metadata


class _Container:
    """A context-KV-only drafter container: fc, hidden_norm and the latent path.

    ``tp_size`` models a TEP context server: a history-only container keeps the
    full fc there (the capture projection needs whole blocks), whatever the TP size.
    """

    def __init__(self, *, context_kv_only=True, tp_size=1, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.context_kv_only = context_kv_only
        self.mapping = Mapping(world_size=tp_size, tp_size=tp_size)
        self.fc = SimpleNamespace(
            weight=torch.randn(OUT, HIDDEN * len(LAYERS), generator=generator) / math.sqrt(HIDDEN)
        )
        self.hidden_norm = SimpleNamespace(eps=1e-6)
        self.kv_proj = torch.randn(LATENT, OUT, generator=generator)
        self.calls = []

    def _norm(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.hidden_norm.eps)

    def capture_projection_blocks(self, num_captures):
        self.calls.append(("blocks", num_captures))
        return list(self.fc.weight.split(HIDDEN, dim=1))

    def finish_capture_projection(self, projected):
        self.calls.append(("finish", tuple(projected.shape)))
        return self._norm(projected)

    def project_target_hidden(self, captured):
        self.calls.append(("project", tuple(captured.shape)))
        return self._norm(captured.float() @ self.fc.weight.t())

    def precompute_context_kv(self, projected, positions):
        latent = projected.float() @ self.kv_proj.t() + positions[:, None].float()
        return latent.view(-1, 1, 1, LATENT), None


def _worker():
    return DFlashWorker(DFlashDecodingConfig(max_draft_len=3), Mapping())


# --- Binding ------------------------------------------------------------------


@pytest.mark.parametrize("tp_size", [1, 8], ids=["ctx", "tep_ctx"])
def test_context_kv_only_drafter_binds_full_fc_blocks_before_the_forward(tp_size):
    worker, metadata, container = _worker(), _metadata(), _Container(tp_size=tp_size)
    # No resource manager: the binding does not depend on a managed draft cache.
    worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)

    assert container.calls == [("blocks", len(LAYERS))]
    blocks = metadata._capture_projection
    assert len(blocks) == len(LAYERS)
    for i, block in enumerate(blocks):
        # Whole [out_features, hidden] blocks of the full fc, also on a TP context server.
        assert tuple(block.shape) == (OUT, HIDDEN)
        assert torch.equal(block, container.fc.weight[:, i * HIDDEN : (i + 1) * HIDDEN])
    assert tuple(metadata.projected_captures.shape) == (metadata.max_num_tokens, OUT)
    assert metadata.projected_captures.dtype == torch.float32
    # The plain capture buffer is never allocated.
    assert metadata.captured_hidden_states is None


def test_binding_repeats_every_forward_and_reuses_the_buffer():
    worker, metadata, container = _worker(), _metadata(), _Container()
    worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)
    buffer = metadata.projected_captures
    container.fc.weight = container.fc.weight * 2  # a reloaded fc
    worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)
    assert metadata.projected_captures is buffer
    assert torch.equal(metadata._capture_projection[0], container.fc.weight[:, :HIDDEN])


@pytest.mark.parametrize("kind", ["not_context_kv_only", "no_projection_blocks"])
def test_drafter_that_cannot_project_its_captures_is_refused(kind):
    """context_kv_only on a drafter that is not context-KV-only fails at the first forward, before
    anything is bound: running the full drafter on the context server would only hide it."""
    if kind == "not_context_kv_only":
        container = _Container(context_kv_only=False)
    else:
        container = SimpleNamespace(context_kv_only=True, calls=[])
    worker, metadata = _worker(), _metadata()
    with pytest.raises(ValueError, match="context_kv_only"):
        worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)
    assert container.calls == []
    assert metadata._capture_projection is None
    assert metadata.captured_hidden_states is None and metadata.projected_captures is None


def test_nothing_is_bound_without_project_captures():
    worker, metadata, container = _worker(), _metadata(project_captures=False), _Container()
    worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)
    assert container.calls == []
    assert metadata._capture_projection is None and metadata.projected_captures is None


# --- Prefill store ------------------------------------------------------------


def _store_worker(num_slots=4, max_ctx=64):
    """A worker on the private dense arena, bound by hand (no GPU, no manager)."""
    worker = _worker()
    worker._ctx_buf_inited = True
    worker._max_ctx = max_ctx
    worker._dummy_slot = num_slots
    worker._free_slots = deque(range(num_slots))
    worker._ctx_len = torch.zeros(num_slots + 1, dtype=torch.long)
    worker._ctx_len_host = [0] * (num_slots + 1)
    worker._ctx_position_offset = torch.zeros(num_slots + 1, dtype=torch.long)
    worker._ctx_paged = False
    worker._ctx_k_buf = torch.zeros(num_slots + 1, 1, max_ctx, 1, LATENT)
    worker._ctx_v_buf = None
    return worker


def _captures(num_tokens, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(num_tokens, HIDDEN, generator=generator) for _ in LAYERS]


def _run_store(*, project):
    lengths = [5, 3]
    num_tokens = sum(lengths)
    container = _Container()
    worker, metadata = _store_worker(), _metadata()
    hidden = _captures(num_tokens)
    if project:
        worker.prepare_managed_draft_cache(container, metadata, SimpleNamespace(), None)
        # What the capture hooks accumulate: sum_i hidden_i @ block_i.T in FP32.
        metadata.projected_captures[:num_tokens] = sum(
            h @ block.t() for h, block in zip(hidden, metadata._capture_projection)
        )
    else:
        metadata.captured_hidden_states = torch.cat(hidden, dim=1)
    container.calls.clear()
    metadata.request_ids = [10, 11]
    metadata.host_position_ids = torch.cat([torch.arange(n) for n in lengths]).int()
    attn = SimpleNamespace(
        num_ctx_tokens=num_tokens,
        num_contexts=len(lengths),
        _seq_lens=torch.tensor(lengths, dtype=torch.int),
        kv_cache_params=None,
    )
    position_ids = metadata.host_position_ids.clone()
    worker._store_prefill_context(container, metadata, attn, position_ids, num_tokens)
    slots = [worker._req_to_slot[rid] for rid in metadata.request_ids]
    stored = [worker._ctx_k_buf[slot, 0, :n, 0] for slot, n in zip(slots, lengths)]
    return container, worker, slots, stored


def test_prefill_store_finishes_the_bound_projection():
    container, worker, slots, stored = _run_store(project=True)
    # The store reads the projected rows of the context tokens and never re-projects.
    assert container.calls == [("finish", (8, OUT))]
    assert [worker._ctx_len_host[slot] for slot in slots] == [5, 3]
    assert worker._ctx_len[slots].tolist() == [5, 3]


def test_projected_latents_equal_the_projection_path_up_to_fp32_reordering():
    container, _, _, projected = _run_store(project=True)
    assert container.calls[0][0] == "finish"
    container, _, _, reference = _run_store(project=False)
    assert container.calls[0][0] == "project"
    for got, want in zip(projected, reference):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
