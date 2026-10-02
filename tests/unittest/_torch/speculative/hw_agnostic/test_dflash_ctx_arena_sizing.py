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
"""The DFlash context arena is sized by the length actually served.

Its ``ctx_len`` is 1:1 with the target's positions, so the served
``max_seq_len`` bounds it and the drafter's ``max_position_embeddings`` does
not -- this checkpoint family advertises 1048576 of them.

The served length is taken from the KV cache manager, which every backend
carries on the metadata base class. Reading a ``max_seq_len`` ATTRIBUTE alone
is the pinned defect: ``TrtllmAttentionMetadata`` declares one and neither the
metadata base nor ``FlashInferAttentionMetadata`` does, so on those backends
the read comes back absent, the position table wins the ``min()``, and the
arena is sized 21x too large (a 165 GiB request that dies in the first warmup
forward).

These tests stop the function at its first allocation and inspect the size it
decided on, so they cover the decision without allocating an arena.
"""

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.cpu_only

SERVED = 49152
POSITION_TABLE = 1048576


class _Stop(Exception):
    """Raised in place of the first allocation, once the size is decided."""


def _worker():
    """A worker carrying only what the sizing path reads before it allocates."""
    worker = object.__new__(DFlashWorker)
    worker._validate_draft_attention_backend = MagicMock()
    worker._ctx_buf_inited = False
    worker._ctx_kv_manager = None
    # max_draft_len is a read-only property reading spec_config, so set the
    # config and let the property answer, rather than shadowing it.
    worker.spec_config = MagicMock(max_draft_len=7, tokens_per_gen_step=8)
    return worker


def _metadata(*, declares_max_seq_len):
    """Metadata with the manager every backend has, and optionally the attribute.

    ``declares_max_seq_len=False`` is the FlashInfer shape: a real KV cache
    manager, no ``max_seq_len`` of its own.
    """
    metadata = MagicMock(
        spec=["kv_cache_manager", "max_seq_len"] if declares_max_seq_len else ["kv_cache_manager"]
    )
    metadata.kv_cache_manager = MagicMock(max_seq_len=SERVED)
    if declares_max_seq_len:
        metadata.max_seq_len = SERVED
    return metadata


def _size_arena(metadata):
    """Run the sizing path and return the ``_max_ctx`` it settled on."""
    worker = _worker()
    draft_model = MagicMock()
    draft_model.config = MagicMock(max_position_embeddings=POSITION_TABLE)
    spec_metadata = MagicMock(max_num_requests=32)

    with patch("tensorrt_llm._torch.speculative.dflash.torch.zeros", side_effect=_Stop):
        with pytest.raises(_Stop):
            worker._lazy_init_ctx_buffers(draft_model, spec_metadata, metadata)
    return worker._max_ctx


def test_backend_without_a_max_seq_len_attribute_still_sizes_to_the_served_length():
    """The pinned defect: the FlashInfer shape, where the attribute read picks 1048576."""
    assert _size_arena(_metadata(declares_max_seq_len=False)) == SERVED


def test_backend_with_the_attribute_is_unchanged():
    """The TRTLLM shape must keep the number it already had."""
    assert _size_arena(_metadata(declares_max_seq_len=True)) == SERVED


def test_both_backends_agree():
    """The whole point of reading the manager: one number, not one per backend."""
    assert _size_arena(_metadata(declares_max_seq_len=False)) == _size_arena(
        _metadata(declares_max_seq_len=True)
    )


def test_the_position_table_still_caps_a_longer_served_length():
    """It is a ceiling, not a fallback: a drafter that covers less still wins."""
    metadata = _metadata(declares_max_seq_len=False)
    metadata.kv_cache_manager = MagicMock(max_seq_len=POSITION_TABLE * 2)

    assert _size_arena(metadata) == POSITION_TABLE
