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

import copy
import pickle

import pytest
import torch

from tensorrt_llm.llmapi.llm import _WhisperSuppressTokensLogitsProcessor

# cpu_only is a collection filter, not an exclusion: this file is still
# collected on GPU stages, so the CUDA cases below really run there rather than
# always skipping.
pytestmark = pytest.mark.cpu_only

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

NEG_INF = float("-inf")


def _make(suppress=(3, 5), begin=(7,)):
    return _WhisperSuppressTokensLogitsProcessor(
        suppress_token_ids=list(suppress), begin_suppress_token_ids=list(begin)
    )


def _logits(vocab=10, device="cpu"):
    return torch.zeros(1, vocab, device=device)


def _masked(logits):
    return {i for i, v in enumerate(logits[0].tolist()) if v == NEG_INF}


# --- behaviour: these hold with or without the index cache -------------------


def test_suppress_masks_exactly_the_listed_tokens():
    proc = _make()
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    # 7 is begin-suppressed: the first callback sees exactly the prompt.
    assert _masked(logits) == {3, 5, 7}


def test_begin_suppress_applies_only_while_the_window_is_open():
    proc = _make()
    # First call captures prompt_len=2 and applies begin-suppression.
    proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    # Second call is past the prompt: begin-suppression must not re-arm.
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1, 2]], stream_ptr=None, client_id=None)
    assert _masked(logits) == {3, 5}


def test_masks_each_beam_independently():
    proc = _make(suppress=(3,), begin=())
    logits = torch.zeros(2, 10)
    proc(req_id=1, logits=logits, token_ids=[[0, 1], [0, 4]], stream_ptr=None, client_id=None)
    assert logits[0][3] == NEG_INF
    assert logits[1][3] == NEG_INF
    assert torch.isinf(logits).sum() == 2


def test_out_of_range_token_id_raises_every_call():
    """Caching must not turn a hard error into a one-shot one.

    An id outside the vocabulary has to fail on the second call exactly as it
    does on the first.
    """
    proc = _make(suppress=(99,), begin=())
    for _ in range(2):
        with pytest.raises(IndexError):
            proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)


# --- the cache and the invariant it depends on ------------------------------


def test_token_ids_are_read_only():
    """The ids must not be changeable after construction, by any route.

    The index caches are keyed on device alone, so an id set that could change
    would leave them masking a stale set of tokens while the attribute reported
    the new one. Item assignment, rebinding and augmented assignment must all
    fail.
    """
    proc = _make()
    with pytest.raises(TypeError):
        proc.suppress_token_ids[0] = 99
    with pytest.raises(AttributeError):
        proc.suppress_token_ids = (7,)
    with pytest.raises(AttributeError):
        proc.begin_suppress_token_ids = [7]
    with pytest.raises(AttributeError):
        proc.suppress_token_ids += (7,)


def test_ids_still_honoured_after_a_rejected_mutation():
    """A rejected mutation must leave the processor masking its original ids."""
    proc = _make(suppress=(3,), begin=())
    proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    with pytest.raises(AttributeError):
        proc.suppress_token_ids = (7,)
    logits = _logits()
    proc(req_id=2, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert _masked(logits) == {3}
    assert proc.suppress_token_ids == (3,)


def test_index_tensor_is_reused_per_device():
    """Repeated calls must reuse one index tensor per device.

    This is the point of the cache: every rebuild is a blocking host-to-device
    copy on the decode path.
    """
    proc = _make()
    for step in range(3):
        proc(
            req_id=1,
            logits=_logits(),
            token_ids=[[0, 1] + [2] * step],
            stream_ptr=None,
            client_id=None,
        )

    device = torch.zeros(1).device
    assert list(proc._suppress_idx) == [device]
    first = proc._suppress_idx[device]
    proc(req_id=2, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert proc._suppress_idx[device] is first


def test_empty_lists_mask_nothing():
    proc = _WhisperSuppressTokensLogitsProcessor(suppress_token_ids=[], begin_suppress_token_ids=[])
    logits = _logits()
    proc(req_id=1, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert not torch.isinf(logits).any()
    assert proc._suppress_idx == {}


# --- the cache must not escape into copies of the owning SamplingParams ------


def test_deepcopy_of_a_warmed_processor_drops_the_cache():
    """Callers deep-copy SamplingParams per request (evaluate/interface.py).

    The index cache is derived device state, so a copy must start cold rather
    than inherit a second allocation of it.
    """
    proc = _make()
    proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert proc._suppress_idx

    clone = copy.deepcopy(proc)
    assert clone._suppress_idx == {}
    assert clone._begin_suppress_idx == {}
    assert clone.suppress_token_ids == proc.suppress_token_ids

    logits = _logits()
    clone(req_id=2, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert _masked(logits) == {3, 5, 7}


def test_pickle_roundtrip_drops_the_cache_and_still_masks():
    proc = _make()
    proc(req_id=1, logits=_logits(), token_ids=[[0, 1]], stream_ptr=None, client_id=None)

    restored = pickle.loads(pickle.dumps(proc))
    assert restored._suppress_idx == {}
    logits = _logits()
    restored(req_id=2, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    assert _masked(logits) == {3, 5, 7}


# --- CUDA: the placement the optimization actually exists for ---------------


@skip_no_cuda
def test_index_is_cached_on_the_logits_device():
    proc = _make()
    logits = _logits(device="cuda")
    proc(req_id=1, logits=logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)

    assert _masked(logits.cpu()) == {3, 5, 7}
    index = proc._suppress_idx[logits.device]
    assert index.device == logits.device
    assert index.dtype == torch.long

    proc(
        req_id=2, logits=_logits(device="cuda"), token_ids=[[0, 1]], stream_ptr=None, client_id=None
    )
    assert proc._suppress_idx[logits.device] is index


@skip_no_cuda
def test_cpu_and_cuda_use_separate_cache_entries():
    proc = _make(suppress=(3,), begin=())
    cpu_logits = _logits()
    cuda_logits = _logits(device="cuda")
    proc(req_id=1, logits=cpu_logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
    proc(req_id=2, logits=cuda_logits, token_ids=[[0, 1]], stream_ptr=None, client_id=None)

    assert set(proc._suppress_idx) == {cpu_logits.device, cuda_logits.device}
    assert _masked(cpu_logits) == {3}
    assert _masked(cuda_logits.cpu()) == {3}


@skip_no_cuda
def test_half_precision_masking_matches_advanced_indexing():
    for dtype in (torch.float16, torch.bfloat16):
        proc = _make(suppress=(3, 5), begin=())
        got = torch.zeros(1, 10, dtype=dtype, device="cuda")
        proc(req_id=1, logits=got, token_ids=[[0, 1]], stream_ptr=None, client_id=None)
        want = torch.zeros(1, 10, dtype=dtype, device="cuda")
        want[..., [3, 5]] = float("-inf")
        assert torch.equal(got, want)
