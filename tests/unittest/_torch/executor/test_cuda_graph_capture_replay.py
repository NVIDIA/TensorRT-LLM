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
"""Tests for CUDAGraphRunner.capture()/replay() and the shared_static_tensors they share.

These guard against a static input being added without a corresponding copy
in replay(): a captured graph reads from shared_static_tensors at fixed
addresses, and replay() is responsible for copying every live input into
those buffers before each graph.replay() call. A missed copy_ silently
leaves stale (or poisoned) data in the region the graph reads.

Five invariants, six tests:
  - Poison-fill completeness: replay() must overwrite every sentinel-poisoned
    static tensor. Catches a key whose copy_ was dropped entirely.
  - input_ids extent agreement: replay() must reject an input_ids whose
    length doesn't match the key's captured extent, rather than silently
    under-copying and leaving a stale tail.
  - mrope_delta_read_seq_slots extent agreement: same invariant, but for
    mrope_delta_read_seq_slots, whose copy extent comes from the caller's
    tensor shape rather than from input_ids' seqlen.
  - mrope_delta_read_seq_slots omission fill: a replay that omits
    mrope_delta_read_seq_slots entirely must still fill the static buffer
    with the dummy seq slot's permanently-zero delta, both right after
    capture (uninitialized buffer) and after a prior replay left real slot
    values in the buffer (two tests, same invariant, two starting states).
  - Staleness detection: two replays with different inputs must produce
    different outputs. A dropped copy_ makes them identical.
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cuda_graph_runner_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

SENTINEL = -12345


class TestCaptureReplayStaticTensors:
    """Invariants CUDAGraphRunner.capture()/replay() must uphold for
    shared_static_tensors: every key present at capture time is fully
    overwritten on every replay() call, so a captured graph never reads
    stale or poisoned data.
    """

    def _make_inputs(self, attn_metadata, num_tokens, batch_size, value, use_mrope=False):
        input_ids = torch.full((num_tokens,), value, device="cuda", dtype=torch.int32)
        if use_mrope:
            position_ids = torch.full((3, 1, num_tokens), value, device="cuda", dtype=torch.int32)
        else:
            position_ids = torch.full((1, num_tokens), value, device="cuda", dtype=torch.int32)
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": input_ids,
            "position_ids": position_ids,
        }
        if use_mrope:
            inputs["mrope_delta_read_seq_slots"] = torch.full(
                (batch_size,), value, device="cuda", dtype=torch.long
            )
        return inputs

    def _captured_region(self, runner, tensor_key, buffer, num_tokens, batch_size):
        # Mirrors the slicing CUDAGraphRunner.capture() applies to each
        # shared static tensor when it builds the graph's fixed-address
        # inputs i.e. the exact region the captured graph reads.
        if tensor_key == "input_ids":
            return buffer[:num_tokens]
        if tensor_key == "position_ids":
            return buffer[..., :num_tokens]
        if tensor_key == "mrope_delta_read_seq_slots":
            return buffer[: batch_size * runner.max_beam_width]
        raise AssertionError(f"Unhandled shared static tensor key: {tensor_key!r}")

    @pytest.mark.parametrize("use_mrope", [False, True])
    def test_replay_overwrites_poisoned_static_tensors(self, use_mrope):
        """replay() must overwrite every shared static tensor's captured
        region; none may be left holding the sentinel poison value.
        """
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=use_mrope)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)

        # Identity, not equality, is what replay() checks against the
        # metadata captured for this key.
        attn_metadata = object()

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=use_mrope),
        )

        for buffer in runner.shared_static_tensors.values():
            buffer.fill_(SENTINEL)
        for tensor_key, buffer in runner.shared_static_tensors.items():
            assert torch.all(buffer == SENTINEL), (
                f"shared_static_tensors[{tensor_key!r}] did not take the "
                "sentinel fill; the poison-fill step is broken, so "
                "this test cannot detect a missing copy_ in replay()."
            )

        runner.replay(
            key,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=2, use_mrope=use_mrope),
        )

        for tensor_key, buffer in runner.shared_static_tensors.items():
            region = self._captured_region(runner, tensor_key, buffer, num_tokens, batch_size)
            assert not torch.any(region == SENTINEL), (
                f"replay() left sentinel values in "
                f"shared_static_tensors[{tensor_key!r}]; a static input may "
                "be missing its copy_ in replay()."
            )

    @pytest.mark.parametrize("use_mrope", [False, True])
    def test_replay_rejects_input_ids_length_mismatch(self, use_mrope):
        """A shorter input_ids must raise, not silently leave a stale tail
        in the static input buffer."""
        batch_size = 4
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=use_mrope, max_num_tokens=128)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=use_mrope),
        )

        with pytest.raises(ValueError, match="tokens"):
            runner.replay(
                key,
                self._make_inputs(
                    attn_metadata, num_tokens - 1, batch_size, value=2, use_mrope=use_mrope
                ),
            )

    @pytest.mark.parametrize("use_mrope", [False, True])
    def test_replay_rejects_position_ids_shape_mismatch(self, use_mrope):
        """A position_ids whose shape doesn't match input_ids' seqlen must
        raise, rather than letting torch.Tensor.copy_() silently broadcast a
        singleton axis across the static buffer.
        """
        batch_size = 4
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=use_mrope, max_num_tokens=128)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=use_mrope),
        )

        replay_inputs = self._make_inputs(
            attn_metadata, num_tokens, batch_size, value=2, use_mrope=use_mrope
        )
        # Collapse the token axis to a broadcastable singleton, mirroring the
        # shape bug copy_() would otherwise silently paper over.
        if use_mrope:
            replay_inputs["position_ids"] = torch.full(
                (3, 1, 1), 2, device="cuda", dtype=torch.int32
            )
        else:
            replay_inputs["position_ids"] = torch.full((1, 1), 2, device="cuda", dtype=torch.int32)

        with pytest.raises(ValueError, match="position_ids"):
            runner.replay(key, replay_inputs)

    def test_replay_rejects_mrope_delta_read_seq_slots_length_mismatch(self):
        """A short mrope_delta_read_seq_slots must raise, not silently leave
        a stale tail in the static buffer.

        Unlike position_ids, its copy extent comes from the caller-supplied
        tensor's own shape rather than from input_ids' seqlen, so it needs
        its own check.
        """
        batch_size = 4
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=True, max_num_tokens=128)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=True),
        )

        with pytest.raises(ValueError, match="mrope_delta_read_seq_slots"):
            runner.replay(
                key,
                self._make_inputs(
                    attn_metadata, num_tokens, batch_size - 2, value=2, use_mrope=True
                ),
            )

    def test_replay_mrope_delta_read_seq_slots_omission_after_capture(self):
        """A replay that omits mrope_delta_read_seq_slots must fill the
        static buffer with the reserved dummy seq slot (a permanently-zero
        delta, per model_engine.py's mrope_dummy_seq_slot fast path) rather
        than leaving it at whatever (here: uninitialized) value it held.
        """
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=True)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()
        dummy_seq_slot = runner.config.max_num_tokens * runner.config.mapping.pp_size

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=True),
        )

        # use_mrope=True keeps position_ids 3D, matching what a real caller
        # always sends for an MRoPE-capable model; only
        # mrope_delta_read_seq_slots is conditionally omitted, so we delete
        # just that key rather than passing use_mrope=False.
        replay_inputs = self._make_inputs(
            attn_metadata, num_tokens, batch_size, value=2, use_mrope=True
        )
        del replay_inputs["mrope_delta_read_seq_slots"]

        runner.replay(key, replay_inputs)

        static_buf = runner.shared_static_tensors["mrope_delta_read_seq_slots"]
        assert static_buf[:batch_size].tolist() == [dummy_seq_slot] * batch_size

    def test_replay_mrope_delta_read_seq_slots_omission_after_prior_replay(self):
        """A replay that omits mrope_delta_read_seq_slots after a prior
        replay carried real slot values must overwrite the static buffer
        with the dummy seq slot, not leave the previous request's stale
        values in place.
        """
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=True)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()
        dummy_seq_slot = runner.config.max_num_tokens * runner.config.mapping.pp_size

        def forward_fn(inputs):
            return inputs["input_ids"].clone()

        runner.capture(
            key,
            forward_fn,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=1, use_mrope=True),
        )

        runner.replay(
            key,
            self._make_inputs(attn_metadata, num_tokens, batch_size, value=2, use_mrope=True),
        )
        static_buf = runner.shared_static_tensors["mrope_delta_read_seq_slots"]
        assert static_buf[:batch_size].tolist() == [2] * batch_size

        # See the omission-after-capture test above for why use_mrope=True is
        # kept here (3D position_ids) while only the delta-slots key is
        # deleted.
        replay_inputs = self._make_inputs(
            attn_metadata, num_tokens, batch_size, value=3, use_mrope=True
        )
        del replay_inputs["mrope_delta_read_seq_slots"]

        runner.replay(key, replay_inputs)

        assert static_buf[:batch_size].tolist() == [dummy_seq_slot] * batch_size

    def test_replay_output_reflects_latest_inputs(self):
        """Replaying twice with different inputs must produce outputs that
        reflect each call's own inputs. A dropped copy_ makes them identical.
        """
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=False)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = object()

        def forward_fn(inputs):
            return inputs["input_ids"].clone() + inputs["position_ids"][...,].clone()

        runner.capture(
            key, forward_fn, self._make_inputs(attn_metadata, num_tokens, batch_size, value=5)
        )

        # run 1
        runner.replay(key, self._make_inputs(attn_metadata, num_tokens, batch_size, value=1))

        # run 2 with new inputs
        logits_cuda_graph = runner.replay(
            key, self._make_inputs(attn_metadata, num_tokens, batch_size, value=2)
        )

        # run 2 eagerly
        logits_eager = forward_fn(self._make_inputs(attn_metadata, num_tokens, batch_size, value=2))

        torch.testing.assert_close(logits_cuda_graph, logits_eager)


class _MetadataStub:
    """Minimal stand-in for attn_metadata carrying one graph-visible CUDA tensor.

    Real attn_metadata/spec_metadata objects have many more attributes, but
    the strict-buffer check is a generic vars()-walk over every CUDA tensor
    attribute, so a stub with a single tracked tensor exercises the same
    code path.
    """

    def __init__(self, value: int):
        self.some_buf = torch.full((1,), value, device="cuda", dtype=torch.int32)


class TestStrictBufferCheck:
    """CUDAGraphRunner._STRICT_BUFFER_CHECK: attn_metadata/spec_metadata tensor
    attributes must keep the same data_ptr() from capture through every
    replay, since the captured graph's kernels read from those fixed
    addresses. In-place updates are fine; rebinding the attribute (to a new
    tensor, or to a non-tensor) invalidates the address the graph reads and
    must raise immediately rather than silently replaying against stale
    memory.
    """

    def _make_runner_and_inputs(self, monkeypatch, value):
        monkeypatch.setattr(cuda_graph_runner_module, "_STRICT_BUFFER_CHECK", True)
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)
        attn_metadata = _MetadataStub(value)
        input_ids = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
        position_ids = torch.zeros((1, num_tokens), device="cuda", dtype=torch.int32)
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": input_ids,
            "position_ids": position_ids,
        }
        return runner, key, attn_metadata, inputs

    def test_replay_accepts_in_place_update_to_graph_tensor(self, monkeypatch):
        """A .copy_() into the same tensor object is accepted, and the
        replayed graph output reflects the new contents.
        """
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)

        def forward_fn(fn_inputs):
            return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].some_buf

        runner.capture(key, forward_fn, inputs)

        attn_metadata.some_buf.copy_(torch.full_like(attn_metadata.some_buf, 99))

        output = runner.replay(key, inputs)

        assert output.item() == 99

    def test_replay_rejects_rebound_graph_tensor(self, monkeypatch):
        """Rebinding the attribute to a freshly allocated tensor must raise,
        naming the attribute, instead of replaying against the old buffer.
        """
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)

        def forward_fn(fn_inputs):
            return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].some_buf

        runner.capture(key, forward_fn, inputs)

        attn_metadata.some_buf = torch.full((1,), 99, device="cuda", dtype=torch.int32)

        with pytest.raises(RuntimeError, match="some_buf"):
            runner.replay(key, inputs)

    def test_replay_rejects_non_tensor_graph_attr(self, monkeypatch):
        """Replacing a graph-visible tensor attribute with a non-tensor is
        rejected, naming the attribute.
        """
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)

        def forward_fn(fn_inputs):
            return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].some_buf

        runner.capture(key, forward_fn, inputs)

        attn_metadata.some_buf = None

        with pytest.raises(RuntimeError, match="some_buf"):
            runner.replay(key, inputs)


class _AttnMetadataStub:
    """Minimal stand-in for attn_metadata's create_cuda_graph_metadata() contract.

    maybe_get_cuda_graph() only calls this when capture is allowed for a new
    key. The real implementation returns a copy of self with
    is_cuda_graph=True; this stub does the same, ignoring its arguments since
    nothing here reads them, only the interface's full signature matters,
    so a caller passing by keyword still works.
    """

    def __init__(self, is_cuda_graph: bool = False):
        self.is_cuda_graph = is_cuda_graph

    def create_cuda_graph_metadata(
        self,
        max_batch_size,
        sub_cross_metadata=False,
        max_draft_tokens=0,
        buffers=None,
        encode_only=False,
    ):
        del max_batch_size, sub_cross_metadata, max_draft_tokens, buffers, encode_only
        return _AttnMetadataStub(is_cuda_graph=True)


def _make_generation_only_batch(req_id: int = 1) -> ScheduledRequests:
    """A batch with no context requests, so ScheduledRequests.can_run_cuda_graph
    is True, the precondition maybe_get_cuda_graph needs before it will even
    consult the _capture_allowed gate.
    """
    request = SimpleNamespace(
        py_request_id=req_id,
        py_draft_tokens=[],
        py_batch_idx=None,
    )
    batch = ScheduledRequests()
    batch.generation_requests = [request]
    return batch


class TestCaptureAllowedGate:
    """CUDAGraphRunner._capture_allowed / allow_capture(): the guard that
    keeps live, on-the-fly capture from resizing the shared workspace/static
    buffers and invalidating addresses baked into every graph captured
    before it. Capture must only be possible inside allow_capture(), and the
    flag must reset even if warmup raises partway through.
    """

    def test_maybe_get_cuda_graph_falls_back_to_eager_outside_allow_capture(self):
        """Requesting an uncaptured key outside allow_capture() must return
        the eager-fallback triple, not start a new capture.
        """
        runner = create_mock_cuda_graph_runner(batch_size=1)
        batch = _make_generation_only_batch()
        assert runner._capture_allowed is False

        result = runner.maybe_get_cuda_graph(
            batch, enable_spec_decode=False, attn_metadata=object()
        )

        assert result == (None, None, None)

    def test_maybe_get_cuda_graph_prepares_capture_inside_allow_capture(self):
        """The same uncaptured key, requested inside allow_capture(), must
        return real graph-ready metadata and a key, proving the eager
        fallback above is the gate blocking capture, not a broken method.
        """
        runner = create_mock_cuda_graph_runner(batch_size=1)
        batch = _make_generation_only_batch()

        with runner.allow_capture():
            attn_metadata, spec_metadata, key = runner.maybe_get_cuda_graph(
                batch, enable_spec_decode=False, attn_metadata=_AttnMetadataStub()
            )

        assert key is not None
        assert attn_metadata is not None and attn_metadata.is_cuda_graph
        assert spec_metadata is None

    def test_allow_capture_resets_flag_on_exception(self):
        """The _capture_allowed reset must run even when the warmup body
        inside allow_capture() raises, not just on a clean exit, otherwise
        an exception mid-warmup would leave capture permanently unguarded.
        """
        runner = create_mock_cuda_graph_runner(batch_size=1)

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom):
            with runner.allow_capture():
                assert runner._capture_allowed is True
                raise _Boom()

        assert runner._capture_allowed is False
