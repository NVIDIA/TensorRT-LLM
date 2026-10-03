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
"""Tests for CUDAGraphRunner.capture()/replay(), the shared_static_tensors they
share, the opt-in strict buffer-stability check, and the capture-allowed gate.

TestCaptureReplayStaticTensors guards against a missing copy_ in replay()
leaving shared_static_tensors stale or poisoned at the fixed addresses the
captured graph reads from.

TestStrictBufferCheck guards a different staleness class: attn_metadata/
spec_metadata tensor attributes aren't copied into by replay() at all, so
TLLM_CUDA_GRAPH_STRICT_BUFFERS checks their data_ptr() stays stable across
replay() calls, catching a rebind (vs. an in-place update) to stale memory.

TestCaptureAllowedGate guards allow_capture(): live capture must stay
confined to warmup, since capturing outside it could resize shared buffers
and invalidate addresses already baked into other graphs.
"""

from types import SimpleNamespace

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as cuda_graph_runner_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import (
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)

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
    """Metadata stand-in with one graph-visible CUDA tensor; the strict check walks vars()."""

    def __init__(self, value: int):
        self.some_buf = torch.full((1,), value, device="cuda", dtype=torch.int32)


class _DraftReplayMetadataStub(TrtllmAttentionMetadata):
    """attn_metadata with only the fields the draft-replay helpers swap; skips __init__."""

    def __init__(self, value: int):
        self.draft_replay_swapped_attrs = {}
        self.kv_cache_manager = object()
        self.kv_cache_block_offsets = torch.full((1,), value, device="cuda", dtype=torch.int32)
        self.host_kv_cache_block_offsets = torch.zeros(1, dtype=torch.int32)
        self.draft_kv_cache_block_offsets = torch.full(
            (1,), value + 1000, device="cuda", dtype=torch.int32
        )
        self.some_alias = self.kv_cache_block_offsets
        self.enable_flash_mla = False

    def prepare_for_draft_forward(self):
        # Mirrors DSA's _fullkv aliases, rebound as a side effect of the swap.
        self.record_draft_swap("some_alias")
        self.some_alias = self.kv_cache_block_offsets
        return None


class TestStrictBufferCheck:
    """Graph-visible metadata tensors must keep their data_ptr() from capture to replay."""

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

    @staticmethod
    def _forward_reading_some_buf(fn_inputs):
        return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].some_buf

    def test_replay_accepts_in_place_update_to_graph_tensor(self, monkeypatch):
        """An in-place copy_() is accepted and the replay reads the new contents."""
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)
        runner.capture(key, self._forward_reading_some_buf, inputs)

        attn_metadata.some_buf.copy_(torch.full_like(attn_metadata.some_buf, 99))

        output = runner.replay(key, inputs)

        assert output.item() == 99

    @pytest.mark.parametrize(
        "owner, to_none, in_postprocess",
        [
            ("attn_metadata", False, False),
            ("attn_metadata", True, False),
            ("spec_metadata", False, False),
            ("attn_metadata", False, True),
        ],
        ids=["rebound_tensor", "non_tensor", "spec_metadata", "postprocess_fn"],
    )
    def test_replay_rejects_rebound_graph_attr(self, monkeypatch, owner, to_none, in_postprocess):
        """Rebinding a graph-visible tensor attribute raises, naming the attribute."""
        runner, key, _, inputs = self._make_runner_and_inputs(monkeypatch, value=10)
        if owner == "spec_metadata":
            inputs["spec_metadata"] = _MetadataStub(value=20)

        def rebind(fn_inputs):
            fn_inputs[owner].some_buf = (
                None if to_none else torch.full((1,), 99, device="cuda", dtype=torch.int32)
            )

        runner.capture(
            key,
            self._forward_reading_some_buf,
            inputs,
            postprocess_fn=rebind if in_postprocess else None,
        )
        if not in_postprocess:
            rebind(inputs)

        with pytest.raises(RuntimeError, match="some_buf"):
            runner.replay(key, inputs)

    def test_replay_accepts_swapped_attr_with_unchanged_saved_target(self, monkeypatch):
        """A swapped attr is validated via its saved target; clearing the swap flags it."""
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)
        runner.capture(key, self._forward_reading_some_buf, inputs)

        # Keep the captured tensor alive so the graph doesn't read freed memory.
        captured_buf = attn_metadata.some_buf
        attn_metadata.some_buf = torch.full((1,), 99, device="cuda", dtype=torch.int32)

        attn_metadata.draft_replay_swapped_attrs = {"some_buf": captured_buf}
        # The graph still reads the captured buffer (10), not the rebound one (99).
        assert runner.replay(key, inputs).item() == 10

        attn_metadata.draft_replay_swapped_attrs = {}
        with pytest.raises(RuntimeError, match="some_buf"):
            runner.replay(key, inputs)
        del captured_buf

    def test_draft_replay_swapped_attrs_does_not_mask_other_rebinds(self, monkeypatch):
        """Exempting one swapped attr still flags a rebind of another attr."""
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)
        attn_metadata.other_buf = torch.full((1,), 1, device="cuda", dtype=torch.int32)
        runner.capture(key, self._forward_reading_some_buf, inputs)

        captured_buf = attn_metadata.some_buf
        attn_metadata.some_buf = torch.full((1,), 99, device="cuda", dtype=torch.int32)
        attn_metadata.other_buf = torch.full((1,), 2, device="cuda", dtype=torch.int32)

        attn_metadata.draft_replay_swapped_attrs = {"some_buf": captured_buf}
        with pytest.raises(RuntimeError, match="other_buf"):
            runner.replay(key, inputs)

    def test_replay_skips_graph_temporary(self, monkeypatch):
        """A rebound graph temporary is ignored; other rebinds are still flagged."""
        runner, key, attn_metadata, inputs = self._make_runner_and_inputs(monkeypatch, value=10)
        attn_metadata.graph_temporary_attrs = frozenset({"tmp_buf"})
        attn_metadata.tmp_buf = torch.full((1,), 10, device="cuda", dtype=torch.int32)

        def forward_fn(fn_inputs):
            return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].tmp_buf

        runner.capture(key, forward_fn, inputs)

        # Keep the captured tensor alive so the graph doesn't read freed memory.
        captured_tmp_buf = attn_metadata.tmp_buf
        attn_metadata.tmp_buf = torch.full((1,), 99, device="cuda", dtype=torch.int32)
        assert runner.replay(key, inputs).item() == 10

        attn_metadata.some_buf = torch.full((1,), 99, device="cuda", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="some_buf"):
            runner.replay(key, inputs)
        del captured_tmp_buf

    def _make_draft_replay_runner_and_inputs(self, monkeypatch, value):
        runner, key, _, inputs = self._make_runner_and_inputs(monkeypatch, value)
        attn_metadata = _DraftReplayMetadataStub(value)
        inputs["attn_metadata"] = attn_metadata
        draft_kv_cache_manager = SimpleNamespace(
            host_kv_cache_block_offsets=torch.zeros(1, dtype=torch.int32)
        )
        return runner, key, attn_metadata, inputs, draft_kv_cache_manager

    @staticmethod
    def _forward_reading_block_offsets(fn_inputs):
        return fn_inputs["input_ids"].clone() + fn_inputs["attn_metadata"].kv_cache_block_offsets

    @pytest.mark.parametrize("attr", ["kv_cache_block_offsets", "some_alias"])
    @pytest.mark.parametrize("replace_target", [False, True])
    def test_draft_swap_lifecycle(self, monkeypatch, attr, replace_target):
        """A valid draft swap replays; a target replaced after capture is flagged."""
        runner, key, attn_metadata, inputs, draft_mgr = self._make_draft_replay_runner_and_inputs(
            monkeypatch, value=10
        )
        runner.capture(key, self._forward_reading_block_offsets, inputs)

        # Keep the captured tensor alive so the graph doesn't read freed memory.
        captured_buf = attn_metadata.kv_cache_block_offsets
        if replace_target:
            setattr(attn_metadata, attr, torch.full((1,), 99, device="cuda", dtype=torch.int32))

        saved = prepare_attn_metadata_for_draft_replay(attn_metadata, draft_mgr)
        try:
            if replace_target:
                with pytest.raises(RuntimeError, match=attr):
                    runner.replay(key, inputs)
            else:
                assert runner.replay(key, inputs).item() == 10
        finally:
            restore_attn_metadata_after_draft_replay(attn_metadata, saved)
        del captured_buf

    def test_restore_clears_swapped_attrs_and_target_buffer(self, monkeypatch):
        """Restore leaves no swapped attrs behind and rebinds the target buffer."""
        _, _, attn_metadata, _, draft_mgr = self._make_draft_replay_runner_and_inputs(
            monkeypatch, value=10
        )
        target_buf = attn_metadata.kv_cache_block_offsets

        saved = prepare_attn_metadata_for_draft_replay(attn_metadata, draft_mgr)
        assert attn_metadata.draft_replay_swapped_attrs["kv_cache_block_offsets"] is target_buf
        assert attn_metadata.draft_replay_swapped_attrs["some_alias"] is target_buf
        assert attn_metadata.kv_cache_block_offsets is attn_metadata.draft_kv_cache_block_offsets

        # A repeated swap keeps the first recorded original.
        attn_metadata.swap_for_draft("kv_cache_block_offsets", torch.zeros(1, device="cuda"))
        assert attn_metadata.draft_replay_swapped_attrs["kv_cache_block_offsets"] is target_buf

        restore_attn_metadata_after_draft_replay(attn_metadata, saved)

        assert attn_metadata.draft_replay_swapped_attrs == {}
        assert attn_metadata.kv_cache_block_offsets is target_buf
        assert attn_metadata.some_alias is target_buf

    def test_cuda_graph_metadata_copy_has_own_swap_record(self):
        """A swap restored on a graph copy does not leak into the eager metadata's restore."""
        eager = TrtllmAttentionMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=None)
        graph = eager.create_cuda_graph_metadata(1)
        assert graph.draft_replay_swapped_attrs is not eager.draft_replay_swapped_attrs

        eager_buf = eager.kv_lens_cuda
        graph.swap_for_draft("kv_lens_cuda", torch.zeros_like(graph.kv_lens_cuda))
        graph.restore_draft_swaps()
        eager.swap_for_draft("kv_lens_cuda", torch.zeros_like(eager_buf))
        eager.restore_draft_swaps()

        assert eager.kv_lens_cuda is eager_buf


class TestStrictBufferCheckEnvVar:
    """TLLM_CUDA_GRAPH_STRICT_BUFFERS parsing; the tests above patch the flag directly."""

    @pytest.mark.parametrize("value, expected", [(None, False), ("1", True), ("true", False)])
    def test_env_var_controls_check(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("TLLM_CUDA_GRAPH_STRICT_BUFFERS", raising=False)
        else:
            monkeypatch.setenv("TLLM_CUDA_GRAPH_STRICT_BUFFERS", value)
        assert cuda_graph_runner_module._strict_buffer_check_enabled() is expected


class _AttnMetadataStub:
    """Stand-in for create_cuda_graph_metadata(); returns a copy with is_cuda_graph=True."""

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
    """A generation-only batch, so maybe_get_cuda_graph consults the capture gate."""
    request = SimpleNamespace(
        py_request_id=req_id,
        py_draft_tokens=[],
        py_batch_idx=None,
    )
    batch = ScheduledRequests()
    batch.generation_requests = [request]
    return batch


class TestCaptureAllowedGate:
    """Capture is only possible inside allow_capture(), and the flag always resets."""

    def test_maybe_get_cuda_graph_falls_back_to_eager_outside_allow_capture(self):
        """An uncaptured key outside allow_capture() falls back to eager."""
        runner = create_mock_cuda_graph_runner(batch_size=1)
        batch = _make_generation_only_batch()
        assert runner._capture_allowed is False

        result = runner.maybe_get_cuda_graph(
            batch, enable_spec_decode=False, attn_metadata=object()
        )

        assert result == (None, None, None)

    def test_maybe_get_cuda_graph_prepares_capture_inside_allow_capture(self):
        """The same key inside allow_capture() returns graph metadata and a key."""
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
        """_capture_allowed resets even when the body inside allow_capture() raises."""
        runner = create_mock_cuda_graph_runner(batch_size=1)

        class _Boom(Exception):
            pass

        with pytest.raises(_Boom):
            with runner.allow_capture():
                assert runner._capture_allowed is True
                raise _Boom()

        assert runner._capture_allowed is False
