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

Four invariants, four tests:
  - Poison-fill completeness: replay() must overwrite every sentinel-poisoned
    static tensor. Catches a key whose copy_ was dropped entirely.
  - input_ids extent agreement: replay() must reject an input_ids whose
    length doesn't match the key's captured extent, rather than silently
    under-copying and leaving a stale tail.
  - mrope_delta_read_seq_slots extent agreement: same invariant, but for
    mrope_delta_read_seq_slots, whose copy extent comes from the caller's
    tensor shape rather than from input_ids' seqlen.
  - Staleness detection: two replays with different inputs must produce
    different outputs. A dropped copy_ makes them identical.
"""

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType

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

        with pytest.raises(ValueError, match="tokens"):
            runner.replay(
                key,
                self._make_inputs(
                    attn_metadata, num_tokens - 1, batch_size, value=2, use_mrope=use_mrope
                ),
            )

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

        # Identity, not equality, is what replay() checks against the
        # metadata captured for this key.
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

    def test_replay_output_reflects_latest_inputs(self):
        """Replaying twice with different inputs must produce outputs that
        reflect each call's own inputs. A dropped copy_ makes them identical.
        """
        batch_size = 1
        runner = create_mock_cuda_graph_runner(batch_size, use_mrope=False)
        key = KeyType(batch_size=batch_size, draft_len=0, is_first_draft=False)
        num_tokens = runner._get_num_tokens_for_key(key)

        # Identity, not equality, is what replay() checks against the
        # metadata captured for this key.
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

        assert torch.allclose(logits_cuda_graph, logits_eager)
