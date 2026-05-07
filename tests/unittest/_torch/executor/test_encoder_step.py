# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the encoder iteration helpers in PyExecutor.

Covers the two pure-Python helpers that drive the encoder branch of
``_executor_loop`` for encoder-decoder models:

* ``_split_encoder_decoder_context_requests`` — splits the scheduler's
  context bucket into encoder-init vs decoder-context subsets.
* ``_scatter_encoder_output`` — slices packed encoder hidden states
  back into per-request tensors and transitions request state from
  ``ENCODER_INIT`` to ``CONTEXT_INIT``.

These helpers do not touch the model engine or KV cache managers, so
the tests run on CPU only.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


def _make_request(req_id: int, *, is_encoder_init: bool, is_last_chunk: bool):
    """Build a lightweight stand-in for ``LlmRequest`` for these helpers.

    Only the attributes that the split / scatter helpers touch are
    populated.  ``state`` is a real :class:`LlmRequestState` so the
    helper can mutate it.
    """
    req = SimpleNamespace()
    req.py_request_id = req_id
    req.is_encoder_init_state = is_encoder_init
    req.is_last_context_chunk = is_last_chunk
    req.state = LlmRequestState.ENCODER_INIT if is_encoder_init else LlmRequestState.CONTEXT_INIT
    req.py_encoder_output = None
    req.py_encoder_output_ready_event = None
    req.py_skip_cross_kv_projection = False
    return req


def _build_scheduled_batch(
    encoder_chunking=(),
    encoder_last_chunk=(),
    decoder_chunking=(),
    decoder_last_chunk=(),
):
    sb = ScheduledRequests()
    sb.context_requests_chunking = list(encoder_chunking) + list(decoder_chunking)
    sb.context_requests_last_chunk = list(encoder_last_chunk) + list(decoder_last_chunk)
    return sb


class TestSplitEncoderDecoderContextRequests:
    def test_no_context_requests(self):
        executor = MagicMock(spec=PyExecutor)
        executor._split_encoder_decoder_context_requests = (
            PyExecutor._split_encoder_decoder_context_requests.__get__(executor, PyExecutor)
        )
        sb = ScheduledRequests()

        encoder_requests = executor._split_encoder_decoder_context_requests(sb)

        assert encoder_requests == []
        assert sb.num_context_requests == 0

    def test_pure_decoder_context_unchanged(self):
        executor = MagicMock(spec=PyExecutor)
        executor._split_encoder_decoder_context_requests = (
            PyExecutor._split_encoder_decoder_context_requests.__get__(executor, PyExecutor)
        )
        d1 = _make_request(1, is_encoder_init=False, is_last_chunk=False)
        d2 = _make_request(2, is_encoder_init=False, is_last_chunk=True)
        sb = _build_scheduled_batch(decoder_chunking=(d1,), decoder_last_chunk=(d2,))

        encoder_requests = executor._split_encoder_decoder_context_requests(sb)

        assert encoder_requests == []
        assert sb.context_requests_chunking == [d1]
        assert sb.context_requests_last_chunk == [d2]

    def test_pure_encoder_init_drained(self):
        executor = MagicMock(spec=PyExecutor)
        executor._split_encoder_decoder_context_requests = (
            PyExecutor._split_encoder_decoder_context_requests.__get__(executor, PyExecutor)
        )
        e1 = _make_request(10, is_encoder_init=True, is_last_chunk=True)
        e2 = _make_request(11, is_encoder_init=True, is_last_chunk=True)
        sb = _build_scheduled_batch(encoder_last_chunk=(e1, e2))

        encoder_requests = executor._split_encoder_decoder_context_requests(sb)

        assert encoder_requests == [e1, e2]
        assert sb.context_requests_chunking == []
        assert sb.context_requests_last_chunk == []

    def test_mixed_preserves_order_and_buckets(self):
        """Encoder-init requests can be admitted alongside decoder context.

        After the split, decoder-context requests must remain in their
        original chunking / last-chunk buckets so the downstream
        decoder forward step is unchanged.
        """
        executor = MagicMock(spec=PyExecutor)
        executor._split_encoder_decoder_context_requests = (
            PyExecutor._split_encoder_decoder_context_requests.__get__(executor, PyExecutor)
        )
        e1 = _make_request(20, is_encoder_init=True, is_last_chunk=True)
        d1 = _make_request(21, is_encoder_init=False, is_last_chunk=False)
        e2 = _make_request(22, is_encoder_init=True, is_last_chunk=True)
        d2 = _make_request(23, is_encoder_init=False, is_last_chunk=True)
        sb = _build_scheduled_batch(
            encoder_chunking=(),
            encoder_last_chunk=(e1, e2),
            decoder_chunking=(d1,),
            decoder_last_chunk=(d2,),
        )

        encoder_requests = executor._split_encoder_decoder_context_requests(sb)

        # Encoder requests are returned in scheduler order across both
        # chunking and last-chunk buckets.
        assert encoder_requests == [e1, e2]
        assert sb.context_requests_chunking == [d1]
        assert sb.context_requests_last_chunk == [d2]

    def test_encoder_init_in_chunking_bucket(self):
        """Encoder-init requests appear in last_chunk in practice, but the
        split helper must still pull them out of the chunking bucket if
        a future scheduler routes them differently."""
        executor = MagicMock(spec=PyExecutor)
        executor._split_encoder_decoder_context_requests = (
            PyExecutor._split_encoder_decoder_context_requests.__get__(executor, PyExecutor)
        )
        e = _make_request(30, is_encoder_init=True, is_last_chunk=False)
        d = _make_request(31, is_encoder_init=False, is_last_chunk=True)
        sb = _build_scheduled_batch(encoder_chunking=(e,), decoder_last_chunk=(d,))

        encoder_requests = executor._split_encoder_decoder_context_requests(sb)

        assert encoder_requests == [e]
        assert sb.context_requests_chunking == []
        assert sb.context_requests_last_chunk == [d]


class TestScatterEncoderOutput:
    def _bind_scatter(self):
        executor = MagicMock(spec=PyExecutor)
        executor._scatter_encoder_output = PyExecutor._scatter_encoder_output.__get__(
            executor, PyExecutor
        )
        return executor

    def test_slices_packed_hidden_states(self):
        executor = self._bind_scatter()
        e1 = _make_request(1, is_encoder_init=True, is_last_chunk=True)
        e2 = _make_request(2, is_encoder_init=True, is_last_chunk=True)
        encoder_seq_lens = [3, 5]
        hidden_size = 4
        packed = torch.arange(sum(encoder_seq_lens) * hidden_size, dtype=torch.float32).reshape(
            sum(encoder_seq_lens), hidden_size
        )

        executor._scatter_encoder_output([e1, e2], packed, encoder_seq_lens)

        torch.testing.assert_close(e1.py_encoder_output, packed[0:3])
        torch.testing.assert_close(e2.py_encoder_output, packed[3:8])

    def test_transitions_state_to_context_init(self):
        executor = self._bind_scatter()
        e = _make_request(1, is_encoder_init=True, is_last_chunk=True)
        encoder_seq_lens = [2]
        packed = torch.zeros(2, 3)

        executor._scatter_encoder_output([e], packed, encoder_seq_lens)

        assert e.state == LlmRequestState.CONTEXT_INIT

    def test_initializes_skip_cross_kv_projection_false(self):
        """The first decoder context step is the only step that writes the
        cross-KV pool; ``py_skip_cross_kv_projection`` must therefore be
        ``False`` at the encoder-to-decoder transition.  The decoder
        step flips it to ``True`` for later steps and chunks."""
        executor = self._bind_scatter()
        e = _make_request(1, is_encoder_init=True, is_last_chunk=True)
        e.py_skip_cross_kv_projection = True  # stale value from a previous run
        packed = torch.zeros(2, 3)

        executor._scatter_encoder_output([e], packed, [2])

        assert e.py_skip_cross_kv_projection is False

    def test_rejects_none_hidden_states(self):
        executor = self._bind_scatter()
        e = _make_request(1, is_encoder_init=True, is_last_chunk=True)

        with pytest.raises(RuntimeError, match="None hidden states"):
            executor._scatter_encoder_output([e], None, [2])

    def test_rejects_mismatched_seq_lens(self):
        executor = self._bind_scatter()
        e = _make_request(1, is_encoder_init=True, is_last_chunk=True)
        packed = torch.zeros(4, 3)

        with pytest.raises(AssertionError):
            executor._scatter_encoder_output([e], packed, [2, 2])  # 2 lens, 1 request

    def test_rejects_packed_size_mismatch(self):
        executor = self._bind_scatter()
        e1 = _make_request(1, is_encoder_init=True, is_last_chunk=True)
        e2 = _make_request(2, is_encoder_init=True, is_last_chunk=True)
        packed = torch.zeros(5, 3)  # claims 5 rows

        with pytest.raises(AssertionError):
            executor._scatter_encoder_output([e1, e2], packed, [2, 2])


class TestAttachEncoderOutputToExecutionStream:
    """Tests for ``_attach_encoder_output_to_execution_stream``.

    Under Option 1 (scheduler-side filter + per-request event), the
    scheduler-side ``filter_unready_decoder_context_requests`` already
    excludes any ``CONTEXT_INIT`` request whose encoder event is not
    complete.  By the time the executor calls this helper, the encoder
    work for every admitted request is finished, so the helper does
    *not* call ``wait_event`` on the execution stream.

    The remaining responsibilities are:
    * call ``record_stream`` on the encoder-output tensor for caching
      allocator safety, and
    * clear ``py_encoder_output_ready_event`` so it cannot be queried
      again on a later iteration.
    """

    def _bind_attach_helper(self):
        executor = MagicMock(spec=PyExecutor)
        executor.execution_stream = MagicMock()
        executor._attach_encoder_output_to_execution_stream = (
            PyExecutor._attach_encoder_output_to_execution_stream.__get__(executor, PyExecutor)
        )
        return executor

    def test_records_stream_and_clears_event_for_context_request(self):
        executor = self._bind_attach_helper()
        req = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        req.py_encoder_output = MagicMock()
        ready_event = MagicMock()
        req.py_encoder_output_ready_event = ready_event
        scheduled = _build_scheduled_batch(decoder_last_chunk=(req,))

        executor._attach_encoder_output_to_execution_stream(scheduled)

        # Filter handles correctness; helper must not wait on the stream.
        executor.execution_stream.wait_event.assert_not_called()
        req.py_encoder_output.record_stream.assert_called_once_with(executor.execution_stream)
        assert req.py_encoder_output_ready_event is None

    def test_skips_requests_without_event(self):
        executor = self._bind_attach_helper()
        req = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        req.py_encoder_output = MagicMock()
        scheduled = _build_scheduled_batch(decoder_last_chunk=(req,))

        executor._attach_encoder_output_to_execution_stream(scheduled)

        executor.execution_stream.wait_event.assert_not_called()
        req.py_encoder_output.record_stream.assert_not_called()
        assert req.py_encoder_output_ready_event is None

    def test_skips_requests_without_encoder_output_tensor(self):
        """An event without a backing tensor is still cleared, but
        ``record_stream`` is not called (nothing to associate)."""
        executor = self._bind_attach_helper()
        req = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        req.py_encoder_output = None
        ready_event = MagicMock()
        req.py_encoder_output_ready_event = ready_event
        scheduled = _build_scheduled_batch(decoder_last_chunk=(req,))

        executor._attach_encoder_output_to_execution_stream(scheduled)

        executor.execution_stream.wait_event.assert_not_called()
        assert req.py_encoder_output_ready_event is None

    def test_only_processes_context_requests(self):
        """Generation requests do not carry encoder events past their
        first decoder-context step; the helper must not touch them."""
        executor = self._bind_attach_helper()
        ctx = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        ctx.py_encoder_output = MagicMock()
        ctx_ready_event = MagicMock()
        ctx.py_encoder_output_ready_event = ctx_ready_event
        gen = _make_request(2, is_encoder_init=False, is_last_chunk=True)
        gen.py_encoder_output = MagicMock()
        gen_ready_event = MagicMock()
        gen.py_encoder_output_ready_event = gen_ready_event
        scheduled = _build_scheduled_batch(decoder_last_chunk=(ctx,))
        scheduled.generation_requests = [gen]

        executor._attach_encoder_output_to_execution_stream(scheduled)

        executor.execution_stream.wait_event.assert_not_called()
        ctx.py_encoder_output.record_stream.assert_called_once_with(executor.execution_stream)
        gen.py_encoder_output.record_stream.assert_not_called()
        assert ctx.py_encoder_output_ready_event is None
        assert gen.py_encoder_output_ready_event is gen_ready_event


class TestMarkCrossKvProjectionConsumed:
    def _bind_helper(self):
        executor = MagicMock(spec=PyExecutor)
        executor._mark_cross_kv_projection_consumed = (
            PyExecutor._mark_cross_kv_projection_consumed.__get__(executor, PyExecutor)
        )
        return executor

    def test_releases_context_encoder_outputs_and_sets_skip_flag(self):
        executor = self._bind_helper()
        req = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        req.py_encoder_output = torch.zeros(2, 3)
        req.py_skip_cross_kv_projection = False
        scheduled = _build_scheduled_batch(decoder_last_chunk=(req,))

        executor._mark_cross_kv_projection_consumed(scheduled)

        assert req.py_encoder_output is None
        assert req.py_skip_cross_kv_projection is True

    def test_clears_stale_output_even_when_projection_already_skipped(self):
        executor = self._bind_helper()
        req = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        req.py_encoder_output = torch.zeros(2, 3)
        req.py_skip_cross_kv_projection = True
        scheduled = _build_scheduled_batch(decoder_last_chunk=(req,))

        executor._mark_cross_kv_projection_consumed(scheduled)

        assert req.py_encoder_output is None
        assert req.py_skip_cross_kv_projection is True

    def test_generation_requests_are_not_touched(self):
        executor = self._bind_helper()
        gen = _make_request(1, is_encoder_init=False, is_last_chunk=True)
        gen.py_encoder_output = torch.zeros(2, 3)
        gen.py_skip_cross_kv_projection = False
        scheduled = _build_scheduled_batch()
        scheduled.generation_requests = [gen]

        executor._mark_cross_kv_projection_consumed(scheduled)

        assert gen.py_encoder_output is not None
        assert gen.py_skip_cross_kv_projection is False


class _FakeCrossAttentionMetadata:
    def __init__(self):
        self.prepared = False

    def prepare(self):
        self.prepared = True


class _FakeAttentionMetadata:
    def __init__(self, num_seqs):
        self.num_seqs = num_seqs
        self.cross_metadata = _FakeCrossAttentionMetadata()
        self.encoder_seq_lens = None
        self.cross_kv_cache_manager = None
        self.encoder_num_cached_tokens_per_seq = None

    def create_cross_metadata(
        self,
        encoder_seq_lens,
        cross_kv_cache_manager,
        *,
        encoder_num_cached_tokens_per_seq=None,
    ):
        self.encoder_seq_lens = encoder_seq_lens
        self.cross_kv_cache_manager = cross_kv_cache_manager
        self.encoder_num_cached_tokens_per_seq = encoder_num_cached_tokens_per_seq
        return self.cross_metadata


class _FakeResourceManager:
    def __init__(self, cross_kv_cache_manager):
        self.cross_kv_cache_manager = cross_kv_cache_manager

    def get_resource_manager(self, key):
        assert key == ResourceManagerType.CROSS_KV_CACHE_MANAGER
        return self.cross_kv_cache_manager


class TestPrepareEncoderDecoderCrossAttentionInputs:
    def _engine(self):
        return object.__new__(PyTorchModelEngine)

    def test_builds_metadata_for_mixed_projection_and_cached_sequences(self):
        engine = self._engine()
        encoder_output = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        metadata = _FakeAttentionMetadata(num_seqs=3)
        cross_manager = object()
        resource_manager = _FakeResourceManager(cross_manager)

        inputs = engine._prepare_encoder_decoder_cross_attention_inputs(
            [encoder_output],
            [2, 0, 0],
            [0, 5, 7],
            metadata,
            resource_manager,
        )

        assert inputs["encoder_hidden_states"] is encoder_output
        assert inputs["skip_cross_kv_projection"] is False
        assert inputs["cross_attn_metadata"] is metadata.cross_metadata
        assert metadata.cross_metadata.prepared is True
        assert metadata.encoder_seq_lens.tolist() == [2, 0, 0]
        assert metadata.cross_kv_cache_manager is cross_manager
        assert metadata.encoder_num_cached_tokens_per_seq == [0, 5, 7]

    def test_all_cached_sequences_skip_projection(self):
        engine = self._engine()
        metadata = _FakeAttentionMetadata(num_seqs=2)
        resource_manager = _FakeResourceManager(object())

        inputs = engine._prepare_encoder_decoder_cross_attention_inputs(
            [],
            [0, 0],
            [3, 4],
            metadata,
            resource_manager,
        )

        assert inputs["encoder_hidden_states"] is None
        assert inputs["skip_cross_kv_projection"] is True
        assert metadata.encoder_seq_lens.tolist() == [0, 0]
        assert metadata.encoder_num_cached_tokens_per_seq == [3, 4]

    def test_rejects_hidden_state_length_mismatch(self):
        engine = self._engine()
        metadata = _FakeAttentionMetadata(num_seqs=1)
        resource_manager = _FakeResourceManager(object())

        with pytest.raises(RuntimeError, match="do not match"):
            engine._prepare_encoder_decoder_cross_attention_inputs(
                [torch.zeros(1, 3)],
                [2],
                [0],
                metadata,
                resource_manager,
            )

    def test_requires_cross_kv_cache_manager(self):
        engine = self._engine()
        metadata = _FakeAttentionMetadata(num_seqs=1)
        resource_manager = _FakeResourceManager(None)

        with pytest.raises(RuntimeError, match="CROSS_KV_CACHE_MANAGER"):
            engine._prepare_encoder_decoder_cross_attention_inputs(
                [],
                [0],
                [2],
                metadata,
                resource_manager,
            )


class _FakeEmbedding:
    def __call__(self, input_ids):
        return input_ids.to(dtype=torch.float32).unsqueeze(-1)


class _CapturingEncoder:
    def __init__(self):
        self.hidden_states = None
        self.position_ids = None

    def __call__(self, hidden_states, attn_metadata, position_ids=None):
        del attn_metadata
        self.hidden_states = hidden_states
        self.position_ids = position_ids
        return hidden_states


class TestPositionIdOffset:
    def test_reads_offset_from_wrapped_model(self):
        engine = object.__new__(PyTorchModelEngine)
        engine.model = SimpleNamespace(model=SimpleNamespace(position_id_offset=2))

        assert engine._get_position_id_offset() == 2
        assert engine._apply_position_id_offset([0, 1, 7]) == [2, 3, 9]

    def test_reads_offset_through_compiled_wrapper(self):
        engine = object.__new__(PyTorchModelEngine)
        engine.model = SimpleNamespace(
            _orig_mod=SimpleNamespace(model=SimpleNamespace(position_id_offset=2))
        )

        assert engine._get_position_id_offset() == 2

    def test_defaults_to_logical_positions(self):
        engine = object.__new__(PyTorchModelEngine)
        engine.model = SimpleNamespace(model=SimpleNamespace())

        position_ids = [0, 1, 7]
        assert engine._get_position_id_offset() == 0
        assert engine._apply_position_id_offset(position_ids) == position_ids


class TestForwardStepEncoder:
    def test_applies_bart_style_embed_scale(self):
        engine = object.__new__(PyTorchModelEngine)
        encoder = _CapturingEncoder()
        inner_model = SimpleNamespace(
            shared_embedding=_FakeEmbedding(),
            embed_scale=3.0,
            encoder=encoder,
        )
        engine.model = SimpleNamespace(model=inner_model)
        position_ids = torch.tensor([[0, 1]])

        output = engine._forward_step_encoder(
            {
                "encoder_input_ids": torch.tensor([2, 5]),
                "encoder_attn_metadata": object(),
                "encoder_position_ids": position_ids,
            }
        )

        expected = torch.tensor([[6.0], [15.0]])
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(encoder.hidden_states, expected)
        torch.testing.assert_close(encoder.position_ids, position_ids.squeeze(0))
