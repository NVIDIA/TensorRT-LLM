# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import collections
import contextlib
import unittest
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
from _torch.pyexecutor.engine.runners.decoder.utils import decoder_config, decoder_graphs
from _torch.pyexecutor.engine_test_utils import _create_request_with_tokens, _make_request_stub

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import (
    DecoderCudaGraphs,
    _make_single_token_context_graph_batch,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runner_type", "graphs_path"),
    [
        (
            DecoderRunner,
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.DecoderCudaGraphs",
        ),
        (
            EncoderDecoderRunner,
            "tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder.EncoderDecoderCudaGraphs",
        ),
    ],
)
def test_cuda_graph_factory_preserves_input_preparer(
    runner_type: type[DecoderRunner], graphs_path: str
) -> None:
    runner = Mock()
    runner._inputs = object()
    runner._cuda_graphs_kwargs.side_effect = lambda: DecoderRunner._cuda_graphs_kwargs(runner)

    with patch(graphs_path) as graphs_type:
        graphs = runner_type._make_cuda_graphs(runner)

    graphs_type.assert_called_once()
    assert graphs_type.call_args.kwargs["inputs"] is runner._inputs
    assert graphs_type.call_args.kwargs["warmup_inputs"] is runner._buffers.input_ids_cuda
    assert graphs is graphs_type.return_value


@pytest.mark.cpu_only
class SingleTokenContextGraphBatchTestCase(unittest.TestCase):
    def test_generation_only_is_identity(self) -> None:
        generation = _make_request_stub(1)
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

        self.assertIs(graph_batch, batch)
        self.assertEqual(promoted_ids, frozenset())

    def test_eligible_batch_has_independent_lists_and_stable_order(self) -> None:
        context_0 = _make_request_stub(10, prompt_len=1)
        context_1 = _make_request_stub(11, prompt_len=8)
        generation = _make_request_stub(12, prompt_len=16)
        paused = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context_0, context_1]
        batch.generation_requests = [generation]
        batch.paused_requests = [paused]
        semantic_lists = (
            batch.encoder_requests,
            batch.context_requests_chunking,
            batch.context_requests_last_chunk,
            batch.generation_requests,
            batch.paused_requests,
        )
        semantic_snapshot = vars(context_1).copy()

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.context_requests, [])
        self.assertEqual(graph_batch.generation_requests, [context_0, context_1, generation])
        self.assertEqual(graph_batch.paused_requests, [paused])
        graph_lists = (
            graph_batch.encoder_requests,
            graph_batch.context_requests_chunking,
            graph_batch.context_requests_last_chunk,
            graph_batch.generation_requests,
            graph_batch.paused_requests,
        )
        for semantic_list, graph_list in zip(semantic_lists, graph_lists):
            self.assertIsNot(semantic_list, graph_list)
        self.assertEqual(promoted_ids, frozenset({10, 11}))
        self.assertEqual(vars(context_1), semantic_snapshot)

        graph_batch.generation_requests.append(object())
        self.assertEqual(batch.context_requests_last_chunk, [context_0, context_1])
        self.assertEqual(batch.generation_requests, [generation])

    def test_structural_fallbacks_return_semantic_batch(self) -> None:
        context = _make_request_stub(1)

        encoder_batch = ScheduledRequests()
        encoder_batch.encoder_requests = [object()]
        encoder_batch.context_requests_last_chunk = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(encoder_batch)
        self.assertIs(graph_batch, encoder_batch)
        self.assertFalse(promoted_ids)

        chunking_batch = ScheduledRequests()
        chunking_batch.context_requests_chunking = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(chunking_batch)
        self.assertIs(graph_batch, chunking_batch)
        self.assertFalse(promoted_ids)

    def test_context_shape_and_mode_fallback_matrix(self) -> None:
        cases = (
            ("multi_token_chunk", "context_chunk_size", 2),
            ("more_context_remaining", "context_remaining_length", 2),
            ("cursor_prompt_mismatch", "py_prompt_len", 5),
            ("beam", "py_beam_width", 2),
            ("draft", "py_draft_tokens", [9]),
            ("context_only", "is_context_only_request", True),
            ("disaggregated", "py_disaggregated_params", object()),
            ("multimodal", "py_multimodal_data", {}),
            ("multimodal_event", "py_mm_encoder_event", object()),
        )
        for name, attribute, value in cases:
            with self.subTest(name=name):
                context = _make_request_stub(1)
                setattr(context, attribute, value)
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [context]

                graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

                self.assertIs(graph_batch, batch)
                self.assertFalse(promoted_ids)

    def test_context_logits_use_final_token_graph_candidate(self) -> None:
        context = _make_request_stub(1)
        context.py_return_context_logits = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))

    def test_generation_only_request_in_context_list_falls_back(self) -> None:
        context = _make_request_stub(1)
        context.is_generation_only_request = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)

    def test_generation_shape_fallback_matrix(self) -> None:
        cases = (
            ("beam", "py_beam_width", 2),
            ("draft", "py_draft_tokens", [9]),
            ("disaggregated", "py_disaggregated_params", object()),
        )
        for name, attribute, value in cases:
            with self.subTest(name=name):
                context = _make_request_stub(1)
                generation = _make_request_stub(2)
                setattr(generation, attribute, value)
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [context]
                batch.generation_requests = [generation]

                graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

                self.assertIs(graph_batch, batch)
                self.assertFalse(promoted_ids)

    def test_mixed_one_and_two_token_contexts_fall_back_together(self) -> None:
        one_token = _make_request_stub(1)
        two_tokens = _make_request_stub(2)
        two_tokens.context_current_position -= 1
        two_tokens.context_remaining_length = 2
        two_tokens.context_chunk_size = 2
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [one_token, two_tokens]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        self.assertEqual(batch.context_requests_last_chunk, [one_token, two_tokens])

    def test_mrope_delta_is_supported_by_decode_provider(self) -> None:
        context = _make_request_stub(1)
        context.py_mrope_position_delta = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))

    def test_multimodal_context_requires_compatible_decode_token(self) -> None:
        context = _make_request_stub(1)
        context.py_multimodal_data = {}
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)

        incompatible = Mock(return_value=False)
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch, incompatible)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        incompatible.assert_called_once_with(context)

        compatible = Mock(return_value=True)
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch, compatible)
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))
        compatible.assert_called_once_with(context)

    def test_multimodal_pending_event_is_rechecked(self) -> None:
        context = _make_request_stub(1)
        context.py_multimodal_data = {}
        context.py_mm_encoder_event = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        compatible = Mock(return_value=True)

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch, compatible)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        compatible.assert_not_called()

        context.py_mm_encoder_event = None
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(batch, compatible)
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))
        compatible.assert_called_once_with(context)

    def test_multimodal_decode_compatibility_uses_final_prompt_token(self) -> None:
        graphs = decoder_graphs(decoder_config())
        graphs.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=100),
            mm_token_ids=torch.tensor([99], dtype=torch.int32),
        )
        request = _create_request_with_tokens([11, 99, 22], 1)

        request.context_current_position = 2
        self.assertTrue(graphs._is_final_multimodal_context_decode_compatible(request))

        request.context_current_position = 1
        self.assertFalse(graphs._is_final_multimodal_context_decode_compatible(request))

        graphs.model.mm_token_ids = None
        request = _create_request_with_tokens([11, 100], 2)
        request.context_current_position = 1
        self.assertFalse(graphs._is_final_multimodal_context_decode_compatible(request))

        graphs.model.mm_token_ids = torch.tensor([99], dtype=torch.int32)
        graphs.model.model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(rope_scaling={"type": "mrope"})
        )
        graphs = decoder_graphs(decoder_config(use_mrope=True), model=graphs.model)
        request = _create_request_with_tokens([11, 22], 3)
        request.context_current_position = 1
        request.py_multimodal_data = {"mrope_config": {}}
        self.assertTrue(graphs._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["multimodal_embedding"] = object()
        self.assertFalse(graphs._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["mrope_config"]["mrope_position_deltas"] = object()
        self.assertTrue(graphs._is_final_multimodal_context_decode_compatible(request))

    def test_lora_graph_variant_selection(self) -> None:
        graphs = decoder_graphs(decoder_config())
        graphs._lora = SimpleNamespace(cuda_graph_manager=object())
        graphs._force_lora_graph_for_capture = None
        graphs = decoder_graphs(decoder_config(cuda_graph_specialize_lora=True), lora=graphs._lora)
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        self.assertFalse(graphs.use_lora_cuda_graph(batch))

        request.lora_task_id = 42
        self.assertTrue(graphs.use_lora_cuda_graph(batch))

        request.lora_task_id = None
        graphs = decoder_graphs(decoder_config(cuda_graph_specialize_lora=False), lora=graphs._lora)
        self.assertTrue(graphs.use_lora_cuda_graph(batch))

        graphs._force_lora_graph_for_capture = False
        self.assertFalse(graphs.use_lora_cuda_graph(batch))


@pytest.mark.cpu_only
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_generation_capture_passes_local_speculation_state(fails: bool) -> None:
    runner = decoder_graphs(
        decoder_config(),
        cuda_graph_runner=Mock(
            spec_set=[
                "enabled",
                "is_warmup_only",
                "short_long_seq_len_threshold",
                "set_capture_sample_type",
            ],
            enabled=True,
            is_warmup_only=False,
            short_long_seq_len_threshold=None,
        ),
    )
    resource_manager = Mock()
    assert not runner._config.is_spec_decode
    batches = {}

    def create_request(
        resources: ResourceManager, batch_size: int, draft_len: int, *args: object, **kwargs: object
    ) -> ScheduledRequests:
        return batches.setdefault(draft_len, ScheduledRequests())

    forward_warmup = Mock(side_effect=RuntimeError("capture failure") if fails else None)
    with (
        patch.object(runner, "_get_graphs_to_capture", return_value=[(1, 0), (1, 2)]),
        patch.object(
            runner._dummy_requests, "create_cuda_graph_warmup_request", side_effect=create_request
        ),
        patch.object(
            runner._dummy_requests,
            "release_batch_context",
            side_effect=lambda request, resources: contextlib.nullcontext(request),
        ),
        patch.object(runner, "_forward_warmup", forward_warmup),
        patch.object(torch.cuda, "synchronize"),
    ):
        if fails:
            with pytest.raises(RuntimeError, match="capture failure"):
                runner._capture_generation_cuda_graphs(resource_manager)
        else:
            runner._capture_generation_cuda_graphs(resource_manager)

    expected_calls = [
        call(batches[2], resource_manager, enable_spec_decode=True, runtime_draft_len=2)
    ]
    if not fails:
        expected_calls.append(
            call(batches[0], resource_manager, enable_spec_decode=False, runtime_draft_len=0)
        )
    assert forward_warmup.call_args_list == expected_calls
    assert runner.force_lora_graph_for_capture is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("schedule", [{4: 4, 8: 2, 16: 1}, {4: 2}, {16: 4}, {4: 0}])
def test_dynamic_warmup_graphs_match_replay_schedule(schedule: dict[int, int]) -> None:
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import (
        compute_dynamic_draft_len_mapping,
    )
    from tensorrt_llm._torch.speculative.utils import get_draft_len_for_batch_size

    batch_sizes = [1, 2, 4, 8, 16]
    mode = SimpleNamespace(support_dynamic_draft_len=lambda: True)
    runner = decoder_graphs(
        decoder_config(
            is_spec_decode=False,
            spec_config=SimpleNamespace(draft_len_schedule=schedule, spec_dec_mode=mode),
            cuda_graph_batch_sizes=batch_sizes,
            original_max_draft_len=max(schedule.values()),
        )
    )
    runner._dynamic_draft_len_mapping = compute_dynamic_draft_len_mapping(runner._config)
    graphs = runner._get_graphs_to_capture(batch_sizes)
    replay = object.__new__(CUDAGraphRunner)
    replay.clear = Mock()
    replay.supported_batch_sizes = batch_sizes
    replay.dynamic_draft_len_mapping = runner._dynamic_draft_len_mapping

    reachable_graphs = set()
    for batch_size in range(1, max(batch_sizes) + 1):
        draft_len = get_draft_len_for_batch_size(schedule, batch_size, max(schedule.values()))
        padded_size = replay._round_up_batch_size_with_draft_len(batch_size, draft_len)
        assert padded_size >= batch_size
        reachable_graphs.add((padded_size, draft_len))

    assert set(graphs) == reachable_graphs
    assert len(graphs) == len(reachable_graphs)


class TestWarmupCleanup(unittest.TestCase):
    @pytest.mark.cpu_only
    def test_cuda_graph_metrics_exclude_piecewise_stages(self) -> None:
        """Verify generation graph timing excludes prefill and includes LoRA warmup cleanup."""
        graphs = DecoderCudaGraphs(
            cuda_graph_runner=SimpleNamespace(enabled=True, is_warmup_only=False),
            breakable_cuda_graph_runner=None,
            config=SimpleNamespace(
                enable_autotuner=True, prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE
            ),
            mapping=Mapping(),
            model=SimpleNamespace(modules=lambda: []),
            sparse_attention_config=None,
            metrics={},
            dummy_requests=None,
            agreement=None,
            metadata=None,
            get_runtime_tokens_per_gen_step=None,
            dynamic_draft_len_mapping=None,
            lora=SimpleNamespace(cuda_graph_manager=object()),
            forward_warmup=None,
            torch_compile_bypass=contextlib.nullcontext,
            context_promotion_supported=True,
        )
        resource_manager = ResourceManager({})
        events = []

        @contextlib.contextmanager
        def record_metric_scope(metric_name: str, _metrics: dict[str, float]) -> Iterator[None]:
            events.append(("enter", metric_name))
            try:
                yield
            finally:
                events.append(("exit", metric_name))

        autotuner = SimpleNamespace(
            cache_pp_recv=lambda: events.append(("lora", "cache_pp_recv")),
            cache_pp_send=lambda: events.append(("lora", "cache_pp_send")),
            clean_pp_flag=lambda: events.append(("lora", "clean_pp_flag")),
        )
        lora_cleanup_events = [
            ("lora", "cache_pp_recv"),
            ("lora", "cache_pp_send"),
            ("lora", "clean_pp_flag"),
            ("exit", "lora_autotune"),
            ("exit", "gen_cuda_graph_warmup_seconds"),
        ]

        with (
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.autotune",
                side_effect=lambda **_: record_metric_scope("lora_autotune", {}),
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.AutoTuner.get",
                return_value=autotuner,
            ),
            patch(
                "tensorrt_llm._torch.modules.linear.flashinfer_mxfp8_decode_graph_capture",
                side_effect=contextlib.nullcontext,
            ),
            patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs.timing_metric",
                side_effect=record_metric_scope,
            ),
            patch.object(
                graphs,
                "_capture_generation_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "generation")),
            ) as generation,
            patch.object(
                graphs,
                "_capture_additional_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "additional")),
            ),
            patch.object(
                graphs,
                "_capture_prefill_cuda_graphs",
                side_effect=lambda _: events.append(("stage", "piecewise")),
            ) as piecewise,
        ):
            graphs.run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_capture_seconds"),
                    ("stage", "generation"),
                    ("stage", "additional"),
                    ("exit", "gen_cuda_graph_capture_seconds"),
                    ("stage", "piecewise"),
                ],
            )

            events.clear()
            piecewise.reset_mock()
            graphs.cuda_graph_runner.is_warmup_only = True
            graphs.run_cuda_graph_warmup(resource_manager)

            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    ("stage", "generation"),
                    ("stage", "additional"),
                    *lora_cleanup_events,
                ],
            )
            piecewise.assert_not_called()

            events.clear()
            generation.side_effect = RuntimeError("warmup failed")
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                graphs.run_cuda_graph_warmup(resource_manager)
            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    *lora_cleanup_events,
                ],
            )
            piecewise.assert_not_called()

            events.clear()
            generation.reset_mock()
            graphs.cuda_graph_runner.enabled = False
            graphs._config.prefill_cuda_graph_backend = PrefillCudaGraphBackend.DISABLED
            graphs.run_cuda_graph_warmup(resource_manager)
            self.assertEqual(
                events,
                [
                    ("enter", "gen_cuda_graph_warmup_seconds"),
                    ("enter", "lora_autotune"),
                    *lora_cleanup_events,
                ],
            )
            generation.assert_not_called()
            piecewise.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("graphable_breaks", [False, True])
def test_prefill_capture_skips_break_graphs_without_graphable_breaks(
    graphable_breaks: bool,
) -> None:
    """Only a bucket whose breakable graph has graphable breaks captures break graphs."""
    dummy_requests = MagicMock()
    dummy_requests.release_batch_context.return_value.__enter__.return_value = object()
    graphs = decoder_graphs(
        decoder_config(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            prefill_cuda_graph_num_tokens=[16, 128],
            max_num_tokens=256,
        ),
        dummy_requests=dummy_requests,
    )
    runner = Mock(metrics={}, num_break_graphs=0)
    runner.has_graphable_breaks.return_value = graphable_breaks
    graphs.breakable_cuda_graph_runner = runner
    graphs._metrics = collections.defaultdict(float)

    with (
        patch.object(graphs, "_capture_break_graphs") as capture_break_graphs,
        patch("torch.cuda.mem_get_info", return_value=(0, 0)),
        patch("torch.cuda.synchronize"),
    ):
        graphs._capture_prefill_cuda_graphs(ResourceManager({}))

    assert [c.args[0] for c in runner.capture.call_args_list] == [128, 16]
    # Bucket 128 exceeds max_num_tokens // 4, so only bucket 16 is a candidate.
    captured = [c.args[1] for c in capture_break_graphs.call_args_list]
    assert captured == ([16] if graphable_breaks else [])
