# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context output storage, bounded projection and mixed-request row ownership."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.context_logits import ContextLogitsPlan, ContextLogitsWorkspace
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm.executor.result import compute_prompt_logprobs


def make_head(
    vocab: int = 97,
    hidden: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
) -> SimpleNamespace:
    torch.manual_seed(17)
    head = SimpleNamespace(
        weight=torch.randn(vocab, hidden, device="cuda", dtype=dtype),
        output_dtype=output_dtype,
        num_embeddings=vocab,
    )
    head.context_logits_workspace = ContextLogitsWorkspace(head)
    return head


def make_request(
    length: int, vocab: int, *, full: bool = True, k: int | None = None, simple: bool = False
) -> SimpleNamespace:
    tokens = [i % vocab for i in range(length)]
    return SimpleNamespace(
        context_chunk_size=length,
        context_current_position=0,
        py_prompt_len=length,
        py_return_context_logits=full or k is not None,
        py_context_logits_explicit=full,
        py_prompt_logprobs=k,
        py_prompt_logprobs_simple_format=simple,
        is_last_context_chunk=True,
        get_tokens=lambda beam: tokens,
        py_result=PyResult(
            prompt_len=length, max_new_tokens=2, return_context_logits=full or k is not None
        ),
    )


def assert_logprobs_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for found, reference in zip(actual, expected):
        if isinstance(reference, float):
            assert found == pytest.approx(reference, abs=1e-5)
        else:
            assert found.keys() == reference.keys()
            for token in reference:
                assert found[token].rank == reference[token].rank
                assert found[token].logprob == pytest.approx(reference[token].logprob, abs=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("host", [False, True])
@torch.inference_mode()
def test_storage_oom_and_chunk_output_oom(monkeypatch, dtype, host) -> None:
    head = make_head(dtype=dtype)
    request = make_request(95, head.num_embeddings)
    hidden = torch.randn(95, 32, device="cuda", dtype=dtype)
    expected = (hidden @ head.weight.t()).float()
    workspace = head.context_logits_workspace
    original_empty = torch.empty
    if host:

        def fail_storage(shape, *args, **kwargs):
            if (
                tuple(shape) == (95, 1, head.num_embeddings)
                and torch.device(kwargs.get("device", "cpu")).type == "cuda"
            ):
                raise torch.OutOfMemoryError("injected context storage failure")
            return original_empty(shape, *args, **kwargs)

        monkeypatch.setattr(torch, "empty", fail_storage)
    attempts = []

    def fail_optional(rows):
        attempts.append(rows)
        raise torch.OutOfMemoryError("injected optional output failure")

    monkeypatch.setattr(workspace, "_allocate", fail_optional)
    pointers = [buffer.data_ptr() for buffer in workspace.resident]
    plan = ContextLogitsPlan(SimpleNamespace(context_requests=[request]))
    sampling = plan.project(hidden, head)
    assert plan.handled
    assert request.py_result.context_logits.device.type == ("cpu" if host else "cuda")
    torch.testing.assert_close(request.py_result.context_logits.cuda(), expected)
    torch.testing.assert_close(sampling, expected[-1:])
    assert [buffer.data_ptr() for buffer in workspace.resident] == pointers
    assert bool(attempts) is not host
    # Mutating resident scratch must not change saved results or sampling rows.
    for buffer in workspace.resident:
        buffer.fill_(123)
    torch.testing.assert_close(request.py_result.context_logits.cuda(), expected)
    torch.testing.assert_close(sampling, expected[-1:])
    # Result synchronization retains owned data, not the reusable output tensor.
    receiver = PyResult(
        prompt_len=95, max_new_tokens=2, return_context_logits=True, use_device_memory=True
    )
    receiver.apply_diff(request.py_result.get_diff())
    assert receiver.context_logits.device.type == ("cpu" if host else "cuda")
    torch.testing.assert_close(receiver.context_logits.cpu(), expected.cpu())


@pytest.mark.parametrize("full_first", [False, True])
@torch.inference_mode()
def test_mixed_prefill_selects_only_requested_rows(full_first) -> None:
    head = make_head()
    requests = [
        make_request(7, head.num_embeddings, full=full_first),
        make_request(11, head.num_embeddings, full=not full_first),
    ]
    hidden = torch.randn(20, 32, device="cuda", dtype=head.weight.dtype)
    reference = (hidden @ head.weight.t()).float()
    result = ContextLogitsPlan(SimpleNamespace(context_requests=requests)).project(hidden, head)
    torch.testing.assert_close(result, reference[[6, 17, 18, 19]])
    for request, start, end in zip(requests, [0, 7], [7, 18]):
        if request.py_context_logits_explicit:
            torch.testing.assert_close(request.py_result.context_logits, reference[start:end])
        else:
            assert request.py_result.context_logits is None


@pytest.mark.parametrize("k,simple", [(0, False), (0, True), (5, False)])
@pytest.mark.parametrize("full", [False, True])
@torch.inference_mode()
def test_prompt_logprobs_across_prefill_chunks(k, simple, full) -> None:
    head = make_head(dtype=torch.float32)
    request = make_request(91, head.num_embeddings, full=full, k=k, simple=simple)
    hidden = torch.randn(91, 32, device="cuda")
    reference = hidden @ head.weight.t()
    for start, end in [(0, 37), (37, 91)]:
        request.context_current_position = start
        request.context_chunk_size = end - start
        request.is_last_context_chunk = end == 91
        ContextLogitsPlan(SimpleNamespace(context_requests=[request])).project(
            hidden[start:end], head
        )
    targets = request.get_tokens(0)[1:] + [13]
    expected = compute_prompt_logprobs(k, reference.cpu(), targets, simple).prompt
    tail = compute_prompt_logprobs(
        k, request.py_result.context_prompt_last_logits, [13], simple
    ).prompt
    actual = request.py_result.context_prompt_logprobs + tail
    assert_logprobs_equal(actual, expected)
    if full:
        torch.testing.assert_close(request.py_result.context_logits, reference)
    else:
        assert request.py_result.context_logits is None
        assert request.py_result._context_logits._storage is None
    receiver = PyResult(
        prompt_len=91, max_new_tokens=2, return_context_logits=True, use_device_memory=False
    )
    receiver.apply_diff(request.py_result.get_diff())
    assert_logprobs_equal(
        receiver.context_prompt_logprobs, request.py_result.context_prompt_logprobs
    )
    torch.testing.assert_close(
        receiver.context_prompt_last_logits, request.py_result.context_prompt_last_logits
    )


@torch.inference_mode()
def test_optional_allocation_backoff(monkeypatch) -> None:
    head = make_head()
    workspace = head.context_logits_workspace
    allocate = workspace._allocate
    attempts = []

    def limited(rows):
        attempts.append(rows)
        if rows > 100:
            raise torch.OutOfMemoryError("injected large output failure")
        return allocate(rows)

    monkeypatch.setattr(workspace, "_allocate", limited)
    output, converted = workspace.acquire(384, host_storage=False)
    assert attempts == [384, 192, 96]
    assert output.shape[0] == converted.shape[0] == 96
    assert output.data_ptr() != workspace.resident[0].data_ptr()


@pytest.mark.parametrize("host", [False, True])
def test_prompt_postprocessing_is_chunked(monkeypatch, host) -> None:
    logits = torch.randn(9, 1024 * 1024, device="cpu" if host else "cuda")
    original = torch.nn.functional.log_softmax
    rows = []

    def record(input, *args, **kwargs):
        rows.append(input.shape[0])
        return original(input, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "log_softmax", record)
    result = compute_prompt_logprobs(0, logits, list(range(9)), True)
    assert len(result.prompt) == 9
    assert max(rows) <= 4
    expected = original(logits, dim=-1)[torch.arange(9), torch.arange(9)].cpu().tolist()
    assert_logprobs_equal(result.prompt, expected)


@torch.inference_mode()
def test_prompt_reduction_oom_falls_back_to_cpu(monkeypatch) -> None:
    logits = torch.randn(7, 97, device="cuda")
    expected = compute_prompt_logprobs(3, logits.cpu(), list(range(7))).prompt
    original = torch.nn.functional.log_softmax
    devices = []

    def fail_cuda(input, *args, **kwargs):
        devices.append(input.device.type)
        if input.is_cuda:
            raise torch.OutOfMemoryError("injected reduction allocation failure")
        return original(input, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "log_softmax", fail_cuda)
    result = compute_prompt_logprobs(3, logits, list(range(7)))
    assert devices == ["cuda", "cpu"]
    assert_logprobs_equal(result.prompt, expected)


@torch.inference_mode()
def test_actual_allocator_limit_preserves_resident_workspace() -> None:
    head = make_head(vocab=4096)
    workspace = head.context_logits_workspace
    hidden = torch.randn(7, 32, dtype=head.weight.dtype, device="cuda")
    expected = (hidden @ head.weight.t()).float()
    old_fraction = torch.cuda.get_per_process_memory_fraction()
    # The allocator limit is process-local; it does not consume other clients'
    # GPU memory. Existing cached blocks remain usable for the resident path.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory
    limit = (torch.cuda.memory_reserved() + 4 * 1024 * 1024) / total
    try:
        torch.cuda.set_per_process_memory_fraction(limit)
        acquired = workspace.acquire(16384, host_storage=False)
        assert acquired is workspace.resident
        result = workspace.project(head, hidden, acquired)
        torch.testing.assert_close(result, expected)
    finally:
        torch.cuda.set_per_process_memory_fraction(old_fraction)


@pytest.mark.parametrize("output_dtype", [None, torch.float32])
@torch.inference_mode()
def test_lm_head_and_logits_processor_use_prepared_workspace(monkeypatch, output_dtype) -> None:
    from tensorrt_llm._torch.modules.context_logits import context_logits_scope
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    head = LMHead(97, 32, dtype=torch.bfloat16, output_dtype=output_dtype).cuda()
    head.weight.normal_()
    assert head.context_logits_workspace is None
    head.prepare_context_logits_workspace()
    resident = head.context_logits_workspace.resident
    head.prepare_context_logits_workspace()
    assert head.context_logits_workspace.resident is resident
    hidden = torch.randn(43, 32, device="cuda", dtype=torch.bfloat16)
    kwargs = {} if output_dtype is None else {"out_dtype": output_dtype}
    reference = torch.mm(hidden, head.weight.t(), **kwargs).float()
    request = make_request(43, 97)
    plan = ContextLogitsPlan(SimpleNamespace(context_requests=[request]))
    processor = LogitsProcessor()

    def fail_optional(rows):
        raise torch.OutOfMemoryError("injected output failure")

    monkeypatch.setattr(head.context_logits_workspace, "_allocate", fail_optional)
    with context_logits_scope(plan):
        sampled = processor(hidden, head, None, return_context_logits=True)
    assert plan.handled
    torch.testing.assert_close(sampled, reference[-1:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(request.py_result.context_logits, reference, rtol=1e-4, atol=1e-4)
    # Models with subsequent vocabulary transforms must retain all rows.
    bypassed_plan = ContextLogitsPlan(SimpleNamespace(context_requests=[request]))
    with context_logits_scope(bypassed_plan):
        complete = LogitsProcessor(allow_context_logits_plan=False)(
            hidden, head, None, return_context_logits=True
        )
    assert not bypassed_plan.handled
    torch.testing.assert_close(complete, reference, rtol=1e-4, atol=1e-4)
    # Ordinary forwards do not see request state or enter the context path.
    ordinary = processor(hidden, head, None)
    torch.testing.assert_close(ordinary.reshape(1, -1), reference[-1:], rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_handled_context_outputs_are_not_appended_twice() -> None:
    from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits

    head = make_head()
    requests = [make_request(5, 97), make_request(3, 97, full=False)]
    for request in requests:
        request.py_return_generation_logits = False
        request.py_use_chunked_generation_logits = False
    hidden = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
    plan = ContextLogitsPlan(SimpleNamespace(context_requests=requests))
    sampling = plan.project(hidden, head)
    HandleLogits()(requests, [], sampling, 1, [0, 1, 2], True, context_logits_handled=True)
    assert requests[0].py_result.context_logits.shape == (5, 97)
    assert requests[1].py_result.context_logits is None


def test_output_allocation_does_not_swallow_non_oom_errors(monkeypatch) -> None:
    head = make_head()

    def invalid(rows):
        raise RuntimeError("invalid output configuration")

    monkeypatch.setattr(head.context_logits_workspace, "_allocate", invalid)
    with pytest.raises(RuntimeError, match="invalid output configuration"):
        head.context_logits_workspace.acquire(256, host_storage=False)


@pytest.mark.parametrize("simple", [False, True])
def test_worker_assembles_compact_prompt_and_caches_streaming_result(simple) -> None:
    from tensorrt_llm.executor.base_worker import _compute_pytorch_prompt_logprobs

    logits = torch.randn(5, 97)
    prompt = [1, 2, 3, 4, 5]
    first_token = 17
    prefix = compute_prompt_logprobs(0, logits[:-1], prompt[1:], simple).prompt
    response = SimpleNamespace(
        result=SimpleNamespace(
            context_logits=None,
            context_prompt_logprobs=prefix,
            context_prompt_last_logits=logits[-1:],
            get_result=lambda: SimpleNamespace(output_token_ids=[[first_token]]),
        )
    )
    generation = SimpleNamespace(
        _streaming=True,
        _logprob_params=SimpleNamespace(prompt_logprobs=0, prompt_logprobs_simple_format=simple),
        _generation_request=SimpleNamespace(prompt_token_ids=prompt),
    )
    result = _compute_pytorch_prompt_logprobs(generation, response)
    expected = compute_prompt_logprobs(0, logits, prompt[1:] + [first_token], simple).prompt
    assert_logprobs_equal(result.prompt, expected)
    response.result.context_prompt_last_logits = None
    cached = _compute_pytorch_prompt_logprobs(generation, response)
    assert cached.prompt is generation._cached_prompt_logprobs
    assert_logprobs_equal(cached.prompt, expected)


def test_receiving_rank_can_fall_back_independently(monkeypatch) -> None:
    source = PyResult(prompt_len=9, max_new_tokens=1, return_context_logits=True)
    logits = torch.randn(9, 97, device="cuda")
    source.append_context_logits(logits)
    diff = source.get_diff()
    assert not diff.context_logits_on_cpu
    original_empty = torch.empty

    def fail_receiver(shape, *args, **kwargs):
        if tuple(shape) == (9, 1, 97) and torch.device(kwargs.get("device", "cpu")).type == "cuda":
            raise torch.OutOfMemoryError("injected receiving-rank allocation failure")
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_receiver)
    receiver = PyResult(prompt_len=9, max_new_tokens=1, return_context_logits=True)
    receiver.apply_diff(diff)
    assert receiver.context_logits.device.type == "cpu"
    torch.testing.assert_close(receiver.context_logits, logits.cpu())


def make_model_and_mapping() -> tuple[torch.nn.Module, SimpleNamespace]:
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

    model = torch.nn.Module()
    model.lm_head = LMHead(97, 32, dtype=torch.float32).cuda()
    with torch.no_grad():
        model.lm_head.weight.normal_()
    model.logits_processor = LogitsProcessor()
    mapping = SimpleNamespace(
        tp_size=1, cp_size=1, enable_attention_dp=False, is_last_pp_rank=lambda: True
    )
    return model, mapping


@pytest.mark.parametrize("legacy_first", [False, True])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("k,simple", [(0, True), (3, False)])
@torch.inference_mode()
def test_prompt_logprobs_survive_batch_dependent_projection_switch(
    legacy_first: bool, full: bool, k: int, simple: bool
) -> None:
    from tensorrt_llm._torch.modules.context_logits import context_logits_scope
    from tensorrt_llm._torch.pyexecutor.handle_logits import HandleLogits

    model, mapping = make_model_and_mapping()
    model.lm_head.prepare_context_logits_workspace()
    request = make_request(91, 97, full=full, k=k, simple=simple)
    peer = make_request(1, 97, full=False)
    for item in (request, peer):
        item.py_return_generation_logits = False
        item.py_use_chunked_generation_logits = False
        item.prepopulated_prompt_len = 0
        item.state = None
    batch = SimpleNamespace(context_requests=[request], all_requests=lambda: [request, peer])
    hidden = torch.randn(92, 32, device="cuda")
    reference = hidden[:91] @ model.lm_head.weight.t()
    for index, (start, end) in enumerate([(0, 37), (37, 91)]):
        request.context_current_position = start
        request.context_chunk_size = end - start
        request.is_last_context_chunk = end == 91
        legacy = legacy_first if index == 0 else not legacy_first
        peer.py_logits_post_processors = [lambda *args: None] if legacy else None
        plan = ContextLogitsPlan.for_batch(model, mapping, batch, is_spec_decode=False)
        assert (plan is None) == legacy
        inputs = torch.cat([hidden[start:end], hidden[-1:]])
        with context_logits_scope(plan):
            logits = model.logits_processor(inputs, model.lm_head, None, True)
        handled = plan is not None and plan.handled
        HandleLogits()(
            [request],
            [peer],
            logits,
            1,
            [0, 1 if handled else end - start],
            True,
            context_logits_handled=handled,
        )
    targets = request.get_tokens(0)[1:] + [13]
    expected = compute_prompt_logprobs(k, reference.cpu(), targets, simple).prompt
    tail = compute_prompt_logprobs(
        k, request.py_result.context_prompt_last_logits, [13], simple
    ).prompt
    assert_logprobs_equal(request.py_result.context_prompt_logprobs + tail, expected)
    if full:
        torch.testing.assert_close(request.py_result.context_logits, reference)
    else:
        assert request.py_result.context_logits is None
        assert request.py_result._context_logits._storage is None


@pytest.mark.parametrize(
    "mode", ["supported", "tp", "cp", "adp", "non_last_pp", "spec", "custom", "opt_out"]
)
def test_warmup_and_execution_share_projection_eligibility(mode: str) -> None:
    from tensorrt_llm._torch.modules.context_logits import context_logits_heads

    model, mapping = make_model_and_mapping()
    if mode == "tp":
        mapping.tp_size = 2
    elif mode == "cp":
        mapping.cp_size = 2
    elif mode == "adp":
        mapping.enable_attention_dp = True
    elif mode == "non_last_pp":
        mapping.is_last_pp_rank = lambda: False
    elif mode == "custom":
        model.logits_processor = torch.nn.Identity()
    elif mode == "opt_out":
        model.logits_processor.allow_context_logits_plan = False
    request = make_request(9, 97)
    batch = SimpleNamespace(context_requests=[request], all_requests=lambda: [request])
    assert ContextLogitsPlan.for_batch(model, mapping, batch, is_spec_decode=mode == "spec") is None
    heads = context_logits_heads(model, mapping, is_spec_decode=mode == "spec")
    for head in heads:
        head.prepare_context_logits_workspace()
    plan = ContextLogitsPlan.for_batch(model, mapping, batch, is_spec_decode=mode == "spec")
    if mode == "supported":
        assert heads == (model.lm_head,)
        assert plan is not None
        assert model.lm_head.context_logits_workspace is not None
    else:
        assert heads == ()
        assert plan is None
        assert model.lm_head.context_logits_workspace is None


@pytest.mark.parametrize(
    "mode",
    ["supported", "unprepared", "custom", "opt_out", "tp", "spec", "postprocessor", "ordinary"],
)
def test_decoder_keeps_graph_replay_unless_bounded_projection_is_active(mode: str) -> None:
    from contextlib import nullcontext
    from unittest.mock import Mock

    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    model, mapping = make_model_and_mapping()
    if mode != "unprepared":
        model.lm_head.prepare_context_logits_workspace()
    if mode == "custom":
        model.logits_processor = torch.nn.Identity()
    elif mode == "opt_out":
        model.logits_processor.allow_context_logits_plan = False
    elif mode == "tp":
        mapping.tp_size = 2
    request = make_request(1, 97, full=mode != "ordinary")
    if mode == "postprocessor":
        request.py_logits_post_processors = [lambda *args: None]
    batch = SimpleNamespace(
        context_requests=[request], num_context_requests=1, all_requests=lambda: [request]
    )
    promoted = SimpleNamespace(context_requests=[], generation_requests=[request])
    promoted_ids = frozenset({17})
    key = object()
    metadata = SimpleNamespace()
    outputs = {"logits": object()}
    replay = Mock(return_value=outputs)
    forward = Mock(return_value=outputs)

    def run(graph_key, inputs, run_model, postprocess_inputs, **kwargs):
        return replay() if graph_key is not None else run_model(inputs)

    graphs = SimpleNamespace(
        attention_dp_execution_batch=lambda value: value,
        promote_final_contexts=lambda *args, **kwargs: (promoted, promoted_ids),
        pad_batch=lambda *args: nullcontext(promoted),
        select=Mock(
            return_value=SimpleNamespace(
                key=key, use_lora_graph=False, attn_metadata=metadata, spec_metadata=None
            )
        ),
        captures_model_body=lambda graph_key: False,
        run=Mock(side_effect=run),
    )
    inputs = SimpleNamespace(
        prepare_inputs=Mock(return_value=SimpleNamespace(kwargs={}, runtime_draft_len=0)),
        record_promoted_iter_states=Mock(),
        record_input_copy=Mock(),
        _dspark_device_budget=None,
    )
    bypass = Mock(side_effect=lambda **kwargs: nullcontext())
    runner = SimpleNamespace(
        _model=model,
        _mapping=mapping,
        _config=SimpleNamespace(is_spec_decode=mode == "spec", spec_config=None, use_mrope=False),
        _graphs=graphs,
        _inputs=inputs,
        _buffers=SimpleNamespace(draft_tokens_cuda=None),
        _metadata=SimpleNamespace(setup_attention=lambda *args: metadata, attn_metadata=metadata),
        _moe_load_balancer=None,
        _stage_in_graph_sampling=None,
        _cuda_graph_metadata_prepare_scope=lambda *args, **kwargs: nullcontext(),
        _start_beam_histories=lambda batch: None,
        _maybe_bypass_torch_compile=bypass,
        _forward_step=forward,
        forward_pass_callable=None,
        _execute_logit_post_processors=Mock(),
        _restore_attention_dp_semantic_outputs=lambda batch, values: values,
    )
    forward_inputs = SimpleNamespace(
        batch=batch,
        new_tensors_device=None,
        cache_indirection_buffer=None,
        gather_context_logits=mode != "ordinary",
        enable_spec_decode=False,
        runtime_draft_len=0,
    )
    manager = SimpleNamespace(get_resource_manager=lambda kind: object())
    result = DecoderRunner._forward_decoder(runner, forward_inputs, manager, is_dummy=False)
    bounded = mode == "supported"
    assert graphs.run.call_args.args[0] is (None if bounded else key)
    assert inputs.prepare_inputs.call_args.args[0] is (batch if bounded else promoted)
    assert inputs.prepare_inputs.call_args.args[7] is not bounded
    assert result["logits"] is outputs["logits"]
    bypass.assert_called_once_with(bypass=bounded)
    if bounded:
        forward.assert_called_once()
        replay.assert_not_called()
        inputs.record_promoted_iter_states.assert_not_called()
    else:
        replay.assert_called_once()
        forward.assert_not_called()
        inputs.record_promoted_iter_states.assert_called_once_with(batch)
