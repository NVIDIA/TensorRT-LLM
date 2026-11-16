# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission ownership and portable generation-only metadata regressions."""

import asyncio
import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2VLInputProcessorBase,
    _prepare_qwen_vl_mrope_config,
)
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2
from tensorrt_llm.grpc.openengine.disagg import _struct_to_dict, disaggregated_params_from_request
from tensorrt_llm.grpc.openengine.preparation import (
    _DECODE_PREPARATION_KEY,
    _decode_preparation,
    _prepared_decode_from_request,
)
from tensorrt_llm.grpc.openengine.server import OpenEngineServer
from tensorrt_llm.grpc.openengine.servicer import _PreparationLease
from tensorrt_llm.grpc.openengine.streaming import _format_result, _OpenEngineFormatState
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.llm import PreprocessedInputs
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_server_rejects_invalid_media_admission_environment(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_OPENENGINE_MEDIA_PREPARATION_LIMIT", value)
    with pytest.raises(ValueError):
        OpenEngineServer("localhost", 0, object(), "model")


@pytest.mark.asyncio
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("fails", [False, True])
async def test_cancelled_worker_retains_admission_until_thread_finishes(decode, fails):
    admission = asyncio.Semaphore(1)
    await admission.acquire()
    lease = _PreparationLease(admission)
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        assert release.wait(5)
        if fails:
            raise ValueError("late worker failure")
        return b"decoded"

    with ThreadPoolExecutor(max_workers=1) as pool:

        async def prepare():
            try:
                if decode:
                    with BaseMediaIO.use_executor(pool, on_submit=lease.track):
                        return await BaseMediaIO._run_in_executor(work)
                return await lease.run(pool, work)
            finally:
                lease.close()

        task = asyncio.create_task(prepare())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert admission.locked()
            waiter = asyncio.create_task(admission.acquire())
            await asyncio.sleep(0)
            assert not waiter.done()
        finally:
            release.set()
        await asyncio.wait_for(waiter, 5)
        admission.release()
        lease.close()
        assert admission._value == 1


def _fixture():
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        model_type="qwen2_vl",
        max_position_embeddings=128,
        to_dict=lambda: {"model_type": "qwen2_vl"},
    )
    processor._processor = None
    llm = SimpleNamespace(input_processor=processor, args=SimpleNamespace(max_seq_len=128))
    request = generation_pb2.GenerateRequest(
        request_id="prefill",
        model="qwen",
        prompt="<|image_pad|> describe",
        media=[
            generation_pb2.MediaItem(
                modality=generation_pb2.MODALITY_IMAGE, raw_bytes=b"image", mime_type="image/png"
            )
        ],
    )
    positions = torch.tensor([0, 1, 1, 1, 2]).reshape(1, 1, 5).expand(3, 1, 5).clone()
    capture = {
        "prompt_token_ids": [1, 2, 3, 4, 5],
        "state": processor.export_decode_state(
            {
                "mrope_config": {
                    "mrope_position_ids": positions,
                    "mrope_position_deltas": torch.tensor([[-2]]),
                }
            }
        ),
    }
    return llm, request, capture, positions


def _handoff(request, payload):
    session = kv_pb2.KvSessionRef(session_id="1")
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    session.attributes_struct.update({_DECODE_PREPARATION_KEY: payload})
    request.kv.session.CopyFrom(session)


def test_qwen_prepared_decode_roundtrip_preserves_tokens_and_positions():
    llm, request, capture, positions = _fixture()
    payload = _decode_preparation(request, llm, capture)
    _handoff(request, payload)
    restored = _prepared_decode_from_request(request, llm)
    assert restored.prompt_token_ids == capture["prompt_token_ids"]
    params = restored.multimodal_params
    try:
        params.to_tensor("multimodal_data")
        assert torch.equal(params.multimodal_data["mrope_config"]["mrope_position_ids"], positions)
        assert params.multimodal_data["mrope_config"]["mrope_position_ids"].dtype == positions.dtype
        assert params.multimodal_data["mrope_config"]["mrope_position_deltas"].tolist() == [[-2]]
        assert params.multimodal_data["mrope_config"]["mrope_position_deltas"].dtype == torch.int32
        assert set(params.multimodal_data) == {"mrope_config"}
    finally:
        params.release_handles()


def test_restored_qwen_delta_writes_to_engine_cache():
    llm, request, capture, _ = _fixture()
    _handoff(request, _decode_preparation(request, llm, capture))
    restored = _prepared_decode_from_request(request, llm)
    params = restored.multimodal_params
    try:
        params.to_tensor("multimodal_data")
        cache = torch.full((4,), 99, dtype=torch.int32)
        slots = torch.tensor([2], dtype=torch.int64)
        config = _prepare_qwen_vl_mrope_config(
            multimodal_params=[params],
            num_generation_requests=1,
            position_ids=None,
            rotary_emb=None,
            mrope_position_deltas_cache=cache,
            mrope_rotary_cos_sin_workspace=torch.empty((1, 0)),
            mrope_delta_write_seq_slots=slots,
            mrope_delta_read_seq_slots=slots,
        )
        assert cache.tolist() == [99, 99, -2, 99]
        assert config["mrope_position_deltas"].dtype == torch.int32
        assert config["mrope_position_deltas"].tolist() == [[-2]]
    finally:
        params.release_handles()


@pytest.mark.parametrize(
    "positions,delta",
    [
        ([0, 1, 1, 1, 2], 1_000_000),
        ([0, 1, 1, 1, 2], -(2**63)),
        ([0, 1, 1, 1, 2], 2**63 - 1),
        ([-1, 1, 1, 1, 2], -2),
        ([0, 1, 1, 1, 128], 124),
        ([0, 1, 1, 1, 2**63 - 1], 2**63 - 5),
        ([0, 1, 1, 1, 5], 1),
    ],
)
def test_prepared_decode_rejects_unsafe_mrope_before_tensor_or_handle_creation(
    positions, delta, monkeypatch
):
    llm, request, capture, _ = _fixture()
    payload = json.loads(_decode_preparation(request, llm, capture))
    config = payload["state"]["mrope_config"]
    for name, values in (
        ("mrope_position_ids", positions * 3),
        ("mrope_position_deltas", [delta]),
    ):
        config[name]["data"] = base64.b64encode(
            torch.tensor(values, dtype=torch.int64).numpy().astype("<i8").tobytes()
        ).decode("ascii")
    _handoff(request, payload)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid mRoPE must be rejected before tensor/IPC handle creation")

    monkeypatch.setattr(torch, "from_numpy", unexpected_allocation)
    monkeypatch.setattr(MultimodalParams, "to_handle", unexpected_allocation)
    with pytest.raises(ValueError, match="Qwen mRoPE"):
        _prepared_decode_from_request(request, llm)


@pytest.mark.parametrize("nested_config", [False, True])
@pytest.mark.parametrize("delta", [-2, 0, 3])
def test_prepared_decode_accepts_mrope_within_generation_budget(delta, nested_config):
    llm, request, capture, _ = _fixture()
    if nested_config:
        llm.input_processor._config.text_config = SimpleNamespace(max_position_embeddings=128)
        del llm.input_processor._config.max_position_embeddings
    positions = torch.tensor([0, 1, 1, 1, 4 + delta]).reshape(1, 1, 5).expand(3, 1, 5)
    capture["state"] = llm.input_processor.export_decode_state(
        {
            "mrope_config": {
                "mrope_position_ids": positions,
                "mrope_position_deltas": torch.tensor([[delta]]),
            }
        }
    )
    # Positive deltas reach exactly the last valid RoPE table row.
    request.stopping.max_tokens = 128 - delta - 5
    _handoff(request, _decode_preparation(request, llm, capture))
    restored = _prepared_decode_from_request(request, llm)
    restored.multimodal_params.release_handles()
    if delta > 0:
        request.stopping.max_tokens += 1
        with pytest.raises(ValueError, match="decode positions exceed"):
            _prepared_decode_from_request(request, llm)


@pytest.mark.parametrize("max_tokens", [None, 8])
def test_prepared_decode_uses_model_bound_when_runtime_limit_is_inferred(max_tokens):
    llm, request, capture, _ = _fixture()
    llm.args.max_seq_len = None
    if max_tokens is not None:
        request.stopping.max_tokens = max_tokens
    _handoff(request, _decode_preparation(request, llm, capture))
    restored = _prepared_decode_from_request(request, llm)
    restored.multimodal_params.release_handles()


def test_prepared_decode_preserves_multimodal_cache_identity_and_item_layout():
    llm, request, capture, _ = _fixture()
    identity = MultimodalInput.from_components(
        [[-1, 2, 3, 4, 5, 6, 7, 8]],
        [1],
        [2],
        ["image-uuid"],
        [0, 1],
        [1],
        [2],
    )
    preprocessed = PreprocessedInputs(
        prompt_token_ids=capture["prompt_token_ids"],
        multimodal_params=MultimodalParams(multimodal_input=identity),
    )
    payload = _decode_preparation(request, llm, capture, preprocessed)
    _handoff(request, payload)
    restored = _prepared_decode_from_request(request, llm)
    try:
        assert restored.multimodal_params.multimodal_input == identity
    finally:
        restored.multimodal_params.release_handles()


@pytest.mark.parametrize("field", ["hash_width", "hash_type", "run_bounds", "position_bounds"])
def test_prepared_decode_rejects_invalid_multimodal_layout(field):
    llm, request, capture, _ = _fixture()
    payload = json.loads(_decode_preparation(request, llm, capture))
    identity = {
        "multimodal_hashes": [[1] * 8],
        "multimodal_positions": [1],
        "multimodal_lengths": [2],
    }
    if field == "hash_width":
        identity["multimodal_hashes"] = [[1] * 7]
    elif field == "hash_type":
        identity["multimodal_hashes"] = [["invalid"] * 8]
    elif field == "position_bounds":
        identity["multimodal_positions"] = [len(capture["prompt_token_ids"])]
    else:
        identity.update(
            multimodal_item_run_cu_offsets=[0, 1],
            multimodal_run_positions=[len(capture["prompt_token_ids"])],
            multimodal_run_lengths=[2],
        )
    payload["multimodal_input"] = identity
    _handoff(request, payload)
    with pytest.raises(ValueError):
        _prepared_decode_from_request(request, llm)


@pytest.mark.parametrize("field", ["media_io", "image_processor"])
def test_prepared_decode_rejects_different_processor_settings(monkeypatch, field):
    llm, request, capture, _ = _fixture()
    _handoff(request, _decode_preparation(request, llm, capture))
    if field == "media_io":
        monkeypatch.setattr(
            llm.input_processor,
            "get_preferred_media_io_kwargs",
            lambda: {"image": {"format": "np"}},
        )
    else:
        llm.input_processor._processor = SimpleNamespace(
            image_processor=SimpleNamespace(to_dict=lambda: {"max_pixels": 1000000})
        )
    with pytest.raises(ValueError, match="model configuration"):
        _prepared_decode_from_request(request, llm)


@pytest.mark.parametrize(
    "mutation",
    ["media", "prompt", "config", "tokens", "shape", "fraction", "fields", "identity_overflow"],
)
def test_prepared_decode_rejects_mismatched_or_invalid_payloads(mutation):
    llm, request, capture, _ = _fixture()
    payload = json.loads(_decode_preparation(request, llm, capture))
    if mutation == "media":
        request.media[0].raw_bytes = b"different-image"
    elif mutation == "prompt":
        request.prompt = "different prompt"
    elif mutation == "config":
        payload["config_digest"] = "different"
    elif mutation == "tokens":
        payload["prompt_token_ids_u32le"] = "%%%"
    elif mutation == "shape":
        payload["state"]["mrope_config"]["mrope_position_ids"] = [1, 2]
    elif mutation == "fraction":
        payload["state"]["mrope_config"]["mrope_position_deltas"]["shape"] = [1, 0.5]
    elif mutation == "identity_overflow":
        payload["multimodal_input"] = {
            "multimodal_hashes": [[1 << 4096, 2, 3, 4, 5, 6, 7, 8]],
            "multimodal_positions": [1],
            "multimodal_lengths": [2],
        }
    else:
        payload["state"]["unrecognized"] = []
    _handoff(request, payload)
    with pytest.raises(ValueError):
        _prepared_decode_from_request(request, llm)


def test_unknown_preparation_version_falls_back_and_missing_metadata_is_unchanged():
    llm, request, capture, _ = _fixture()
    assert _prepared_decode_from_request(request, llm) is None
    payload = json.loads(_decode_preparation(request, llm, capture))
    payload["version"] = 99
    _handoff(request, payload)
    assert _prepared_decode_from_request(request, llm) is None


@pytest.mark.parametrize("invalid", ["syntax", "object", "size", "kind", "depth"])
def test_prepared_decode_rejects_invalid_json_envelopes(invalid):
    llm, request, _, _ = _fixture()
    payload = {
        "syntax": "{",
        "object": "[]",
        "size": "x" * (2 * 1024 * 1024),
        "kind": 17,
        "depth": "[" * 2000 + "0" + "]" * 2000,
    }[invalid]
    _handoff(request, payload)
    with pytest.raises(ValueError):
        _prepared_decode_from_request(request, llm)


def test_oversized_preparation_keeps_the_ordinary_decode_path():
    llm, request, capture, _ = _fixture()
    llm.args.max_seq_len = 131072
    capture["prompt_token_ids"] = [1] * 65536
    capture["state"] = llm.input_processor.export_decode_state(
        {
            "mrope_config": {
                "mrope_position_ids": torch.zeros((3, 1, 65536), dtype=torch.int64),
                "mrope_position_deltas": torch.zeros((1, 1), dtype=torch.int64),
            }
        }
    )
    assert _decode_preparation(request, llm, capture) is None


def test_prefill_handoff_carries_portable_preparation_metadata():
    llm, request, capture, _ = _fixture()
    payload = _decode_preparation(request, llm, capture)
    params = SimpleNamespace(
        ctx_request_id=1,
        ctx_dp_rank=0,
        ctx_info_endpoint="tcp://localhost:1234",
        opaque_state=None,
        first_gen_tokens=[7],
        draft_tokens=None,
        disagg_request_id=None,
        schedule_style=None,
        ctx_usage=None,
        first_gen_log_probs=None,
    )
    result = SimpleNamespace(
        error=None,
        prompt_token_ids=capture["prompt_token_ids"],
        cached_tokens=0,
        finished=False,
        outputs=[
            SimpleNamespace(
                index=0, token_ids=[], text="", finish_reason=None, disaggregated_params=params
            )
        ],
    )
    state = _OpenEngineFormatState(
        request_id="prefill",
        sampling_params=SamplingParams(),
        is_context_only=True,
        kv_transfer_backend="nixl",
        decode_preparation=payload,
    )
    responses = _format_result(result, state)
    assert len(responses) == 1
    response = responses[0]
    assert response.WhichOneof("event") == "prefill_ready"
    assert response.prefill_ready.kv_session.attributes_struct[_DECODE_PREPARATION_KEY] == payload
    request.kv.session.CopyFrom(response.prefill_ready.kv_session)
    restored = _prepared_decode_from_request(request, llm)
    assert restored.prompt_token_ids == capture["prompt_token_ids"]
    restored.multimodal_params.release_handles()


def test_disaggregated_session_filters_preparation_and_preserves_generation_tokens():
    request = generation_pb2.GenerateRequest(request_id="decode")
    session = request.kv.session
    session.session_id = "1"
    session.attributes_struct.update(
        {
            _DECODE_PREPARATION_KEY: "prepared-image-state",
            "ctx_info_endpoint": "tcp://localhost:1234",
            "first_gen_tokens": [7, 8],
        }
    )

    attributes = _struct_to_dict(session.attributes_struct)
    assert _DECODE_PREPARATION_KEY not in attributes
    assert attributes["first_gen_tokens"] == [7, 8]
    params = disaggregated_params_from_request(request)
    assert params.request_type == "generation_only"
    assert params.ctx_request_id == 1
    assert params.first_gen_tokens == [7, 8]
    assert params.ctx_info_endpoint == "tcp://localhost:1234"
    assert session.attributes_struct[_DECODE_PREPARATION_KEY] == "prepared-image-state"


def test_capture_is_request_scoped_and_does_not_modify_processor_output():
    llm, _, capture, positions = _fixture()
    processor = llm.input_processor
    output = (
        capture["prompt_token_ids"],
        {
            "multimodal_data": {
                "mrope_config": {
                    "mrope_position_ids": positions,
                    "mrope_position_deltas": torch.tensor([[-2]]),
                }
            }
        },
    )
    with processor.capture_decode_state() as result:
        assert processor._capture_prepared_decode(output) is output
    assert result == capture
    assert processor._decode_state_capture.get() is None
    assert processor._capture_prepared_decode(output) is output
