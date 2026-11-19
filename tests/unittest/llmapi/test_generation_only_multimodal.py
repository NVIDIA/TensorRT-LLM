# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class Processor:
    uses_mrope = True

    def get_mrope_config(self):
        raise AssertionError("mRoPE must be supplied by context")

    def get_mm_token_ids(self):
        return torch.tensor([91, 92])

    def __call__(self, *args, **kwargs):
        raise AssertionError("input processor must not run")


class PlainRopeProcessor(Processor):
    uses_mrope = False


class VocabProcessor(Processor):
    """Reports no media token IDs; media tokens are at or above the vocabulary size."""

    def get_mm_token_ids(self):
        return None

    def get_vocab_size(self):
        return 100


class RotaryTableProcessor(Processor):
    """Checks mRoPE positions like the Qwen-VL processors, with a 16-row rotary table."""

    config = SimpleNamespace(max_position_embeddings=16)

    def check_mrope_positions(self, *args):
        from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase

        return Qwen2VLInputProcessorBase.check_mrope_positions(self, *args)


class Submitted(Exception):
    pass


class Executor:
    """Records each request that reaches executor submission, then stops it."""

    def __init__(self):
        self.requests = []

    def is_shutdown(self):
        return False

    def shutdown(self):
        pass

    def generate_async(self, prompt_token_ids, **kwargs):
        self.requests.append(kwargs)
        raise Submitted


@pytest.fixture
def llm():
    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = Processor()
    llm.args = SimpleNamespace()
    llm._is_encoder_decoder_model = lambda: False
    llm._encode_only = False
    llm._executor = Executor()
    llm._prepare_sampling_params = lambda sampling_params=None: sampling_params or SamplingParams()
    llm._check_arguments = lambda *args, **kwargs: None
    return llm


def submit(llm, inputs, disagg, max_tokens=4):
    """Run generate_async up to the executor and return the multimodal params it got."""
    vars(llm.args).setdefault("mm_encoder_only", False)
    vars(llm.args).setdefault("return_perf_metrics", False)
    with pytest.raises(Submitted):
        llm.generate_async(
            inputs, SamplingParams(max_tokens=max_tokens), disaggregated_params=disagg
        )
    return llm._executor.requests[-1]["multimodal_params"]


def params(prompt_len=3):
    return DisaggregatedParams(
        request_type="generation_only",
        mrope_position_ids_handle=SharedTensorContainer.from_tensor_bytes(
            torch.arange(3 * prompt_len).reshape(3, 1, prompt_len)
        ).dump_to_dict(),
        mrope_position_deltas_handle=SharedTensorContainer.from_tensor_bytes(
            torch.tensor([-128], dtype=torch.int32)
        ).dump_to_dict(),
    )


def table_params(positions, delta):
    """Generation-only parameters with byte handles for a 4-token prompt."""
    return DisaggregatedParams(
        request_type="generation_only",
        mrope_position_ids_handle=SharedTensorContainer.from_tensor_bytes(
            torch.tensor(positions).expand(3, 1, 4)
        ).dump_to_dict(),
        mrope_position_deltas_handle=SharedTensorContainer.from_tensor_bytes(
            torch.tensor([[delta]], dtype=torch.int32)
        ).dump_to_dict(),
    )


def test_expanded_tokens_skip_input_processor(llm):
    ids, prompt, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        params(),
    )
    assert ids == [1, 91, 2]
    assert prompt is None
    mm.to_tensor("multimodal_data")
    assert mm.multimodal_data["mrope_config"]["mrope_position_deltas"].item() == -128


@pytest.mark.parametrize("missing", ["mrope_position_ids_handle", "mrope_position_deltas_handle"])
@pytest.mark.parametrize("processor", [Processor, PlainRopeProcessor])
def test_missing_mrope_fails_loudly(llm, processor, missing):
    # With plain RoPE too: a request that sends one handle must send both.
    llm.input_processor = processor()
    disagg = params()
    setattr(disagg, missing, None)
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess(
            {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None},
            SamplingParams(max_tokens=4),
            disagg,
        )


@pytest.mark.parametrize("processor", [Processor, PlainRopeProcessor])
def test_cuda_ipc_mrope_is_rejected(llm, processor):
    llm.input_processor = processor()
    disagg = params()
    disagg.mrope_position_ids_handle = {"method_key": 1}
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        submit(llm, {"prompt_token_ids": [1, 91, 2]}, disagg)


@pytest.mark.parametrize("key", ["tensor_size", "dtype", "data"])
def test_incomplete_byte_mrope_is_rejected(llm, key):
    disagg = params()
    del disagg.mrope_position_deltas_handle[key]
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        llm._preprocess({"prompt_token_ids": [1, 91, 2]}, SamplingParams(max_tokens=4), disagg)


@pytest.mark.parametrize(
    "prompt, name, mutate, error",
    [
        ([1, 91, 2], "mrope_position_ids_handle", {"tensor_size": [3, 1, 4]}, "shape"),
        ([1, 91, 2], "mrope_position_deltas_handle", {"tensor_size": [2]}, "shape"),
        ([1, 91, 2], "mrope_position_ids_handle", {"dtype": "torch.float32"}, "integers"),
        ([1, 91, 2], "mrope_position_ids_handle", {"dtype": "torch.int32"}, "of 36 bytes"),
        ([1, 91, 2], "mrope_position_ids_handle", lambda data: data + "AAAA", "of 72 bytes"),
        ([1, 91, 2], "mrope_position_ids_handle", lambda data: "*" + data[1:], "of 72 bytes"),
        ([1, 2, 3], "mrope_position_ids_handle", lambda data: data + "AAAA", "of 72 bytes"),
    ],
    ids=["ids-shape", "delta-shape", "float", "dtype-size", "too-long", "not-base64", "text-only"],
)
def test_malformed_byte_mrope_is_rejected_before_decoding(llm, prompt, name, mutate, error):
    disagg = params()
    handle = getattr(disagg, name)
    if callable(mutate):
        handle["data"] = mutate(handle["data"])
    else:
        handle.update(mutate)
    with pytest.raises(ValueError, match=f"{name}.*{error}"):
        llm._preprocess({"prompt_token_ids": prompt}, SamplingParams(max_tokens=4), disagg)


def test_media_tokens_above_vocab_are_detected(llm):
    llm.input_processor = VocabProcessor()
    disagg = DisaggregatedParams(request_type="generation_only")
    with pytest.raises(ValueError, match="requires portable mRoPE"):
        submit(llm, {"prompt_token_ids": [1, 101, 2]}, disagg)
    assert submit(llm, {"prompt_token_ids": [1, 2, 99]}, disagg) is None


class NoFlagProcessor:
    """A processor that does not say whether it uses mRoPE, like a text model's."""

    def get_mm_token_ids(self):
        return torch.tensor([91, 92])


@pytest.mark.parametrize(
    "processor, required",
    [(Processor, True), (PlainRopeProcessor, False), (NoFlagProcessor, False)],
    ids=["mrope", "plain-rope", "no-flag"],
)
def test_input_processor_decides_if_mrope_is_required(llm, processor, required):
    llm.input_processor = processor()
    # The config is not read: only the processor flag counts.
    llm.input_processor.config = SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 1]})
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=False))
    request = (
        {"prompt_token_ids": [1, 91, 2]},
        DisaggregatedParams(request_type="generation_only"),
    )
    if required:
        with pytest.raises(ValueError, match="requires portable mRoPE"):
            submit(llm, *request)
    else:
        assert submit(llm, *request) is None


def test_every_mrope_model_has_a_processor_that_says_so():
    """A model reads mRoPE state exactly when its input processor sets uses_mrope."""
    from importlib import import_module

    from tensorrt_llm._torch.models._arch_index import (
        MULTIMODAL_MODEL_TYPE_TO_MODULE,
        is_builtin_zoo_module,
    )
    from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY

    for module in sorted(set(MULTIMODAL_MODEL_TYPE_TO_MODULE.values())):
        import_module(f"tensorrt_llm._torch.models.{module}")
    reads, declares = set(), set()
    for model, processor in INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type.items():
        if not is_builtin_zoo_module(model.__module__):
            continue
        # The engine moves these paths to the GPU; mRoPE models list the position IDs.
        paths = getattr(model, "multimodal_data_device_paths", None)
        if isinstance(paths, property):
            paths = paths.fget(None)
        if "mrope_config.mrope_position_ids" in (paths or ()):
            reads.add(model.__name__)
        if getattr(processor, "uses_mrope", False):
            declares.add(model.__name__)
            # So the frontend can check the handed-over positions.
            assert callable(getattr(processor, "check_mrope_positions", None)), processor
    assert declares == reads
    assert {"Qwen2VLModel", "Qwen2_5_VLModel", "Qwen3VLModel", "Qwen3_5VLModel"} <= declares
    # EXAONE 4.5 reuses the Qwen2-VL processor but has plain RoPE.
    assert "Exaone4_5_ForConditionalGeneration" not in declares


def test_text_only_does_not_require_mrope(llm):
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 2, 3]},
        SamplingParams(max_tokens=4),
        DisaggregatedParams(request_type="generation_only"),
    )
    assert ids == [1, 2, 3]
    assert mm is None


def test_context_exports_portable_mrope_before_executor_submission(llm, monkeypatch):
    positions = torch.arange(12).reshape(3, 1, 4)
    deltas = torch.tensor([-32], dtype=torch.int32)

    def processor(*args, **kwargs):
        return [1, 91, 91, 2], {
            "multimodal_data": {
                "mrope_config": {"mrope_position_ids": positions, "mrope_position_deltas": deltas}
            }
        }

    monkeypatch.setattr(
        "tensorrt_llm.llmapi.llm.create_input_processor_with_hash",
        lambda *args, **kwargs: processor,
    )
    llm.args = SimpleNamespace(multimodal_config=None)
    disagg = DisaggregatedParams(request_type="context_only")
    _, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2], "multi_modal_data": {"image": [object()]}},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert disagg.mrope_position_ids_handle["method_key"] == 4
    assert disagg.mrope_position_deltas_handle["method_key"] == 4
    deltas.zero_()
    restored = SharedTensorContainer.from_dict(disagg.mrope_position_deltas_handle).get_local_view()
    assert restored.item() == -32
    mm.to_tensor("multimodal_data")


def test_generation_only_preserves_multimodal_cache_identity(llm):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    disagg = params(4)
    disagg.multimodal_input = asdict(identity)
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert ids == [1, 91, 2, 91]
    assert asdict(mm.multimodal_input) == asdict(identity)


@pytest.mark.parametrize("reuse", [False, True])
def test_plain_rope_model_does_not_require_mrope(llm, reuse):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    llm.input_processor = PlainRopeProcessor()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=reuse))
    disagg = DisaggregatedParams(request_type="generation_only", multimodal_input=asdict(identity))
    ids, _, mm, _ = llm._preprocess(
        {"prompt_token_ids": [1, 91, 2, 91], "multi_modal_data": None},
        SamplingParams(max_tokens=4),
        disagg,
    )
    assert ids == [1, 91, 2, 91]
    assert asdict(mm.multimodal_input) == asdict(identity)
    assert "mrope_config" not in mm.multimodal_data


def test_context_result_carries_cache_identity():
    from dataclasses import asdict

    from tensorrt_llm.executor import GenerationRequest, GenerationResult
    from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    request = GenerationRequest(
        prompt_token_ids=[1, 91, 2, 91],
        sampling_params=SamplingParams(max_tokens=1),
        multimodal_params=MultimodalParams(multimodal_input=identity),
    )
    request.set_id(1)
    original = DisaggregatedParams(request_type="context_only")
    result = GenerationResult(request, disaggregated_params=original)
    assert original.multimodal_input is None
    assert result.disaggregated_params.multimodal_input == asdict(identity)
    identity.multimodal_hashes[0][0] = 99
    assert result.disaggregated_params.multimodal_input["multimodal_hashes"][0][0] == 1


def test_generation_only_cache_reuse_requires_multimodal_identity(llm):
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=True))
    request = {"prompt_token_ids": [1, 91, 2], "multi_modal_data": None}
    with pytest.raises(ValueError, match="KV reuse requires cache identity") as error:
        submit(llm, request, params())
    # The error names the setting that serves the request without an identity.
    assert "kv_cache_config.enable_block_reuse=False" in str(error.value)
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=False))
    assert submit(llm, request, params()).multimodal_input is None


def test_identity_spans_must_pair_up():
    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components([[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2])
    # Changed after the constructor checks: the extra length must not be dropped silently.
    identity.multimodal_lengths.append(1)
    with pytest.raises(ValueError, match="zip"):
        identity.validate_for_prompt(4)


@pytest.mark.parametrize(
    "update, error",
    [
        ({"multimodal_hashes": [[1] * 7]}, "eight int32"),
        ({"multimodal_hashes": [[2**31] + [1] * 7]}, "eight int32"),
        ({"multimodal_positions": [3]}, "exceed the prompt"),
        ({"multimodal_run_positions": [1, 4]}, "exceed the prompt"),
    ],
    ids=["hash-width", "hash-int32", "span", "run"],
)
def test_invalid_multimodal_identity_is_rejected(llm, update, error):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    disagg = params(4)
    disagg.multimodal_input = {**asdict(identity), **update}
    with pytest.raises(ValueError, match=error):
        llm._preprocess({"prompt_token_ids": [1, 91, 2, 91]}, SamplingParams(max_tokens=4), disagg)


@pytest.mark.parametrize(
    "positions, delta, max_tokens, error",
    [
        ([0, 1, 1, 2], -1, 4, None),
        ([-1, 0, 0, 1], -2, 4, "Qwen mRoPE positions exceed"),
        ([0, 1, 1, 16], 13, 4, "Qwen mRoPE positions exceed"),
        ([0, 1, 1, 2], -100, 4, "inconsistent"),
        ([0, 1, 1, 10], 7, 6, "decode positions exceed"),
        ([0, 1, 1, 2], -1, None, None),
        ([0, 1, 1, 5], 2, None, "decode positions exceed"),
    ],
    ids=[
        "valid",
        "negative",
        "past-table",
        "delta",
        "decode-past-table",
        "no-max-tokens",
        "no-max-tokens-past-table",
    ],
)
def test_mrope_positions_must_fit_the_rotary_table(llm, positions, delta, max_tokens, error):
    # The worker indexes its rotary table with these values: a bad one fails the request here.
    llm.input_processor = RotaryTableProcessor()
    # Above the 16-row table: the worker caps it, so the check must too.
    llm.args.max_seq_len = 32
    disagg = table_params(positions, delta)
    request = {"prompt_token_ids": [1, 91, 91, 2]}
    if error is None:
        mm = submit(llm, request, disagg, max_tokens=max_tokens)
        assert mm.multimodal_data["mrope_config"]["mrope_position_ids"] is (
            disagg.mrope_position_ids_handle
        )
        return
    with pytest.raises(ValueError, match=error):
        submit(llm, request, disagg, max_tokens=max_tokens)
    assert llm._executor.requests == []


@pytest.mark.parametrize(
    "processor, disagg, error",
    [
        (Processor, params, "requires portable mRoPE"),
        (PlainRopeProcessor, None, "KV reuse requires cache identity"),
    ],
    ids=["mrope", "identity"],
)
def test_prepared_inputs_without_handoff_state_are_not_submitted(llm, processor, disagg, error):
    # preprocess() without disaggregated parameters, then a generation-only submission.
    llm.input_processor = processor()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=True))
    prepared = llm.preprocess({"prompt_token_ids": [1, 91, 2]})
    assert prepared.multimodal_params is None
    disagg = disagg() if disagg else DisaggregatedParams(request_type="generation_only")
    with pytest.raises(ValueError, match=error):
        submit(llm, prepared, disagg)
    assert llm._executor.requests == []


def test_prepared_inputs_with_handoff_state_are_submitted(llm):
    from dataclasses import asdict

    from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams

    identity = MultimodalInput.from_components(
        [[1, 2, 3, 4, 5, 6, 7, 8]], [1], [2], ["image-id"], [0, 2], [1, 3], [1, 1]
    )
    llm.input_processor = RotaryTableProcessor()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(enable_block_reuse=True))
    request = {"prompt_token_ids": [1, 91, 2, 91]}
    sampling_params = SamplingParams(max_tokens=4)
    # As trtllm-serve does: preprocess with the request's parameters, then submit.
    with pytest.raises(ValueError, match="positions exceed"):
        llm.preprocess(request, sampling_params, table_params([-1, 0, 0, 1], -2))
    disagg = table_params([0, 1, 1, 2], -1)
    disagg.multimodal_input = asdict(identity)
    prepared = llm.preprocess(request, sampling_params, disagg)
    assert submit(llm, prepared, disagg) is prepared.multimodal_params
    # Local state with no handles, like the gRPC prepared-decode path.
    local = MultimodalParams(
        multimodal_input=identity,
        multimodal_data={
            "mrope_config": {
                "mrope_position_ids": torch.tensor([0, 1, 1, 2]).expand(3, 1, 4),
                "mrope_position_deltas": torch.tensor([[-1]], dtype=torch.int32),
            }
        },
    )
    prepared = PreprocessedInputs(prompt_token_ids=[1, 91, 2, 91], multimodal_params=local)
    assert submit(llm, prepared, DisaggregatedParams(request_type="generation_only")) is local
