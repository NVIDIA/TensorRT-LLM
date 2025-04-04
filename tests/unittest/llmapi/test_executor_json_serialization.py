from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParamsSafe
from tensorrt_llm.executor.utils import ExecutorResponseTensorsSafe, ExecutorResponseSafe

# Test json serialization for various executor data param classes that are used in ZMQ socket IPC communication

def test_DisaggregatedParams_json_serialization():
    # Test with all fields populated
    params = DisaggregatedParamsSafe(
        request_type="context_only",
        first_gen_tokens=[123, 456],
        ctx_request_id=789,
        opaque_state=b"1234567890",
        draft_tokens=[1, 2, 3]
    )

    # Test JSON serialization
    json_str = params.model_dump_json()
    assert isinstance(json_str, str)
    
    # Test deserialization
    deserialized = DisaggregatedParamsSafe.model_validate_json(json_str)
    
    # Verify the deserialized data matches the original
    assert deserialized.request_type == params.request_type
    assert deserialized.first_gen_tokens == params.first_gen_tokens
    assert deserialized.ctx_request_id == params.ctx_request_id
    assert deserialized.opaque_state == params.opaque_state
    assert deserialized.draft_tokens == params.draft_tokens

    # Test with None values
    params_none = DisaggregatedParamsSafe(
        request_type=None,
        first_gen_tokens=None,
        ctx_request_id=None,
        opaque_state=None,
        draft_tokens=None
    )
    
    json_str_none = params_none.model_dump_json()
    deserialized_none = DisaggregatedParamsSafe.model_validate_json(json_str_none)
    
    assert deserialized_none.request_type is None
    assert deserialized_none.first_gen_tokens is None
    assert deserialized_none.ctx_request_id is None
    assert deserialized_none.opaque_state is None
    assert deserialized_none.draft_tokens is None

def test_ExecutorResponseTensors_json_serialization():
    tensors = ExecutorResponseTensorsSafe(
        output_token_ids=[[64, 128], [256, 512]],
        context_logits="random_context_tensor_123",
        generation_logits="random_generation_tensor_456",
        log_probs=[0.1, 0.2, 0.3, 0.4],
        cum_log_probs=[0.1, 0.3, 0.6, 1.0],
    )

    # context_logits and generation_logits cannot be torch.Tensor as pydantic is not be able to serialize it
    assert isinstance(tensors.context_logits, str)
    assert isinstance(tensors.generation_logits, str)

    # Test JSON serialization
    json_str = tensors.model_dump_json()
    assert isinstance(json_str, str)
    
    # Test deserialization
    deserialized = ExecutorResponseTensorsSafe.model_validate_json(json_str)
    
    # Verify the deserialized data matches the original
    assert deserialized.output_token_ids == tensors.output_token_ids
    assert deserialized.context_logits == tensors.context_logits
    assert deserialized.generation_logits == tensors.generation_logits
    assert deserialized.log_probs == tensors.log_probs
    assert deserialized.cum_log_probs == tensors.cum_log_probs

    # Test with None values
    tensors_none = ExecutorResponseTensorsSafe(
        output_token_ids=[[64, 128]],
        context_logits="random_context_tensor_123",
        generation_logits="random_generation_tensor_456",
        log_probs=None,
        cum_log_probs=None,
    )
    
    json_str_none = tensors_none.model_dump_json()
    deserialized_none = ExecutorResponseTensorsSafe.model_validate_json(json_str_none)
    
    assert deserialized_none.output_token_ids == tensors_none.output_token_ids
    assert deserialized_none.context_logits == tensors_none.context_logits
    assert deserialized_none.generation_logits == tensors_none.generation_logits
    assert deserialized_none.log_probs is None
    assert deserialized_none.cum_log_probs is None

def _convert_FinishReason_enum_to_int(finish_reasons: list[tllm.FinishReason]) -> list[int]:
    return [reason.value for reason in finish_reasons]

def _convert_int_to_FinishReason_enum(finish_reasons: list[int]) -> list[tllm.FinishReason]:
    return [tllm.FinishReason(reason) for reason in finish_reasons]

def _convert_Exception_to_str(error: Exception) -> str:
    return f"{error.__class__.__name__}:{','.join(str(arg) for arg in error.args)}"

def _convert_str_to_Exception(error: str) -> Exception:
    exception_type, exception_args = error.split(":", 1)
    exception_type = eval(exception_type)
    exception_args = exception_args.split(",") if exception_args else []
    return exception_type(*exception_args)

def _check_two_exceptions_equal(e: Exception, e2: Exception) -> bool:
    assert type(e) is type(e2) and e.args == e2.args

def test_ExecutorResponse_json_serialization():
    # Create sample tensors
    tensors = ExecutorResponseTensorsSafe(
        output_token_ids=[[64, 128], [256, 512]],
        context_logits="random_context_tensor_123",
        generation_logits="random_generation_tensor_456",
        log_probs=[0.1, 0.2, 0.3, 0.4],
        cum_log_probs=[0.1, 0.3, 0.6, 1.0],
    )

    # Create sample disaggregated params
    disaggregated_params = DisaggregatedParamsSafe(
        request_type="context_only",
        first_gen_tokens=[123, 456],
        ctx_request_id=789,
        opaque_state=b"1234567890",
        draft_tokens=[1, 2, 3]
    )

    finish_reason_lst = [
        tllm.FinishReason.NOT_FINISHED,
        tllm.FinishReason.END_ID,
        tllm.FinishReason.STOP_WORDS,
        tllm.FinishReason.LENGTH,
        tllm.FinishReason.TIMED_OUT,
        tllm.FinishReason.CANCELLED
    ]

    error_handle = TypeError("This is a type error.")

    # Test with all fields populated
    response = ExecutorResponseSafe(
        client_id=1,
        tensors=tensors,
        finish_reasons=_convert_FinishReason_enum_to_int(finish_reason_lst),
        is_final=True,
        sequence_index=0,
        error=_convert_Exception_to_str(error_handle),
        timestamp=1234.5678,
        disaggregated_params=disaggregated_params
    )

    # Test JSON serialization
    json_str = response.model_dump_json()
    assert isinstance(json_str, str)
    
    # Test deserialization
    deserialized = ExecutorResponseSafe.model_validate_json(json_str)
    deserialized.finish_reasons = _convert_int_to_FinishReason_enum(deserialized.finish_reasons)
    deserialized.error = _convert_str_to_Exception(deserialized.error)
    
    # Verify the deserialized data matches the original
    assert deserialized.client_id == response.client_id
    assert deserialized.tensors.output_token_ids == response.tensors.output_token_ids
    assert deserialized.tensors.context_logits == response.tensors.context_logits
    assert deserialized.tensors.generation_logits == response.tensors.generation_logits
    assert deserialized.tensors.log_probs == response.tensors.log_probs
    assert deserialized.tensors.cum_log_probs == response.tensors.cum_log_probs
    assert deserialized.finish_reasons == finish_reason_lst
    assert deserialized.is_final == response.is_final
    assert deserialized.sequence_index == response.sequence_index
    assert isinstance(deserialized.error, Exception)
    _check_two_exceptions_equal(deserialized.error, error_handle)
    assert deserialized.timestamp == response.timestamp
    assert deserialized.disaggregated_params.request_type == response.disaggregated_params.request_type
    assert deserialized.disaggregated_params.first_gen_tokens == response.disaggregated_params.first_gen_tokens
    assert deserialized.disaggregated_params.ctx_request_id == response.disaggregated_params.ctx_request_id
    assert deserialized.disaggregated_params.opaque_state == response.disaggregated_params.opaque_state
    assert deserialized.disaggregated_params.draft_tokens == response.disaggregated_params.draft_tokens
