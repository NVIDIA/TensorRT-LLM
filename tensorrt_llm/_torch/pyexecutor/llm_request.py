from typing import List, Optional

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm.bindings import executor as tllm_executor

SamplingConfig = tensorrt_llm.bindings.SamplingConfig
'''
CONTEXT_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.CONTEXT_INIT: 2>
ENCODER_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.ENCODER_INIT: 1>
GENERATION_COMPLETE: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_COMPLETE: 5>
GENERATION_IN_PROGRESS: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_IN_PROGRESS: 3>
GENERATION_TO_COMPLETE: typing.ClassVar[
    LlmRequestState]  # value = <LlmRequestState.GENERATION_TO_COMPLETE: 4>
UNKNOWN: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.UNKNOWN: 0>
'''
LlmRequestState = tensorrt_llm.bindings.LlmRequestState
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType

ExecutorRequest = tllm_executor.Request
ExecutorResponse = tllm_executor.Response
ExecutorSamplingConfig = tllm_executor.SamplingConfig

REQUEST_TYPE_MAPPING = {
    tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION:
    LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
    tllm_executor.RequestType.REQUEST_TYPE_CONTEXT_ONLY:
    LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
    tllm_executor.RequestType.REQUEST_TYPE_GENERATION_ONLY:
    LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
}


class LlmRequest(tensorrt_llm.bindings.internal.batch_manager.LlmRequest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.py_request_id = self.request_id
        self.py_end_id = self.end_id
        self.py_prompt_len = self.prompt_len
        self.py_orig_prompt_len = self.orig_prompt_len
        self.py_max_new_tokens = self.max_new_tokens
        self.py_batch_idx = None
        self.py_rewind_len = 0
        self.py_draft_tokens = None


def executor_request_to_llm_request(req_id: int,
                                    executor_request: ExecutorRequest,
                                    input_token_ids: Optional[List] = None):
    executor_sampling_config = executor_request.sampling_config
    sampling_config = SamplingConfig(executor_sampling_config)

    # todo: remove this when have pytorch tensor binding
    assert executor_request.embedding_bias is None, "Tensor not supported now."
    assert executor_request.bad_words is None or len(
        executor_request.bad_words) == 0, "Tensor not supported now."

    input_tokens = input_token_ids if input_token_ids is not None else executor_request.input_token_ids

    llm_request_type = REQUEST_TYPE_MAPPING[executor_request.request_type]

    llm_request = LlmRequest(
        request_id=req_id,
        max_new_tokens=executor_request.max_tokens,
        input_tokens=input_tokens,
        sampling_config=sampling_config,
        is_streaming=executor_request.streaming,
        end_id=executor_request.end_id,
        pad_id=executor_request.pad_id,
        embedding_bias=None,
        bad_words_list=None,
        stop_words_list=None,
        prompt_embedding_table=None if executor_request.prompt_tuning_config
        is None else executor_request.prompt_tuning_config.embedding_table,
        prompt_vocab_size=None if executor_request.prompt_tuning_config is None
        else executor_request.prompt_tuning_config.embedding_table.shape[0],
        lora_task_id=None,
        lora_weights=None,
        lora_config=None,
        lookahead_config=None,
        return_log_probs=False,
        return_context_logits=executor_request.output_config.
        return_context_logits,
        return_generation_logits=False,
        draft_tokens=getattr(executor_request, "draft_tokens", None),
        draft_logits=None,
        exclude_input_from_output=executor_request.output_config.
        exclude_input_from_output,
        logits_post_processor=None,
        apply_logits_post_processor_batched=False,
        encoder_input_tokens=None,
        return_encoder_output=False,
        client_id=executor_request.client_id,
        priority=0.5,
        llm_request_type=llm_request_type,
        context_phase_params=executor_request.context_phase_params)

    # TODO: remove this when use DynamicDecodeOp in pytorch flow.
    # currently, keep py_stop_workds_list as python list, rather than tensor.
    llm_request.py_stop_words_list = executor_request.stop_words

    return llm_request
