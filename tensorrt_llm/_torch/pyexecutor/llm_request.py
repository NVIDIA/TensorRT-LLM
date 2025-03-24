from typing import List, Optional

import torch

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
        self.py_draft_tokens = self.draft_tokens


def convert_wordlist(word_list) -> List[List[int]]:
    """Converts a wordlist from format:

    [[word_0 token_0, word_0 token_1, ...], [word_1 token_0, ...], ...]]

    into the TRTLLM word list format. The TRTLLM format expects a list of tokens,
    and an inclusive prefix sum. A word_list (either bad_words or stop_words) is
    a list that encodes the list of words that have to be banned / trigger the stop from generated sequences.
    Its shape is [2, badWordsLength], as explained below, or [batchSize, 2, badWordsLength]
    when there is a different list for each sequence in the batch.

    The badWordsList and stopWordsList tensors have the same shape [2, length]. Let's consider an example with three words to describe the
    representation of those lists.  The first word contains tokens [5, 7, 3], the
    second one contains [9, 2] and the third one is composed of tokens [6, 2, 4, 1]. In total, there are 9 tokens. That's the length. The shape of the tensor
    is [2, 9].  The first row of the tensor must contain the 9 token IDs and the
    second row must store the inclusive prefix-sum of the word lengths as shown on the following diagram:

        0           3       5              9
        |           |       |              |
        V           V       V              V
    [  5,  7,  3,  9,  2,  6,  2,  4,  1]
    [  3,  5,  9, -1, -1, -1, -1, -1, -1]
    """
    if not word_list:
        return []
    tokens = []
    offsets = []
    current_offset = 0
    for word_tokens in word_list:
        tokens.extend(word_tokens)
        current_offset += len(word_tokens)
        offsets.append(current_offset)
    if len(tokens) > len(offsets):
        offsets.extend([-1] * (len(tokens) - len(offsets)))
    return [tokens, offsets]


def executor_request_to_llm_request(req_id: int,
                                    executor_request: ExecutorRequest,
                                    input_token_ids: Optional[List] = None):
    executor_sampling_config = executor_request.sampling_config
    sampling_config = SamplingConfig(executor_sampling_config)

    input_tokens = input_token_ids if input_token_ids is not None else executor_request.input_token_ids

    llm_request_type = REQUEST_TYPE_MAPPING[executor_request.request_type]
    py_stop_words_list = convert_wordlist(
        executor_request.stop_words) if executor_request.stop_words else None

    llm_request = LlmRequest(
        request_id=req_id,
        max_new_tokens=executor_request.max_tokens,
        input_tokens=input_tokens,
        sampling_config=sampling_config,
        is_streaming=executor_request.streaming,
        end_id=executor_request.end_id,
        pad_id=executor_request.pad_id,
        embedding_bias=torch.tensor(executor_request.embedding_bias,
                                    dtype=torch.int32)
        if executor_request.embedding_bias else None,
        bad_words_list=torch.tensor(
            convert_wordlist(executor_request.bad_words), dtype=torch.int32)
        if executor_request.bad_words else None,
        stop_words_list=torch.tensor(py_stop_words_list, dtype=torch.int32)
        if executor_request.stop_words else None,
        prompt_embedding_table=None if executor_request.prompt_tuning_config
        is None else executor_request.prompt_tuning_config.embedding_table,
        prompt_vocab_size=None if executor_request.prompt_tuning_config is None
        else executor_request.prompt_tuning_config.embedding_table.shape[0],
        mrope_rotary_cos_sin=None if executor_request.mrope_config is None else
        executor_request.mrope_config.mrope_rotary_cos_sin,
        mrope_position_deltas=None if executor_request.mrope_config is None else
        executor_request.mrope_config.mrope_position_deltas,
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
        guided_decoding_params=executor_request.guided_decoding_params,
        encoder_input_tokens=None,
        return_encoder_output=False,
        client_id=executor_request.client_id,
        priority=0.5,
        llm_request_type=llm_request_type,
        context_phase_params=executor_request.context_phase_params)

    # TODO: remove this when use DynamicDecodeOp in pytorch flow.
    # currently, keep py_stop_words_list as python list, rather than tensor.
    llm_request.py_stop_words_list = py_stop_words_list

    return llm_request
