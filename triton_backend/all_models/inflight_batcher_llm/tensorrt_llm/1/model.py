import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from random import randint
from threading import Lock, Thread
from typing import Any, List

import numpy as np
import pandas as pd
import torch
import triton_python_backend_utils as pb_utils
from torch import from_numpy
from torch.utils.dlpack import from_dlpack

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.llmapi.tokenizer import _xgrammar_tokenizer_info

METRIC_TOTAL_OUTPUT_TOKENS = "total_output_tokens"
METRIC_TOTAL_INPUT_TOKENS = "total_input_tokens"
import tensorrt_llm.logger as logger

# From https://github.com/pytorch/pytorch/blob/39425feac799905402abe4d15667fa47c344f2d7/torch/testing/_internal/common_utils.py#L1761
# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {
    value: key
    for (key, value) in numpy_to_torch_dtype_dict.items()
}
torch_to_numpy_dtype_dict.update({
    torch.bfloat16: np.float32,
    torch.complex32: np.complex64
})


@dataclass
class RequestData:
    triton_req_id: int
    triton_user_id: str
    batch_index: int
    batch_size: int
    num_return_sequences: int
    num_input_tokens: int
    num_output_tokens: int
    response_sender: Any
    return_num_input_tokens: bool = False
    return_num_output_tokens: bool = False


def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank()


def get_input_tensor_by_name(request,
                             name,
                             expected_batch_size=None,
                             batch_index=None,
                             force_on_torch=False):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None

    if tensor.is_cpu() and not force_on_torch:
        tensor = tensor.as_numpy()
    else:
        tensor = from_dlpack(tensor.to_dlpack())

    if expected_batch_size is not None and tensor.shape[
            0] != expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Expected batch size doesn't match batch size for tensor {name}. Expected {expected_batch_size} got {tensor.shape[0]}"
        )

    if batch_index is not None and expected_batch_size is not None and batch_index >= expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Invalid batch index in get_input_tensor_by_name for {name}")

    if batch_index is not None:
        # Add leading 1 batch dimension
        if isinstance(tensor, np.ndarray):
            return np.expand_dims(tensor[batch_index], axis=0)
        elif isinstance(tensor, torch.Tensor):
            return torch.unsqueeze(tensor[batch_index], dim=0)
    else:
        return tensor


def get_input_scalar_by_name(request,
                             name,
                             expected_batch_size=1,
                             batch_index=0):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    tensor = tensor.as_numpy()

    if tensor.size != expected_batch_size:
        raise pb_utils.TritonModelException(
            f"Expected a scalar tensor for tensor {name}")

    return tensor.item(batch_index)


def read_parameter_as_type(value, name, pytype=str):
    if value == "":
        return None
    if value.startswith("${") and value.endswith("}"):
        return None
    if pytype is bool:
        return value.lower() in ["1", "true"]
    try:
        result = pytype(value)
        return result
    except:
        pb_utils.Logger.log_warning(
            f"Could not read parameter '{name}' with value '{value}', will use default."
        )
        return None


def get_parameter(model_config, name, pytype=str):
    if name not in model_config['parameters']:
        return None
    return read_parameter_as_type(
        model_config['parameters'][name]['string_value'], name, pytype)


def convert_word_list(word_list):
    if word_list is None:
        return None
    word_list = word_list.tolist()
    if len(word_list) == 0 or len(word_list[0]) != 2:
        raise pb_utils.TritonModelException(f"Invalid format for word list.")
    words, indices = word_list[0]
    result = []
    current_index = 0
    for i in indices:
        if i == -1:
            continue
        if i > len(words):
            raise pb_utils.TritonModelException(
                f"Invalid format for word list.")
        current_word = []
        while current_index < i:
            current_word.append(words[current_index])
            current_index += 1
        result.append(current_word)
    return result


def parse_medusa_choices(medusa_choices):
    if medusa_choices is None:
        return None
    try:
        result = json.loads("[" +
                            medusa_choices.replace("{", "[").replace("}", "]") +
                            "]")
        assert isinstance(result, list) and len(result) > 0
        assert all([isinstance(x, list) for x in result])
        assert all([isinstance(y, int) for x in result for y in x])
    except Exception:
        raise pb_utils.TritonModelException("Invalid format for medusa_choices")
    return result


def parse_eagle_choices(eagle_choices):
    return parse_medusa_choices(eagle_choices)


def get_sampling_config_from_request(request, batch_size=1, batch_index=0):
    kwargs = {}
    kwargs['beam_width'] = get_input_scalar_by_name(
        request, 'beam_width', batch_size, batch_index) or 1
    kwargs['top_k'] = get_input_scalar_by_name(request, 'runtime_top_k',
                                               batch_size, batch_index)
    kwargs['top_p'] = get_input_scalar_by_name(request, 'runtime_top_p',
                                               batch_size, batch_index)
    kwargs['top_p'] = None if kwargs['top_p'] is None or kwargs[
        'top_p'] <= 0 else kwargs['top_p']
    kwargs['seed'] = get_input_scalar_by_name(request, 'seed', batch_size,
                                              batch_index)
    kwargs['temperature'] = get_input_scalar_by_name(request, 'temperature',
                                                     batch_size, batch_index)
    kwargs['min_tokens'] = get_input_scalar_by_name(request, 'min_tokens',
                                                    batch_size, batch_index)
    kwargs['repetition_penalty'] = get_input_scalar_by_name(
        request, 'repetition_penalty', batch_size, batch_index)
    kwargs['presence_penalty'] = get_input_scalar_by_name(
        request, 'presence_penalty', batch_size, batch_index)
    kwargs['frequency_penalty'] = get_input_scalar_by_name(
        request, 'frequency_penalty', batch_size, batch_index)
    kwargs['length_penalty'] = get_input_scalar_by_name(request, 'len_penalty',
                                                        batch_size, batch_index)
    kwargs['top_p_min'] = get_input_scalar_by_name(request, 'runtime_top_p_min',
                                                   batch_size, batch_index)
    kwargs['top_p_reset_ids'] = get_input_scalar_by_name(
        request, 'runtime_top_p_reset_ids', batch_size, batch_index)
    kwargs['top_p_decay'] = get_input_scalar_by_name(request,
                                                     'runtime_top_p_decay',
                                                     batch_size, batch_index)
    kwargs['beam_search_diversity_rate'] = get_input_scalar_by_name(
        request, 'beam_search_diversity_rate', batch_size, batch_index)
    kwargs['early_stopping'] = get_input_scalar_by_name(request,
                                                        'early_stopping',
                                                        batch_size, batch_index)
    kwargs['num_return_sequences'] = get_input_scalar_by_name(
        request, 'num_return_sequences', batch_size, batch_index) or 1
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return trtllm.SamplingConfig(**kwargs)


def get_output_config_from_request(request, batch_size=1, batch_index=0):
    kwargs = {}
    kwargs["return_log_probs"] = get_input_scalar_by_name(
        request, 'return_log_probs', batch_size, batch_index)
    kwargs["return_context_logits"] = get_input_scalar_by_name(
        request, 'return_context_logits', batch_size, batch_index)
    kwargs["return_generation_logits"] = get_input_scalar_by_name(
        request, 'return_generation_logits', batch_size, batch_index)
    kwargs["return_perf_metrics"] = get_input_scalar_by_name(
        request, 'return_perf_metrics', batch_size, batch_index)
    if get_input_scalar_by_name(request, 'return_kv_cache_reuse_stats',
                                batch_size, batch_index):
        pb_utils.Logger.log_warn(
            "return_kv_cache_reuse_stats is deprecated, please use return_perf_metrics instead."
        )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return trtllm.OutputConfig(**kwargs)


def get_external_draft_tokens_config_from_request(request,
                                                  batch_size=1,
                                                  batch_index=0):
    kwargs = {}
    draft_input_ids = get_input_tensor_by_name(request, 'draft_input_ids',
                                               batch_size, batch_index)
    if draft_input_ids is not None:
        kwargs['tokens'] = draft_input_ids[0].tolist()
    draft_logits = get_input_tensor_by_name(request, 'draft_logits', batch_size,
                                            batch_index)
    if draft_logits is not None:
        kwargs['logits'] = from_numpy(draft_logits).squeeze(dim=0)
    kwargs['acceptance_threshold'] = get_input_scalar_by_name(
        request, 'draft_acceptance_threshold', batch_size, batch_index)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.ExternalDraftTokensConfig(**kwargs)
    return None


def get_prompt_tuning_config_from_request(request,
                                          batch_size=1,
                                          batch_index=0,
                                          input_length=0):
    # prompt_vocab_size is unused by executor.
    kwargs = {}
    prompt_embedding_table = get_input_tensor_by_name(request,
                                                      'prompt_embedding_table',
                                                      batch_size,
                                                      batch_index,
                                                      force_on_torch=True)
    prompt_table_extra_ids = get_input_tensor_by_name(request,
                                                      'prompt_table_extra_ids',
                                                      batch_size, batch_index)
    if prompt_embedding_table is not None:
        if isinstance(prompt_embedding_table, np.ndarray):
            kwargs["embedding_table"] = from_numpy(
                prompt_embedding_table).squeeze(dim=0)
        elif isinstance(prompt_embedding_table, torch.Tensor):
            kwargs["embedding_table"] = prompt_embedding_table.squeeze(dim=0)

        if prompt_table_extra_ids is not None:
            prompt_table_extra_ids = prompt_table_extra_ids[0].tolist()
            if len(prompt_table_extra_ids) != 0:
                kwargs["input_token_extra_ids"] = prompt_table_extra_ids[
                    0:input_length]
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.PromptTuningConfig(**kwargs)
    return None


def get_lora_config_from_request(request, batch_size=1, batch_index=0):
    kwargs = {}
    kwargs["task_id"] = get_input_scalar_by_name(request, 'lora_task_id',
                                                 batch_size, batch_index)
    lora_weights = get_input_tensor_by_name(request, 'lora_weights', batch_size,
                                            batch_index)
    if lora_weights is not None:
        kwargs["weights"] = from_numpy(lora_weights).squeeze(dim=0)
    lora_config = get_input_tensor_by_name(request, 'lora_config', batch_size,
                                           batch_index)
    if lora_config is not None:
        kwargs["config"] = from_numpy(lora_config).squeeze(dim=0)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.LoraConfig(**kwargs)
    return None


def get_guided_decoding_params_from_request(request,
                                            batch_size=1,
                                            batch_index=0):
    kwargs = {}
    guided_decoding_guide_type = get_input_tensor_by_name(
        request, 'guided_decoding_guide_type', batch_size, batch_index)
    if guided_decoding_guide_type is not None:
        guided_decoding_guide_type = guided_decoding_guide_type.squeeze(
            axis=0)[0].decode()
        guided_decoding_guide_type_mapping = {
            "json": trtllm.GuidedDecodingParams.GuideType.JSON,
            "json_schema": trtllm.GuidedDecodingParams.GuideType.JSON_SCHEMA,
            "regex": trtllm.GuidedDecodingParams.GuideType.REGEX,
            "ebnf_grammar": trtllm.GuidedDecodingParams.GuideType.EBNF_GRAMMAR
        }
        guided_decoding_guide_type = guided_decoding_guide_type_mapping.get(
            guided_decoding_guide_type)
    kwargs['guide_type'] = guided_decoding_guide_type

    guided_decoding_guide = get_input_tensor_by_name(request,
                                                     'guided_decoding_guide',
                                                     batch_size, batch_index)
    if guided_decoding_guide is not None:
        kwargs['guide'] = guided_decoding_guide.squeeze(axis=0)[0].decode()
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.GuidedDecodingParams(**kwargs)
    return None


def get_kv_cache_retention_config_from_request(request,
                                               batch_size=1,
                                               batch_index=0):

    def get_tensor_and_check_length(name: str, expected_length: int):
        tensor = get_input_tensor_by_name(request, name, batch_size,
                                          batch_index)

        if tensor is None:
            raise RuntimeError(f"{name} must be provided.")

        tensor = np.squeeze(tensor, axis=0)

        if len(tensor) != expected_length:
            raise RuntimeError(
                f"Invalid {name} length. Expected length {expected_length}, got length {len(tensor)}"
            )

        return tensor

    token_range_starts = get_input_tensor_by_name(
        request, "retention_token_range_starts", batch_size, batch_index)

    if token_range_starts is not None:
        token_range_starts = np.squeeze(token_range_starts, axis=0)

        token_range_ends = get_tensor_and_check_length(
            "retention_token_range_ends", len(token_range_starts))
        token_range_ends = [
            None if end == -1 else end for end in token_range_ends
        ]

        token_range_priorities = get_tensor_and_check_length(
            "retention_token_range_priorities", len(token_range_starts))

        token_range_durations_ms = get_input_tensor_by_name(
            request, "retention_token_range_durations_ms", batch_size,
            batch_index)

        if token_range_durations_ms is None:
            token_range_durations_ms = [None] * len(token_range_starts)
        else:
            token_range_durations_ms = np.squeeze(token_range_durations_ms,
                                                  axis=0)
            token_range_durations_ms = [
                None if duration == -1 else duration
                for duration in token_range_durations_ms
            ]

            if len(token_range_durations_ms) != len(token_range_starts):
                raise RuntimeError(
                    f"Invalid retention_token_range_durations length. Expected length {len(token_range_starts)}, got length {len(token_range_durations_ms)}"
                )

        ranges = []

        for start, end, priority, duration_ms in zip(token_range_starts,
                                                     token_range_ends,
                                                     token_range_priorities,
                                                     token_range_durations_ms):
            ranges.append(
                trtllm.KvCacheRetentionConfig.TokenRangeRetentionConfig(
                    token_start=start,
                    token_end=end,
                    priority=priority.item(),
                    duration_ms=None if duration_ms is None else
                    datetime.timedelta(milliseconds=duration_ms.item())))

        decode_args = {}

        decode_priority = get_input_scalar_by_name(request,
                                                   "retention_decode_priority",
                                                   batch_size, batch_index)
        if decode_priority is not None:
            decode_args['decode_retention_priority'] = decode_priority

        decode_duration_ms = get_input_scalar_by_name(
            request, "retention_decode_duration_ms", batch_size, batch_index)
        if decode_duration_ms is not None:
            decode_args[
                'decode_duration_ms'] = decode_duration_ms if decode_duration_ms != -1 else None

        return trtllm.KvCacheRetentionConfig(
            token_range_retention_configs=ranges, **decode_args)

    return None


def get_lookahead_decoding_config_from_request(request,
                                               executor_lookahead_config,
                                               batch_size=1,
                                               batch_index=0):
    lookahead_window_size = get_input_tensor_by_name(request,
                                                     "lookahead_window_size",
                                                     batch_size, batch_index)

    lookahead_ngram_size = get_input_tensor_by_name(request,
                                                    "lookahead_ngram_size",
                                                    batch_size, batch_index)

    lookahead_verification_set_size = get_input_tensor_by_name(
        request, "lookahead_verification_set_size", batch_size, batch_index)

    # None lookahead config for requests.
    if all(x is None for x in [
            lookahead_window_size, lookahead_ngram_size,
            lookahead_verification_set_size
    ]):
        return None

    # Have request lookahead config but no executor config.
    if executor_lookahead_config is None:
        raise RuntimeError(
            "The request lookahead decoding input tensors (window_size, ngram_size and verification_set_size) can only be set if the model instance lookahead parameters are also specified"
        )

    return trtllm.LookaheadDecodingConfig(lookahead_window_size,
                                          lookahead_ngram_size,
                                          lookahead_verification_set_size)


def get_mrope_config_from_request(request, batch_size=1, batch_index=0):
    mrope_rotary_cos_sin = get_input_tensor_by_name(request,
                                                    'mrope_rotary_cos_sin',
                                                    batch_size, batch_index)
    mrope_position_deltas = get_input_tensor_by_name(request,
                                                     'mrope_position_deltas',
                                                     batch_size,
                                                     batch_index,
                                                     force_on_torch=False)
    assert (mrope_rotary_cos_sin is None) == (
        mrope_position_deltas is None
    ), "Both mrope_rotary_cos_sin and mrope_position_detals must be either None or not None."

    if mrope_rotary_cos_sin is not None and mrope_position_deltas is not None:
        mrope_config = trtllm.MropeConfig(
            mrope_rotary_cos_sin=mrope_rotary_cos_sin[0],
            mrope_position_deltas=mrope_position_deltas[0])
        return mrope_config
    return None


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values (1, 5), starting from 10 until the value exceeds
    the specified maximum.

    Example:
    >>> build_1_2_5_buckets(1000)
    [10, 50, 100, 500, 1000]
    """
    mantissa_lst = [1, 5]
    exponent = 1  # Start from exponent 1 instead of 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def convert_request(request,
                    exclude_input_from_output,
                    decoupled,
                    executor_lookahead_config=None):
    inputs = {}
    input_token_ids = get_input_tensor_by_name(request, 'input_ids')
    if input_token_ids is None:
        raise pb_utils.TritonModelException("A value is required for input_ids")
    if len(input_token_ids.shape) != 2:
        raise pb_utils.TritonModelException(f"Invalid format for input_ids")
    batch_size = input_token_ids.shape[0]
    requests = []
    for batch_index in range(0, batch_size):
        input_token_ids = get_input_tensor_by_name(request, 'input_ids',
                                                   batch_size, batch_index)[0]
        if input_token_ids is None:
            raise pb_utils.TritonModelException(
                "A value is required for input_ids")
        input_token_ids = input_token_ids.tolist()
        if len(input_token_ids) == 0:
            raise pb_utils.TritonModelException(f"Invalid format for input_ids")

        input_length = get_input_scalar_by_name(request, 'input_lengths',
                                                batch_size, batch_index)
        if input_length is None:
            input_length = len(input_token_ids)
        # Trim input token ids with input_lengths
        inputs['input_token_ids'] = input_token_ids[0:input_length]
        inputs['max_tokens'] = get_input_scalar_by_name(request,
                                                        'request_output_len',
                                                        batch_size, batch_index)
        if inputs['max_tokens'] is None:
            raise pb_utils.TritonModelException(
                "A value is required for request_output_len")
        inputs['streaming'] = get_input_scalar_by_name(request, 'streaming',
                                                       batch_size, batch_index)
        if inputs['streaming'] and not decoupled:
            raise pb_utils.TritonModelException(
                "Streaming is only supported in decoupled mode.")

        inputs['end_id'] = get_input_scalar_by_name(request, 'end_id',
                                                    batch_size, batch_index)
        inputs['pad_id'] = get_input_scalar_by_name(request, 'pad_id',
                                                    batch_size, batch_index)
        inputs['stop_words'] = convert_word_list(
            get_input_tensor_by_name(request, 'stop_words_list', batch_size,
                                     batch_index))
        inputs['bad_words'] = convert_word_list(
            get_input_tensor_by_name(request, 'bad_words_list', batch_size,
                                     batch_index))
        embedding_bias = get_input_tensor_by_name(request, 'embedding_bias',
                                                  batch_size, batch_index)
        if embedding_bias is not None and embedding_bias.size != 0:
            inputs['embedding_bias'] = from_numpy(embedding_bias).squeeze(dim=0)

        sampling_config = get_sampling_config_from_request(
            request, batch_size, batch_index)
        output_config = get_output_config_from_request(request, batch_size,
                                                       batch_index)
        req_exclude_input_from_output = get_input_scalar_by_name(
            request, 'exclude_input_in_output', batch_size, batch_index)
        if req_exclude_input_from_output is None:
            # if request doesn't specify exclude_input_from_output, try to use the parameter
            output_config.exclude_input_from_output = (
                exclude_input_from_output
                if exclude_input_from_output is not None else False)
        else:
            output_config.exclude_input_from_output = req_exclude_input_from_output

        external_draft_tokens_config = get_external_draft_tokens_config_from_request(
            request, batch_size, batch_index)
        prompt_tuning_config = get_prompt_tuning_config_from_request(
            request, batch_size, batch_index, input_length)
        mrope_config = get_mrope_config_from_request(request, batch_size,
                                                     batch_index)
        lora_config = get_lora_config_from_request(request, batch_size,
                                                   batch_index)
        kv_cache_retention_config = get_kv_cache_retention_config_from_request(
            request, batch_size, batch_index)
        request_lookahead_config = get_lookahead_decoding_config_from_request(
            request, executor_lookahead_config, batch_size, batch_index)

        # Inputs for mllama support
        encoder_input_features = get_input_tensor_by_name(
            request, 'encoder_input_features', batch_size, batch_index)
        if encoder_input_features is not None:
            if isinstance(encoder_input_features, np.ndarray):
                encoder_input_features = from_numpy(
                    encoder_input_features).squeeze(dim=0)
            elif isinstance(encoder_input_features, torch.Tensor):
                encoder_input_features = encoder_input_features.squeeze(dim=0)
            inputs['encoder_input_features'] = encoder_input_features
            logger.debug(
                f"inputs to llm: encoder_input_features ({encoder_input_features.shape}"
            )

            encoder_output_length = get_input_tensor_by_name(
                request, 'encoder_output_lengths', batch_size, batch_index)
            if encoder_output_length is not None:
                inputs['encoder_output_length'] = np.squeeze(
                    encoder_output_length, axis=0)

            cross_attention_mask = get_input_tensor_by_name(
                request, 'cross_attention_mask', batch_size, batch_index)
            if cross_attention_mask is not None:
                inputs['cross_attention_mask'] = cross_attention_mask[0]
                logger.debug(
                    f"inputs to llm: cross_attention_mask ({ cross_attention_mask.shape})"
                )

            skip_cross_attn_blocks = get_input_tensor_by_name(
                request,
                'skip_cross_attn_blocks',
                batch_size,
                batch_index,
                force_on_torch=True)
            if skip_cross_attn_blocks is not None:
                inputs['skip_cross_attn_blocks'] = skip_cross_attn_blocks[0]
                logger.debug(
                    f"inputs to llm: skip_cross_attn_blocks ({ skip_cross_attn_blocks.shape})"
                )

        guided_decoding_params = get_guided_decoding_params_from_request(
            request, batch_size, batch_index)

        requests.append(
            trtllm.Request(
                **inputs,
                sampling_config=sampling_config,
                output_config=output_config,
                external_draft_tokens_config=external_draft_tokens_config,
                prompt_tuning_config=prompt_tuning_config,
                mrope_config=mrope_config,
                lora_config=lora_config,
                guided_decoding_params=guided_decoding_params,
                lookahead_config=request_lookahead_config,
                kv_cache_retention_config=kv_cache_retention_config))
    return requests


def convert_response(response,
                     batch_index,
                     batch_size,
                     num_return_sequences,
                     expected_logits_dtype=torch.float32,
                     input_token_count=None,
                     return_num_input_tokens=False,
                     return_num_output_tokens=False):

    if response.has_error():
        return pb_utils.InferenceResponse(output_tensors=[],
                                          error=pb_utils.TritonError(
                                              response.error_msg)), True, 0
    result = response.result
    beam_lengths = np.expand_dims(
        np.array([len(beam) for beam in result.output_token_ids], np.int32), 0)
    max_beam_length = max([len(beam) for beam in result.output_token_ids])
    output_ids = np.full((1, len(result.output_token_ids), max_beam_length), -1,
                         np.int32)
    for idx, beam in enumerate(result.output_token_ids):
        output_ids[0, idx, :len(beam)] = beam

    output_lengths = output_ids.size
    output_tensors = [
        pb_utils.Tensor("output_ids", output_ids),
        pb_utils.Tensor("sequence_length", beam_lengths),
    ]

    if result.cum_log_probs is not None:
        output_tensors.append(
            pb_utils.Tensor(
                "cum_log_probs",
                np.expand_dims(np.array(result.cum_log_probs, np.float32), 0)))

    if result.log_probs is not None:
        output_tensors.append(
            pb_utils.Tensor(
                "output_log_probs",
                np.expand_dims(np.array(result.log_probs, np.float32), 0)))

    if result.context_logits is not None:
        assert (result.context_logits.dtype is expected_logits_dtype)
        output_tensors.append(
            pb_utils.Tensor(
                "context_logits",
                np.expand_dims(
                    np.array(
                        result.context_logits,
                        torch_to_numpy_dtype_dict[result.context_logits.dtype]),
                    0)))

    if result.generation_logits is not None:
        assert (result.generation_logits.dtype is expected_logits_dtype)
        output_tensors.append(
            pb_utils.Tensor(
                "generation_logits",
                np.expand_dims(
                    np.array(
                        result.generation_logits, torch_to_numpy_dtype_dict[
                            result.generation_logits.dtype]), 0)))

    if batch_size > 1:
        output_tensors.append(
            pb_utils.Tensor(
                "batch_index",
                np.expand_dims(np.array([batch_index], np.int32), 0)))

    if num_return_sequences > 1:
        output_tensors.append(
            pb_utils.Tensor(
                "sequence_index",
                np.expand_dims(np.array([result.sequence_index], np.int32), 0)))

    # Add token count outputs if requested
    if return_num_input_tokens and input_token_count is not None:
        triton_output_tensor = pb_utils.Tensor(
            "num_input_tokens",
            np.expand_dims(np.array([input_token_count], np.int32), 0))
        output_tensors.append(triton_output_tensor)
    if return_num_output_tokens:
        triton_output_tensor = pb_utils.Tensor(
            "num_output_tokens",
            np.expand_dims(np.array([output_lengths], np.int32), 0))
        output_tensors.append(triton_output_tensor)

    if result.request_perf_metrics is not None:
        kv_cache_metrics = result.request_perf_metrics.kv_cache_metrics
        output_tensors.append(
            pb_utils.Tensor(
                "kv_cache_alloc_new_blocks",
                np.expand_dims(
                    np.array([kv_cache_metrics.num_new_allocated_blocks],
                             np.int32), 0)))
        output_tensors.append(
            pb_utils.Tensor(
                "kv_cache_reused_blocks",
                np.expand_dims(
                    np.array([kv_cache_metrics.num_reused_blocks], np.int32),
                    0)))
        output_tensors.append(
            pb_utils.Tensor(
                "kv_cache_alloc_total_blocks",
                np.expand_dims(
                    np.array([kv_cache_metrics.num_total_allocated_blocks],
                             np.int32), 0)))

        timing_metrics = result.request_perf_metrics.timing_metrics
        output_tensors.append(
            pb_utils.Tensor(
                "arrival_time_ns",
                np.expand_dims(
                    np.array([pd.Timedelta(timing_metrics.arrival_time).value],
                             np.int64), 0)))
        output_tensors.append(
            pb_utils.Tensor(
                "first_scheduled_time_ns",
                np.expand_dims(
                    np.array([
                        pd.Timedelta(timing_metrics.first_scheduled_time).value
                    ], np.int64), 0)))
        output_tensors.append(
            pb_utils.Tensor(
                "first_token_time_ns",
                np.expand_dims(
                    np.array(
                        [pd.Timedelta(timing_metrics.first_token_time).value],
                        np.int64), 0)))
        output_tensors.append(
            pb_utils.Tensor(
                "last_token_time_ns",
                np.expand_dims(
                    np.array(
                        [pd.Timedelta(timing_metrics.last_token_time).value],
                        np.int64), 0)))

        spec_dec_metrics = result.request_perf_metrics.speculative_decoding
        output_tensors.append(
            pb_utils.Tensor(
                "acceptance_rate",
                np.expand_dims(
                    np.array([spec_dec_metrics.acceptance_rate], np.float32),
                    0)))
        output_tensors.append(
            pb_utils.Tensor(
                "total_accepted_draft_tokens",
                np.expand_dims(
                    np.array([spec_dec_metrics.total_accepted_draft_tokens],
                             np.int32), 0)))
        output_tensors.append(
            pb_utils.Tensor(
                "total_draft_tokens",
                np.expand_dims(
                    np.array([spec_dec_metrics.total_draft_tokens], np.int32),
                    0)))

    return pb_utils.InferenceResponse(
        output_tensors), result.is_final, output_lengths


def convert_scheduler_policy(batch_scheduler_policy: str):
    if batch_scheduler_policy.lower() == "max_utilization":
        return trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    elif batch_scheduler_policy.lower() == "guaranteed_no_evict":
        return trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    raise pb_utils.TritonModelException(
        f"batch_scheduler_policy value of '{batch_scheduler_policy}' is not supported."
    )


def convert_batching_type(gpt_model_type: str):
    if gpt_model_type is None:
        return None
    if gpt_model_type.lower(
    ) == "inflight_fused_batching" or gpt_model_type.lower(
    ) == "inflight_batching":
        return trtllm.BatchingType.INFLIGHT
    elif gpt_model_type.lower() == "v1":
        return trtllm.BatchingType.STATIC
    raise pb_utils.TritonModelException(
        f"gpt_model_type value of '{gpt_model_type}' is not supported.")


def convert_decoding_mode(decoding_mode: str):
    if decoding_mode is None:
        return None
    elif decoding_mode == "auto":
        return trtllm.DecodingMode.Auto()
    elif decoding_mode == "top_k":
        return trtllm.DecodingMode.TopK()
    elif decoding_mode == "top_p":
        return trtllm.DecodingMode.TopP()
    elif decoding_mode == "top_k_top_p":
        return trtllm.DecodingMode.TopKTopP()
    elif decoding_mode == "beam_search":
        return trtllm.DecodingMode.BeamSearch()
    elif decoding_mode == "medusa":
        return trtllm.DecodingMode.Medusa()
    elif decoding_mode == "redrafter":
        return trtllm.DecodingMode.ExplicitDraftTokens()
    elif decoding_mode == "lookahead":
        return trtllm.DecodingMode.Lookahead()
    elif decoding_mode == "eagle":
        return trtllm.DecodingMode.Eagle()
    raise pb_utils.TritonModelException(
        f"decoding_mode value of '{decoding_mode}' is not supported.")


def convert_timestamp_to_seconds(timestamp: str):
    return int(
        datetime.datetime.strptime(timestamp,
                                   "%m-%d-%Y %H:%M:%S.%f").timestamp())


def triton_string_to_torch(dtype):
    type_map = {
        "TYPE_BOOL": torch.bool,
        "TYPE_UINT8": torch.uint8,
        "TYPE_INT8": torch.int8,
        "TYPE_INT16": torch.int16,
        "TYPE_INT32": torch.int32,
        "TYPE_INT64": torch.int64,
        "TYPE_FP16": torch.float16,
        "TYPE_FP32": torch.float32,
        "TYPE_FP64": torch.float64,
        "TYPE_BF16": torch.bfloat16
    }
    return type_map[dtype]


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def get_scheduler_config(self, model_config):
        batch_scheduler_policy = get_parameter(model_config,
                                               "batch_scheduler_policy")
        if batch_scheduler_policy is None:
            return trtllm.SchedulerConfig()
        return trtllm.SchedulerConfig(
            convert_scheduler_policy(batch_scheduler_policy))

    def get_kv_cache_config(self, model_config):
        kwargs = {
            "enable_block_reuse":
            get_parameter(model_config, "enable_kv_cache_reuse", bool),
            "max_tokens":
            get_parameter(model_config, "max_tokens_in_paged_kv_cache", int),
            "sink_token_length":
            get_parameter(model_config, "sink_token_length", int),
            "free_gpu_memory_fraction":
            get_parameter(model_config, "kv_cache_free_gpu_mem_fraction",
                          float),
            "cross_kv_cache_fraction":
            get_parameter(model_config, "cross_kv_cache_fraction", float),
            "host_cache_size":
            get_parameter(model_config, "kv_cache_host_memory_bytes", int),
            "onboard_blocks":
            get_parameter(model_config, "kv_cache_onboard_blocks", bool),
        }
        max_attention_window_size = get_parameter(model_config,
                                                  "max_attention_window_size")
        if max_attention_window_size:
            kwargs["max_attention_window"] = [
                int(x) for x in max_attention_window_size.split(",")
            ]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.KvCacheConfig(**kwargs)

    def get_parallel_config(self, model_config):
        kwargs = {}
        gpu_device_ids = get_parameter(model_config, "gpu_device_ids")
        if gpu_device_ids:
            kwargs["device_ids"] = [int(x) for x in gpu_device_ids.split(",")]
        self.use_orchestrator_mode = os.environ.get("TRTLLM_ORCHESTRATOR",
                                                    "0") == "1"
        if self.use_orchestrator_mode:
            kwargs["communication_mode"] = trtllm.CommunicationMode.ORCHESTRATOR
            worker_path = get_parameter(model_config, "worker_path")
            spawn_processes = os.environ.get(
                "TRTLLM_ORCHESTRATOR_SPAWN_PROCESSES", "1") == "1"
            if not spawn_processes:
                raise pb_utils.TritonModelException(
                    "Orchestrator mode with --disable-spawn-processes is not supported in the Python backend."
                )
            is_orchestrator = (mpi_rank() == 0) if spawn_processes else True
            if worker_path is not None:
                raise pb_utils.TritonModelException(
                    "worker_path parameter is specified, but this is no longer supported. Please specify executor_worker_path instead to specify the location of the trtllmExecutorWorker executable."
                )
            executor_worker_path = get_parameter(model_config,
                                                 "executor_worker_path")
            kwargs["orchestrator_config"] = trtllm.OrchestratorConfig(
                is_orchestrator, executor_worker_path)
        if len(kwargs) > 0:
            return trtllm.ParallelConfig(**kwargs)
        return None

    def get_peft_cache_config(self, model_config):
        kwargs = {
            "optimal_adapter_size":
            get_parameter(model_config, "lora_cache_optimal_adapter_size", int),
            "max_adapter_size":
            get_parameter(model_config, "lora_cache_max_adapter_size", int),
            "device_cache_percent":
            get_parameter(model_config, "lora_cache_gpu_memory_fraction",
                          float),
            "host_cache_size":
            get_parameter(model_config, "lora_cache_host_memory_bytes", int),
            "lora_prefetch_dir":
            get_parameter(model_config, "lora_prefetch_dir", int),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.PeftCacheConfig(**kwargs)

    def get_executor_lookahead_config(self, model_config):
        lookahead_window_size = get_parameter(model_config,
                                              "lookahead_window_size", int)
        lookahead_ngram_size = get_parameter(model_config,
                                             "lookahead_ngram_size", int)
        lookahead_verification_set_size = get_parameter(
            model_config, "lookahead_verification_set_size", int)
        # executor_lookahead_config is not set
        if all(item is None for item in [
                lookahead_window_size, lookahead_ngram_size,
                lookahead_verification_set_size
        ]):
            return None

        incomplete_config = None in [
            lookahead_window_size, lookahead_ngram_size,
            lookahead_verification_set_size
        ]

        assert (
            not incomplete_config
        ), "Please set executor_lookahead_window_size, executor_lookahead_ngram_size and executor_lookahead_verification_set_size together."

        return trtllm.LookaheadDecodingConfig(lookahead_window_size,
                                              lookahead_ngram_size,
                                              lookahead_verification_set_size)

    def get_decoding_config(self, model_config):

        decoding_mode = convert_decoding_mode(
            get_parameter(model_config, "decoding_mode"))
        self.executor_lookahead_config = None
        if decoding_mode == trtllm.DecodingMode.Lookahead():
            # Add LAD config
            self.executor_lookahead_config = self.get_executor_lookahead_config(
                model_config)
        eagle_choices = parse_eagle_choices(
            get_parameter(model_config, "eagle_choices"))
        kwargs = {
            "medusa_choices":
            parse_medusa_choices(get_parameter(model_config, "medusa_choices")),
            "eagle_config":
            None
            if eagle_choices is None else trtllm.EagleConfig(eagle_choices),
            "lookahead_decoding_config":
            self.executor_lookahead_config,
            "decoding_mode":
            decoding_mode,
        }
        print(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.DecodingConfig(**kwargs)

    def get_extended_runtime_perf_knob_config(self, model_config):
        kwargs = {
            "multi_block_mode":
            get_parameter(model_config, "multi_block_mode", bool),
            "enable_context_fmha_fp32_acc":
            get_parameter(model_config, "enable_context_fmha_fp32_acc", bool),
            "cuda_graph_mode":
            get_parameter(model_config, "cuda_graph_mode", bool),
            "cuda_graph_cache_size":
            get_parameter(model_config, "cuda_graph_cache_size", int),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExtendedRuntimePerfKnobConfig(**kwargs)

    def get_guided_decoding_config(self, model_config):

        guided_decoding_backend = get_parameter(model_config,
                                                "guided_decoding_backend", str)

        tokenizer_dir = get_parameter(model_config, "tokenizer_dir", str)
        if guided_decoding_backend not in ['xgrammar']:
            if tokenizer_dir:
                pb_utils.Logger.log_warn(
                    f"Guided decoding backend has not been set but tokenizer_dir is given. Tokenizer_dir will be ignored."
                )
            return None

        if guided_decoding_backend == 'xgrammar':
            guided_decoding_backend = trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR

        if not tokenizer_dir:
            raise ValueError(
                "Guided decoding requires tokenizer's information. Please provide 'tokenizer_dir'."
            )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        pb_utils.Logger.log_info(
            f"Guided decoding has been set with {guided_decoding_backend} backend"
        )
        return trtllm.GuidedDecodingConfig(
            backend=guided_decoding_backend,
            **_xgrammar_tokenizer_info(tokenizer))

    def get_executor_config(self, model_config):
        kwargs = {
            "max_beam_width":
            get_parameter(model_config, "max_beam_width", int),
            "scheduler_config":
            self.get_scheduler_config(model_config),
            "kv_cache_config":
            self.get_kv_cache_config(model_config),
            "enable_chunked_context":
            get_parameter(model_config, "enable_chunked_context", bool),
            "normalize_log_probs":
            get_parameter(model_config, "normalize_log_probs", bool),
            "batching_type":
            convert_batching_type(get_parameter(model_config,
                                                "gpt_model_type")),
            "parallel_config":
            self.get_parallel_config(model_config),
            "peft_cache_config":
            self.get_peft_cache_config(model_config),
            "decoding_config":
            self.get_decoding_config(model_config),
            "max_queue_size":
            model_config.get(
                "dynamic_batching",
                {},
            ).get(
                "default_queue_policy",
                {},
            ).get("max_queue_size"),
            "extended_runtime_perf_knob_config":
            self.get_extended_runtime_perf_knob_config(model_config),
            "guided_decoding_config":
            self.get_guided_decoding_config(model_config)
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)

    def create_metrics(self, model: str, version: str, is_v1_model: bool):
        self.request_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_request_metrics",
            description="TRT LLM request metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.runtime_memory_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_runtime_memory_metrics",
            description="TRT LLM runtime memory metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.kv_cache_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_kv_cache_block_metrics",
            description="TRT LLM KV cache block metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        model_type = "v1" if is_v1_model else "inflight_batcher"
        self.model_type_metric_family = pb_utils.MetricFamily(
            name=f"nv_trt_llm_{model_type}_metrics",
            description=f"TRT LLM {model_type}-specific metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.general_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_general_metrics",
            description="General TRT LLM metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        # Set the metric using self.general_metric_output_family.observe(string_size)
        self.request_tokens_metric_family = pb_utils.MetricFamily(
            name="nv_llm_input_token_len",
            description="TRT LLM response metrics",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        self.response_tokens_metric_family = pb_utils.MetricFamily(
            name="nv_llm_output_token_len",
            description="TRT LLM response metrics",
            kind=pb_utils.MetricFamily.HISTOGRAM,
        )
        common_labels = {"model": model, "version": version}
        self.all_metrics = {
            # Request metrics
            "num_active_requests":
            self.request_metric_family.Metric(labels={
                "request_type": "active",
                **common_labels
            }),
            "max_num_active_requests":
            self.request_metric_family.Metric(labels={
                "request_type": "max",
                **common_labels
            }),
            "num_scheduled_requests":
            self.request_metric_family.Metric(labels={
                "request_type": "scheduled",
                **common_labels
            }),
            "num_context_requests":
            self.request_metric_family.Metric(labels={
                "request_type": "context",
                **common_labels
            }),
            "num_waiting_requests":
            self.request_metric_family.Metric(labels={
                "request_type": "waiting",
                **common_labels
            }),
            # Runtime metrics
            "cpu_mem_usage":
            self.runtime_memory_metric_family.Metric(labels={
                "memory_type": "cpu",
                **common_labels
            }),
            "gpu_mem_usage":
            self.runtime_memory_metric_family.Metric(labels={
                "memory_type": "gpu",
                **common_labels
            }),
            "pinned_mem_usage":
            self.runtime_memory_metric_family.Metric(labels={
                "memory_type": "pinned",
                **common_labels
            }),
            # KV cache metrics
            "max_num_blocks":
            self.kv_cache_metric_family.Metric(labels={
                "kv_cache_block_type": "max",
                **common_labels
            }),
            "free_num_blocks":
            self.kv_cache_metric_family.Metric(labels={
                "kv_cache_block_type": "free",
                **common_labels
            }),
            "used_num_blocks":
            self.kv_cache_metric_family.Metric(labels={
                "kv_cache_block_type": "used",
                **common_labels
            }),
            "tokens_per_block":
            self.kv_cache_metric_family.Metric(labels={
                "kv_cache_block_type": "tokens_per",
                **common_labels
            }),
            "fraction_used_blocks":
            self.kv_cache_metric_family.Metric(labels={
                "kv_cache_block_type": "fraction",
                **common_labels
            }),
            # General metrics
            "timestamp":
            self.general_metric_family.Metric(labels={
                "general_type": "timestamp",
                **common_labels
            }),
            "iter":
            self.general_metric_family.Metric(labels={
                "general_type": "iteration_counter",
                **common_labels
            }),
            METRIC_TOTAL_OUTPUT_TOKENS:
            self.response_tokens_metric_family.Metric(
                labels={
                    "response_metric_type": METRIC_TOTAL_OUTPUT_TOKENS,
                    **common_labels
                },
                buckets=build_1_2_5_buckets(1000)),
            METRIC_TOTAL_INPUT_TOKENS:
            self.request_tokens_metric_family.Metric(
                labels={
                    "response_metric_type": METRIC_TOTAL_INPUT_TOKENS,
                    **common_labels
                },
                buckets=build_1_2_5_buckets(1000)),
        }
        if is_v1_model:
            self.all_metrics.update({
                "num_ctx_tokens":
                self.model_type_metric_family.Metric(labels={
                    "v1_specific_metric": "total_context_tokens",
                    **common_labels
                }),
                "num_gen_tokens":
                self.model_type_metric_family.Metric(
                    labels={
                        "v1_specific_metric": "total_generation_tokens",
                        **common_labels
                    }),
                "empty_gen_slots":
                self.model_type_metric_family.Metric(labels={
                    "v1_specific_metric": "empty_generation_slots",
                    **common_labels
                }),
            })
        else:
            self.all_metrics.update({
                "num_ctx_tokens":
                self.model_type_metric_family.Metric(
                    labels={
                        "inflight_batcher_specific_metric":
                        "total_context_tokens",
                        **common_labels
                    }),
                "num_gen_requests":
                self.model_type_metric_family.Metric(
                    labels={
                        "inflight_batcher_specific_metric":
                        "generation_requests",
                        **common_labels
                    }),
                "micro_batch_id":
                self.model_type_metric_family.Metric(
                    labels={
                        "inflight_batcher_specific_metric": "micro_batch_id",
                        **common_labels
                    }),
                "num_paused_requests":
                self.model_type_metric_family.Metric(
                    labels={
                        "inflight_batcher_specific_metric": "paused_requests",
                        **common_labels
                    }),
            })

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_config = json.loads(args['model_config'])
        gpt_model_path = get_parameter(model_config, "gpt_model_path")
        if get_parameter(model_config, "enable_trt_overlap", bool):
            raise pb_utils.TritonModelException(
                f"enable_trt_overlap=true is not supported.")
        self.exclude_input_from_output = get_parameter(
            model_config, "exclude_input_in_output", bool)
        executor_config = self.get_executor_config(model_config)
        self.executor = trtllm.Executor(gpt_model_path,
                                        trtllm.ModelType.DECODER_ONLY,
                                        executor_config)
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)
        self.cancellation_check_period_ms = get_parameter(
            model_config, "cancellation_check_period_ms", int) or 100
        self.stats_check_period_ms = get_parameter(
            model_config, "stats_check_period_ms", int) or 100

        self.logits_dtype = None
        for output in model_config['output']:
            if output['name'] == 'context_logits' or output[
                    'name'] == 'generation_logits':
                self.logits_dtype = triton_string_to_torch(output['data_type'])

        self.create_metrics(args["model_name"],
                            args["model_version"],
                            is_v1_model=executor_config.batching_type ==
                            trtllm.BatchingType.STATIC)
        self.triton_user_id_to_req_ids = {}
        self.triton_req_id_to_req_ids = {}
        self.req_id_to_request_data = {}
        self.lock = Lock()
        self.running = False
        self.awaiter_thread = Thread(target=self.awaiter_loop)
        self.cancellation_thread = Thread(target=self.cancellation_loop)
        self.metrics_thread = Thread(target=self.metrics_loop)
        if self.executor.can_enqueue_requests():
            self.running = True
            self.awaiter_thread.start()
            self.cancellation_thread.start()
            self.metrics_thread.start()
        else:
            # In leader mode, worker ranks will wait here until leader is done.
            self.executor.shutdown()

    def handle_stop_request(self, triton_user_id, response_sender):
        if triton_user_id is None or triton_user_id == "":
            response_sender.send(
                pb_utils.InferenceResponse(error=pb_utils.TritonError(
                    "A request id must be provided for request cancellation")),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            return

        with self.lock:
            if triton_user_id in self.triton_user_id_to_req_ids:
                req_ids = self.triton_user_id_to_req_ids[triton_user_id]
                for req_id in req_ids:
                    self.executor.cancel_request(req_id)

        response_sender.send(
            pb_utils.InferenceResponse(),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        if not self.executor.can_enqueue_requests():
            return

        # Convert to executor requests.

        triton_requests = []
        executor_requests = []
        batch_indices = []
        triton_user_ids = []
        triton_req_ids = []

        for request in requests:

            triton_user_id = request.request_id()

            response_sender = request.get_response_sender()
            stop = get_input_scalar_by_name(request, 'stop')

            if stop:
                self.handle_stop_request(triton_user_id, response_sender)
            else:
                #Unique request id used to identify each triton request
                triton_req_id = str(randint(0, sys.maxsize))
                self.triton_req_id_to_req_ids[triton_req_id] = set()
                if triton_user_id is not None and triton_user_id != "":
                    self.triton_user_id_to_req_ids[triton_user_id] = set()

                try:
                    converted_reqs = convert_request(
                        request, self.exclude_input_from_output, self.decoupled,
                        self.executor_lookahead_config)
                except Exception as e:
                    response_sender.send(
                        pb_utils.InferenceResponse(error=pb_utils.TritonError(
                            f"An error occurred when processing the input values for request id {request.request_id()}, the error was '{e}'"
                        )),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    for batch_index, converted_req in enumerate(converted_reqs):
                        triton_requests.append(request)
                        executor_requests.append(converted_req)
                        triton_user_ids.append(triton_user_id)
                        triton_req_ids.append(triton_req_id)
                        batch_indices.append(batch_index)

        with self.lock:
            request_ids = self.executor.enqueue_requests(executor_requests)
            for req_id, triton_req_id, triton_user_id, executor_request, triton_request, batch_index in zip(
                    request_ids, triton_req_ids, triton_user_ids,
                    executor_requests, triton_requests, batch_indices):

                self.req_id_to_request_data[req_id] = RequestData(
                    triton_req_id, triton_user_id, batch_index,
                    len(batch_indices),
                    executor_request.sampling_config.num_return_sequences, 0, 0,
                    triton_request.get_response_sender(),
                    get_input_scalar_by_name(triton_request,
                                             'return_num_input_tokens',
                                             batch_index=batch_index),
                    get_input_scalar_by_name(triton_request,
                                             'return_num_output_tokens',
                                             batch_index=batch_index))
                self.triton_req_id_to_req_ids[triton_req_id].add(req_id)
                input_len = len(
                    executor_request.input_token_ids
                ) if executor_request.input_token_ids is not None else 0
                self.req_id_to_request_data[
                    req_id].num_input_tokens += input_len
                # This checks both request level and instance config level
                if executor_request.output_config.exclude_input_from_output == False and executor_request.streaming == False:
                    self.req_id_to_request_data[
                        req_id].num_output_tokens -= self.req_id_to_request_data[
                            req_id].num_input_tokens * executor_request.sampling_config.beam_width
                if triton_user_id is not None and triton_user_id != "":
                    self.triton_user_id_to_req_ids[triton_user_id].add(req_id)

        return None

    def awaiter_loop(self):
        """Gets responses from executor and returns the results."""
        while self.running:
            for response in self.executor.await_responses(
                    timeout=datetime.timedelta(milliseconds=1)):
                req_id = response.request_id
                request_data = None
                with self.lock:
                    if req_id not in self.req_id_to_request_data:
                        continue
                    request_data = self.req_id_to_request_data[req_id]

                triton_response, is_final, output_length = convert_response(
                    response, request_data.batch_index, request_data.batch_size,
                    request_data.num_return_sequences, self.logits_dtype,
                    request_data.num_input_tokens,
                    request_data.return_num_input_tokens,
                    request_data.return_num_output_tokens)
                with self.lock:
                    self.req_id_to_request_data[
                        req_id].num_output_tokens += output_length
                triton_request_final = False
                if is_final:
                    with self.lock:
                        # Check if all executor requests part of that triton request are finished
                        self.triton_req_id_to_req_ids[
                            request_data.triton_req_id].remove(req_id)
                        if len(self.triton_req_id_to_req_ids[
                                request_data.triton_req_id]) == 0:
                            pb_utils.Logger.log_info(
                                f"DELETING Req id {req_id}, triton_req_id {request_data.triton_req_id} "
                            )
                            triton_request_final = True
                            del self.triton_req_id_to_req_ids[
                                request_data.triton_req_id]
                            if request_data.triton_user_id is not None and request_data.triton_user_id != "":
                                del self.triton_user_id_to_req_ids[
                                    request_data.triton_user_id]
                        self.update_metrics_per_request(req_id)
                        del self.req_id_to_request_data[req_id]

                request_data.response_sender.send(
                    triton_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    if triton_request_final else 0)

    def cancellation_loop(self):
        """Checks if any pending requests have been cancelled."""
        while self.running:
            time.sleep(self.cancellation_check_period_ms / 1000.0)
            with self.lock:
                for req_id, request_data in self.req_id_to_request_data.items():
                    if request_data.response_sender.is_cancelled():
                        self.executor.cancel_request(req_id)

    def update_metrics_per_request(self, req_id):
        """Updates triton metrics after completing one request"""
        output_tokens = self.req_id_to_request_data[req_id].num_output_tokens
        input_tokens = self.req_id_to_request_data[req_id].num_input_tokens

        self.all_metrics[METRIC_TOTAL_OUTPUT_TOKENS].observe(output_tokens)
        self.all_metrics[METRIC_TOTAL_INPUT_TOKENS].observe(input_tokens)

    def get_composite_metric_map(self, stat):

        def get_metric(metric_name, family_stats=None):
            if family_stats is None:
                if hasattr(stat, metric_name):
                    return getattr(stat, metric_name)
                elif stat.kv_cache_stats is not None and hasattr(
                        stat.kv_cache_stats, metric_name):
                    return getattr(stat.kv_cache_stats, metric_name)
                elif stat.static_batching_stats is not None and hasattr(
                        stat.static_batching_stats, metric_name):
                    return getattr(stat.static_batching_stats, metric_name)
                elif stat.inflight_batching_stats is not None and hasattr(
                        stat.inflight_batching_stats, metric_name):
                    return getattr(stat.inflight_batching_stats, metric_name)
            elif family_stats is not None and hasattr(family_stats,
                                                      metric_name):
                return getattr(family_stats, metric_name)
            pb_utils.Logger.log_warn(
                f"Constituent metric \"{metric_name}\" not found.")
            return None

        composite_metrics = {}

        # compute fraction_used_blocks
        max_blocks = get_metric("max_num_blocks", stat.kv_cache_stats)
        used_blocks = get_metric("used_num_blocks", stat.kv_cache_stats)
        if max_blocks is not None and used_blocks is not None:
            composite_metrics[
                "fraction_used_blocks"] = 0.0 if max_blocks <= 0 else used_blocks / max_blocks
        else:
            pb_utils.Logger.log_warn(
                f"fraction_used_blocks is missing one or more constituent metric."
            )

        # compute num_waiting_requests
        active_requests = get_metric("num_active_requests")
        scheduled_requests = get_metric("num_scheduled_requests")
        if active_requests is not None and scheduled_requests is not None:
            composite_metrics[
                "num_waiting_requests"] = active_requests - scheduled_requests
        else:
            pb_utils.Logger.log_warn(
                f"num_waiting_requests is missing one or more constituent metric."
            )

        return composite_metrics

    def metrics_loop(self):
        """Updates triton metrics using stats from the executor."""
        while self.running:
            time.sleep(self.stats_check_period_ms / 1000.0)
            for stat in self.executor.get_latest_iteration_stats():
                try:
                    composite_metrics = self.get_composite_metric_map(stat)
                    for key, metric in self.all_metrics.items():
                        # Skip processing for both histogram metrics
                        if isinstance(key, str) and key in [
                                METRIC_TOTAL_OUTPUT_TOKENS,
                                METRIC_TOTAL_INPUT_TOKENS
                        ]:
                            continue
                        value = None
                        if hasattr(stat, key):
                            value = getattr(stat, key)
                        elif stat.kv_cache_stats is not None and hasattr(
                                stat.kv_cache_stats, key):
                            value = getattr(stat.kv_cache_stats, key)
                        elif stat.static_batching_stats is not None and hasattr(
                                stat.static_batching_stats, key):
                            value = getattr(stat.static_batching_stats, key)
                        elif stat.inflight_batching_stats is not None and hasattr(
                                stat.inflight_batching_stats, key):
                            value = getattr(stat.inflight_batching_stats, key)
                        elif key in composite_metrics:
                            value = composite_metrics[key]
                        if value is not None:
                            if key == "timestamp":
                                value = convert_timestamp_to_seconds(value)
                            metric.set(value)
                        else:
                            pb_utils.Logger.log_warn(
                                f"Metric \"{key}\" not found.")
                except Exception as e:
                    pb_utils.Logger.log_warn(
                        f"Error while processing metrics: {e}")

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        if self.executor.can_enqueue_requests():
            self.running = False
            self.awaiter_thread.join()
            self.cancellation_thread.join()
            self.metrics_thread.join()
            self.executor.shutdown()
