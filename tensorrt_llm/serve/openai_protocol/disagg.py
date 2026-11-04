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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

import base64
from typing import List, Optional, Union

import torch

from tensorrt_llm.llmapi import ConversationParams as LlmConversationParams
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.serve.openai_protocol.chat import (ChatCompletionRequest,
                                                     ChatCompletionResponse)
from tensorrt_llm.serve.openai_protocol.common import (ConversationParams,
                                                       DisaggregatedParams,
                                                       UsageInfo)
from tensorrt_llm.serve.openai_protocol.completions import (CompletionRequest,
                                                            CompletionResponse)


def encode_opaque_state(opaque_state: Optional[bytes]) -> Optional[str]:
    if opaque_state is None:
        return None
    return base64.b64encode(opaque_state).decode("utf-8")


def decode_opaque_state(encoded_opaque_state: Optional[str]) -> Optional[bytes]:
    if encoded_opaque_state is None:
        return None
    return base64.b64decode(encoded_opaque_state)


def _serialize_first_gen_log_probs(
        first_gen_log_probs: Optional[list]) -> Optional[List]:
    """Serialize ``list[dict[int, Logprob]] | list[float]`` to a JSON-safe form.

    - Default (verbose) format: each position is a ``dict[int, Logprob]`` and is
      serialized as a list of ``{token_id, logprob, rank}`` dicts.
    - Simple format: each position is a ``float``; passed through verbatim.
    """
    if first_gen_log_probs is None:
        return None
    if not isinstance(first_gen_log_probs, list):
        raise ValueError("first_gen_log_probs must be a list")
    result = []
    for i, pos in enumerate(first_gen_log_probs):
        if isinstance(pos, dict):
            result.append([{
                "token_id": tid,
                "logprob": lp.logprob,
                "rank": lp.rank
            } for tid, lp in pos.items()])
        elif isinstance(pos, (float, int)):
            # Simple format: per-token sampled logprob.
            result.append(float(pos))
        else:
            raise ValueError(
                f"first_gen_log_probs[{i}] must be a dict or float, got {type(pos)}"
            )
    return result


def _deserialize_first_gen_log_probs(
    serialized: Optional[List], ) -> Optional[list]:
    """Inverse of :func:`_serialize_first_gen_log_probs`.

    Returns either ``list[dict[int, Logprob]]`` (default format) or
    ``list[float]`` (simple format) depending on the serialized payload.
    """
    if serialized is None:
        return None
    from tensorrt_llm.executor.result import Logprob
    result = []
    for i, pos in enumerate(serialized):
        if isinstance(pos, (float, int)):
            result.append(float(pos))
            continue
        if not isinstance(pos, list):
            raise ValueError(
                f"first_gen_log_probs[{i}] must be a list or float, got {type(pos)}"
            )
        token_map = {}
        for j, item in enumerate(pos):
            if not isinstance(item, dict):
                raise ValueError(
                    f"first_gen_log_probs[{i}][{j}] must be a dict")
            if "token_id" not in item or "logprob" not in item:
                raise ValueError(
                    f"first_gen_log_probs[{i}][{j}] missing required keys "
                    "'token_id' and/or 'logprob'")
            token_map[item["token_id"]] = Logprob(logprob=item["logprob"],
                                                  rank=item.get("rank"))
        result.append(token_map)
    return result


def _serialize_first_gen_logits(
    first_gen_logits: Optional[list], ) -> Optional[List]:
    """Serialize list[torch.Tensor] to JSON-safe list[dict] with base64 data."""
    if first_gen_logits is None:
        return None
    result = []
    for i, tensor in enumerate(first_gen_logits):
        t = tensor.contiguous().cpu()
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float16)
        np_array = t.numpy()
        result.append({
            "data": base64.b64encode(np_array.tobytes()).decode(),
            "shape": list(np_array.shape),
            "dtype": str(np_array.dtype),
        })
    return result


def _deserialize_first_gen_logits(
    serialized: Optional[List], ) -> Optional[list]:
    """Deserialize JSON list[dict] back to list[torch.Tensor]."""
    if serialized is None:
        return None
    import numpy as np
    result = []
    for i, item in enumerate(serialized):
        if not isinstance(item, dict):
            raise ValueError(
                f"first_gen_logits[{i}] must be a dict, got {type(item)}")
        for key in ("data", "shape", "dtype"):
            if key not in item:
                raise ValueError(
                    f"first_gen_logits[{i}] missing required key '{key}'")
        np_array = np.frombuffer(
            base64.b64decode(item["data"]),
            dtype=np.dtype(item["dtype"]),
        ).reshape(item["shape"])
        result.append(torch.from_numpy(np_array.copy()))
    return result


def to_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams) -> DisaggregatedParams:
    if tllm_disagg_params is None:
        return None
    ctx_usage = tllm_disagg_params.ctx_usage
    if ctx_usage is not None and not isinstance(ctx_usage, UsageInfo):
        ctx_usage = UsageInfo.model_validate(ctx_usage)
    return DisaggregatedParams(
        request_type=tllm_disagg_params.request_type,
        first_gen_tokens=tllm_disagg_params.first_gen_tokens,
        first_gen_log_probs=_serialize_first_gen_log_probs(
            tllm_disagg_params.first_gen_log_probs),
        first_gen_logits=_serialize_first_gen_logits(
            tllm_disagg_params.first_gen_logits),
        ctx_request_id=tllm_disagg_params.ctx_request_id,
        encoded_opaque_state=encode_opaque_state(
            tllm_disagg_params.opaque_state),
        draft_tokens=tllm_disagg_params.draft_tokens,
        disagg_request_id=tllm_disagg_params.disagg_request_id,
        ctx_dp_rank=tllm_disagg_params.ctx_dp_rank,
        ctx_info_endpoint=tllm_disagg_params.ctx_info_endpoint,
        schedule_style=tllm_disagg_params.schedule_style,
        ctx_usage=ctx_usage,
        remote_tail_start=tllm_disagg_params.remote_tail_start,
        multimodal_positions=tllm_disagg_params.multimodal_positions,
        multimodal_lengths=tllm_disagg_params.multimodal_lengths,
    )


def to_llm_disaggregated_params(
        disaggregated_params: DisaggregatedParams) -> LlmDisaggregatedParams:
    if disaggregated_params is None:
        return None
    ctx_usage = disaggregated_params.ctx_usage
    return LlmDisaggregatedParams(
        request_type=disaggregated_params.request_type,
        first_gen_tokens=disaggregated_params.first_gen_tokens,
        first_gen_log_probs=_deserialize_first_gen_log_probs(
            disaggregated_params.first_gen_log_probs),
        first_gen_logits=_deserialize_first_gen_logits(
            disaggregated_params.first_gen_logits),
        ctx_request_id=disaggregated_params.ctx_request_id,
        opaque_state=decode_opaque_state(
            disaggregated_params.encoded_opaque_state),
        draft_tokens=disaggregated_params.draft_tokens,
        disagg_request_id=disaggregated_params.disagg_request_id,
        ctx_dp_rank=disaggregated_params.ctx_dp_rank,
        ctx_info_endpoint=disaggregated_params.ctx_info_endpoint,
        schedule_style=disaggregated_params.schedule_style,
        ctx_usage=None if ctx_usage is None else ctx_usage.model_dump(),
        remote_tail_start=disaggregated_params.remote_tail_start,
        multimodal_positions=disaggregated_params.multimodal_positions,
        multimodal_lengths=disaggregated_params.multimodal_lengths,
        orig_elapsed_s=disaggregated_params.orig_elapsed_s,
    )


def to_llm_conversation_params(
    conversation_params: Optional[ConversationParams]
) -> Optional[LlmConversationParams]:
    if conversation_params is None:
        return None
    return LlmConversationParams(
        conversation_id=conversation_params.conversation_id)


UCompletionRequest = Union[CompletionRequest, ChatCompletionRequest]

UCompletionResponse = Union[CompletionResponse, ChatCompletionResponse]
