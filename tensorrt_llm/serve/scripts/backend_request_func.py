# Adopted from
# https://github.com/vllm-project/vllm/blob/200bbf92e8861e2458a6f90bca73f40cc3b1ad1f/benchmarks/backend_request_func.py
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
from tqdm.asyncio import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    ignore_eos: bool = False
    language: Optional[str] = None
    multi_modal_content: Optional[dict] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    avg_decoded_tokens_per_iter: float = 0.0  # Average tokens decoded per iteration
    exception_type: str = None  # unset


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    streaming: bool = True,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    if not api_url.endswith("generate_stream"):
        raise ValueError(
            f"TRT-LLM API URL must end with 'generate_stream', but got: {api_url}"
        )

    request_session = aiohttp.ClientSession(
        trust_env=True,
        timeout=AIOHTTP_TIMEOUT,
        connector=aiohttp.TCPConnector(
            limit=0, limit_per_host=0)) if session is None else session

    payload = {
        "accumulate_tokens": True,
        "text_input": request_func_input.prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": request_func_input.output_len,
        "stream": streaming,
    }
    if request_func_input.ignore_eos:
        payload["min_length"] = request_func_input.output_len
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with request_session.post(url=api_url, json=payload) as response:
            if response.status == 200:
                output.success = True
                if streaming:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                        # Extract avg_decoded_tokens_per_iter from TensorRT-LLM response
                        if "avg_decoded_tokens_per_iter" in data:
                            output.avg_decoded_tokens_per_iter = data[
                                "avg_decoded_tokens_per_iter"]

                    output.latency = most_recent_timestamp - st

                else:
                    content = await response.content.read()
                    data = json.loads(content.decode())
                    output.ttft = -1
                    output.itl = []
                    output.generated_text = data["text_output"]
                    output.latency = time.perf_counter() - st

                    # Extract avg_decoded_tokens_per_iter from non-streaming TensorRT-LLM response
                    if "avg_decoded_tokens_per_iter" in data:
                        output.avg_decoded_tokens_per_iter = data[
                            "avg_decoded_tokens_per_iter"]

            else:
                output.error = response.reason or ""
                output.success = False
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__
    finally:
        if session is None:
            await request_session.close()

    if pbar:
        pbar.update(1)

    return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    streaming: bool = True,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    if not api_url.endswith(("completions", "profile")):
        raise ValueError(
            "OpenAI Completions API URL must end with 'completions' or 'profile'."
        )

    request_session = aiohttp.ClientSession(
        trust_env=True,
        timeout=AIOHTTP_TIMEOUT,
        connector=aiohttp.TCPConnector(
            limit=0, limit_per_host=0)) if session is None else session

    payload = {
        "model": request_func_input.model_name \
            if request_func_input.model_name else request_func_input.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": request_func_input.output_len,
        "logprobs": request_func_input.logprobs,
        "stream": streaming,
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with request_session.post(url=api_url,
                                        json=payload,
                                        headers=headers) as response:
            if response.status == 200:
                if streaming:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""

                                # Extract avg_decoded_tokens_per_iter from streaming response
                                if "avg_decoded_tokens_per_iter" in choices[0]:
                                    output.avg_decoded_tokens_per_iter = choices[
                                        0]["avg_decoded_tokens_per_iter"]
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    content = await response.content.read()
                    data = json.loads(content.decode())
                    generated_text = data["choices"][0]["text"]
                    output.success = True
                    output.generated_text = generated_text
                    output.latency = time.perf_counter() - st
                    output.ttft = -1
                    output.itl = []
                    output.output_tokens = data["usage"]["completion_tokens"]
                    # Extract avg_decoded_tokens_per_iter if available
                    choice = data["choices"][0]
                    if "avg_decoded_tokens_per_iter" in choice:
                        output.avg_decoded_tokens_per_iter = choice[
                            "avg_decoded_tokens_per_iter"]
            else:
                print(f"HTTP Error {response.status}: {response}")
                output.error = response.reason or ""
                output.success = False
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__
    finally:
        if session is None:
            await request_session.close()

    if pbar:
        pbar.update(1)

    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    streaming: bool = True,
    pbar: Optional[tqdm] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    if not api_url.endswith(("chat/completions", "profile")):
        raise ValueError(
            "OpenAI Chat Completions API URL must end with 'chat/completions'.")

    request_session = aiohttp.ClientSession(
        trust_env=True,
        timeout=AIOHTTP_TIMEOUT,
        connector=aiohttp.TCPConnector(
            limit=0, limit_per_host=0)) if session is None else session

    payload = {
        "model": request_func_input.model_name \
            if request_func_input.model_name else request_func_input.model,
        "messages": [
        ],
        "temperature": 0.0,
        "max_completion_tokens": request_func_input.output_len,
        "stream": streaming,
    }

    if isinstance(request_func_input.prompt, list) and all(
        [isinstance(i, int) for i in request_func_input.prompt]):
        payload["prompt_token_ids"] = request_func_input.prompt
    else:
        if not isinstance(request_func_input.prompt, str):
            raise ValueError("Prompt must be a string or a list of integers")
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.extend(request_func_input.multi_modal_content)
        payload["messages"].append({"role": "user", "content": content})

    if streaming:
        payload["stream_options"] = {"include_usage": True}
    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    try:
        async with request_session.post(url=api_url,
                                        json=payload,
                                        headers=headers) as response:
            if response.status == 200:
                output.success = True
                if streaming:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""

                                # Extract avg_decoded_tokens_per_iter from streaming chat response
                                if "avg_decoded_tokens_per_iter" in choices[0]:
                                    output.avg_decoded_tokens_per_iter = choices[
                                        0]["avg_decoded_tokens_per_iter"]
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    content = await response.content.read()
                    data = json.loads(content.decode())
                    output.generated_text = data["choices"][0]["message"][
                        "content"]
                    output.output_tokens = data["usage"]["completion_tokens"]
                    output.itl = []
                    output.latency = time.perf_counter() - st
                    output.ttft = -1

                    # Extract avg_decoded_tokens_per_iter if available
                    choice = data["choices"][0]
                    if "avg_decoded_tokens_per_iter" in choice:
                        output.avg_decoded_tokens_per_iter = choice[
                            "avg_decoded_tokens_per_iter"]

            else:
                # TODO: Need to store the status code to debug and report
                output.error = response.reason or ""
                output.success = False
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__
    finally:
        if session is None:
            await request_session.close()

    if pbar:
        pbar.update(1)

    return output


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


ASYNC_REQUEST_FUNCS = {
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)
]
