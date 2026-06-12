import asyncio
import copy
import json
import os
import time
import types
from abc import ABC
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

import httpx
import openai
from mcp import ClientSession
from mcp.client.sse import sse_client
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.executor import GenerationExecutor, GenerationResult
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SchedulerConfig
from tensorrt_llm.sampling_params import SamplingParams

from .result import ScaffoldingOutput
from .task import (AssistantMessage, ChatTask, DropKVCacheTask, GenerationTask,
                   MCPCallTask, StreamGenerationTask, Task, TaskStatus,
                   TokenizeTask)

ExecutorCls = GenerationExecutor


# Helper function to check if deterministic mode is enabled
def is_deterministic_mode():
    """Check if SCAFFOLDING_DETERMINISTIC environment variable is set to enable deterministic inference."""
    return int(os.environ.get("SCAFFOLDING_DETERMINISTIC", 0)) == 1


class Worker(ABC):
    # user can use this api to register/add/override task handle function
    def register_task_handler(self, task_cls: type[Task],
                              handler: Callable[[object, Task], TaskStatus]):
        worker_cls = type(self)
        worker_cls.task_handlers[task_cls] = handler

    async def run_task(self, task: Task) -> TaskStatus:
        worker_cls = type(self)
        if type(task) not in worker_cls.task_handlers:
            return TaskStatus.WORKER_NOT_SUPPORTED
        return await worker_cls.task_handlers[type(task)](self, task)

    task_handlers = {}

    async def on_scope_end(self, scope_id: str) -> None:
        """Called when an :class:`ExecutionScope` finishes.

        Override in subclasses to release resources (SSE connections,
        sandboxes, etc.) that were acquired under *scope_id*.  The
        default implementation is a no-op.
        """

    def shutdown(self):
        pass

    async def async_shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# helper function
# add first non-None candidate_values to params with key
def add_param_if_not_none(params, key, candidate_values):
    for value in candidate_values:
        if value is not None:
            params[key] = value
            return


# helper function
# add first non-None candidate_values to the attribute of the object with key
def add_attr_if_not_none(obj, attr, candidate_values):
    for value in candidate_values:
        if value is not None:
            setattr(obj, attr, value)
            return


# Worker for standard openai api
class OpenaiWorker(Worker):

    def __init__(
        self,
        async_client: openai.AsyncOpenAI,
        model: str,
        kv_cache_hint_enabled: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        # Dynamic patch to support KV cache hint
        async def send_kv_cache_hint(self, task: DropKVCacheTask, params: dict):
            base_url = str(self.base_url)
            if not base_url.endswith("/"):
                base_url += "/"
            url = base_url + "kv_cache_hints"

            headers = {}
            if self.api_key is not None:
                headers["Authorization"] = f"Bearer {self.api_key}"

            kv_cache_hint_params = {
                "action":
                "truncate",
                "messages":
                [message.to_dict() for message in task.chat_task.messages],
                "messages_to_retain":
                [message.to_dict() for message in task.messages_to_retain],
            }

            # Spread extra_body contents into the request (like OpenAI client does)
            extra_body = params.pop("extra_body", {})
            kv_cache_hint_params.update(params)
            kv_cache_hint_params.update(
                extra_body)  # Spread extra_body contents

            async with httpx.AsyncClient() as client:
                return await client.post(
                    url,
                    json=kv_cache_hint_params,
                    headers=headers,
                )

        async_client.create_kv_cache_hint = types.MethodType(
            send_kv_cache_hint, async_client)

        self.model = model
        self.async_client = async_client
        self.kv_cache_hint_enabled = kv_cache_hint_enabled
        self.extra_body = copy.deepcopy(
            extra_body) if extra_body is not None else {}

    async def send_kv_cache_truncate_tokens(
        self,
        prefixes: List[List[int]],
        num_tokens_to_keep: List[int],
        request_timeout_s: float = 60.0,
    ) -> None:
        """POST a token-level KV cache truncate batch to trtllm-serve.

        Free radix-tree blocks the worker previously caused the server to
        commit (by sending the full ``prefix`` on a generation request).
        ``num_tokens_to_keep[i]`` is the prefix length to retain in the
        radix tree for ``prefixes[i]``; everything past that point on the
        same chain has its refcount decremented, and blocks whose
        refcount drops to zero are returned to the free pool.

        Hits ``/_control/kv_cache/truncate_tokens`` (added by
        :class:`tensorrt_llm.serve.control_plane.KVCacheControlPlane`) —
        a sibling of ``/_control/kv_cache/truncate`` that bypasses
        ``apply_chat_template`` so the radix-tree walk hashes the exact
        token-id bytes the caller previously sent on a ``/v1/completions``
        request.

        The endpoint is rooted at the server's host (NOT under ``/v1``),
        so we strip the ``/v1`` suffix that the OpenAI client's
        ``base_url`` typically carries.

        Raises :class:`RuntimeError` on a non-200 response.
        """
        if len(prefixes) != len(num_tokens_to_keep):
            raise ValueError(
                f"prefixes ({len(prefixes)}) and num_tokens_to_keep "
                f"({len(num_tokens_to_keep)}) length mismatch")
        if not prefixes:
            return

        base_url = str(self.async_client.base_url)
        # The OpenAI client's base_url is typically ``http://host:port/v1``
        # or ``http://host:port/v1/``; the control plane lives at
        # ``http://host:port/_control/...``, one level up.
        for v1_suffix in ("/v1/", "/v1"):
            if base_url.endswith(v1_suffix):
                base_url = base_url[:-len(v1_suffix)]
                break
        url = base_url.rstrip("/") + "/_control/kv_cache/truncate_tokens"

        headers = {"Content-Type": "application/json"}
        api_key = getattr(self.async_client, "api_key", None)
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "prefixes": [list(p) for p in prefixes],
            "num_tokens_to_keep": list(num_tokens_to_keep),
        }

        async with httpx.AsyncClient(timeout=request_timeout_s) as http:
            response = await http.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(
                    f"truncate_tokens failed: HTTP {response.status_code} "
                    f"body={response.text!r}")

    def convert_task_params(self, task: GenerationTask | ChatTask):
        params = {
            "model": self.model,
            "extra_body": copy.deepcopy(self.extra_body),
        }

        if hasattr(task, "sub_request_markers") and os.environ.get(
                'DEBUG_AGENT_HIERARCHY') == '1':
            print(f"task.sub_request_markers is {task.sub_request_markers}")

        if not isinstance(task, ChatTask):
            # Prefer pre-tokenized prompts when available (e.g. trace replay
            # via ReplayEngine fills ``input_tokens`` with synthetic token ids
            # and leaves ``input_str`` unset). The OpenAI completions API
            # natively accepts ``prompt`` as ``List[int]`` alongside ``str``.
            if task.input_tokens is not None:
                params["prompt"] = task.input_tokens
            else:
                params["prompt"] = task.input_str
            add_param_if_not_none(params, "echo", [task.echo])

        add_param_if_not_none(params, "best_of", [task.best_of])
        add_param_if_not_none(params, "frequency_penalty",
                              [task.frequency_penalty])
        add_param_if_not_none(params, "logit_bias", [task.logit_bias])
        add_param_if_not_none(params, "logprobs", [task.num_logprobs])
        add_param_if_not_none(params, "max_tokens", [task.max_tokens])
        add_param_if_not_none(params, "min_tokens", [task.min_tokens])
        add_param_if_not_none(params, "n", [task.n])
        add_param_if_not_none(params, "presence_penalty",
                              [task.presence_penalty])
        add_param_if_not_none(params, "seed", [task.seed])
        add_param_if_not_none(params, "stop", [task.stop])
        add_param_if_not_none(params, "suffix", [task.suffix])
        add_param_if_not_none(params, "temperature", [task.temperature])
        add_param_if_not_none(params, "top_p", [task.top_p])
        add_param_if_not_none(params, "user", [task.user])

        # Forward ignore_eos so trace replay (and any caller that must emit an
        # exact token budget) can bypass EOS early-stop. trtllm-serve surfaces
        # ignore_eos via extra_body -> SamplingParams.
        if task.ignore_eos:
            params["extra_body"]["ignore_eos"] = True

        # Forward skip_detokenizer so trace replay can retrieve the server's
        # actual decoded token ids in each stream chunk's ``token_ids`` field
        # (trtllm-serve gates this behind ``CompletionRequest.detokenize``;
        # see openai_protocol.py and postprocess_handlers.py). Without this,
        # the replay's per-turn segment store ends up holding RNG placeholder
        # ids that sibling-fork off the server's real decode chain in the
        # radix tree, dropping every subsequent turn's prefix hit at the
        # prev-prompt block boundary. See replay.py for the consequence.
        if getattr(task, "skip_detokenizer", False):
            params["extra_body"]["detokenize"] = False

        # Override parameters for deterministic inference
        if is_deterministic_mode():
            params["temperature"] = 0.0  # Deterministic sampling
            params["top_p"] = 1.0  # Disable nucleus sampling
            params["n"] = 1  # Only return one result
            if "seed" not in params or params["seed"] is None:
                params["seed"] = 42  # Fixed seed for reproducibility

        if hasattr(task, "sub_request_markers") and len(
                task.sub_request_markers) > 0:
            params["extra_body"]["agent_hierarchy"] = [
                task.sub_request_markers[-1]
            ]

        return params

    @staticmethod
    def _request_params_for_trace(params: dict) -> dict:
        return {
            key: copy.deepcopy(value)
            for key, value in params.items()
            if key not in ("messages", "tools", "prompt")
        }

    @staticmethod
    def _get_response_field(obj: Any, field: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    @classmethod
    def _reasoning_content_from_thinking_blocks(cls,
                                                message: Any) -> Optional[str]:
        thinking_blocks = cls._get_response_field(message, "thinking_blocks")
        if not thinking_blocks:
            return None
        thinking = cls._get_response_field(thinking_blocks[0], "thinking")
        return thinking if isinstance(thinking, str) else None

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        """Stream one completion and record per-request timing on *task*.

        Uses ``stream=True`` + ``stream_options={"include_usage": True}`` so we
        can measure TTFT (wall-time from request start to the first chunk that
        carries a token) and the full per-request ``latency`` (to the final
        chunk), and pick up ``usage.completion_tokens`` from the trailing
        usage-only chunk. These are the same per-request quantities
        SemiAnalysis's ``benchmark_serving.py`` records for the
        InferenceMAX ``intvty`` headline, letting the trace-replay Pareto
        pipeline emit ``1000 / median_TPOT`` per LLM call instead of a
        per-session proxy. See ``intvty_alignment_handoff.md`` §7 (Phase A).

        The trtllm-serve ``/v1/completions`` backend emits ``token_ids`` in
        each stream chunk only when the request opts out of detokenization
        (``CompletionRequest.detokenize=False``). Callers that need the
        actual decoded token ids (e.g. trace replay, which must align its
        per-conversation segment store with the server's KV-cache radix
        tree) should set ``task.skip_detokenizer=True`` — see
        :meth:`convert_task_params`. Callers that only need the count use
        ``task.usage_completion_tokens`` (authoritative, from the server's
        ``usage`` chunk) or the known ``max_tokens`` budget.
        """
        params = self.convert_task_params(task)
        task.llm_request_params = self._request_params_for_trace(params)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        text_parts: List[str] = []
        token_ids_acc: List[int] = []
        logprobs_acc: List = []
        finish_reason: Optional[str] = None
        usage_completion_tokens: Optional[int] = None
        usage_prompt_tokens: Optional[int] = None
        ttft_s: Optional[float] = None
        request_id: Optional[str] = None

        t_start = time.perf_counter()
        try:
            stream = await self.async_client.completions.create(**params)
            async for chunk in stream:
                now = time.perf_counter()
                # Every chunk carries the server-assigned request id. Capture
                # it once (the first non-None value) so callers can later
                # correlate per-request perf metrics drained from
                # ``/perf_metrics`` with the GenerationTask that issued them.
                if request_id is None:
                    cid = getattr(chunk, "id", None)
                    if cid is not None:
                        request_id = cid
                for choice in (chunk.choices or []):
                    delta_text = getattr(choice, "text", "") or ""
                    delta_token_ids = getattr(choice, "token_ids", None)
                    if ttft_s is None and (delta_text or delta_token_ids):
                        ttft_s = now - t_start
                    if delta_text:
                        text_parts.append(delta_text)
                    if delta_token_ids:
                        token_ids_acc.extend(delta_token_ids)
                    delta_logprobs = getattr(choice, "logprobs", None)
                    if delta_logprobs is not None:
                        logprobs_acc.append(delta_logprobs)
                    fr = getattr(choice, "finish_reason", None)
                    if fr is not None:
                        finish_reason = fr
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    ct = getattr(usage, "completion_tokens", None)
                    pt = getattr(usage, "prompt_tokens", None)
                    if ct is not None:
                        usage_completion_tokens = int(ct)
                    if pt is not None:
                        usage_prompt_tokens = int(pt)
            latency_s = time.perf_counter() - t_start

            task.output_str = "".join(text_parts)
            task.output_tokens = token_ids_acc if token_ids_acc else None
            task.finish_reason = finish_reason
            task.logprobs = logprobs_acc if logprobs_acc else None
            task.ttft_s = ttft_s
            task.latency_s = latency_s
            task.usage_completion_tokens = usage_completion_tokens
            task.usage_prompt_tokens = usage_prompt_tokens
            task.request_id = request_id

            return TaskStatus.SUCCESS

        except Exception as e:
            print('Openai client get exception: ' + str(e))
            return TaskStatus.WORKER_EXECEPTION

    async def chat_handler(self, task: ChatTask) -> TaskStatus:
        params = self.convert_task_params(task)
        params["messages"] = task.messages_to_dict_content()
        params["model"] = self.model
        if task.tools is not None:
            params["tools"] = [tool.to_dict() for tool in task.tools]
        task.llm_request_params = self._request_params_for_trace(params)

        try:
            response = await self.async_client.chat.completions.create(**params)
            finish_reason = response.choices[0].finish_reason
            task.finish_reason = finish_reason
            message = response.choices[0].message
            content = self._get_response_field(message, "content")
            reasoning = self._get_response_field(message, "reasoning")
            reasoning_content = self._get_response_field(
                message, "reasoning_content")
            if reasoning_content is None:
                reasoning_content = self._reasoning_content_from_thinking_blocks(
                    message)
            tool_calls = self._get_response_field(message, "tool_calls")
            task.messages.append(
                AssistantMessage(content, reasoning, reasoning_content,
                                 tool_calls, finish_reason))
            if task.enable_token_counting:
                usage = self._get_response_field(response, "usage")
                if usage is not None:
                    task.prompt_tokens_num = self._get_response_field(
                        usage, "prompt_tokens") or 0
                    task.completion_tokens_num = self._get_response_field(
                        usage, "completion_tokens") or 0
                    details = self._get_response_field(
                        usage, "completion_tokens_details")
                    if details is not None:
                        task.reasoning_tokens_num = self._get_response_field(
                            details, "reasoning_tokens") or 0

            return TaskStatus.SUCCESS

        except Exception as e:
            # Handle errors
            print('Openai chat client get exception: ' + str(e))
            return TaskStatus.WORKER_EXECEPTION

    async def drop_kv_cache_handler(self, task: DropKVCacheTask) -> TaskStatus:
        if not self.kv_cache_hint_enabled:
            return TaskStatus.SUCCESS

        params = self.convert_task_params(task.chat_task)
        params["messages"] = task.chat_task.messages_to_dict_content()
        params["model"] = self.model
        if task.chat_task.tools is not None:
            params["tools"] = [tool.to_dict() for tool in task.chat_task.tools]

        response = await self.async_client.create_kv_cache_hint(task, params)
        if response.status_code != 200:
            return TaskStatus.WORKER_EXECEPTION
        return TaskStatus.SUCCESS

    async def tokenize_handler(self, task: TokenizeTask) -> TaskStatus:
        base_url = str(self.async_client.base_url).rstrip("/")
        candidate_urls = [f"{base_url}/tokenize"]
        if base_url.endswith("/v1"):
            candidate_urls.append(f"{base_url[:-3]}/tokenize")
        else:
            candidate_urls.append(f"{base_url}/v1/tokenize")
        candidate_urls = list(dict.fromkeys(candidate_urls))
        failures = []
        task.tokenize_error = None
        headers = {}
        if self.async_client.api_key is not None:
            headers["Authorization"] = f"Bearer {self.async_client.api_key}"
        payload = {
            "prompt": task.content,
            # Include model for multi-model routers and gateways.
            "model": self.model,
        }
        try:
            async with httpx.AsyncClient() as client:
                for url in candidate_urls:
                    try:
                        response = await client.post(url,
                                                     json=payload,
                                                     headers=headers)
                    except Exception as e:
                        failures.append(f"{url} raised {type(e).__name__}: {e}")
                        continue
                    if response.status_code != 200:
                        text_preview = response.text[:200].replace("\n", " ")
                        failures.append(
                            f"{url} returned {response.status_code}: {text_preview}"
                        )
                        continue
                    body = response.json()
                    if "count" in body:
                        task.token_count = body.get("count", 0)
                        task.tokenize_error = None
                    elif "tokens" in body and isinstance(body["tokens"], list):
                        task.token_count = len(body["tokens"])
                        task.tokenize_error = None
                    else:
                        failures.append(
                            f"{url} returned malformed response keys={list(body.keys())}"
                        )
                        continue
                    return TaskStatus.SUCCESS
        except Exception as e:
            task.tokenize_error = f"Tokenize request got exception: {e}"
            print('Tokenize request got exception: ' + str(e))
            return TaskStatus.WORKER_EXECEPTION
        if failures:
            task.tokenize_error = "; ".join(failures)
            print("Tokenize request failed across candidate endpoints: " +
                  "; ".join(failures))
        return TaskStatus.WORKER_EXECEPTION

    task_handlers = {
        GenerationTask: generation_handler,
        ChatTask: chat_handler,
        DropKVCacheTask: drop_kv_cache_handler,
        TokenizeTask: tokenize_handler,
    }


# worker inherit from OpenaiWorker
# add TRT-LLM openai server special params
class TRTOpenaiWorker(OpenaiWorker):

    def convert_task_params(self, task: GenerationTask):
        params = super().convert_task_params(task)
        if task.top_k is not None:
            params["extra_body"]["top_k"] = task.top_k
        return params


class TRTLLMWorker(Worker):

    def __init__(
        self,
        llm: LLM,
        tokenizer: AutoTokenizer,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.own_llm = False

    @classmethod
    def init_with_new_llm(
        cls,
        model_dir: str,
        backend: str = "pytorch",
        max_batch_size: int = 32,
        max_num_tokens: int = 4096,
        kv_cache_free_gpu_memory_fraction: float = 0.9,
        disable_overlap_scheduler: bool = False,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        if scheduler_config is None:
            scheduler_config = SchedulerConfig()

        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction, )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )

        llm = LLM(model_dir,
                  tokenizer=tokenizer,
                  disable_overlap_scheduler=disable_overlap_scheduler,
                  kv_cache_config=kv_cache_config,
                  max_batch_size=max_batch_size,
                  max_num_tokens=max_num_tokens,
                  scheduler_config=scheduler_config)

        worker = cls(llm, tokenizer)
        worker.own_llm = True
        return worker

    def convert_task_params(self, task: GenerationTask):
        sampling_params = SamplingParams(
            max_tokens=task.max_tokens,
            temperature=task.temperature,
            top_p=task.top_p,
            top_k=task.top_k,
            return_context_logits=task.return_context_logits,
            logprobs=task.num_logprobs,
            ignore_eos=task.ignore_eos)
        return sampling_params

    async def streaming_generate_helper(self, generate_result, step_at_least,
                                        streaming_output_list):
        step = 0
        while not generate_result._done:
            async_task = asyncio.create_task(generate_result._aresult_step())
            if step_at_least and step >= step_at_least and not async_task.done(
            ):
                async_task.cancel()
                break
            await async_task
            step += 1
            # do not put the last token to the streaming_output_list
            if streaming_output_list is not None and not generate_result._done:
                streaming_output_list.append(
                    ScaffoldingOutput(
                        generate_result.outputs[0].text,
                        copy.deepcopy(generate_result.outputs[0].token_ids)))

    def fill_task_with_result(self, task: GenerationTask,
                              result: GenerationResult):
        task.output_str = result.outputs[0].text
        task.output_tokens = result.outputs[0].token_ids
        task.context_logits = result.context_logits
        task.logprobs = result.outputs[0].logprobs

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        sampling_params = self.convert_task_params(task)

        if task.streaming_output_flag:
            result = self.llm.generate_async(task.input_str,
                                             sampling_params=sampling_params,
                                             streaming=True)
            await self.streaming_generate_helper(result, None,
                                                 task.streaming_output_list)
        else:
            result = await self.llm.generate_async(
                task.input_str, sampling_params=sampling_params)

        self.fill_task_with_result(task, result)

        # TODO: error handle
        return TaskStatus.SUCCESS

    async def stream_generation_handler(
            self, task: StreamGenerationTask) -> TaskStatus:
        sampling_params = self.convert_task_params(task)
        if task.request_handle is None:
            task.request_handle = self.llm.generate_async(
                task.input_str, sampling_params=sampling_params, streaming=True)

        if task.cancel_flag:
            task.end_flag = True
            task.request_handle.abort()
            return TaskStatus.SUCCESS

        await self.streaming_generate_helper(
            task.request_handle, task.streaming_step,
            task.streaming_output_queue if task.streaming_output_flag else None)

        self.fill_task_with_result(task, task.request_handle)

        if task.request_handle._done:
            task.end_flag = True
        return TaskStatus.SUCCESS

    def shutdown(self):
        if self.own_llm:
            self.llm.shutdown()

    task_handlers = {
        GenerationTask: generation_handler,
        StreamGenerationTask: stream_generation_handler
    }


def _apply_mcp_tool_text_payload(task: MCPCallTask,
                                 text: Optional[str]) -> None:
    """If *text* is a Coder-style JSON envelope, set chat content and stdio fields.

    Envelope shape: ``{"content": str, "stdout": str, "stderr": str}`` (from Apiary
    MCP).  Otherwise *task*.result_str is *text* and stdio fields stay None.
    """
    task.result_str = text
    task.result_stdout = None
    task.result_stderr = None
    if not text or not text.lstrip().startswith("{"):
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    if not all(k in payload for k in ("content", "stdout", "stderr")):
        return
    content = payload["content"]
    if not isinstance(content, str):
        return
    task.result_str = content
    out = payload.get("stdout")
    err = payload.get("stderr")
    task.result_stdout = out if isinstance(
        out, str) else (str(out) if out is not None else "")
    task.result_stderr = err if isinstance(
        err, str) else (str(err) if err is not None else "")


class MCPWorker(Worker):

    class ToolCall:

        def __init__(self, tool_name: str, args: dict):
            self.tool_name = tool_name
            self.args = args
            self.ready = asyncio.Event()
            self.result = None

        def set_result(self, result: Optional[str]):
            self.result = result
            self.ready.set()

    def __init__(
        self,
        urls: List[str],
    ):
        self.urls = urls
        self.queues = [asyncio.Queue() for _ in urls]
        self._background_tasks: List[asyncio.Task] = []

    @classmethod
    def init_with_urls(cls, urls: List[str]):
        worker = cls(urls)
        return worker

    async def _main_loop_async_client_iter(self, url: str, index: int):

        class TaskType(Enum):
            TOOL_CALL = "tool_call"
            WAIT_QUEUE = "wait_queue"

        async with sse_client(url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.list_tools()
                tools = response.tools
                pending_dict = {
                    asyncio.create_task(self.queues[index].get()):
                    (TaskType.WAIT_QUEUE, None)
                }
                pending = pending_dict.keys()
                while True:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED)

                    for task in done:
                        task_type, obj = pending_dict[task]
                        if task_type == TaskType.TOOL_CALL:
                            response = task.result()
                            tool_call = obj
                            tool_call.set_result(response.content[0].text)
                        else:  # TaskType.WAIT_QUEUE
                            queue_obj = task.result()
                            if queue_obj is None:
                                # Shutdown signal received
                                # Cancel all pending tasks before exit to avoid blocking
                                for pending_task in pending:
                                    pending_task.cancel()
                                if pending:
                                    await asyncio.gather(*pending,
                                                         return_exceptions=True)
                                return
                            else:
                                tool_name = queue_obj.tool_name
                                args = queue_obj.args
                                if tool_name in [tool.name for tool in tools]:
                                    new_task = asyncio.create_task(
                                        session.call_tool(tool_name, args))
                                    pending_dict[new_task] = (
                                        TaskType.TOOL_CALL, queue_obj)
                                    pending.add(new_task)
                                else:
                                    queue_obj.set_result(None)
                                # Wait next queue object
                                new_wait_queue_task = asyncio.create_task(
                                    self.queues[index].get())
                                pending_dict[new_wait_queue_task] = (
                                    TaskType.WAIT_QUEUE, None)
                                pending.add(new_wait_queue_task)

    async def init_in_asyncio_event_loop(self):
        for index in range(len(self.urls)):
            task = asyncio.create_task(
                self._main_loop_async_client_iter(self.urls[index], index))
            self._background_tasks.append(task)

    async def async_shutdown(self):
        """Async shutdown MCP worker and wait for all background tasks to complete."""
        # Signal all background tasks to stop
        for queue in self.queues:
            queue.put_nowait(None)
        # Wait for all background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks,
                                 return_exceptions=True)
            self._background_tasks.clear()

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        tool_name = task.tool_name
        tool_args = json.loads(task.args)
        for index in range(len(self.urls)):
            tool_call = self.ToolCall(tool_name, tool_args)
            self.queues[index].put_nowait(tool_call)
            await tool_call.ready.wait()
            result = tool_call.result
            if result is not None:
                _apply_mcp_tool_text_payload(task, result)
                break

        return TaskStatus.SUCCESS

    task_handlers = {MCPCallTask: call_handler}


class ApiaryMCPWorker(Worker):
    """MCP worker with per-scope SSE connections for Apiary sandbox isolation.

    Unlike :class:`MCPWorker` which maintains a single shared SSE connection,
    ``ApiaryMCPWorker`` creates a separate SSE connection (and therefore a
    separate Apiary sandbox session) for each :class:`ExecutionScope`.
    The scope is read from the :data:`current_scope` ContextVar that
    :class:`ScaffoldingLlm` sets automatically for every request and
    parallel branch.  Connections are released when the scope ends.

    Usage::

        worker = ApiaryMCPWorker("http://localhost:8082/sse")
        # No init_in_asyncio_event_loop() needed -- connections are lazy.
    """

    class _ToolCall:

        def __init__(self, tool_name: str, args: dict):
            self.tool_name = tool_name
            self.args = args
            self.ready = asyncio.Event()
            self.result: Optional[str] = None
            self.error: bool = False

        def set_result(self, result: Optional[str], *, error: bool = False):
            self.result = result
            self.error = error
            self.ready.set()

    class _ConnState:
        __slots__ = ("queue", "task", "ready")

        def __init__(self, queue: asyncio.Queue, task: asyncio.Task,
                     ready: asyncio.Event):
            self.queue = queue
            self.task = task
            self.ready = ready

    def __init__(self, base_url: str, max_connections: int = 200):
        self.base_url = base_url.rstrip("/")
        self._max_connections = max_connections
        self._conns: dict[str, ApiaryMCPWorker._ConnState] = {}
        self._scope_params: dict[str, dict[str, str | list[str]]] = {}
        # Lazy-init for asyncio primitives (must be created inside the loop).
        self._lock: Optional[asyncio.Lock] = None
        self._sem: Optional[asyncio.Semaphore] = None

    def _ensure_primitives(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
            self._sem = asyncio.Semaphore(self._max_connections)

    def set_scope_params(self, request_id: str,
                         **params: str | list[str]) -> None:
        """Associate extra SSE URL query parameters with *request_id*.

        These parameters are appended to the SSE URL when the connection
        for this request (or any of its child scopes) is created.  Call
        this **after** :meth:`ScaffoldingLlm.generate_async` returns, using
        ``result.id`` as the *request_id*.

        Values may be strings or lists of strings.  List values are
        emitted as repeated query parameters (e.g.
        ``base_image=/p1&base_image=/p2``).

        Typical usage for SWE-bench::

            result = llm.generate_async(prompt)
            mcp_worker.set_scope_params(
                result.id,
                base_image=["/layer/base", "/layer/top"],
            )
        """
        self._scope_params[request_id] = params

    async def _conn_loop(self, url: str, queue: asyncio.Queue,
                         ready: asyncio.Event):
        try:
            async with sse_client(url) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    ready.set()
                    while True:
                        tc = await queue.get()
                        if tc is None:
                            return
                        try:
                            resp = await session.call_tool(
                                tc.tool_name, tc.args)
                            tc.set_result(resp.content[0].text)
                        except Exception as exc:
                            tc.set_result(f"Error: {exc}", error=True)
        except Exception:
            ready.set()  # unblock waiters even on connection failure

    @staticmethod
    def _root_request_id(scope_id: str) -> str:
        """Extract the root request_id from a scope_id.

        Child scopes have the format ``request_id:branch.path``.
        """
        return scope_id.split(":")[0]

    async def _get_conn(self, scope_id: str) -> "_ConnState":
        self._ensure_primitives()
        async with self._lock:
            if scope_id in self._conns:
                conn = self._conns[scope_id]
            else:
                await self._sem.acquire()
                url = f"{self.base_url}?client_id=coder-{scope_id}"
                extra = self._scope_params.get(self._root_request_id(scope_id),
                                               {})
                if extra:
                    url += "&" + urlencode(extra, doseq=True)
                queue: asyncio.Queue = asyncio.Queue()
                ready = asyncio.Event()
                task = asyncio.create_task(self._conn_loop(url, queue, ready))
                conn = self._ConnState(queue=queue, task=task, ready=ready)
                self._conns[scope_id] = conn
        await conn.ready.wait()
        return conn

    async def release_connection(self, scope_id: str) -> None:
        """Close the SSE connection for *scope_id* and release the slot."""
        self._ensure_primitives()
        async with self._lock:
            conn = self._conns.pop(scope_id, None)
        if conn is None:
            return
        conn.queue.put_nowait(None)
        try:
            await asyncio.wait_for(conn.task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            conn.task.cancel()
        self._sem.release()
        root_id = self._root_request_id(scope_id)
        if scope_id == root_id:
            self._scope_params.pop(root_id, None)

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        from .execution_scope import current_scope
        scope = current_scope.get()
        scope_id = scope.scope_id if scope is not None else "default"

        conn = await self._get_conn(scope_id)

        tc = self._ToolCall(task.tool_name, json.loads(task.args))
        conn.queue.put_nowait(tc)
        await tc.ready.wait()

        if tc.result is not None:
            _apply_mcp_tool_text_payload(task, tc.result)

        if tc.error:
            return TaskStatus.WORKER_EXECEPTION
        return TaskStatus.SUCCESS

    task_handlers = {MCPCallTask: call_handler}

    async def on_scope_end(self, scope_id: str) -> None:
        await self.release_connection(scope_id)

    async def async_shutdown(self):
        """Close all SSE connections and wait for background tasks."""
        self._ensure_primitives()
        async with self._lock:
            ids = list(self._conns.keys())
        for rid in ids:
            await self.release_connection(rid)

    def shutdown(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(self.async_shutdown(),
                                                       loop)
                fut.result(timeout=30)
            else:
                loop.run_until_complete(self.async_shutdown())
        except Exception:
            pass
