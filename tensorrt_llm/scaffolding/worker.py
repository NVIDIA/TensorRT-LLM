import asyncio
import copy
import os
import types
from abc import ABC
from enum import Enum
from typing import Callable, List, Optional

import openai
import requests
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.executor import GenerationExecutor, GenerationResult
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SchedulerConfig
from tensorrt_llm.sampling_params import SamplingParams

from .result import ScaffoldingOutput
from .task import (AssistantMessage, ChatTask, DropKVCacheTask, GenerationTask,
                   MCPCallTask, StreamGenerationTask, Task, TaskStatus)

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

    def shutdown(self):
        pass

    async def async_shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self):
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

            return requests.post(
                url,
                json=kv_cache_hint_params,
                headers=headers,
            )

        async_client.create_kv_cache_hint = types.MethodType(
            send_kv_cache_hint, async_client)

        self.model = model
        self.async_client = async_client
        self.kv_cache_hint_enabled = kv_cache_hint_enabled

    def convert_task_params(self, task: GenerationTask | ChatTask):
        params = {
            "model": self.model,
            "extra_body": {},
        }

        if hasattr(task, "sub_request_markers") and os.environ.get(
                'DEBUG_AGENT_HIERARCHY') == '1':
            print(f"task.sub_request_markers is {task.sub_request_markers}")

        if not isinstance(task, ChatTask):
            params["prompt"] = task.input_str
            add_param_if_not_none(params, "echo", [task.echo])

        add_param_if_not_none(params, "best_of", [task.best_of])
        add_param_if_not_none(params, "frequency_penalty",
                              [task.frequency_penalty])
        add_param_if_not_none(params, "logit_bias", [task.logit_bias])
        add_param_if_not_none(params, "logprobs", [task.num_logprobs])
        add_param_if_not_none(params, "max_tokens", [task.max_tokens])
        add_param_if_not_none(params, "n", [task.n])
        add_param_if_not_none(params, "presence_penalty",
                              [task.presence_penalty])
        add_param_if_not_none(params, "seed", [task.seed])
        add_param_if_not_none(params, "stop", [task.stop])
        add_param_if_not_none(params, "suffix", [task.suffix])
        add_param_if_not_none(params, "temperature", [task.temperature])
        add_param_if_not_none(params, "top_p", [task.top_p])
        add_param_if_not_none(params, "user", [task.user])

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

    def fill_generation_task_with_response(self, task: GenerationTask,
                                           response: openai.Completion):
        task.output_str = response.choices[0].text
        task.output_tokens = response.choices[0].token_ids
        task.finish_reason = response.choices[0].finish_reason
        task.logprobs = response.choices[0].logprobs
        task.perf_metrics = response.perf_metrics

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        params = self.convert_task_params(task)

        # Make the API call
        try:
            response = await self.async_client.completions.create(**params)
            self.fill_generation_task_with_response(task, response)

            return TaskStatus.SUCCESS

        except Exception as e:
            # Handle errors
            print('Openai client get exception: ' + str(e))
            return TaskStatus.WORKER_EXECEPTION

    async def chat_handler(self, task: ChatTask) -> TaskStatus:
        params = self.convert_task_params(task)
        params["messages"] = task.messages_to_dict_content()
        params["model"] = self.model
        if task.tools is not None:
            params["tools"] = [tool.to_dict() for tool in task.tools]

        try:
            response = await self.async_client.chat.completions.create(**params)
            task.finish_reason = response.choices[0].finish_reason
            task.perf_metrics = response.perf_metrics
            content = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning
            reasoning_content = response.choices[0].message.reasoning_content
            tool_calls = response.choices[0].message.tool_calls
            task.messages.append(
                AssistantMessage(content, reasoning, reasoning_content,
                                 tool_calls))
            if task.enable_token_counting:
                task.prompt_tokens_num = response.usage.prompt_tokens
                task.completion_tokens_num = response.usage.completion_tokens
                if hasattr(
                        response.usage, "completion_tokens_details"
                ) and response.usage.completion_tokens_details is not None:
                    task.reasoning_tokens_num = response.usage.completion_tokens_details.reasoning_tokens

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

    task_handlers = {
        GenerationTask: generation_handler,
        ChatTask: chat_handler,
        DropKVCacheTask: drop_kv_cache_handler
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
            logprobs=task.num_logprobs)
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


import asyncio
import json

from mcp import ClientSession
from mcp.client.sse import sse_client


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
                task.result_str = result
                break

        return TaskStatus.SUCCESS

    task_handlers = {MCPCallTask: call_handler}
