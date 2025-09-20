from abc import ABC
from typing import Callable

import openai
from transformers import AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

from .task import GenerationTask, Task, TaskStatus

ExecutorCls = GenerationExecutor


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
    ):
        self.model = model
        self.async_client = async_client

    def convert_task_params(self, task: GenerationTask):
        params = {
            "model": self.model,
            "prompt": task.input_str,
        }
        add_param_if_not_none(params, "best_of", [task.best_of])
        add_param_if_not_none(params, "echo", [task.echo])
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

        return params

    def fill_generation_task_with_response(self, task: GenerationTask,
                                           response: openai.Completion):
        task.output_str = response.choices[0].text
        task.logprobs = response.choices[0].logprobs

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

    def shutdown(self):
        # OpenAI client doesn't require explicit cleanup
        pass

    task_handlers = {GenerationTask: generation_handler}


# worker inherit from OpenaiWorker
# add TRT-LLM openai server special params
class TRTOpenaiWorker(OpenaiWorker):

    def convert_task_params(self, task: GenerationTask):
        params = super().convert_task_params(task)
        if task.top_k is not None:
            params["extra_body"] = {"top_k": task.top_k}
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
    ):
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
                  enable_mixed_sampler=True,
                  disable_overlap_scheduler=disable_overlap_scheduler,
                  kv_cache_config=kv_cache_config,
                  max_batch_size=max_batch_size,
                  max_num_tokens=max_num_tokens)

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

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        sampling_params = self.convert_task_params(task)

        # If the task is streaming, we will return result directly for
        # async iteration outside. Otherwise, we will wait.
        if task.streaming:
            result = self.llm.generate_async(task.input_str,
                                             sampling_params=sampling_params,
                                             streaming=True)
        else:
            result = await self.llm.generate_async(
                task.input_str, sampling_params=sampling_params)
        task.result = result

        # TODO: error handle
        return TaskStatus.SUCCESS

    def shutdown(self):
        if self.own_llm:
            self.llm.shutdown()

    task_handlers = {GenerationTask: generation_handler}
