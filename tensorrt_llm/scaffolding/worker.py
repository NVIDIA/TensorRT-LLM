from abc import ABC
from copy import deepcopy
from typing import Callable, List, Optional, Union

import openai
from transformers import AutoTokenizer

from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.executor import GenerationExecutor
from tensorrt_llm.llmapi.llm import LLM
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
        max_tokens: int = 2048,
        temperature: float = 0.9,
        top_p: Optional[float] = None,
    ):
        self.model = model
        self.async_client = async_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def combine_params_with_generation_task(self, params: dict,
                                            task: GenerationTask):
        params["prompt"] = task.input_str

        add_param_if_not_none(params, "max_tokens",
                              [task.max_tokens, self.max_tokens])
        add_param_if_not_none(params, "temperature",
                              [task.temperature, self.temperature])
        add_param_if_not_none(params, "top_p", [task.top_p, self.top_p])

    def fill_generation_task_with_response(self, task: GenerationTask,
                                           response: openai.Completion):
        task.output_str = response.choices[0].text
        task.logprobs = response.choices[0].logprobs

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        params = {}

        # Set required parameters
        params["model"] = self.model

        self.combine_params_with_generation_task(params, task)

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
    # just manager the TRT-LLM openai server special params
    def __init__(self, top_k: Optional[float] = None, **kwargs):
        self.top_k = top_k
        super().__init__(**kwargs)

    def combine_params_with_generation_task(self, params: dict,
                                            task: GenerationTask):
        super().combine_params_with_generation_task(params, task)
        extra_body = {}
        add_param_if_not_none(extra_body, "top_k", [task.top_k, self.top_k])
        params["extra_body"] = extra_body


class TRTLLMWorker(Worker):

    def __init__(
        self,
        llm: LLM,
        tokenizer: AutoTokenizer,
        max_num_tokens: int = 2048,
        temperature: float = 0.9,
        top_p: Optional[float] = None,
        topk: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.default_sampling_params = SamplingParams(max_tokens=max_num_tokens,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      top_k=topk,
                                                      stop=stop)

        self.default_sampling_params._setup(self.tokenizer)
        self.own_llm = False

    @classmethod
    def init_with_new_llm(cls,
                          model_dir: str,
                          backend: str = None,
                          max_batch_size: int = 32,
                          **kwargs):
        pytorch_backend_config = PyTorchConfig(
            mixed_decoder=True,
            enable_overlap_scheduler=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )

        llm = LLM(model_dir,
                  backend=backend,
                  tokenizer=tokenizer,
                  pytorch_backend_config=pytorch_backend_config,
                  max_batch_size=max_batch_size,
                  max_num_tokens=kwargs.get("max_num_tokens", 2048))

        worker = cls(llm, tokenizer, **kwargs)
        worker.own_llm = True
        return worker

    def combine_sampling_params_with_generation_task(self,
                                                     task: GenerationTask):
        sampling_params = deepcopy(self.default_sampling_params)

        add_attr_if_not_none(sampling_params, "max_tokens", [task.max_tokens])
        add_attr_if_not_none(sampling_params, "temperature", [task.temperature])
        add_attr_if_not_none(sampling_params, "top_p", [task.top_p])
        add_attr_if_not_none(sampling_params, "top_k", [task.top_k])

        return sampling_params

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        sampling_params = self.combine_sampling_params_with_generation_task(
            task)

        result = await self.llm.generate_async(task.input_str,
                                               sampling_params=sampling_params)

        task.output_tokens = result.outputs[0].token_ids
        task.cumulative_logprob = result.outputs[0].cumulative_logprob
        task.logprobs = result.outputs[0].logprobs
        task.output_str = result.outputs[0].text

        # TODO: error handle
        return TaskStatus.SUCCESS

    def shutdown(self):
        if self.own_llm:
            self.llm.shutdown()

    task_handlers = {GenerationTask: generation_handler}
