import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch

from tensorrt_llm.executor.result import TokenLogprobs

from .result import ScaffoldingOutput


@dataclass
class Task:
    # Scaffolding delivers the task to the Worker by worker_tag.
    worker_tag: str = field(default=None)

    # For streaming output.
    streaming_output_flag: bool = field(default=False)
    streaming_output_list: list[Any] = field(default_factory=list)

    # Reserve for custom input params.
    custom_input_params: Optional[dict] = None

    # Reserve for custom output params.
    custom_output_params: Optional[dict] = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "Task":
        pass

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        pass

    def create_scaffolding_output_stream(self) -> List[ScaffoldingOutput]:
        pass


class TaskStatus(Enum):
    SUCCESS = "success"
    WORKER_NOT_SUPPORTED = "worker_not_supported"
    WORKER_EXECEPTION = "worker_exception"


@dataclass
class GenerationTask(Task):
    # input field
    input_tokens: Optional[List[int]] = None
    input_str: Optional[str] = None
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False
    #streaming: bool = False

    # sampling params for openai
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    # The special case is `num_logprobs`, its original name si `logprobs` but conflicted by the result field
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    num_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # sampling params
    top_k: Optional[int] = None
    return_context_logits: Optional[bool] = False

    # suggest to use Controller.WorkerTag
    # anyway, users need to ensure that the value of the worker_tag can be found in the scaffoldingLlm's workers map
    worker_tag: Union[str, "Controller.WorkerTag"] = None

    # result field
    output_str: Optional[str] = None
    output_tokens: Optional[List[int]] = None
    # TODO: support openai API format context logits
    context_logits: Optional[torch.Tensor] = None
    # TODO: don't not use TokenLogprobs for general support
    logprobs: Optional[TokenLogprobs] = None
    customized_result_fields: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        return ScaffoldingOutput(self.output_str, self.output_tokens)


@dataclass
class StreamGenerationTask(GenerationTask):
    # input field
    # if the flag is set to True, the worker will cancel the generation work
    cancel_flag: Optional[bool] = field(default=False)
    # the task will be returned to the controller with at least new streaming_step tokens
    # if the streaming_step is set to 0,
    # the task will be returned to the controller immediately with
    # new tokens that have already been generated.
    streaming_step: Optional[int] = field(default=1)

    #result field
    # worker set this field and identify the same task by this field
    request_handle: Any = field(default=None)
    # worker set this field to True when the generation is finished
    end_flag: bool = field(default=False)

    @staticmethod
    def create_from_generation_task(task: GenerationTask,
                                    streaming_step) -> "StreamGenerationTask":
        stream_task = StreamGenerationTask()
        stream_task.__dict__ = copy.deepcopy(task.__dict__)
        stream_task.streaming_step = streaming_step
        return stream_task


@dataclass
class RewardTask(Task):
    # input field
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)


@dataclass
class StreamGenerationTask(GenerationTask):
    # input field
    # if the flag is set to True, the worker will cancel the generation work
    cancel_flag: Optional[bool] = field(default=False)
    # the task will be returned to the controller with at least new streaming_step tokens
    # if the streaming_step is set to 0,
    # the task will be returned to the controller immediately with
    # new tokens that have already been generated.
    streaming_step: Optional[int] = field(default=1)

    #result field
    # worker set this field and identify the same task by this field
    request_handle: Any = field(default=None)
    # worker set this field to True when the generation is finished
    end_flag: bool = field(default=False)
