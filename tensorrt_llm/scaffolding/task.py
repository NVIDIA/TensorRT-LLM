from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union


class ScaffoldingOutput:

    def __init__(self):
        self.output_str = None


@dataclass
class Task:
    # Reserve for custom input params.
    custom_input_params: Optional[dict] = None

    # Scaffolding delivers the task to the Worker by worker_tag.
    worker_tag: str = field(default=None)

    # Reserve for custom output params.
    custom_output_params: Optional[dict] = None


class TaskStatus(Enum):
    SUCCESS = "success"
    WORKER_NOT_SUPPORTED = "worker_not_supported"
    WORKER_EXECEPTION = "worker_exception"


@dataclass
class GenerationTask(Task):
    # input field
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False
    # sampling params
    #custom_sampling_params: Optional[dict] = None
    max_tokens: Optional[int] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)

    # suggest to use Controller.WorkerTag
    # anyway, users need to ensure that the value of the worker_tag can be found in the scaffoldingLlm's workers map
    worker_tag: Union[str, "Controller.WorkerTag"] = None

    # result field
    output_tokens: List[int] = None
    output_str: Optional[str] = None
    cumulative_logprob: Optional[float] = None
    logprobs: List[float] = field(default_factory=list)

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> "ScaffoldingOutput":
        output = ScaffoldingOutput()
        output.output_str = self.output_str
        return output


@dataclass
class RewardTask(Task):
    # input field
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)
