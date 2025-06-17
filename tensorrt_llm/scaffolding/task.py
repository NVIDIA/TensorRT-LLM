from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

import torch

from tensorrt_llm.executor.result import GenerationResult


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
    input_tokens: Optional[List[int]] = None
    input_str: Optional[str] = None
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False
    streaming: bool = False

    # sampling params for openai
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    # The special case is `num_logprobs`, its original name si `logprobs` but conflicted by the result field
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    num_logprobs: Optional[int] = None
    max_tokens: Optional[int] = 2048
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
    _outputs: Optional[List[dict]] = None

    # link to TRTLLM's GenerationResult, for async update in streaming mode
    _result: Optional[GenerationResult] = None

    @property
    def result(self) -> GenerationResult:
        return self._result

    @result.setter
    def result(self, result: GenerationResult) -> None:
        self._result = result
        self._outputs = result.outputs

    @property
    def output_tokens(self) -> List[int]:
        return self._outputs[
            0].token_ids if self.result and self._outputs else None

    @property
    def output_str(self) -> Optional[str]:
        return self._outputs[0].text if self.result and self._outputs else None

    @output_str.setter
    def output_str(self, output) -> Optional[str]:
        assert self.result and self._outputs
        self._outputs[0].text = output

    @property
    def cumulative_logprob(self) -> Optional[float]:
        return self._outputs[
            0].cumulative_logprob if self.result and self._outputs else None

    @property
    def logprobs(self) -> Optional[List[float]]:
        return self._outputs[
            0].logprobs if self.result and self._outputs else None

    @property
    def context_logits(self) -> Optional[torch.Tensor]:
        return self.result.context_logits if self.result else None

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> GenerationResult:
        return self.result


@dataclass
class RewardTask(Task):
    # input field
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)
