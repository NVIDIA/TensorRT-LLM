from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Task:
    # Scaffolding delivers the task to the Worker by role.
    role: str = field(default=None)
    # Worker delivers the task to its own function by this.
    type: str = field(default=None)


@dataclass
class TaskStatus:
    # TODO: add fields or maybe update it to enum
    pass


@dataclass
class GenerationTask(Task):
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False
    # custom sampling params to override worker's sampling params in each generation.
    custom_sampling_params: Optional[dict] = None

    # overwrite base class default value
    type: str = field(default="generate")
    role: str = field(default="generation")

    # result field
    output_tokens: List[int] = None
    output_str: Optional[str] = None
    cumulative_logprob: Optional[float] = None
    logprobs: List[float] = field(default_factory=list)
