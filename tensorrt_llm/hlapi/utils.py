import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union

import torch

from tensorrt_llm.bindings import executor as tllme

from ..bindings.executor import OutputConfig


def print_traceback_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e

    return wrapper


class SamplingConfig(tllme.SamplingConfig):

    def __init__(self,
                 end_id: Optional[int] = None,
                 pad_id: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Patch some configs from cpp Request here, since they are attached to each prompt while
        # it we won't introduce the Request concept in the generate() API.
        self.end_id = end_id
        self.pad_id = pad_id if pad_id is not None else end_id
        self.max_new_tokens = max_new_tokens

    def _get_member_names(self):
        return ("beam_width", "top_k", "top_p", "top_p_min", "top_p_reset_ids",
                "top_p_decay", "random_seed", "temperature", "min_length",
                "beam_search_diversity_rate", "repetition_penalty",
                "presence_penalty", "frequency_penalty", "length_penalty",
                "early_stopping")

    def __eq__(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in self._get_member_names()) and \
               self.end_id == other.end_id and self.pad_id == other.pad_id and self.max_new_tokens == other.max_new_tokens

    @print_traceback_on_error
    def __setstate__(self, args: tuple) -> None:
        assert len(args) == 3 + len(self._get_member_names())
        kwargs = {
            'end_id': args[0],
            'pad_id': args[1],
            'max_new_tokens': args[2]
        }
        kwargs.update({
            key: value
            for key, value in zip(self._get_member_names(), args[3:])
            if value is not None
        })
        # The C++ class's constructor is not properly called by pickle, so we need to call it manually.
        self.__init__(**kwargs)

    @print_traceback_on_error
    def __getstate__(self) -> tuple:
        args = (self.end_id, self.pad_id, self.max_new_tokens) + tuple(
            getattr(self, name) for name in self._get_member_names())
        return args

    def __repr__(self):
        return f"SamplingConfig({', '.join(f'{name}={getattr(self, name)}' for name in self._get_member_names())})"


class OutputConfig(OutputConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Alter several default values for HLAPI usage.
        if "exclude_input_from_output" not in kwargs:
            self.exclude_input_from_output = True

    def __eq__(self, other: "OutputConfig") -> bool:
        return self.__getstate__() == other.__getstate__()

    def __getstate__(self):
        return {
            "exclude_input_from_output": self.exclude_input_from_output,
            "return_log_probs": self.return_log_probs,
            "return_context_logits": self.return_context_logits,
            "return_generation_logits": self.return_generation_logits,
        }

    def __setstate__(self, state):
        self.exclude_input_from_output = state["exclude_input_from_output"]
        self.return_log_probs = state["return_log_probs"]
        self.return_context_logits = state["return_context_logits"]
        self.return_generation_logits = state["return_generation_logits"]


@dataclass
class GenerationOutput:
    text: str = ""
    token_ids: Union[List[int], List[List[int]]] = field(default_factory=list)
    log_probs: Optional[List[float]] = None
    context_logits: Optional[torch.Tensor] = None
    generation_logits: Optional[torch.Tensor] = None


def print_colored(message, color: str = None):
    colors = dict(
        grey="\x1b[38;20m",
        yellow="\x1b[33;20m",
        red="\x1b[31;20m",
        bold_red="\x1b[31;1m",
        bold_green="\033[1;32m",
        green="\033[0;32m",
    )
    reset = "\x1b[0m"

    if color:
        sys.stderr.write(colors[color] + message + reset)
    else:
        sys.stderr.write(message)


def file_with_glob_exists(directory, glob) -> bool:
    path = Path(directory)
    for file_path in path.glob(glob):
        if file_path.is_file():
            return True
    return False


def file_with_suffix_exists(directory, suffix) -> bool:
    return file_with_glob_exists(directory, f'*{suffix}')


def get_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_total_gpu_memory(device: int) -> float:
    return torch.cuda.get_device_properties(device).total_memory


class GpuArch:

    @staticmethod
    def is_post_hopper() -> bool:
        return get_gpu_arch() >= 9

    @staticmethod
    def is_post_ampere() -> bool:
        return get_gpu_arch() >= 8

    @staticmethod
    def is_post_volta() -> bool:
        return get_gpu_arch() >= 7


def get_gpu_arch(device: int = 0) -> int:
    return torch.cuda.get_device_properties(device).major


class ContextManager:
    ''' A helper to create a context manager for a resource. '''

    def __init__(self, resource):
        self.resource = resource

    def __enter__(self):
        return self.resource.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.resource.__exit__(exc_type, exc_value, traceback)


def is_directory_empty(directory: Path) -> bool:
    return not any(directory.iterdir())


def suppress_runtime_log():
    ''' Suppress the runtime log if the environment variable is not set.  '''

    if "TLLM_LOG_LEVEL" not in os.environ:
        os.environ["TLLM_LOG_LEVEL"] = "ERROR"
    if "TLLM_LOG_FIRST_RANK_ONLY" not in os.environ:
        os.environ["TLLM_LOG_FIRST_RANK_ONLY"] = "ON"


def sigint_handler(signal, frame):
    sys.stderr.write("\nSIGINT received, quit LLM!\n")
    sys.exit(1)


# Register the signal handler to handle SIGINT
# This helps to deal with user's Ctrl+C
signal.signal(signal.SIGINT, sigint_handler)
