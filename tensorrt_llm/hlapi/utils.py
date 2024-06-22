import sys
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Optional, Union

import torch

import tensorrt_llm.bindings as tllm


class SamplingConfig(tllm.SamplingConfig):
    ''' The sampling config for the generation. '''

    # TODO[chunweiy]: switch to the cpp executor's once ready
    def __init__(self,
                 end_id: Optional[int] = None,
                 pad_id: Optional[int] = None,
                 beam_width: int = 1,
                 max_new_tokens: Optional[int] = None) -> None:
        super().__init__(beam_width)
        self.end_id = end_id
        self.pad_id = pad_id if pad_id is not None else end_id
        self.max_new_tokens = max_new_tokens

    def __setstate__(self, arg0: tuple) -> None:
        self.end_id = arg0[0]
        self.pad_id = arg0[1]
        self.max_new_tokens = arg0[2]
        super().__setstate__(arg0[3:])

    def __getstate__(self) -> tuple:
        return (self.end_id, self.pad_id,
                self.max_new_tokens) + super().__getstate__()

    def get_attr_names(self):
        return list(self.__dict__.keys()) + [
            "beam_search_diversity_rate",
            "beam_width",
            "early_stopping",
            "frequency_penalty",
            "length_penalty",
            "min_length",
            "presence_penalty",
            "random_seed",
            "repetition_penalty",
            "temperature",
            "top_k",
            "top_p",
            "top_p_decay",
            "top_p_min",
            "top_p_reset_ids",
        ]

    def __repr__(self):
        return f"SamplingConfig(" + ", ".join(
            f"{k}={getattr(self, k)}" for k in self.get_attr_names()
            if getattr(self, k) is not None) + ")"


@dataclass
class GenerationOutput:
    text: str = ""
    token_ids: Union[List[int], List[List[int]]] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)


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


def print_traceback_on_error(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise e

    return wrapper


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
