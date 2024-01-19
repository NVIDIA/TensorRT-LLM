import sys
import traceback
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List

import torch


@dataclass
class GenerationOutput:
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)


def print_colored(message, color: str = None):
    colors = dict(
        grey="\x1b[38;20m",
        yellow="\x1b[33;20m",
        red="\x1b[31;20m",
        bold_red="\x1b[31;1m",
        bold_green="\033[1;32m",
    )
    reset = "\x1b[0m"

    if color:
        sys.stderr.write(colors[color] + message + reset)
    else:
        sys.stderr.write(message)


def file_with_suffix_exists(directory, suffix) -> bool:
    path = Path(directory)
    for file_path in path.glob(f'*{suffix}'):
        if file_path.is_file():
            return True
    return False


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
