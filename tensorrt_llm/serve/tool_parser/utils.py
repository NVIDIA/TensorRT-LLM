# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/qwen25_detector.py
import json
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any

import partial_json_parser
from partial_json_parser.core.options import Allow


def find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False
