import json

import pytest

from tensorrt_llm.evaluate.json_mode_eval import _load_json_from_generation


def test_load_json_from_generation_accepts_plain_json():
    assert _load_json_from_generation('{"answer": 1}') == {"answer": 1}


def test_load_json_from_generation_accepts_gpt_oss_prefixed_json():
    text = 'analysis hidden reasoning assistantfinal json{"answer": 1}'

    assert _load_json_from_generation(text) == {"answer": 1}


def test_load_json_from_generation_raises_without_json_value():
    with pytest.raises(json.JSONDecodeError):
        _load_json_from_generation("analysis no final payload")
