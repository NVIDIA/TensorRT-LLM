import os
import sys

import pytest

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples",
                 "apps"))
from chat import LLM, AutoTokenizer, LlmConsole, SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import llama_model_path


@pytest.fixture
def interactive_console():
    model_dir = llama_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model_dir)

    sampling_params = SamplingParams()

    return LlmConsole(llm, tokenizer, sampling_params)


def test_interactive_console(interactive_console):
    console = interactive_console
    console.runsource('A B C')
    assert len(console.history) == 2
    assert console.history[0]['content'] == 'A B C'
    assert console.history[0]['role'] == 'user'
    assert console.history[1]['role'] == 'assistant'
    assert console.history[1]['content']  # reply not empty
