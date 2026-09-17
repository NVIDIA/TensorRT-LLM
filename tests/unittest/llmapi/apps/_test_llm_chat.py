import pytest

from ..test_llm import llama_model_path

__extra_import_path__ = ["~/examples"]
from apps.chat import LLM, AutoTokenizer, LlmConsole, SamplingParams


@pytest.fixture
def interactive_console():
    model_dir = llama_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model_dir, max_batch_size=8, max_seq_len=512)

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
    console.llm.shutdown()
