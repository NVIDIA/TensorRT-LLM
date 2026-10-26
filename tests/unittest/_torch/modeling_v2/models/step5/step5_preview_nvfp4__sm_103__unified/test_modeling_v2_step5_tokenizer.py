# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine tokenizes a Step-5 prompt exactly as the checkpoint's ``tokenizer.json`` does.

The checkpoint ships a ByteLevel BPE ``tokenizer.json`` (a ``Sequence`` pre-tokenizer
ending in ``ByteLevel``) under a ``tokenizer_config.json`` that names
``LlamaTokenizerFast``. Under transformers 5.x a bare ``AutoTokenizer`` resolves
``LlamaTokenizer``, which replaces the pre-tokenizer with a Metaspace one and drops the
spaces ("The capital of France is" becomes ``[0, 671, 79666, 2154, 51725, 278]`` instead
of ``[0, 671, 6102, 294, 8760, 344]``). Every launcher (LLM API, ``trtllm-eval``,
``trtllm-serve``) loads the tokenizer through ``tokenizer_factory`` /
``TransformersTokenizer.from_pretrained``, whose ``maybe_fix_byte_level_tokenizer``
reloads such a tokenizer through ``PreTrainedTokenizerFast``. These checks pin that the
result is ``tokenizer.json`` verbatim: the same ids as ``PreTrainedTokenizerFast`` and as
the raw ``tokenizers.Tokenizer``, BOS from ``tokenizer_config.json``'s ``add_bos_token``,
a lossless decode, and the checkpoint's ``chat_template.jinja``.

CPU only. Needs the checkpoint: ``STEP5_PREVIEW_CHECKPOINT``, else
``$LLM_MODELS_ROOT/Step-5-Preview-NVFP4``; skipped without one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from utils.llm_data import llm_models_root

from tensorrt_llm.tokenizer import TransformersTokenizer, tokenizer_factory

pytestmark = pytest.mark.cpu_only

#: The checkpoint's BOS (`<｜begin▁of▁sentence｜>`), prepended by `add_bos_token`.
_BOS_ID = 0

#: Measured on the checkpoint's tokenizer.json (the oracle's ids for the boot prompt).
_CAPITAL = "The capital of France is"
_CAPITAL_IDS = [0, 671, 6102, 294, 8760, 344]

#: An lm-eval gsm8k few-shot prompt in the accuracy gate's raw format: shots joined by a
#: blank line, then the question with an open answer.
_GSM8K = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes "
    "of babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
    "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10\n\n"
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. How much in dollars "
    "does she make every day at the farmers' market?\nAnswer:"
)

_PROMPTS = {
    "boot": _CAPITAL,
    "single_spaces": "a b c d e f g",
    "repeated_spaces": "two  spaces,   three   spaces,\t\ttabs and    four",
    "leading_trailing": "   leading and trailing   ",
    "newlines": "line one\nline two\n\nafter a blank line\r\nwindows\n\n\n",
    "digits": "1 22 333 4444 55555 3.14159 1e-5 2026-10-23 $1,234,567.89",
    "punctuation": "Hello, world! (yes?) [no] {maybe}; a/b\\c 'q' \"qq\" -- ... #@%^&*_+=~`|<>",
    "cjk": "中文测试：你好，世界。日本語のテスト、カタカナ。한국어 테스트",
    "mixed": "GPT-4o vs. 第5代 model: 92 layers × 4096 hidden → ok?",
    "gsm8k": _GSM8K,
}


def _checkpoint_dir() -> Path | None:
    explicit = os.environ.get("STEP5_PREVIEW_CHECKPOINT")
    if explicit:
        return Path(explicit)
    root = llm_models_root()
    return None if root is None else root / "Step-5-Preview-NVFP4"


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = _checkpoint_dir()
    if path is None or not (path / "tokenizer.json").is_file():
        pytest.skip(
            "no Step-5-Preview-NVFP4 checkpoint: set STEP5_PREVIEW_CHECKPOINT or stage "
            "Step-5-Preview-NVFP4 under LLM_MODELS_ROOT"
        )
    return path


@pytest.fixture(scope="module")
def engine(checkpoint: Path) -> TransformersTokenizer:
    """The tokenizer every launcher builds: `tokenizer_factory` on the checkpoint dir."""
    tok = tokenizer_factory(str(checkpoint))
    assert isinstance(tok, TransformersTokenizer), type(tok)
    return tok


@pytest.fixture(scope="module")
def fast(checkpoint: Path) -> PreTrainedTokenizerFast:
    """tokenizer.json through transformers' fast class, the way StepFun's reference loads it."""
    return PreTrainedTokenizerFast.from_pretrained(str(checkpoint))


@pytest.fixture(scope="module")
def raw(checkpoint: Path) -> Tokenizer:
    """tokenizer.json verbatim, with no transformers class in between."""
    return Tokenizer.from_file(str(checkpoint / "tokenizer.json"))


def test_from_pretrained_is_the_factory(checkpoint: Path, engine: TransformersTokenizer) -> None:
    direct = TransformersTokenizer.from_pretrained(str(checkpoint))
    for text in _PROMPTS.values():
        assert direct.encode(text) == engine.encode(text)


def test_pre_tokenizer_is_tokenizer_json(checkpoint: Path, engine: TransformersTokenizer) -> None:
    """The Metaspace replacement is what drops the spaces; the loaded backend must keep
    tokenizer.json's own pre-tokenizer."""
    declared = json.loads((checkpoint / "tokenizer.json").read_text())["pre_tokenizer"]
    loaded = json.loads(engine.tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]
    assert loaded == declared


def test_boot_prompt_ids(engine: TransformersTokenizer) -> None:
    assert engine.encode(_CAPITAL) == _CAPITAL_IDS


@pytest.mark.parametrize("name", sorted(_PROMPTS))
def test_ids_match_tokenizer_json(
    name: str, engine: TransformersTokenizer, fast: PreTrainedTokenizerFast, raw: Tokenizer
) -> None:
    text = _PROMPTS[name]
    ids = engine.encode(text)
    assert ids == fast.encode(text), name
    assert ids == raw.encode(text, add_special_tokens=True).ids, name
    # add_bos_token is true in tokenizer_config.json: exactly one BOS, first.
    assert ids[0] == _BOS_ID and _BOS_ID not in ids[1:], (name, ids[:4])
    assert engine.encode(text, add_special_tokens=False) == ids[1:], name


@pytest.mark.parametrize("name", sorted(_PROMPTS))
def test_decode_round_trips(name: str, engine: TransformersTokenizer) -> None:
    text = _PROMPTS[name]
    ids = engine.encode(text)
    assert engine.decode(ids, skip_special_tokens=True) == text, name
    assert engine.decode(ids[1:]) == text, name


def test_spaces_survive(engine: TransformersTokenizer) -> None:
    """The space-dropping defect, stated directly: every word after the first carries
    its leading space."""
    pieces = [engine.decode([i]) for i in engine.encode(_CAPITAL, add_special_tokens=False)]
    assert pieces == ["The", " capital", " of", " France", " is"], pieces


def test_chat_template_is_the_checkpoints(
    checkpoint: Path, engine: TransformersTokenizer, fast: PreTrainedTokenizerFast
) -> None:
    template = (checkpoint / "chat_template.jinja").read_text()
    assert engine.get_chat_template() == template
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "What is 48 + 24?"},
    ]
    rendered = engine.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == fast.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=template
    )
    assert "What is 48 + 24?" in rendered and "You are a careful assistant." in rendered
    ids = engine.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    assert ids == fast.encode(rendered, add_special_tokens=False)
