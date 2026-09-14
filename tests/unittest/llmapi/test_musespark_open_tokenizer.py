# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer/template contracts for the MuseSparkOpen checkpoint.

Runs against the real checkpoint (no fixture copy): the point of every case is
that the shipped files say something the defaults get wrong. Not wired into any
L0 list — it needs the local checkpoint.
"""

import hashlib
import json
import os

import pytest

from tensorrt_llm.tokenizer import TOKENIZER_ALIASES, load_custom_tokenizer
from tensorrt_llm.tokenizer.musespark_open import (
    CHAT_TEMPLATE_FILENAME,
    END_OF_MESSAGE_TOKEN_ID,
    EOS_TOKEN_IDS,
    PAD_TOKEN_ID,
    MuseSparkOpenTokenizer,
    read_chat_template,
    read_generation_eos_ids,
)

pytestmark = pytest.mark.cpu_only

# Full path to the NVFP4 reference checkpoint; unset means skip.
CHECKPOINT = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")

# Prompt token ids the reference implementation produced for its
# raw-completion smoke: raw text, no chat template, no BOS.
ORACLE_RAW_PROMPTS = [
    ("The capital of France is", [954, 7963, 323, 11698, 373]),
    (
        'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
        [1251, 105436, 2736, 1960, 277, 10938, 9586, 290, 307, 9588, 75714, 1416, 34807],
    ),
]


@pytest.fixture(scope="module")
def tokenizer():
    if not CHECKPOINT or not os.path.isdir(CHECKPOINT):
        pytest.skip("checkpoint not present; set MUSESPARK_OPEN_ROOT")
    return load_custom_tokenizer("musespark_open", CHECKPOINT)


def test_musespark_open_tokenizer_alias_is_registered():
    assert TOKENIZER_ALIASES["musespark_open"] == (
        "tensorrt_llm.tokenizer.musespark_open.MuseSparkOpenTokenizer"
    )


def test_musespark_open_tokenizer_class(tokenizer):
    assert isinstance(tokenizer, MuseSparkOpenTokenizer)


def test_musespark_open_chat_template_comes_from_the_checkpoint_file(tokenizer):
    """The chat template comes from the .jinja beside the weights.

    tokenizer_config.json has no chat_template, so that file is the only
    source and it has to be used verbatim.
    """
    config_path = os.path.join(CHECKPOINT, "tokenizer_config.json")
    with open(config_path, encoding="utf-8") as handle:
        assert json.load(handle).get("chat_template") is None, (
            "checkpoint now wires the template into tokenizer_config.json; "
            "this test's premise needs revisiting"
        )

    template = read_chat_template(CHECKPOINT)
    assert template is not None
    assert tokenizer.tokenizer.chat_template == template
    on_disk = open(os.path.join(CHECKPOINT, CHAT_TEMPLATE_FILENAME), "rb").read()
    assert (
        hashlib.sha256(template.encode("utf-8")).hexdigest() == hashlib.sha256(on_disk).hexdigest()
    )


def test_musespark_open_chat_template_renders_the_generation_header(tokenizer):
    rendered = tokenizer.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    # The generation prompt stops right after `<|start|>assistant`, so the
    # model's first emitted text is the ` to=RECIPIENT<|message|>` header the
    # reasoning parser keys off.
    assert rendered.endswith("<|start|>assistant")
    assert "<|start|>user<|message|>hi<|eot|>" in rendered


@pytest.mark.parametrize("text,expected_ids", ORACLE_RAW_PROMPTS)
def test_musespark_open_raw_completion_ids_match_the_reference(tokenizer, text, expected_ids):
    """Raw completion means no template and no BOS.

    These ids come from the reference vLLM port, so a mismatch is a real
    tokenizer difference, not a self-comparison.
    """
    assert tokenizer.encode(text, add_special_tokens=False) == expected_ids


def test_musespark_open_pad_token_is_configured(tokenizer):
    inner = tokenizer.tokenizer
    assert inner.pad_token_id == PAD_TOKEN_ID
    assert inner.convert_ids_to_tokens(PAD_TOKEN_ID) == "<|finetune_right_pad|>"
    batch = inner(["a", "bbbb cccc dddd"], padding=True)
    lengths = {len(ids) for ids in batch["input_ids"]}
    assert len(lengths) == 1, "batched encoding must pad to one length"
    padded = batch["input_ids"][0]
    assert PAD_TOKEN_ID in padded


def test_musespark_open_stop_ids_are_both_eos_and_exclude_end_of_message(tokenizer):
    assert sorted(tokenizer.stop_token_ids) == sorted(EOS_TOKEN_IDS)
    assert read_generation_eos_ids(CHECKPOINT) == list(EOS_TOKEN_IDS)
    assert END_OF_MESSAGE_TOKEN_ID not in tokenizer.stop_token_ids
    inner = tokenizer.tokenizer
    assert inner.convert_ids_to_tokens(200001) == "<|end_of_text|>"
    assert inner.convert_ids_to_tokens(200008) == "<|eot|>"
    assert inner.convert_ids_to_tokens(END_OF_MESSAGE_TOKEN_ID) == "<|eom|>"


def test_musespark_open_end_of_message_survives_detokenization(tokenizer):
    """The parsers key off `<|eom|>`, so it has to reach them as text."""
    ids = [200007]
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<|eom|>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == ""


def test_musespark_open_digit_tokenization_is_not_regex_fixed(tokenizer):
    """``fix_mistral_regex`` has to stay off for this tokenizer.

    Its warning is a false positive here, and enabling it changes how digits
    are split.
    """
    from transformers import AutoTokenizer

    text = "12345 and 987 plus 1000000"
    ours = tokenizer.encode(text, add_special_tokens=False)
    fixed = AutoTokenizer.from_pretrained(CHECKPOINT, fix_mistral_regex=True).encode(
        text, add_special_tokens=False
    )
    assert ours != fixed, (
        "fix_mistral_regex no longer changes digit tokenization; this guard "
        "cannot detect the flag being turned on any more"
    )
    # And the untouched path is what the reference tokenizer produces.
    plain = AutoTokenizer.from_pretrained(CHECKPOINT).encode(text, add_special_tokens=False)
    assert ours == plain
