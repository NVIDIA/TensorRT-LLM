# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which ids end a MuseSparkOpen turn, read off the checkpoint rather than assumed.

Three ids matter and two of them stop:

* 200001 ``<|end_of_text|>`` -- stops. The only one ``tokenizer_config.json``
  names.
* 200008 ``<|eot|>`` -- stops. Named by ``generation_config.json`` only, so a
  runtime that reads the tokenizer config alone never terminates on it and
  runs every request to the length cap.
* 200007 ``<|eom|>`` -- does **not** stop. It ends a reasoning block or a
  non-final tool call. Treating it as EOS truncates every reasoning block at
  its first boundary, which looks like a short but well-formed answer rather
  than like a bug.

These cover the file-reading and the configuration guard on synthetic
checkpoint directories; ``test_musespark_open_tokenizer.py`` covers the same
contract against the real tokenizer.
"""

import json

import pytest

from tensorrt_llm.tokenizer.musespark_open import (
    END_OF_MESSAGE_TOKEN_ID,
    EOS_TOKEN_IDS,
    MuseSparkOpenTokenizer,
    read_generation_eos_ids,
)


def _checkpoint(tmp_path, eos):
    """A directory with just the ``generation_config.json`` this reads."""
    payload = {} if eos is None else {"eos_token_id": eos}
    (tmp_path / "generation_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path)


def test_the_module_constants_are_the_checkpoint_ids():
    assert EOS_TOKEN_IDS == (200001, 200008)
    assert END_OF_MESSAGE_TOKEN_ID == 200007
    assert END_OF_MESSAGE_TOKEN_ID not in EOS_TOKEN_IDS


def test_both_eos_ids_are_read_from_the_generation_config(tmp_path):
    assert read_generation_eos_ids(_checkpoint(tmp_path, [200001, 200008])) == [200001, 200008]


def test_a_scalar_eos_id_becomes_a_one_element_list(tmp_path):
    """``eos_token_id`` is allowed to be a bare int; the caller wants a list."""
    assert read_generation_eos_ids(_checkpoint(tmp_path, 200001)) == [200001]


@pytest.mark.parametrize("eos", [["200001", "200008"], [200001.0]])
def test_string_and_float_ids_are_coerced_to_int(tmp_path, eos):
    ids = read_generation_eos_ids(_checkpoint(tmp_path, eos))
    assert all(isinstance(i, int) for i in ids)


def test_a_missing_key_or_file_reads_as_none(tmp_path):
    assert read_generation_eos_ids(str(tmp_path)) is None
    assert read_generation_eos_ids(_checkpoint(tmp_path, None)) is None


class _InnerTokenizer:
    """The few attributes ``configure_for_musespark_open`` touches."""

    def __init__(self, eos_token_id=200001):
        self.chat_template = "{{ messages }}"
        self.pad_token_id = 0
        self.eos_token_id = eos_token_id
        self.pad_token = None

    def convert_ids_to_tokens(self, _id):  # pragma: no cover - not reached
        return "<|pad|>"


class _FakeWrapper:
    """Stands in for the tokenizer wrapper.

    ``configure_for_musespark_open`` is called on an already-constructed
    ``TransformersTokenizer``, so calling it unbound on a stub tests exactly
    the configuration step without downloading or building a real tokenizer.
    """

    def __init__(self, eos_token_id=200001):
        self.tokenizer = _InnerTokenizer(eos_token_id)

    configure = MuseSparkOpenTokenizer.configure_for_musespark_open
    stop_token_ids = MuseSparkOpenTokenizer.stop_token_ids


def test_stop_ids_are_both_eos_ids_and_never_end_of_message(tmp_path):
    wrapper = _FakeWrapper()
    wrapper.configure(_checkpoint(tmp_path, [200001, 200008]))

    assert wrapper.stop_token_ids == [200001, 200008]
    assert END_OF_MESSAGE_TOKEN_ID not in wrapper.stop_token_ids


def test_a_checkpoint_listing_end_of_message_as_eos_is_rejected(tmp_path):
    """Fail loudly. Silently stopping at ``<|eom|>`` looks like a short answer."""
    wrapper = _FakeWrapper()
    with pytest.raises(ValueError, match="200007"):
        wrapper.configure(_checkpoint(tmp_path, [200001, 200007, 200008]))


def test_a_checkpoint_without_generation_eos_falls_back_to_both_ids(tmp_path):
    """The fallback keeps 200008 terminal even with no generation config."""
    wrapper = _FakeWrapper()
    wrapper.configure(str(tmp_path))

    assert wrapper.stop_token_ids == [200001, 200008]


def test_stop_ids_default_before_configuration_runs():
    """A wrapper that was never configured still stops on both ids."""
    assert _FakeWrapper().stop_token_ids == [200001, 200008]


def test_a_tokenizer_config_eos_outside_the_generation_set_only_warns(tmp_path):
    """A mismatch is worth saying, not worth refusing to load over."""
    wrapper = _FakeWrapper(eos_token_id=199999)
    wrapper.configure(_checkpoint(tmp_path, [200001, 200008]))

    assert wrapper.stop_token_ids == [200001, 200008]
