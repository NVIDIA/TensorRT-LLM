"""Unit tests for `BaseMultimodalDummyInputsBuilder`.

Verifies the default behaviors that subclasses inherit:

* ``spatial_merge_unit`` defaults to 1 (no merging).
* ``get_num_mm_tokens`` / ``get_size_with_most_features`` raise
  ``NotImplementedError`` so subclasses must opt into deterministic dummy
  sizing explicitly.
* ``get_dummy_prompt`` catches that ``NotImplementedError`` and returns
  ``None`` so ``_create_dummy_mm_context_request`` falls back to text-only.
"""

import pytest

from tensorrt_llm.inputs.registry import BaseMultimodalDummyInputsBuilder


class _StubBuilder(BaseMultimodalDummyInputsBuilder):
    """Minimal concrete builder that satisfies the abstract properties.

    Doesn't run any real processor / tokenizer — these tests only exercise
    the default base-class behavior, never the deterministic path.
    """

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        return None

    @property
    def model_path(self):
        return ""


def test_spatial_merge_unit_defaults_to_one():
    builder = _StubBuilder()
    assert builder.spatial_merge_unit == 1


def test_get_num_mm_tokens_default_raises_not_implemented():
    builder = _StubBuilder()
    with pytest.raises(NotImplementedError):
        builder.get_num_mm_tokens(width=224, height=224)


def test_get_size_with_most_features_default_raises_not_implemented():
    builder = _StubBuilder()
    with pytest.raises(NotImplementedError):
        builder.get_size_with_most_features(max_tokens=1024)


def test_get_dummy_prompt_returns_none_when_subclass_did_not_implement():
    """``get_dummy_prompt`` returns ``None`` for non-migrated subclasses.

    Caller falls back to a text-only dummy rather than seeing an unhandled
    exception or an iterative halving probe.
    """
    builder = _StubBuilder()
    assert builder.get_dummy_prompt(input_seq_len=512) is None


def test_get_dummy_prompt_returns_none_for_non_positive_seq_len():
    builder = _StubBuilder()
    assert builder.get_dummy_prompt(input_seq_len=0) is None
    assert builder.get_dummy_prompt(input_seq_len=-1) is None
