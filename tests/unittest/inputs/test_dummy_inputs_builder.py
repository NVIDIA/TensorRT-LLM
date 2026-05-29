"""Unit tests for `BaseMultimodalDummyInputsBuilder`.

Verifies the default behaviors that subclasses inherit:

* ``get_size_with_most_features`` raises ``NotImplementedError`` so
  subclasses must opt into deterministic dummy sizing explicitly.
* ``get_dummy_prompt`` catches that ``NotImplementedError`` and returns
  ``None`` so ``_create_dummy_mm_context_request`` falls back to text-only.

Note: ``get_num_mm_tokens`` and ``spatial_merge_unit`` live on
:class:`BaseMultimodalInputProcessor` — see
``tests/unittest/inputs/test_input_processor_deterministic_math.py`` —
because the hashing path (``get_num_tokens_per_image``) and the dummy
path share that math.
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
