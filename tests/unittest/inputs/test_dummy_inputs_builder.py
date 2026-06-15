"""Unit tests for `BaseMultimodalDummyInputsBuilder`.

Verifies the default behaviors that subclasses inherit:

* ``get_mm_max_tokens_per_item`` returns ``{}`` and
  ``get_dummy_mm_data_for_tokens`` raises ``NotImplementedError`` — the two
  modality-agnostic profiler hooks default to "no direct profiling for this
  model" (text-only dummy fallback) until a subclass opts in. Modality-specific
  helpers (e.g. vision's ``get_size_for_max_tokens`` /
  ``get_dummy_mm_data_for_size``) live on the concrete processor, not this base.

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


def test_get_mm_max_tokens_per_item_default_empty():
    assert _StubBuilder().get_mm_max_tokens_per_item() == {}


def test_get_dummy_mm_data_for_tokens_default_raises_not_implemented():
    builder = _StubBuilder()
    with pytest.raises(NotImplementedError):
        builder.get_dummy_mm_data_for_tokens(max_tokens_per_modality={"image": 1024})
