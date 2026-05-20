"""Unit tests for `MultimodalEncoderBudget`.

Validates:

* ``from_llm_args`` reads the encoder runtime sizes (with fallback to
  LLM-side knobs) into the budget.
* ``llm_visible_tokens`` converts the encoder-side budget to the
  LLM-visible (post-merger) units used to cap dummy prompt lengths.
* ``iter_modality_dummies`` yields one ``(modality, size)`` pair per
  supported modality, skips modalities the processor has not implemented,
  and yields nothing when ``supported_modalities`` is empty (so
  ``_create_dummy_mm_context_request`` falls back to text-only).
"""

from typing import Dict, Tuple

from tensorrt_llm._torch.pyexecutor.multimodal_budget import MultimodalEncoderBudget
from tensorrt_llm.inputs import Modality


class _StubLlmArgs:
    """``llm_args.get_encoder_runtime_sizes`` returns (max_items, max_tokens)."""

    def __init__(self, *, max_items: int, max_tokens: int):
        self._items = max_items
        self._tokens = max_tokens

    def get_encoder_runtime_sizes(self) -> Tuple[int, int]:
        return (self._items, self._tokens)


class _StubProcessor:
    """Stub processor exposing ``supported_modalities`` + size math.

    ``size_returner`` is a mapping {modality -> size_dict or "skip"}; when
    the value is the string ``"skip"`` the get_size method raises
    NotImplementedError (mimicking a partially-migrated processor).
    """

    def __init__(self, *, supported, size_returner: Dict[Modality, object]):
        self._supported = tuple(supported)
        self._size_returner = size_returner

    @property
    def supported_modalities(self):
        return self._supported

    def get_size_with_most_features(self, modality, *, max_tokens):
        v = self._size_returner.get(modality)
        if v == "skip" or v is None:
            raise NotImplementedError(modality)
        return v


def test_from_llm_args_reads_runtime_sizes():
    args = _StubLlmArgs(max_items=4, max_tokens=8192)
    budget = MultimodalEncoderBudget.from_llm_args(args)
    assert budget.max_items_per_step == 4
    assert budget.max_tokens_per_step == 8192


def test_llm_visible_tokens_uses_spatial_merge_unit():
    budget = MultimodalEncoderBudget(max_tokens_per_step=8192, max_items_per_step=4)
    assert budget.llm_visible_tokens(spatial_merge_unit=4) == 2048
    # Default merge_unit=1 (no merger) leaves the budget untouched.
    assert budget.llm_visible_tokens(spatial_merge_unit=1) == 8192


def test_llm_visible_tokens_min_one_for_safety():
    budget = MultimodalEncoderBudget(max_tokens_per_step=4, max_items_per_step=1)
    # spatial_merge_unit=8 > budget → math would round to 0, clamp to 1.
    assert budget.llm_visible_tokens(spatial_merge_unit=8) == 1


def test_iter_modality_dummies_yields_each_supported():
    processor = _StubProcessor(
        supported=(Modality.IMAGE, Modality.VIDEO),
        size_returner={
            Modality.IMAGE: {"width": 224, "height": 224},
            Modality.VIDEO: {"width": 128, "height": 128, "num_frames": 1},
        },
    )
    budget = MultimodalEncoderBudget(max_tokens_per_step=4096, max_items_per_step=1)
    pairs = list(budget.iter_modality_dummies(processor))
    assert len(pairs) == 2
    assert pairs[0][0] == Modality.IMAGE
    assert pairs[1][0] == Modality.VIDEO
    assert pairs[0][1]["width"] == 224
    assert pairs[1][1]["num_frames"] == 1


def test_iter_modality_dummies_skips_unimplemented():
    """Multi-modal processor partially migrated: AUDIO not yet implemented.

    ``iter_modality_dummies`` should yield IMAGE only, silently skipping
    the modality whose ``get_size_with_most_features`` raises
    NotImplementedError.
    """
    processor = _StubProcessor(
        supported=(Modality.IMAGE, Modality.AUDIO),
        size_returner={
            Modality.IMAGE: {"width": 224, "height": 224},
            Modality.AUDIO: "skip",
        },
    )
    budget = MultimodalEncoderBudget(max_tokens_per_step=4096, max_items_per_step=1)
    pairs = list(budget.iter_modality_dummies(processor))
    assert [m for m, _ in pairs] == [Modality.IMAGE]


def test_iter_modality_dummies_empty_when_no_supported():
    processor = _StubProcessor(supported=(), size_returner={})
    budget = MultimodalEncoderBudget(max_tokens_per_step=4096, max_items_per_step=1)
    assert list(budget.iter_modality_dummies(processor)) == []


def test_multi_encoder_dummy_iteration_includes_audio():
    """Audio + image processor (e.g. Nemotron Nano VL2 shape) must
    produce a dummy for *both* encoders so neither workspace ends up
    sized at zero after memory profiling."""
    processor = _StubProcessor(
        supported=(Modality.IMAGE, Modality.AUDIO),
        size_returner={
            Modality.IMAGE: {"width": 224, "height": 224},
            Modality.AUDIO: {"audio_length": 16000},
        },
    )
    budget = MultimodalEncoderBudget(max_tokens_per_step=4096, max_items_per_step=1)
    seen = [m for m, _ in budget.iter_modality_dummies(processor)]
    assert seen == [Modality.IMAGE, Modality.AUDIO]
