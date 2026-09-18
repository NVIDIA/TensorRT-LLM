# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A text-only request must reach the decoder unchanged by the vision wrapper.

The multimodal wrapper sits in front of the same decoder the text-only model
uses, and four banked parity gates are pinned to that decoder's behaviour. If a
text request picks up so much as an embedding lookup on the way through, those
gates are measuring something other than what they were banked against.

This checks the wrapper's own text path by construction. It does not need the
architecture registry, so it is available before the registration takeover: the
end-to-end half of that criterion -- identical answers through a served engine
-- still does.

The claim being pinned is narrow and exact: with no multimodal parameters, the
wrapper hands the decoder the same ``input_ids`` it received, no
``inputs_embeds``, and the same everything else. Not "similar", not "equivalent
after a lookup" -- the same object.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import fuse_input_embeds
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm.sampling_params import SamplingParams


class _RecordingLLM(torch.nn.Module):
    """Stands in for the decoder and records exactly what it was handed."""

    def __init__(self, hidden_size=16, vocab_size=32):
        super().__init__()
        self.calls = []
        self.model = SimpleNamespace(
            embed_tokens=Embedding(vocab_size, hidden_size, dtype=torch.float32)
        )
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(1)


def _wrapper_with(llm):
    """A wrapper instance whose only wired part is the piece under test.

    Built with ``__new__`` on purpose: constructing the real thing would build a
    62-layer decoder and a 50-layer tower to exercise one branch that touches
    neither.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
        MuseSparkOpenOmniForConditionalGeneration,
    )

    wrapper = MuseSparkOpenOmniForConditionalGeneration.__new__(
        MuseSparkOpenOmniForConditionalGeneration
    )
    object.__setattr__(wrapper, "_parameters", {})
    object.__setattr__(wrapper, "_buffers", {})
    object.__setattr__(wrapper, "_modules", {})
    object.__setattr__(wrapper, "_non_persistent_buffers_set", set())
    wrapper.llm = llm
    wrapper.mm_encoder = None
    return wrapper


@pytest.mark.parametrize("disable_encoder", [False, True])
def test_encoder_disable_controls_construction_and_weight_loading(monkeypatch, disable_encoder):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.models import modeling_musespark_open_omni as omni

    config = PretrainedConfig()
    model_config = SimpleNamespace(pretrained_config=config, disable_mm_encoder=disable_encoder)
    decoder = _RecordingLLM()
    decoder.config = config
    encoder = torch.nn.Identity()
    constructed = []
    loaded = []
    weights = {"vision_model.unused": object(), "model.text": object()}
    mapper = object()
    summary = {"loaded": 1}

    def build_encoder(cfg):
        constructed.append(cfg)
        return encoder

    def load_decoder(actual_weights, actual_mapper):
        loaded.append(("text", actual_weights, actual_mapper))
        return summary

    def load_omni(wrapper, actual_weights):
        assert wrapper.mm_encoder is encoder
        loaded.append(("omni", actual_weights, None))
        return summary

    monkeypatch.setattr(omni, "MuseSparkOpenVisionEncoder", build_encoder)
    monkeypatch.setattr(omni, "MuseSparkOpenForCausalLM", lambda cfg: decoder)
    monkeypatch.setattr(decoder, "load_weights", load_decoder, raising=False)
    monkeypatch.setattr(omni, "load_musespark_open_omni_weights", load_omni)
    wrapper = omni.MuseSparkOpenOmniForConditionalGeneration(model_config)

    assert wrapper.llm is decoder
    assert wrapper.mm_encoder is (None if disable_encoder else encoder)
    assert constructed == ([] if disable_encoder else [model_config])
    assert wrapper.load_weights(weights, mapper) is summary
    expected = [("text", weights, mapper)] if disable_encoder else [("omni", weights, None)]
    assert loaded == expected


def _attn(num_contexts, num_generations=0):
    """The attention-metadata fields the forward actually reads.

    The wrapper's own forward read only ``num_contexts``. The mixin's forward
    also reads ``num_generations`` (it passes the count down to the
    model-specific extra-kwargs hook), so a stub carrying only ``num_contexts``
    raises ``AttributeError`` under the mixin. Both are real fields on
    ``AttentionMetadata`` (``_torch/attention/backends/interface.py:262``), so
    the stub is completed to match the contract rather than the contract being
    narrowed to the stub.
    """
    return SimpleNamespace(num_contexts=num_contexts, num_generations=num_generations)


def test_fuse_input_embeds_is_a_pass_through_when_there_is_no_media():
    """The framework helper the wrapper leans on returns the ids untouched.

    If this ever changes to return embeddings for the empty case, every
    text-only request through the wrapper would start paying an embedding
    lookup the decoder then repeats -- so the wrapper's correctness rests on
    this, and it is checked here rather than assumed.
    """
    embedding = Embedding(32, 16, dtype=torch.float32)
    input_ids = torch.arange(7, dtype=torch.int32)

    returned_ids, returned_embeds = fuse_input_embeds(embedding, input_ids, [])[:2]

    assert returned_embeds is None
    assert returned_ids is input_ids, "the ids were copied or rebuilt, not passed through"


def test_a_text_only_request_reaches_the_decoder_unchanged():
    """No multimodal params: the decoder sees exactly the text-only call."""
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)

    input_ids = torch.arange(5, dtype=torch.int32)
    position_ids = torch.arange(5, dtype=torch.int32)
    attn_metadata = _attn(num_contexts=1)

    wrapper.forward(
        attn_metadata=attn_metadata,
        input_ids=input_ids,
        position_ids=position_ids,
        return_context_logits=False,
    )

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["input_ids"] is input_ids
    assert call["inputs_embeds"] is None
    assert call["position_ids"] is position_ids
    assert call["attn_metadata"] is attn_metadata
    assert call["return_context_logits"] is False


def test_an_empty_multimodal_list_is_also_the_text_path():
    """A text turn in a multimodal conversation carries an empty list.

    After the prefill that consumed the image, later turns arrive with
    ``multimodal_params=[]``. That must take the same path as a request that
    never had media, not a degenerate version of the vision path.
    """
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)
    input_ids = torch.arange(4, dtype=torch.int32)

    wrapper.forward(
        attn_metadata=_attn(num_contexts=1),
        input_ids=input_ids,
        position_ids=torch.arange(4, dtype=torch.int32),
        multimodal_params=[],
    )

    call = llm.calls[0]
    assert call["input_ids"] is input_ids
    assert call["inputs_embeds"] is None


def test_the_text_path_never_touches_the_encoder():
    """`mm_encoder` is None here, so any use of it would raise.

    That is the assertion: the two tests above passing with no encoder wired at
    all is proof the text path does not reach for one.
    """
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)
    assert wrapper.mm_encoder is None

    wrapper.forward(
        attn_metadata=_attn(num_contexts=1),
        input_ids=torch.arange(3, dtype=torch.int32),
        position_ids=torch.arange(3, dtype=torch.int32),
    )
    assert len(llm.calls) == 1


def test_generation_phase_requests_do_not_run_the_encoder():
    """num_contexts=0 is decode; the slice of multimodal params is empty.

    The encoder is only ever fed `multimodal_params[:num_context_requests]`, so
    a decode-only iteration must not reach it even when the batch still carries
    multimodal params from its prefill.
    """
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)

    def _explode(*_args, **_kwargs):
        raise AssertionError("the encoder ran during a generation-only iteration")

    wrapper.mm_encoder = SimpleNamespace(forward=_explode)
    input_ids = torch.arange(2, dtype=torch.int32)

    wrapper.forward(
        attn_metadata=_attn(num_contexts=0, num_generations=1),
        input_ids=input_ids,
        position_ids=torch.arange(2, dtype=torch.int32),
        multimodal_params=[SimpleNamespace(multimodal_data={})],
    )

    call = llm.calls[0]
    assert call["input_ids"] is input_ids
    assert call["inputs_embeds"] is None


@pytest.mark.parametrize("num_contexts", [0, 1, 4])
def test_the_text_path_is_shape_agnostic(num_contexts):
    """Batch shape does not change the pass-through property."""
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)
    input_ids = torch.arange(11, dtype=torch.int32)

    wrapper.forward(
        attn_metadata=_attn(num_contexts=num_contexts),
        input_ids=input_ids,
        position_ids=torch.arange(11, dtype=torch.int32),
    )
    assert llm.calls[0]["input_ids"] is input_ids
    assert llm.calls[0]["inputs_embeds"] is None


# --------------------------------------------------------------------------
# The input processor must actually be constructible
#
# Both base classes declare abstract properties. Leaving any of them out does
# not fail at import or at definition -- it makes the class abstract, and the
# framework simply cannot build it. The rest of this repository's MuseSparkOpen
# tests only ever called the module-level preprocessing helpers, so nothing
# instantiated a processor and nothing noticed. That is the gap these close.
# --------------------------------------------------------------------------
def _tiny_vision_config():
    from test_modeling_musespark_open_vision import TINY_VISION_CONFIG  # noqa: PLC0415

    from tensorrt_llm._torch.configs.musespark_open import MuseSparkOpenConfig

    return MuseSparkOpenConfig(
        vocab_size=128,
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=24,
        vision_config=TINY_VISION_CONFIG,
        torch_dtype=torch.bfloat16,
    )


class _StubTokenizer:
    """Tokenizes the way the real one does where it matters.

    Three properties are load-bearing for the marker splice and all three are
    true of the checkpoint tokenizer: ``<image>`` and ``<video>`` are ORDINARY
    TEXT (no added token exists for either), so each becomes a multi-id run;
    the two runs differ; and ``><`` merges into a single token, so two adjacent
    markers do NOT tokenize as two copies of one marker. A stub without that
    last property would let the fixed-subsequence matcher this replaced pass.
    """

    _VOCAB = {
        "><": 2696,
        "<": 60,
        ">": 62,
        "image": 771,
        "video": 772,
        "audio": 773,
        "describe": 900,
        "this": 901,
        "and": 902,
        "that": 903,
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        rest = text
        while rest:
            for word, token in self._VOCAB.items():
                if rest.startswith(word):
                    ids.append(token)
                    rest = rest[len(word) :]
                    break
            else:
                if rest[0] == " ":
                    rest = rest[1:]
                    continue
                ids.append(1000 + ord(rest[0]))
                rest = rest[1:]
        return ids


def _build_processor(model_path="/nonexistent/checkpoint"):
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import MuseSparkOpenOmniInputProcessor

    return MuseSparkOpenOmniInputProcessor(
        model_path=model_path,
        config=_tiny_vision_config(),
        tokenizer=_StubTokenizer(),
    )


def _patch_items(data):
    """The `(modality, item)` pairs of `get_dummy_mm_data` that carry patches.

    The returned mapping is deliberately not uniform: a multi-modality call
    also carries an `mm_item_order` manifest (a list, so that a mixed request
    can be replayed in prompt order) and an audio entry holds waveforms rather
    than pixel patches. Only the vision entries have a grid and a
    `pixel_values` tensor, so only they can answer the grid-vs-allocation
    questions below.
    """
    return [
        (modality, item)
        for modality, item in data.items()
        if isinstance(item, dict) and "pixel_values" in item
    ]


def test_the_input_processor_can_be_instantiated():
    """The defect this file exists for: an abstract class cannot be built.

    If any of `processor`, `tokenizer`, `config`, `dtype` or `model_path` goes
    missing again, this raises TypeError at construction and names the ones
    that are still abstract.
    """
    processor = _build_processor()
    assert processor is not None


def test_the_processor_exposes_every_property_the_framework_asks_for():
    processor = _build_processor()
    assert processor.tokenizer is not None
    assert processor.config is not None
    assert processor.model_path == "/nonexistent/checkpoint"
    assert processor.dtype is torch.bfloat16


def test_asking_for_an_hf_processor_says_why_there_is_none():
    """There is no AutoProcessor for this model, and saying so beats lying.

    Returning the tokenizer here would satisfy the interface and then surface
    much later as a confusing attribute error inside someone else's code.
    """
    processor = _build_processor()
    with pytest.raises(AttributeError, match="no HF AutoProcessor"):
        _ = processor.processor


def test_a_still_image_counts_one_clip_not_patch_temporal_clips():
    """`patch_temporal` packs frames into a clip; it is not a clip count.

    A still image is duplicated to fill one clip, so its temporal extent is 1.
    Using `patch_temporal` here reported exactly twice the real row count -- to
    the KV-cache profiler and to anything else sizing work from this number.
    The check is against the preprocessing module's own answer for the same
    image, which is what the tower will actually be handed.
    """
    from PIL import Image  # noqa: PLC0415

    from tensorrt_llm._torch.models.musespark_open_vision_processing import (
        least_distortion_fit,
        num_placeholder_rows,
    )

    processor = _build_processor()
    cfg = processor.config.vision_config

    for width, height in ((448, 448), (640, 480), (84, 448), (1296, 968)):
        reported = processor.get_num_tokens_per_image(image=Image.new("RGB", (width, height)))
        canvas_w, canvas_h = least_distortion_fit(
            width,
            height,
            patch_size=cfg.patch_size,
            downsample_factor=cfg.downsample_factor,
            min_tokens=cfg.min_num_image_tokens,
            max_tokens=cfg.max_num_image_tokens,
        )
        truth = num_placeholder_rows(
            (1, canvas_h // cfg.patch_size, canvas_w // cfg.patch_size), cfg
        )
        assert reported == truth, (
            f"{width}x{height}: get_num_tokens_per_image says {reported}, "
            f"the preprocessing says {truth}"
        )


# --------------------------------------------------------------------------
# The registration takeover
#
# The checkpoint's only architecture string is "MuseSparkOpenOmniForCausalLM". It
# used to name the pure-text CausalLM, which was right while there was no
# tower. Registration is first-wins and the omni module imports the text one,
# so the takeover is only correct if the text module stops registering: if both
# register, the text class wins the slot, the perception weights are never
# built, and an image request answers about nothing without erroring.
#
# AC-MM-08 asks for a statement of how the architecture string now resolves.
# These are that statement, executable.
# --------------------------------------------------------------------------
def _omni_class():
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration,
    )

    return MuseSparkOpenOmniForConditionalGeneration


def test_the_architecture_string_resolves_to_the_omni_class():
    from tensorrt_llm._torch.models.modeling_utils import (  # noqa: PLC0415
        get_registered_model_class,
    )

    assert get_registered_model_class("MuseSparkOpenOmniForCausalLM") is _omni_class()


def test_the_text_decoder_no_longer_holds_the_architecture_slot():
    """The negative half, and the one that catches a re-added decorator."""
    from tensorrt_llm._torch.models.modeling_musespark_open import (  # noqa: PLC0415
        MuseSparkOpenForCausalLM,
    )
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING  # noqa: PLC0415

    assert MODEL_CLASS_MAPPING["MuseSparkOpenOmniForCausalLM"] is not MuseSparkOpenForCausalLM
    assert MuseSparkOpenForCausalLM not in MODEL_CLASS_MAPPING.values()


def test_the_vision_encoder_is_registered_against_that_architecture():
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenVisionEncoder,
    )
    from tensorrt_llm._torch.models.modeling_utils import (  # noqa: PLC0415
        MODEL_CLASS_VISION_ENCODER_MAPPING,
    )

    encoder_cls, _ = MODEL_CLASS_VISION_ENCODER_MAPPING["MuseSparkOpenOmniForCausalLM"]
    assert encoder_cls is MuseSparkOpenVisionEncoder


def test_the_input_processor_is_registered_against_the_omni_class():
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniInputProcessor,
    )
    from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY  # noqa: PLC0415

    registered = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type[_omni_class()]
    assert registered is MuseSparkOpenOmniInputProcessor


def test_the_lazy_index_points_at_the_omni_module():
    """The static index is what a fresh process reads before importing anything.

    Left pointing at ``modeling_musespark_open`` it would import the text module,
    whose registration is now gone, and the architecture would resolve to
    nothing at all.
    """
    from tensorrt_llm._torch.models._arch_index import (  # noqa: PLC0415
        MODEL_ARCH_TO_MODULE,
        MULTIMODAL_MODEL_TYPE_TO_MODULE,
    )

    assert MODEL_ARCH_TO_MODULE["MuseSparkOpenOmniForCausalLM"] == "modeling_musespark_open_omni"
    assert MULTIMODAL_MODEL_TYPE_TO_MODULE["musespark_open_omni"] == "modeling_musespark_open_omni"


def test_the_chat_template_keeps_its_content_parts():
    """OPENAI, not STRING, and the reason is in the template.

    Under STRING the framework merges the placeholder into the message text,
    and this checkpoint's template escapes a literal "<image>" in text to
    "< image>" precisely so a prompt cannot forge one. The media would then be
    silently dropped -- no error, an answer about nothing.
    """
    from tensorrt_llm.inputs.registry import (  # noqa: PLC0415
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        ContentFormat,
    )

    assert (
        MULTIMODAL_PLACEHOLDER_REGISTRY.get_content_format("musespark_open_omni")
        is ContentFormat.OPENAI
    )


# --------------------------------------------------------------------------
# What the splice actually looks for
#
# The marker left in the prompt is the plain string "<image>", which the
# tokenizer has no added token for. Two matchers were tried on the id stream
# and both find NOTHING:
#   - the special <|image|> 200090, which a templated prompt never contains;
#   - tokenizer.encode("<image>") as a fixed subsequence, which stops matching
#     the moment two media are adjacent, because BPE merges the seam into a
#     "><" token. Adjacent media are the norm, not an edge case: every MindCube
#     item carries 2 to 4 images and the template writes them with no separator.
# So the marker is found in the TEXT and the segments between markers are
# tokenized separately.
# --------------------------------------------------------------------------
def _spliced(processor, text, expansions):
    """Just the fused ids; the prompt-order manifest has its own tests below."""
    return processor._splice_placeholders(text, expansions)[0]


def test_the_plain_text_marker_is_what_gets_replaced():
    processor = _build_processor()
    out = _spliced(processor, "describe <image> this", [("image", [10, 11, 12])])
    assert out == [900, 10, 11, 12, 901]


def test_adjacent_markers_are_all_found():
    """The defect this replaced: a seam-merged run matched zero markers.

    The stub tokenizer models the merge the real one performs -- "><" is one
    token -- so a fixed-subsequence matcher fails here exactly as it failed on
    the reference multi-image prompts.
    """
    processor = _build_processor()
    seam = processor.tokenizer.encode("><", add_special_tokens=False)
    assert len(seam) == 1, "the stub must merge the seam or this proves nothing"

    out = _spliced(
        processor,
        "<image><image><image>",
        [("image", [10]), ("image", [11]), ("image", [12])],
    )
    assert out == [10, 11, 12]


def test_the_special_image_token_is_not_a_marker():
    """Negative control for the first wrong matcher.

    "<|image|>" is what 200090 spells, and the template escapes any literal
    occurrence to "< |image|>" precisely so a prompt cannot forge one. If the
    matcher went looking for it, this prompt would splice.
    """
    processor = _build_processor()
    with pytest.raises(ValueError, match="0 image markers but 1 image items"):
        _spliced(processor, "<|image|>", [("image", [10, 11])])


def test_markers_keep_prompt_order_across_modalities():
    """Interleaved media must land where they were written, not modality-major."""
    processor = _build_processor()
    out = _spliced(
        processor,
        "<video> and <image> that",
        [("image", [30, 31]), ("video", [20, 21])],
    )
    assert out == [20, 21, 902, 30, 31, 903]


def test_a_missing_item_is_an_error_not_a_silent_drop():
    processor = _build_processor()
    with pytest.raises(ValueError, match="more image markers than image items"):
        _spliced(processor, "<image> <image>", [("image", [10])])


# --------------------------------------------------------------------------
# Both spellings of a media slot
#
# The two shipped checkpoints mark a media slot differently: the earlier
# template writes the plain "<image>" / "<video>" / "<audio>", the open-export
# template writes "<|patch|>" / "<|video|>" / "<|audio|>". A scan that knows
# only one spelling finds zero markers against the other template, so every
# media request dies with "prompt has 0 <modality> markers". Both spellings are
# accepted and whichever the active template emitted is the one that matches.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "modality, marker, expansion",
    [
        ("image", "<|patch|>", [10, 11, 12]),
        ("video", "<|video|>", [20, 21]),
        ("audio", "<|audio|>", [30, 31]),
    ],
)
def test_the_open_export_marker_spelling_is_spliced(modality, marker, expansion):
    processor = _build_processor()
    out = _spliced(processor, f"describe {marker} this", [(modality, expansion)])
    assert expansion == out[1:-1], f"{marker} was not spliced as a {modality} marker"


def test_both_marker_spellings_keep_prompt_order():
    """Items land on markers in TEXT order, whichever spelling each one used."""
    processor = _build_processor()
    out = _spliced(
        processor,
        "<|video|> and <image> that",
        [("image", [30, 31]), ("video", [20, 21])],
    )
    assert out == [20, 21, 902, 30, 31, 903]


def test_the_open_export_marker_is_consumed_whole():
    """The cursor advances past the marker it MATCHED, not a fixed length.

    With one hard-coded marker length the scan leaves the tail of a longer
    spelling behind and re-tokenizes it as ordinary text, so adjacent markers
    splice in the wrong place.
    """
    processor = _build_processor()
    out = _spliced(processor, "<|patch|><|patch|>", [("image", [10]), ("image", [11])])
    assert out == [10, 11]


def test_an_open_export_audio_prompt_is_not_read_as_zero_markers():
    """The exact failure the released checkpoint's template produced.

    Before both spellings were accepted, an audio request rendered by the
    open-export template contained no "<audio>" anywhere, so the splice saw
    zero markers against one supplied item and every audio request failed --
    which would have taken the whole audio suite down.
    """
    processor = _build_processor()
    out = _spliced(processor, "<|audio|>transcribe", [("audio", [30, 31])])
    assert out[:2] == [30, 31]


# --------------------------------------------------------------------------
# The contracts the runtime calls on the hashing path
#
# These four are called once per media item per request, and every exception
# they raise is CAUGHT and logged as "Multimodal hashing failed:" rather than
# propagated. So a wrong one does not break a request -- it silently turns off
# embedding caching and re-runs the tower on every chunked-prefill chunk, which
# is exactly what AC-MM-04's chunked-prefill check forbids. Nothing here would
# have failed loudly on its own.
# --------------------------------------------------------------------------
def _pil(width, height):
    from PIL import Image  # noqa: PLC0415

    return Image.new("RGB", (width, height))


def test_the_framework_asks_for_an_image_not_for_its_dimensions():
    """The call the framework actually makes, in each shape media loaders emit.

    PIL from a direct caller, CHW tensor from the hashing path (which skips the
    ToPIL round-trip), HWC uint8 from the "np" media format. All three describe
    the same image, so all three must count the same.
    """
    import numpy as np  # noqa: PLC0415

    processor = _build_processor()
    width, height = 640, 480

    from_pil = processor.get_num_tokens_per_image(image=_pil(width, height))
    from_chw = processor.get_num_tokens_per_image(image=torch.zeros(3, height, width))
    from_hwc = processor.get_num_tokens_per_image(
        image=np.zeros((height, width, 3), dtype=np.uint8)
    )

    assert from_pil > 0
    assert from_pil == from_chw == from_hwc


def test_the_framework_can_count_a_video():
    """Counting must agree with what preprocessing will actually produce."""
    from tensorrt_llm._torch.models.musespark_open_vision_processing import (  # noqa: PLC0415
        num_placeholder_rows,
        preprocess_video,
    )

    processor = _build_processor()
    frames = [_pil(112, 112) for _ in range(8)]

    counted = processor.get_num_tokens_per_video(video=frames, video_metadata={"fps": 4.0})
    _, grid, _ = preprocess_video(frames, 4.0, processor.config.vision_config)
    assert counted == num_placeholder_rows(grid, processor.config.vision_config)


def test_the_checkpoints_declared_video_budget_reaches_the_count(tmp_path):
    """The wiring, at the call the serve path actually makes.

    A checkpoint that declares `max_video_frame_tokens` in
    `processor_config.json` must have that budget govern the token count, not
    the module default. Counting is where it matters first: the scheduler and
    the KV-cache profiler size their work from this number before a single
    pixel is preprocessed, so a count taken at the wrong budget mis-sizes the
    request even when the preprocessing that follows is correct.
    """
    import json  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniInputProcessor,
    )

    # 4 frames at 1 fps survive sampling whole and pair into 2 clips. The tiny
    # config's token patch is 4 px, so a 64x64 frame is exactly 16x16 cells --
    # over the 144 default (shrunk to 12x12) and exactly on a 256 declaration.
    frames = [_pil(64, 64) for _ in range(4)]
    metadata = {"fps": 1.0}

    default_count = _build_processor().get_num_tokens_per_video(
        video=frames, video_metadata=metadata
    )
    assert default_count == 2 * 144, "two clips at the 144-token default"

    (tmp_path / "processor_config.json").write_text(
        json.dumps({"video_processor": {"max_video_frame_tokens": 256}})
    )
    declared = MuseSparkOpenOmniInputProcessor(
        model_path=str(tmp_path), config=_tiny_vision_config(), tokenizer=_StubTokenizer()
    )
    assert declared.get_num_tokens_per_video(video=frames, video_metadata=metadata) == 2 * 256


def test_counting_a_video_without_a_frame_rate_says_why():
    """No safe default: 2 fps out of the source rate decides the frame count."""
    processor = _build_processor()
    with pytest.raises(ValueError, match="frame rate"):
        processor.get_num_tokens_per_video(video=[_pil(112, 112)], video_metadata={})


def test_the_multimodal_ids_do_not_route_through_the_absent_processor():
    """The base class would look for `self.processor.mm_token_ids` and raise.

    Only the embed-carrying ids belong here, and no specials: an image is one
    unbroken run of <|patch|>, a video's per-clip timestamp text is ordinary
    text that no id set can cover, and the audio delimiters
    (<|soft_audio_start|> / <|soft_audio_end|>) keep their own text embeddings
    while the interior <|audio|> run carries the encoder rows.
    """
    from tensorrt_llm._torch.models.musespark_open_audio import AUDIO_TOKEN_IDS  # noqa: PLC0415
    from tensorrt_llm._torch.models.musespark_open_vision_processing import (  # noqa: PLC0415
        VISION_TOKEN_IDS,
    )

    processor = _build_processor()
    assert sorted(processor.get_mm_token_ids().tolist()) == sorted(
        [VISION_TOKEN_IDS["patch"], VISION_TOKEN_IDS["video"], AUDIO_TOKEN_IDS["audio"]]
    )
    assert processor.get_mm_special_token_ids() is None


def test_the_dummy_grid_describes_exactly_the_patches_it_allocates():
    """A grid claiming more patches than were allocated walks off the tensor.

    The tower segments the packed batch by `t * h * w` read off the grid, so
    the two must agree exactly -- `t` of `patch_temporal` described twice the
    rows, because `patch_temporal` is already inside `patch_dim`.
    """
    processor = _build_processor()
    cfg = processor.config.vision_config

    data = processor.get_dummy_mm_data(
        max_num_encoder_tokens=256, mm_counts={"image": 2, "video": 1}, dtype=torch.bfloat16
    )
    patch_items = _patch_items(data)
    assert [modality for modality, _ in patch_items] == ["image", "video"]
    for modality, item in patch_items:
        grid = item[f"{modality}_grid_thw"]
        described = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
        allocated = item["pixel_values"].shape[0]
        assert described == allocated, f"{modality}: grid says {described}, allocated {allocated}"
        assert item["pixel_values"].shape[1] == (
            cfg.num_channels * cfg.patch_temporal * cfg.patch_size**2
        )
        # Both grid sides stay even: the pixel shuffle merges 2x2 patches.
        assert int(grid[0, 1]) % cfg.downsample_factor == 0
        assert int(grid[0, 2]) % cfg.downsample_factor == 0


def test_a_video_item_carries_its_own_frame_rate():
    """The serving path hands over a VideoData, whose metadata holds the rate.

    Reading it off `mm_processor_kwargs` only would mean every OpenAI-endpoint
    video request had to restate a rate the decoder already read off the
    container.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        _frames_and_fps,
    )

    frames = [_pil(112, 112) for _ in range(4)]
    video_data = SimpleNamespace(frames=frames, metadata={"fps": 24.0})

    got_frames, got_fps, sampling = _frames_and_fps(video_data, {})
    assert got_frames is frames
    assert got_fps == 24.0
    assert sampling == {}, "a decoder that reported no sampling must not get a guess"


def test_the_encoder_budget_is_converted_before_it_bounds_an_item():
    """Rows and patches are different units and min() of them is wrong.

    The config caps count decoder rows; `max_num_encoder_tokens` counts encoder
    patches, and one row is downsample_factor**2 patches. Comparing them
    directly let an item four times too large past the bound, which is the
    number the startup profiler then sizes work from.
    """
    processor = _build_processor()
    cfg = processor.config.vision_config
    ds = cfg.downsample_factor

    budget = 4 * ds**2  # room for exactly four rows
    caps = processor.get_mm_max_tokens_per_item(budget)
    assert caps["image"] == min(cfg.max_num_image_tokens, 4)
    assert caps["video"] == min(cfg.max_num_video_tokens, 4)

    # Unbounded still reports the config's own caps.
    unbounded = processor.get_mm_max_tokens_per_item()
    assert unbounded["image"] == cfg.max_num_image_tokens
    assert unbounded["video"] == cfg.max_num_video_tokens

    # And the bound is honoured end to end: the dummy item the profiler builds
    # from it must not allocate more patches than the budget allows.
    data = processor.get_dummy_mm_data(
        max_num_encoder_tokens=budget, mm_counts={"image": 1}, dtype=torch.bfloat16
    )
    assert data["image"]["pixel_values"].shape[0] <= budget


def test_the_geometry_only_copy_builds_without_a_tokenizer():
    """The engine builds one with `tokenizer=None` and asks it for geometry.

    Construction must not tokenize -- it used to, which made the engine's own
    copy unbuildable -- and the geometry calls must all work without one.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniInputProcessor,
    )

    processor = MuseSparkOpenOmniInputProcessor(
        model_path="/nonexistent/checkpoint", config=_tiny_vision_config(), tokenizer=None
    )
    assert processor.get_num_tokens_per_image(image=_pil(448, 448)) > 0
    assert processor.get_mm_max_tokens_per_item()["image"] > 0
    assert (
        processor.get_dummy_mm_data(
            max_num_encoder_tokens=256, mm_counts={"image": 1}, dtype=torch.bfloat16
        )["image"]["pixel_values"].shape[0]
        > 0
    )

    # And it never touched the tokenizer: a geometry-only copy that loaded one
    # would make every rank pay for a load it does not use.
    assert processor._tokenizer is None


def test_a_missing_tokenizer_is_built_from_the_checkpoint():
    """The framework hands the engine's copy `tokenizer=None` and expects the
    processor to supply its own.

    `create_input_processor` wraps construction in `hf_remote_code_lock()`
    precisely because "input processors build an AutoTokenizer/AutoProcessor",
    and the sibling processors do. Without it a media-free request through an
    engine-built processor died on `NoneType has no attribute 'encode'` before
    a single weight loaded -- which is exactly the regression the text-path
    criterion exists to catch.
    """
    import os  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniInputProcessor,
    )

    checkpoint = os.environ.get("MUSESPARK_OPEN_TOKENIZER")
    if not checkpoint:
        pytest.skip("MUSESPARK_OPEN_TOKENIZER unset: no checkpoint to build a tokenizer from")

    processor = MuseSparkOpenOmniInputProcessor(
        model_path=checkpoint, config=_tiny_vision_config(), tokenizer=None
    )
    assert processor._tokenizer is None, "must not load until something needs it"
    assert processor._encode("hello") != []
    assert processor._tokenizer is not None


# --------------------------------------------------------------------------
# The encoder wrapper is where the dtypes meet
#
# Preprocessing normalizes `(x/255 - 0.5)/0.5` in fp32 on purpose -- that is
# the reference's arithmetic -- and the engine stages `image.pixel_values` to
# the device WITHOUT changing dtype, because `multimodal_data_device_paths`
# only moves tensors. So the cast to the tower's dtype is this wrapper's job
# and nothing upstream does it. Every module-level parity harness applied the
# cast by hand before calling the tower, which is exactly why none of them saw
# a bf16 tower being handed fp32 and raising
# "expected mat1 and mat2 to have the same dtype" on every image request that
# reached a real engine.
# --------------------------------------------------------------------------
class _DtypeRecordingTower(torch.nn.Module):
    """Stands in for the tower and refuses a dtype it would not accept."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.seen = []

    def forward(self, pixel_values, grids):
        self.seen.append(pixel_values.dtype)
        if pixel_values.dtype is not self.dtype:
            raise RuntimeError(
                "expected mat1 and mat2 to have the same dtype, but got: "
                f"{pixel_values.dtype} != {self.dtype}"
            )
        return torch.zeros(int(grids.sum()), 8, dtype=self.dtype)


def _encoder_with(tower):
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenVisionEncoder,
    )

    encoder = MuseSparkOpenVisionEncoder.__new__(MuseSparkOpenVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.tower = tower
    return encoder


def test_the_encoder_casts_preprocessed_pixels_to_the_towers_dtype():
    tower = _DtypeRecordingTower(torch.bfloat16)
    encoder = _encoder_with(tower)

    params = SimpleNamespace(
        multimodal_data={
            "image": {
                # fp32, exactly as preprocess_image returns it.
                "pixel_values": torch.zeros(4, 1176, dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int32),
            }
        }
    )

    out = encoder.forward([params])
    assert tower.seen == [torch.bfloat16], f"tower was handed {tower.seen}"
    assert out.dtype is torch.bfloat16


def test_the_cast_follows_the_tower_rather_than_being_hardcoded():
    """An fp32 tower must not be handed bf16 either -- the cast is a match."""
    tower = _DtypeRecordingTower(torch.float32)
    encoder = _encoder_with(tower)

    params = SimpleNamespace(
        multimodal_data={
            "image": {
                "pixel_values": torch.zeros(4, 1176, dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int32),
            }
        }
    )
    encoder.forward([params])
    assert tower.seen == [torch.float32]


# --------------------------------------------------------------------------
# The image representations the media loaders actually produce
#
# The hashing path feeds CHW tensors to skip a ToPIL round-trip, the "np"
# media format yields HWC uint8, and direct callers pass PIL. The
# preprocessing is written against PIL, so a tensor reaching it raised
# "'Tensor' object has no attribute 'convert'" -- and because the hashing path
# CATCHES that and logs "Multimodal hashing failed:", the symptom was not a
# failed request but a served engine that answered while re-running the tower
# on every chunked-prefill chunk.
# --------------------------------------------------------------------------
def test_every_loader_representation_reaches_preprocessing_as_pil():
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_musespark_open_omni import _to_pil  # noqa: PLC0415

    width, height = 56, 28
    pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

    from_pil = _to_pil(Image.fromarray(pixels))
    from_hwc = _to_pil(pixels)
    from_chw = _to_pil(torch.from_numpy(pixels).permute(2, 0, 1))

    for got in (from_pil, from_hwc, from_chw):
        assert isinstance(got, Image.Image)
        assert got.size == (width, height), f"got {got.size}, expected {(width, height)}"
        assert got.mode == "RGB"
    # Same pixels in, same pixels out: the CHW case has to be transposed, not
    # reinterpreted, or the image silently comes out rotated.
    assert np.array_equal(np.asarray(from_chw), pixels)
    assert np.array_equal(np.asarray(from_hwc), pixels)


def test_a_float_image_is_scaled_rather_than_truncated():
    """Float loaders use 0..1; casting straight to uint8 would black the image."""
    import numpy as np  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_musespark_open_omni import _to_pil  # noqa: PLC0415

    got = np.asarray(_to_pil(torch.full((3, 8, 8), 1.0, dtype=torch.float32)))
    assert got.max() == 255, f"a fully-white float image came out at {got.max()}"


def test_an_alpha_image_keeps_its_alpha_for_the_compositor():
    """RGBA must survive to `to_rgb`, which composites it over the reference
    background; flattening it here would drop that decision on the floor."""
    import numpy as np  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_musespark_open_omni import _to_pil  # noqa: PLC0415

    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    assert _to_pil(rgba).mode == "RGBA"


def test_a_decoders_sampling_metadata_is_carried_not_dropped():
    """`frames` is not the whole video when a decoder already subsampled.

    `frames_indices` are the ORIGINAL indices of what survived, so a timestamp
    is `original_index / fps` and not `position / fps`, and `total_num_frames`
    is the source count the 2-fps target is computed against. Dropping them
    gives a decoder-subsampled video both the wrong clip timestamps and the
    wrong number of clips, and nothing downstream can tell.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        _frames_and_fps,
    )

    frames = [_pil(112, 112) for _ in range(4)]
    video_data = SimpleNamespace(
        frames=frames,
        metadata={"fps": 30.0, "frames_indices": [0, 30, 60, 90], "total_num_frames": 120},
    )

    _, fps, sampling = _frames_and_fps(video_data, {})
    assert fps == 30.0
    assert sampling == {"frame_indices": [0, 30, 60, 90], "total_num_frames": 120}


def test_the_dummy_batch_fits_the_aggregate_encoder_budget():
    """The budget is for the whole batch, not an allowance per item.

    Sizing each item at the per-item maximum overshot it by exactly the item
    count -- a budget of 256 across 4 items allocated 4 x 256 patches -- and
    the profiler reserves encoder memory from what this returns.
    """
    processor = _build_processor()
    budget = 256

    for counts in ({"image": 4}, {"image": 2, "video": 2}, {"image": 1}):
        data = processor.get_dummy_mm_data(
            max_num_encoder_tokens=budget, mm_counts=counts, dtype=torch.bfloat16
        )
        patch_items = _patch_items(data)
        assert [modality for modality, _ in patch_items] == list(counts)
        allocated = sum(item["pixel_values"].shape[0] for _, item in patch_items)
        assert allocated <= budget, f"{counts}: allocated {allocated} patches against {budget}"
        for modality, item in patch_items:
            grid = item[f"{modality}_grid_thw"]
            described = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
            assert described == item["pixel_values"].shape[0]


def test_the_model_asks_the_loaders_for_pil():
    """The framework's default is `pt` -- a float32 CHW tensor in 0..1 -- and
    this preprocessing is written against PIL. Declaring the format is the
    framework's own mechanism rather than converting on every request."""
    processor = _build_processor()
    prefs = processor.get_preferred_media_io_kwargs()
    assert prefs["image"]["format"] == "pil"
    assert prefs["video"]["format"] == "pil"


def test_the_model_declares_its_own_video_sampling():
    """The video loader defaults to `num_frames=10, fps=30` -- 10 frames per
    video regardless of length -- while this model's sampling policy is
    `sampling_fps` capped at `num_frames_per_video`. Without declaring both
    here, a bare serve boot hands the model 10 frames per video with no error
    anywhere: a silent quality collapse on video inputs."""
    prefs = _build_processor().get_preferred_media_io_kwargs()
    assert prefs["video"]["num_frames"] == 96
    assert prefs["video"]["fps"] == 2.0
    # Frame sampling is a video concept; the image loader takes neither key.
    assert set(prefs["image"]) == {"format"}


def test_video_sampling_prefs_follow_the_processor_config(tmp_path):
    """A checkpoint-declared sampling rate must reach the loader defaults, the
    same way the frame token budget reaches the row math: through
    `with_processor_config`, so the checkpoint decides its own policy."""
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"video_processor": {"fps": 1.0, "max_video_frame_tokens": 256}})
    )
    prefs = _build_processor(model_path=str(tmp_path)).get_preferred_media_io_kwargs()
    assert prefs["video"]["fps"] == 1.0
    # The frame cap is deliberately not read from processor_config.json
    # (`num_frames` is in `_PROCESSOR_KEYS_NOT_ADOPTED`); it stays at the
    # module policy.
    assert prefs["video"]["num_frames"] == 96


def test_a_null_sampling_fps_omits_the_fps_key():
    """`sampling_fps=None` is a legal preprocess-config value meaning "no
    rate-based sampling". Emitting `fps=None` would override the loader's
    built-in with a value its arithmetic cannot take; omitting the key leaves
    the frame cap as the only bound."""
    from dataclasses import replace

    from tensorrt_llm._torch.models.musespark_open_vision_processing import MuseSparkOpenPreprocessConfig

    processor = _build_processor()
    processor._vision_config = replace(MuseSparkOpenPreprocessConfig(), sampling_fps=None)
    prefs = processor.get_preferred_media_io_kwargs()
    assert "fps" not in prefs["video"]
    assert prefs["video"]["num_frames"] == 96


# --------------------------------------------------------------------------
# The class-level contracts a takeover can silently drop
#
# Registration moved the architecture string from MuseSparkOpenForCausalLM to the
# omni wrapper. Three framework hooks are looked up on the registered CLASS and
# every one of them is getattr/hasattr guarded, because "no preference" is a
# legal answer -- so a wrapper that simply does not define them is indist-
# inguishable from a model that does not care. That cost a V1 KV cache manager
# on a model whose whole long-context story is 46 windowed + 16 full-context
# layers, with no error and no warning anywhere in the boot.
# --------------------------------------------------------------------------
def _framework_class_hooks():
    """The hooks the framework asks of the registered class, by name.

    Derived from the decoder rather than hardcoded, so a fourth hook added
    upstream fails the delegation test instead of being quietly dropped.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenForCausalLM

    return sorted(
        name
        for name in dir(MuseSparkOpenForCausalLM)
        if name.startswith("get_preferred_") or name == "get_model_defaults"
    )


def test_the_wrapper_answers_every_class_hook_the_decoder_does():
    """Structural: the takeover must not narrow the decoder's class contract.

    Name-based on purpose. The values are checked below, but a hook that the
    wrapper does not define at all is the failure mode that stayed silent, so
    it gets its own assertion with its own message.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration,
    )

    hooks = _framework_class_hooks()
    assert hooks, "the decoder exposes no class hooks -- has the base class moved?"

    missing = [
        name
        for name in hooks
        if not any(
            name in klass.__dict__ for klass in MuseSparkOpenOmniForConditionalGeneration.__mro__
        )
    ]
    assert not missing, (
        f"the omni wrapper holds the architecture string but does not answer {missing}; "
        "the framework getattr-guards these, so the answer would silently be 'no preference'"
    )


def test_the_wrapper_prefers_the_same_kv_cache_manager_as_its_decoder():
    from tensorrt_llm._torch.models.modeling_musespark_open import (  # noqa: PLC0415
        MuseSparkOpenForCausalLM,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration,
    )

    assert MuseSparkOpenForCausalLM.get_preferred_kv_cache_manager_version() == "V2"
    assert (
        MuseSparkOpenOmniForConditionalGeneration.get_preferred_kv_cache_manager_version()
        == MuseSparkOpenForCausalLM.get_preferred_kv_cache_manager_version()
    )
    assert (
        MuseSparkOpenOmniForConditionalGeneration.get_preferred_transceiver_runtime()
        == MuseSparkOpenForCausalLM.get_preferred_transceiver_runtime()
    )
    assert MuseSparkOpenOmniForConditionalGeneration.get_model_defaults(
        None
    ) == MuseSparkOpenForCausalLM.get_model_defaults(None)


def test_the_resolver_builds_v2_for_the_registered_architecture():
    """The real resolver, on the class the checkpoint actually loads through.

    This is the assertion that would have caught the defect: the two tests
    above pass on the decoder alone, and the engine does not consult the
    decoder -- it consults whatever `get_registered_model_class` returns for
    the checkpoint's architecture string. `_resolve_kv_cache_manager_v2_auto`
    is invoked here the way `pyexecutor/model_loader.py:739` invokes it, with
    a stand-in llm_args, which the resolver's own docstring sanctions.
    """
    from tensorrt_llm._torch.models.modeling_utils import (  # noqa: PLC0415
        get_registered_model_class,
    )
    from tensorrt_llm.llmapi.llm_utils import _resolve_kv_cache_manager_v2_auto  # noqa: PLC0415

    model_cls = get_registered_model_class("MuseSparkOpenOmniForCausalLM")

    llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2="auto"),
        cache_transceiver_config=None,
        speculative_config=None,
    )
    assert _resolve_kv_cache_manager_v2_auto(llm_args, model_cls) is True
    assert llm_args.kv_cache_config.use_kv_cache_manager_v2 is True


# --------------------------------------------------------------------------
# A video request through a processor the engine built
#
# The engine always constructs its copy with tokenizer=None. The lazy build
# covers the text path, but the video branch read the raw attribute and handed
# None to a helper that rejects it -- so images went through and videos failed,
# on exactly the copy the server uses.
# --------------------------------------------------------------------------
def test_a_video_request_builds_the_tokenizer_it_needs(monkeypatch):
    from tensorrt_llm._torch.models import modeling_musespark_open_omni as omni  # noqa: PLC0415

    built = {}

    class _Auto:
        @staticmethod
        def from_pretrained(path, **kwargs):
            built["path"] = path
            built["kwargs"] = kwargs
            return _StubTokenizer()

    monkeypatch.setattr(omni, "AutoTokenizer", _Auto)

    processor = omni.MuseSparkOpenOmniInputProcessor(
        model_path="/nonexistent/checkpoint",
        config=_tiny_vision_config(),
        tokenizer=None,
    )
    video = SimpleNamespace(
        frames=[_pil(112, 112) for _ in range(4)],
        metadata={"fps": 2.0, "total_num_frames": 4},
    )

    ids, extra = processor.call_with_text_prompt(
        {"prompt": "describe <video>", "multi_modal_data": {"video": [video]}},
        SamplingParams(),
    )

    assert built["path"] == "/nonexistent/checkpoint"
    assert built["kwargs"]["trust_remote_code"] is True
    assert built["kwargs"]["use_fast"] is True
    assert extra["multimodal_data"]["video"]["pixel_values"].shape[0] > 0

    from tensorrt_llm._torch.models.musespark_open_vision_processing import (  # noqa: PLC0415
        VISION_TOKEN_IDS,
    )

    assert VISION_TOKEN_IDS["vid_start"] in ids
    assert VISION_TOKEN_IDS["vid_end"] in ids
    assert ids.count(VISION_TOKEN_IDS["video"]) > 0
    # The timestamp text is ordinary text, so it must have gone through a real
    # tokenizer rather than being skipped: the stub emits ids for it.
    timestamped = [i for i in ids if i not in VISION_TOKEN_IDS.values()]
    assert timestamped, "no timestamp text was tokenized into the video expansion"


def test_the_wrapper_satisfies_the_encoder_profiling_gate():
    """`_create_dummy_encoder_inputs` returns [] for a non-mixin model.

    That gate is the only door to startup encoder profiling, so failing it
    silently hands the tower's whole activation footprint to the KV cache. It
    is an isinstance test, which no amount of duck typing satisfies.
    """
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (  # noqa: PLC0415
        MultimodalModelMixin,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration,
    )

    assert issubclass(MuseSparkOpenOmniForConditionalGeneration, MultimodalModelMixin)
    # The profiler calls this one directly (`pyexecutor/_util.py`), so the
    # mixin's `raise NotImplementedError` body would turn the gate into a boot
    # failure rather than a fix.
    assert (
        MuseSparkOpenOmniForConditionalGeneration.encode_multimodal_inputs
        is not MultimodalModelMixin.encode_multimodal_inputs
    )


def test_the_wrapper_uses_the_mixins_forward():
    """The wrapper defines no `forward` of its own; the mixin's is the one that runs.

    The inverse of the pin this test used to carry. Adopting the mixin's forward
    is merge-neutral only because both paths reach the same `fuse_input_embeds`
    with the same two arguments, so what has to be pinned now is that the class
    really does inherit it -- an edit that reintroduced a local `forward` would
    fork the merge again, silently.
    """
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (  # noqa: PLC0415
        MultimodalModelMixin,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration,
    )

    assert "forward" not in vars(MuseSparkOpenOmniForConditionalGeneration)
    assert MuseSparkOpenOmniForConditionalGeneration.forward is MultimodalModelMixin.forward


def test_the_mixin_sits_ahead_of_pretrainedmodel_in_the_mro():
    """Resolution has to find the mixin first, or none of this runs.

    `PreTrainedModel` also defines `forward`. Ordered the other way round the
    class would inherit that one, and the failure would surface as a
    NotImplementedError deep inside a serve rather than as an import error.
    """
    from transformers import PreTrainedModel  # noqa: PLC0415

    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (  # noqa: PLC0415
        MultimodalModelMixin,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration as Omni,
    )

    mro = Omni.__mro__
    assert mro.index(MultimodalModelMixin) < mro.index(PreTrainedModel)


def test_the_merge_hooks_are_the_raw_embedding_and_the_embed_carrying_ids():
    """The two invariants that make the mixin's merge identical to the old one.

    `fuse_input_embeds` with non-None `mm_token_ids` embeds the whole prompt
    through `embedding_layer` and then overwrites the media rows. So the merge
    is unchanged exactly as long as (i) `text_embedding_layer` is the decoder's
    RAW `Embedding` -- handing it `embed_norm` would norm the text twice, which
    is not a no-op in bf16 at small row scale -- and (ii)
    `multimodal_token_ids` names exactly the ids that carry encoder rows, in
    the order the processor publishes them.
    """
    from tensorrt_llm._torch.models.musespark_open_audio import AUDIO_TOKEN_IDS  # noqa: PLC0415
    from tensorrt_llm._torch.models.musespark_open_vision_processing import (  # noqa: PLC0415
        VISION_TOKEN_IDS,
    )

    llm = _RecordingLLM()
    llm.model.embed_norm = object()  # what must NOT be returned
    wrapper = _wrapper_with(llm)

    assert wrapper.text_embedding_layer is llm.model.embed_tokens
    assert isinstance(wrapper.text_embedding_layer, Embedding)
    assert wrapper.text_embedding_layer is not llm.model.embed_norm

    ids = wrapper.multimodal_token_ids
    assert ids.tolist() == [
        VISION_TOKEN_IDS["patch"],
        VISION_TOKEN_IDS["video"],
        AUDIO_TOKEN_IDS["audio"],
    ]
    # The old spelling stays as an alias for one release: the executor prefers
    # the new name and falls back to this one.
    assert wrapper.mm_token_ids.tolist() == ids.tolist()

    assert wrapper.embedding_dim == llm.model.embed_tokens.embedding_dim
    assert wrapper.embedding_dtype == llm.model.embed_tokens.weight.dtype


def test_the_wrapper_no_longer_repeats_the_decoders_size_hooks():
    """`vocab_size_padded` / `infer_max_seq_len` come from the mixin, not a copy.

    Both delegate to `language_model`, which is the decoder, so a local copy
    would be a second definition of the same answer -- one more thing to keep in
    sync for no behaviour.
    """
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import (  # noqa: PLC0415
        MultimodalModelMixin,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration as Omni,
    )

    assert "vocab_size_padded" not in vars(Omni)
    assert "infer_max_seq_len" not in vars(Omni)
    assert Omni.infer_max_seq_len is MultimodalModelMixin.infer_max_seq_len


def test_spec_metadata_and_lora_params_reach_the_decoder():
    """The wrapper used to eat both, silently.

    The mixin's default `get_language_model_extra_forward_kwargs` returns `{}`
    and the wrapper's old `forward` passed five arguments only, so neither
    speculative decoding nor LoRA could reach the decoder through this class --
    and `DecoderModelForCausalLM.forward` defaults both to None rather than
    complaining, so nothing would have said so.
    """
    llm = _RecordingLLM()
    wrapper = _wrapper_with(llm)

    spec = object()
    lora = {"whatever": 1}
    extra = wrapper.get_language_model_extra_forward_kwargs(
        raw_input_ids=None,
        position_ids=None,
        mm_inputs=None,
        spec_metadata=spec,
        lora_params=lora,
        text_token_indices=None,
    )

    assert extra == {"spec_metadata": spec, "lora_params": lora}

    # And end to end through the mixin's forward: the decoder stub records them.
    wrapper.forward(
        attn_metadata=_attn(num_contexts=0, num_generations=1),
        input_ids=torch.arange(3, dtype=torch.int32),
        position_ids=torch.arange(3, dtype=torch.int32),
        spec_metadata=spec,
        lora_params=lora,
    )
    call = llm.calls[0]
    assert call["spec_metadata"] is spec
    assert call["lora_params"] is lora


def test_the_mixins_opt_in_features_stay_off():
    """Encoder cache and item scheduling are separate decisions, not free riders.

    Both are ClassVars defaulting to False, and both bring contracts this model
    does not implement (`build_multimodal_encoder_input`, an item-metadata
    provider). Turning either on by accident would engage a code path with no
    gate behind it.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenOmniForConditionalGeneration as Omni,
    )

    assert Omni.supports_encoder_cache is False
    assert Omni.supports_mm_encoder_item_scheduling is False


# -- encoder row order ------------------------------------------------------
#
# The rows the encoder returns are consumed POSITIONALLY: `fuse_input_embeds`
# writes them into the prompt's `mm_token_ids` positions from left to right.
# So the encoder owes prompt order, and it used to owe it only by accident --
# it iterated ("image", "video") per request, which is prompt order exactly
# while a request carries one modality and wrong the moment a prompt writes
# <video> before <image>. Nothing raised: the video's placeholder run received
# the image's rows, and the answer described the wrong medium.


class _OrderRecordingTower(torch.nn.Module):
    """Stands in for the NaViT tower and reports which item each row came from.

    Emits one row per grid entry holding that item's fill value, so the vector
    it returns *is* the order the encoder packed the items in. Building the
    real tower would cost 50 layers to observe a list order.
    """

    dtype = torch.float32

    def forward(self, pixel_values, grid_thw, **kwargs):
        rows, start = [], 0
        for row in grid_thw:
            patches = int(row[0]) * int(row[1]) * int(row[2])
            rows.append(pixel_values[start].clone())
            start += patches
        assert start == pixel_values.shape[0], "tower stub was handed a ragged batch"
        return torch.stack(rows)


def _encoder_with_stub_tower():
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        MuseSparkOpenVisionEncoder,
    )

    encoder = MuseSparkOpenVisionEncoder.__new__(MuseSparkOpenVisionEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.tower = _OrderRecordingTower()
    return encoder


def _modality_data(modality, fills, side=2, patch_dim=3):
    """One modality's packed tensors, item *i* filled with ``fills[i]``."""
    patches = side * side
    return {
        "pixel_values": torch.cat(
            [torch.full((patches, patch_dim), float(fill)) for fill in fills]
        ),
        f"{modality}_grid_thw": torch.tensor([[1, side, side]] * len(fills), dtype=torch.int32),
    }


def _params(data, item_order=None):
    from tensorrt_llm.inputs.multimodal import MultimodalParams  # noqa: PLC0415

    return MultimodalParams(multimodal_data=data, mm_item_order=item_order)


def _encoded_order(params_list):
    encoder = _encoder_with_stub_tower()
    return encoder.forward(params_list)[:, 0].tolist()


def test_a_mixed_request_is_encoded_in_prompt_order_not_modality_order():
    """`<video><image>` must encode the video first. The bug this pins.

    Both media are attached to one request, the manifest says video came first,
    and the returned rows have to agree with the manifest and not with the
    order the modalities happen to be iterated in.
    """
    params = _params(
        {
            "image": _modality_data("image", [1.0]),
            "video": _modality_data("video", [2.0]),
        },
        item_order=[{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
    )
    assert _encoded_order([params]) == [2.0, 1.0]


def test_an_interleaved_request_keeps_every_item_where_the_prompt_put_it():
    """`<image><video><image>` -- the modality-order iteration cannot express it.

    Two items of one modality straddling an item of the other is the case a
    single "which modality first" decision gets wrong no matter which way it
    is made, so it is the one worth pinning.
    """
    params = _params(
        {
            "image": _modality_data("image", [1.0, 3.0]),
            "video": _modality_data("video", [2.0]),
        },
        item_order=[
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
            {"modality": "image", "index": 1},
        ],
    )
    assert _encoded_order([params]) == [1.0, 2.0, 3.0]


def test_requests_stay_in_batch_order_and_items_in_prompt_order_within_them():
    """The output is split back per request by row count, so both must hold.

    `_store_chunked_prefill_embeddings` slices this one tensor request by
    request; if the batch order slipped, every request would receive another
    request's rows.
    """
    first = _params(
        {"video": _modality_data("video", [1.0]), "image": _modality_data("image", [2.0])},
        item_order=[{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
    )
    second = _params({"image": _modality_data("image", [3.0, 4.0])})
    assert _encoded_order([first, second]) == [1.0, 2.0, 3.0, 4.0]


def test_a_single_modality_request_needs_no_manifest():
    """Within one modality, item order already is prompt order.

    Raw-prompt entrypoints attach no manifest, and the startup KV-cache
    profiler builds its dummy batch that way too, so requiring one everywhere
    would break both.
    """
    params = _params({"image": _modality_data("image", [1.0, 2.0, 3.0])})
    assert _encoded_order([params]) == [1.0, 2.0, 3.0]


def test_a_mixed_request_without_a_manifest_is_refused_not_guessed():
    """Silently guessing is what produced the wrong answers in the first place."""
    params = _params(
        {"image": _modality_data("image", [1.0]), "video": _modality_data("video", [2.0])}
    )
    with pytest.raises(ValueError, match="mm_item_order"):
        _encoded_order([params])


def test_a_manifest_that_does_not_cover_every_item_is_refused():
    """A short manifest would drop rows and misalign every placeholder after it."""
    params = _params(
        {"image": _modality_data("image", [1.0, 3.0]), "video": _modality_data("video", [2.0])},
        item_order=[{"modality": "image", "index": 0}, {"modality": "video", "index": 0}],
    )
    with pytest.raises(ValueError, match="every item must appear exactly once"):
        _encoded_order([params])


def test_a_grid_that_disagrees_with_the_pixels_says_so():
    """The split is read off the grid, so a mismatch has to be an error here.

    Left unchecked it silently hands the tower a short last item and shifts
    every row after it.
    """
    data = _modality_data("image", [1.0, 2.0])
    data["pixel_values"] = data["pixel_values"][:-1]
    with pytest.raises(ValueError, match="disagree on geometry"):
        _encoded_order([_params({"image": data})])


def test_an_empty_batch_still_returns_a_tensor():
    """The profiler's no-media path and generation-phase batches both land here."""
    encoded = _encoder_with_stub_tower().forward([])
    assert isinstance(encoded, torch.Tensor)
    assert encoded.numel() == 0


# -- where prompt order comes from -----------------------------------------
#
# The encoder needs a prompt-order manifest. `MultimodalParams.mm_item_order`
# is the framework's field for it and the chat front end fills it in correctly,
# but on the served path it did not reach the tower: a real image+video request
# arrived with the field empty and the encoder refused it, which -- because the
# refusal is raised inside `forward` -- took the whole executor down rather than
# failing that one request. So the processor now writes the order into
# `multimodal_data`, the same dict as the pixels, and the encoder prefers it.


def test_the_processor_records_the_order_it_spliced():
    """The splice loop walks the TEXT, so the order it consumes IS prompt order."""
    processor = _build_processor()
    _, order = processor._splice_placeholders(
        "a <image> b <video> c <image> d",
        [("image", [10]), ("image", [11]), ("video", [20])],
    )
    assert order == [
        {"modality": "image", "index": 0},
        {"modality": "video", "index": 0},
        {"modality": "image", "index": 1},
    ]


def test_the_manifest_indexes_each_modality_separately():
    """`index` is per-modality, because that is how the payloads are packed."""
    processor = _build_processor()
    _, order = processor._splice_placeholders(
        "<video><video><image>",
        [("image", [10]), ("video", [20]), ("video", [21])],
    )
    assert order == [
        {"modality": "video", "index": 0},
        {"modality": "video", "index": 1},
        {"modality": "image", "index": 0},
    ]


def test_a_prompt_with_no_media_records_an_empty_manifest():
    processor = _build_processor()
    fused, order = processor._splice_placeholders("nothing to see", [])
    assert order == []
    assert fused == processor._encode("nothing to see")


def test_the_encoder_reads_the_manifest_the_processor_attached():
    """The regression test for the crash: no framework field, and it still works.

    `MultimodalParams.mm_item_order` is left None here on purpose -- that is
    exactly the state a served mixed-modality request arrived in.
    """
    data = {
        "image": _modality_data("image", [1.0]),
        "video": _modality_data("video", [2.0]),
        "mm_item_order": [
            {"modality": "video", "index": 0},
            {"modality": "image", "index": 0},
        ],
    }
    params = _params(data)
    assert getattr(params, "mm_item_order", None) is None
    assert _encoded_order([params]) == [2.0, 1.0]


def test_the_attached_manifest_wins_over_the_framework_field():
    """Ours is the order the placeholders were actually spliced in.

    The framework's field is the order the chat handler decoded the media in.
    They normally agree; when they cannot both be honoured, the token stream is
    what the encoder rows have to line up with.
    """
    data = {
        "image": _modality_data("image", [1.0]),
        "video": _modality_data("video", [2.0]),
        "mm_item_order": [
            {"modality": "video", "index": 0},
            {"modality": "image", "index": 0},
        ],
    }
    params = _params(
        data,
        item_order=[{"modality": "image", "index": 0}, {"modality": "video", "index": 0}],
    )
    assert _encoded_order([params]) == [2.0, 1.0]


def test_the_framework_field_still_works_on_its_own():
    """Entrypoints that do not go through our processor keep working."""
    params = _params(
        {"image": _modality_data("image", [1.0]), "video": _modality_data("video", [2.0])},
        item_order=[{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
    )
    assert _encoded_order([params]) == [2.0, 1.0]


def test_the_manifest_key_is_not_mistaken_for_a_modality():
    """`mm_item_order` sits beside `image`/`video` in the same dict.

    Anything that walks `multimodal_data` looking for media must not pick it up;
    the encoder iterates a fixed modality list rather than the dict's keys.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (  # noqa: PLC0415
        _VISION_MODALITIES,
    )

    assert "mm_item_order" not in {modality for modality, _ in _VISION_MODALITIES}
    params = _params(
        {"image": _modality_data("image", [1.0, 2.0]), "mm_item_order": []},
    )
    # An empty manifest with one modality present falls back to item order.
    assert _encoded_order([params]) == [1.0, 2.0]
