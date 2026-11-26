# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Precomputed image / video embeddings for the Qwen-VL family (CPU)."""

import re
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5VLInputProcessorBase,
    Qwen2VLModelBase,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase, Qwen3VLModelBase
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.inputs.external_embeddings import ExternalEmbeddingItem
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_VISION_START = 151652
_VISION_END = 151653
_IMAGE_PAD = 151655
_VIDEO_PAD = 151656
_SPECIAL_IDS = {
    "<|vision_start|>": _VISION_START,
    "<|vision_end|>": _VISION_END,
    "<|image_pad|>": _IMAGE_PAD,
    "<|video_pad|>": _VIDEO_PAD,
}
_HIDDEN = 8


class _FakeTokenizer:
    """Maps special tokens to their ids, each timestamp to one id, other text to 7."""

    _PIECE_RE = re.compile(r"<\|[a-z_]+\|>|<[0-9.]+ seconds>|[^<]+")

    def __call__(self, text: str, return_tensors: str = None) -> SimpleNamespace:
        """Tokenize ``text`` into a [1, n] id tensor."""
        ids = [self.piece_id(piece) for piece in self._PIECE_RE.findall(text)]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def convert_tokens_to_ids(self, token: str) -> int:
        """Id of one special token."""
        return _SPECIAL_IDS[token]

    def decode(self, ids: list) -> str:
        """Inverse of the special-token mapping, enough for token-id prompts."""
        inverse = {v: k for k, v in _SPECIAL_IDS.items()}
        return "".join(inverse.get(i, "x") for i in ids)

    @staticmethod
    def piece_id(piece: str) -> int:
        """Id of one piece."""
        if piece in _SPECIAL_IDS:
            return _SPECIAL_IDS[piece]
        match = re.fullmatch(r"<([0-9.]+) seconds>", piece)
        if match:
            return 1000 + round(float(match.group(1)) * 10)
        return 7


def _make_processor(
    cls: type, deepstack_levels: int = 0, tokens_per_second: Optional[float] = None
) -> Any:
    """Bare processor with a minimal Qwen config and the fake tokenizer."""
    processor = object.__new__(cls)
    vision_config = SimpleNamespace(
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=list(range(deepstack_levels)),
    )
    if tokens_per_second is not None:
        vision_config.tokens_per_second = tokens_per_second
    processor._config = SimpleNamespace(
        image_token_id=_IMAGE_PAD,
        video_token_id=_VIDEO_PAD,
        vision_start_token_id=_VISION_START,
        vision_end_token_id=_VISION_END,
        text_config=SimpleNamespace(
            hidden_size=_HIDDEN, vocab_size=200000, max_position_embeddings=128000
        ),
        vision_config=vision_config,
    )
    processor._processor = SimpleNamespace()
    processor._tokenizer = _FakeTokenizer()
    return processor


def _attach(
    processor: Any, prompt: Any, embeddings: dict, **extra: Any
) -> tuple[list[int], MultimodalParams]:
    """Run the public attach hook and wrap the result as executor params."""
    tokens, extra_inputs = processor.attach_multimodal_embeddings(
        {"prompt": prompt, **extra}, embeddings, SamplingParams()
    )
    return tokens, MultimodalParams(**extra_inputs)


def _video(t: int, h: int, w: int, width: int = _HIDDEN, seed: int = 0, **meta: Any) -> dict:
    rows = t * (h // 2) * (w // 2)
    generator = torch.Generator().manual_seed(seed)
    return {
        "mm_embeddings": torch.randn(rows, width, generator=generator),
        "video_grid_thw": [t, h, w],
        **meta,
    }


def test_qwen3_video_prompt_has_per_frame_timestamps_and_runs() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    video = _video(2, 4, 2, timestamps=[0.25, 1.25])
    tokens, params = _attach(
        processor, "a<|vision_start|><|video_pad|><|vision_end|>b", {"video": [video]}
    )

    frame = [_VISION_START, _VIDEO_PAD, _VIDEO_PAD, _VISION_END]
    # HF rounds timestamps with "{:.1f}".
    assert tokens == [7, 1002, *frame, 1012, *frame, 7]
    mm_input = params.multimodal_input
    assert mm_input.multimodal_positions == [3]
    assert mm_input.multimodal_lengths == [4]
    assert mm_input.multimodal_run_positions == [3, 8]
    assert mm_input.multimodal_run_lengths == [2, 2]
    assert params.get_item_layout() == ((("video", 0),), (4,))
    assert params.multimodal_data["multimodal_embedding"][0] is video["mm_embeddings"]
    # Qwen3 MRoPE: each frame is a (1, h=2, w=1) block placed after its timestamp
    # and vision-start text tokens; text resumes at the block maximum + 1.
    expected = [
        [0, 1, 2, 3, 3, 5, 6, 7, 8, 8, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 3, 5, 6, 7, 8, 8, 10, 11],
    ]
    mrope = params.multimodal_data["mrope_config"]
    assert mrope["mrope_position_ids"][:, 0, :].tolist() == expected
    assert mrope["mrope_position_deltas"].tolist() == [[0]]


def test_qwen3_timestamps_from_fps_and_frame_indices_match_hf() -> None:
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    processor = _make_processor(Qwen3VLInputProcessorBase)
    indices = [0, 7, 15, 22, 30]  # odd count: HF pads with the last index
    hf = Qwen3VLProcessor._calculate_timestamps(None, list(indices), 29.97, 2)
    item = ExternalEmbeddingItem(
        "video", torch.zeros(3, _HIDDEN), (3, 2, 2), fps=29.97, frames_indices=tuple(indices)
    )
    assert processor._resolve_external_video_metadata(item)["timestamps"] == hf


def test_mixed_image_and_video_follow_prompt_order() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase, deepstack_levels=1)
    width = 2 * _HIDDEN
    image = {"mm_embeddings": torch.ones(2, width), "image_grid_thw": [1, 2, 4]}
    video = _video(1, 2, 2, width=width, timestamps=[0.0])
    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>x<|vision_start|><|image_pad|><|vision_end|>"
    )
    tokens, params = _attach(processor, prompt, {"image": [image], "video": [video]})

    assert tokens == [
        1000,
        _VISION_START,
        _VIDEO_PAD,
        _VISION_END,
        7,
        _VISION_START,
        _IMAGE_PAD,
        _IMAGE_PAD,
        _VISION_END,
    ]
    assert params.get_item_layout() == ((("video", 0), ("image", 0)), (1, 2))
    assert params.multimodal_input.multimodal_positions == [2, 6]
    embeddings = params.multimodal_data["multimodal_embedding"]
    assert embeddings[0] is video["mm_embeddings"] and embeddings[1] is image["mm_embeddings"]


def test_mixed_bare_and_framed_video_placeholders_are_rejected() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    videos = [_video(1, 2, 2, timestamps=[0.0]), _video(1, 2, 2, timestamps=[1.0])]
    with pytest.raises(ValueError, match="framed"):
        _attach(
            processor,
            "<|video_pad|><|vision_start|><|video_pad|><|vision_end|>",
            {"video": videos},
        )


def test_non_square_image_grid_drives_mrope() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    image = {"mm_embeddings": torch.ones(2, _HIDDEN), "image_grid_thw": [1, 2, 4]}
    tokens, params = _attach(
        processor, "<|vision_start|><|image_pad|><|vision_end|>", {"image": [image]}
    )
    mrope = params.multimodal_data["mrope_config"]["mrope_position_ids"]
    # One row, two columns: height index constant, width index increments.
    assert mrope[:, 0, 1:3].tolist() == [[1, 1], [1, 1], [1, 2]]


def test_qwen3_5_config_takes_hidden_size_rows() -> None:
    """Qwen3.5 vision configs have no deepstack levels, so rows are hidden-size wide."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    config = Qwen3_5Config(
        text_config={"hidden_size": _HIDDEN, "vocab_size": 200000},
        vision_config={"out_hidden_size": _HIDDEN},
        image_token_id=_IMAGE_PAD,
        video_token_id=_VIDEO_PAD,
        vision_start_token_id=_VISION_START,
        vision_end_token_id=_VISION_END,
    )
    assert getattr(config.vision_config, "deepstack_visual_indexes", None) is None
    processor = _make_processor(Qwen3VLInputProcessorBase)
    processor._config = config
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    image = {"mm_embeddings": torch.ones(2, _HIDDEN), "image_grid_thw": [1, 2, 4]}
    tokens, _ = _attach(processor, prompt, {"image": [image]})
    assert tokens.count(_IMAGE_PAD) == 2
    wide = {"mm_embeddings": torch.ones(2, 2 * _HIDDEN), "image_grid_thw": [1, 2, 4]}
    with pytest.raises(ValueError, match="width"):
        _attach(processor, prompt, {"image": [wide]})


def test_bare_image_tensor_needs_a_grid() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with pytest.raises(ValueError, match="requires 'image_grid_thw'"):
        _attach(processor, "<|image_pad|>", {"image": torch.ones(4, _HIDDEN)})


def test_qwen2_5_video_scales_time_by_second_per_grid() -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    video = _video(2, 2, 2, second_per_grid_ts=2.0)
    tokens, params = _attach(
        processor, "<|vision_start|><|video_pad|><|vision_end|>", {"video": [video]}
    )
    assert tokens == [_VISION_START, _VIDEO_PAD, _VIDEO_PAD, _VISION_END]
    mrope = params.multimodal_data["mrope_config"]["mrope_position_ids"]
    # Temporal index advances by second_per_grid * tokens_per_second = 4 per patch.
    assert mrope[0, 0, 1:3].tolist() == [1, 5]


@pytest.mark.parametrize(
    "timing, match",
    [
        ({}, "requires 'second_per_grid_ts'"),
        # VideoData's fps is the source rate; Qwen2.5 needs the sampled one.
        ({"fps": 24.0}, "not 'timestamps' / 'fps'"),
        ({"fps": 2.0, "second_per_grid_ts": 1.0}, "not 'timestamps' / 'fps'"),
        ({"timestamps": [0.0, 1.0]}, "not 'timestamps'"),
    ],
)
def test_qwen2_5_video_requires_second_per_grid_ts(timing: dict, match: str) -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match=match):
        _attach(processor, "<|video_pad|>", {"video": [_video(2, 2, 2, **timing)]})


def _attach_framed_video(processor: Any, video: dict) -> tuple:
    """Attach one framed video."""
    return processor.attach_multimodal_embeddings(
        {"prompt": "<|vision_start|><|video_pad|><|vision_end|>"},
        {"video": [video]},
        SamplingParams(),
    )


@pytest.mark.parametrize(
    "timing, match",
    [
        ({"second_per_grid_ts": 1e9}, "do not fit int32"),
        ({"second_per_grid_ts": 1e300}, "overflows float32"),
    ],
)
def test_qwen2_5_video_positions_past_int32_are_rejected(timing: dict, match: str) -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match=match):
        _attach_framed_video(processor, _video(8, 2, 2, **timing))


def _admitted(
    mrope_config: dict,
    prompt_len: int,
    max_new_tokens: int,
    draft_tokens: int = 0,
    max_positions: int = 128000,
    generation_only: bool = False,
) -> LlmRequest:
    """A request after the Qwen-VL model's admission hook."""
    request = LlmRequest(
        request_id=1,
        max_new_tokens=max_new_tokens,
        input_tokens=[0] * prompt_len,
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
        py_multimodal_data={"mrope_config": mrope_config},
        llm_request_type=(
            LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
            if generation_only
            else LlmRequestType.LLMREQUEST_TYPE_CONTEXT_AND_GENERATION
        ),
    )
    pretrained_config = _make_processor(Qwen2_5VLInputProcessorBase).config
    pretrained_config.text_config.max_position_embeddings = max_positions
    spec_config = SimpleNamespace(max_total_draft_tokens=draft_tokens) if draft_tokens else None
    model = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=pretrained_config, spec_config=spec_config)
    )
    Qwen2VLModelBase.validate_request(model, request)
    return request


def test_qwen2_5_video_decoding_is_capped_at_admission() -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    # The last video position is 1 + 7 * 9141 * 2 = 127975, below 128000.
    tokens, extra = _attach_framed_video(processor, _video(8, 2, 2, second_per_grid_ts=9141.0))
    mrope_config = extra["multimodal_data"]["mrope_config"]
    capped = _admitted(mrope_config, len(tokens), 128000 - len(tokens))
    assert capped.max_new_tokens == capped.py_max_new_tokens == 24
    assert _admitted(mrope_config, len(tokens), 5).py_max_new_tokens == 5
    # Positions that trail the token count leave max_seq_len as the only limit.
    trailing = {"mrope_position_deltas": torch.tensor([[-50]], dtype=torch.int32)}
    assert _admitted(trailing, 10, 10**6).max_new_tokens == 10**6


def test_qwen2_5_video_prompt_past_rotary_table_is_rejected_at_admission() -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    # Preprocessing accepts it (positions fit int32); the engine's table bounds it.
    tokens, extra = _attach_framed_video(processor, _video(8, 2, 2, second_per_grid_ts=1e5))
    mrope_config = extra["multimodal_data"]["mrope_config"]
    with pytest.raises(ValueError, match="past the model's 128000 max_position_embeddings"):
        _admitted(mrope_config, len(tokens), 1)
    # A larger table, as model_kwargs may set on the engine, admits it.
    assert _admitted(mrope_config, len(tokens), 1, max_positions=2 * 10**6).max_new_tokens == 1


def test_mrope_decode_cap_counts_forwarded_positions() -> None:
    # Prompt positions 0..99 shifted by 32667 end at 32766; the last sampled token is not
    # forwarded, so two new tokens write only position 32767.
    deltas = {"mrope_position_deltas": torch.tensor([[32667]], dtype=torch.int32)}
    assert _admitted(deltas, 100, 2, max_positions=32768).max_new_tokens == 2
    assert _admitted(deltas, 100, 3, max_positions=32768).max_new_tokens == 2
    # A speculative step forwards its draft tokens past the last accepted one.
    assert _admitted(deltas, 100, 10, draft_tokens=1, max_positions=32768).max_new_tokens == 1
    # The prompt forward samples the first token without forwarding drafts.
    assert _admitted(deltas, 100, 10, draft_tokens=2, max_positions=32768).max_new_tokens == 1
    # A generation-only request forwards its first token and drafts first.
    assert _admitted(deltas, 100, 10, max_positions=32768, generation_only=True).max_new_tokens == 2
    with pytest.raises(ValueError, match="1 draft tokens starts past"):
        _admitted(deltas, 100, 10, draft_tokens=1, max_positions=32768, generation_only=True)
    # The prompt's last position, 100 - 1 + 32669, is past the table.
    past = {"mrope_position_deltas": torch.tensor([[32669]], dtype=torch.int32)}
    with pytest.raises(ValueError, match="last mRoPE position 32768"):
        _admitted(past, 100, 1, max_positions=32768)


def test_qwen2_5_video_without_max_tokens_is_capped_not_rejected() -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    # trtllm-serve leaves max_tokens unset; the worker then fills max_seq_len - prompt_len.
    tokens, extra = processor.attach_multimodal_embeddings(
        {"prompt": "<|vision_start|><|video_pad|><|vision_end|>"},
        {"video": [_video(8, 2, 2, second_per_grid_ts=9141.0)]},
        SamplingParams(max_tokens=None),
    )
    request = _admitted(extra["multimodal_data"]["mrope_config"], len(tokens), 4096 - len(tokens))
    assert request.max_new_tokens == request.py_max_new_tokens == 24


@pytest.mark.parametrize("timestamps", [[0.0, 1e308], [-1.0, 0.0]])
def test_qwen3_out_of_range_timestamps_are_rejected(timestamps: list[float]) -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with pytest.raises(ValueError, match=r"\[0, 1e6\] seconds"):
        _attach_framed_video(processor, _video(2, 4, 2, timestamps=timestamps))


def test_qwen2_5_bare_video_placeholder_is_rejected() -> None:
    processor = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)
    with pytest.raises(ValueError, match="framed"):
        _attach(processor, "<|video_pad|>", {"video": [_video(2, 2, 2, second_per_grid_ts=1.0)]})


@pytest.mark.parametrize(
    "embeddings, prompt, match",
    [
        ({"video": [_video(2, 4, 2)]}, "<|video_pad|>", "timestamps"),
        ({"video": [_video(2, 4, 2, timestamps=[0.0])]}, "<|video_pad|>", "expected 2 timestamps"),
        (
            {"video": [{"mm_embeddings": torch.ones(4, _HIDDEN), "video_grid_thw": [2, 4, 3]}]},
            "<|video_pad|>",
            "multiples of the spatial merge size",
        ),
        (
            {"video": [{"mm_embeddings": torch.ones(3, _HIDDEN), "video_grid_thw": [2, 4, 2]}]},
            "<|video_pad|>",
            "has 3 rows but its grid",
        ),
        ({"video": [{"mm_embeddings": torch.ones(4, _HIDDEN)}]}, "<|video_pad|>", "video_grid_thw"),
        ({"image": [torch.ones(4, _HIDDEN + 1)]}, "<|image_pad|>", "width"),
        (
            {"image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]},
            "no placeholder",
            "0 image placeholders",
        ),
        (
            {"image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]},
            "<|vision_start|><|image_pad|><|vision_end|>" * 2,
            "more image",
        ),
        (
            {"image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]},
            "<|image_pad|>",
            "needs image placeholders framed",
        ),
        ({"image": None}, "<|image_pad|>", "must be a tensor, dict, or a list of them"),
        (
            {"image": [torch.zeros(1, _HIDDEN).expand(64, _HIDDEN)]},
            "<|image_pad|>",
            "self-overlapping view",
        ),
        ({"image": [torch.empty(4, _HIDDEN, device="meta")]}, "<|image_pad|>", "cpu or cuda"),
        (
            {
                "image": [
                    {
                        "mm_embeddings": torch.ones(4, _HIDDEN),
                        "image_grid_thw": torch.ones(10**6, dtype=torch.long),
                    }
                ]
            },
            "<|image_pad|>",
            "image_grid_thw has 1000000 entries, at most 3",
        ),
        (
            {"video": [_video(2, 4, 2, timestamps=torch.tensor([0.25, 0.45]))]},
            "<|video_pad|>",
            "timestamps must be float64 or integer, got torch.float32",
        ),
        (
            {"video": [_video(1, 2, 2, fps=1.0, frames_indices=list(range(10**6)))]},
            "<|video_pad|>",
            "frames_indices has 1000000 entries for 1 temporal patches",
        ),
        (
            {"video": [_video(1, 2, 2, timestamps=[0.0], fps=None)]},
            "<|video_pad|>",
            r"omit \['fps'\] instead of passing None",
        ),
        ({"image": [torch.ones(4, _HIDDEN, dtype=torch.int32)]}, "<|image_pad|>", "bfloat16"),
        (
            {"image": [torch.ones(4, _HIDDEN).to(torch.float8_e4m3fn)]},
            "<|image_pad|>",
            "bfloat16",
        ),
        ({"audio": [torch.ones(4, _HIDDEN)]}, "<|image_pad|>", "Unsupported modalities"),
        (
            {"image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "timestamps": [0.0]}]},
            "<|image_pad|>",
            "only applies to video",
        ),
    ],
)
def test_invalid_embeddings_are_rejected(embeddings: dict, prompt: str, match: str) -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    with pytest.raises((ValueError, TypeError), match=match):
        _attach(processor, prompt, embeddings)


def _hashes(
    processor: Any, prompt: Any, embeddings: dict, **extra: Any
) -> tuple[list[int], list[list[int]]]:
    """Prompt tokens and per-item cache keys of one attached request."""
    tokens, params = _attach(processor, prompt, embeddings, **extra)
    return tokens, params.multimodal_input.multimodal_hashes


def test_position_metadata_joins_the_cache_key() -> None:
    """Same rows and prompt tokens but different video timing never share KV."""
    rows = torch.ones(4, _HIDDEN)
    qwen3 = _make_processor(Qwen3VLInputProcessorBase)
    video = "<|vision_start|><|video_pad|><|vision_end|>"
    qwen2_5 = _make_processor(Qwen2_5VLInputProcessorBase, tokens_per_second=2)

    def clip(**timing: float | list[float]) -> dict:
        return {"video": [{"mm_embeddings": rows, "video_grid_thw": [2, 2, 4], **timing}]}

    slow = _hashes(qwen2_5, video, clip(second_per_grid_ts=2.0))
    fast = _hashes(qwen2_5, video, clip(second_per_grid_ts=1.0))
    assert slow[0] == fast[0]
    assert slow[1] != fast[1]

    # Qwen3 prints frame timestamps into the prompt, so they change the tokens.
    early = _hashes(qwen3, video, clip(timestamps=[0.0, 1.0]))
    late = _hashes(qwen3, video, clip(timestamps=[0.0, 2.0]))
    assert early[0] != late[0]


def test_float64_and_integer_timestamp_tensors_are_accepted() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    for timestamps in (torch.tensor([0.25, 1.25], dtype=torch.float64), torch.tensor([0, 1])):
        tokens, _ = _attach_framed_video(processor, _video(2, 4, 2, timestamps=timestamps))
        assert tokens[0] == _FakeTokenizer.piece_id(f"<{float(timestamps[0]):.1f} seconds>")


def test_rows_past_the_rotary_table_are_rejected_before_expansion() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    processor.config.text_config.max_position_embeddings = 6
    image = {"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}
    with pytest.raises(ValueError, match="8 rows in total, more than the model's 6 positions"):
        _attach(
            processor, "<|vision_start|><|image_pad|><|vision_end|>" * 2, {"image": [image] * 2}
        )


def test_embeddings_without_a_local_encoder_or_handoff_are_rejected_at_admission() -> None:
    model = SimpleNamespace(mm_encoder=None)
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        Qwen3VLModelBase.validate_multimodal_request_data(
            model, {"multimodal_embedding": [torch.ones(4, _HIDDEN)]}
        )
    # Models that take encoder handoffs accept them; raw media is untouched.
    Qwen3VLModelBase.validate_multimodal_request_data(
        SimpleNamespace(mm_encoder=None, support_mm_disagg=True),
        {"multimodal_embedding": [torch.ones(4, _HIDDEN)]},
    )
    Qwen3VLModelBase.validate_multimodal_request_data(model, {"image": {}})


def test_uuids_of_absent_modalities_are_ignored() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    prompt = "<|vision_start|><|image_pad|><|vision_end|>"
    image = {"image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]}
    _, plain = _hashes(processor, prompt, image)
    # As for raw media, a UUID key with no items of that modality is ignored.
    _, other = _hashes(processor, prompt, image, multi_modal_uuids={"video": ["clip-1"]})
    assert other == plain


def test_llm_preprocess_carries_hashes_and_uuids_from_token_ids() -> None:
    processor = _make_processor(Qwen3VLInputProcessorBase)
    llm = object.__new__(BaseLLM)
    llm.input_processor = processor
    llm._hf_model_config = SimpleNamespace(is_encoder_decoder=False)
    prompt_ids = [7, _VISION_START, _IMAGE_PAD, _VISION_END, 7]
    inputs = {
        "prompt_token_ids": prompt_ids,
        "multi_modal_embeddings": {
            "image": [{"mm_embeddings": torch.ones(4, _HIDDEN), "image_grid_thw": [1, 4, 4]}]
        },
        "multi_modal_uuids": {"image": ["img-1"]},
    }
    tokens, _, params, _ = llm._preprocess(inputs, SamplingParams(), None)

    assert tokens == [7, _VISION_START, *([_IMAGE_PAD] * 4), _VISION_END, 7]
    mm_input = params.multimodal_input
    assert mm_input is not None
    assert mm_input.multimodal_uuids == ["img-1"]
    _, from_text = _hashes(
        processor,
        "x<|vision_start|><|image_pad|><|vision_end|>x",
        inputs["multi_modal_embeddings"],
        multi_modal_uuids={"image": ["img-1"]},
    )
    assert mm_input.multimodal_hashes == from_text
    assert "mrope_config" in params.multimodal_data
    assert "multimodal_embed_mask_cumsum" in params.multimodal_data
