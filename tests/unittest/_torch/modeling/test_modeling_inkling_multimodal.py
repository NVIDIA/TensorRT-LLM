# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the Inkling multimodal towers and input processor.

Covers the three media paths and the shared placeholder plumbing:

* vision  -- the hMLP scale plan / module tree, the fold's value preservation,
  and the tower's one-row-per-patch output;
* audio   -- dMel preprocessing shape/range, the codebook-sum + norm forward,
  and the bin-count guard;
* video   -- frame sampling (bounded by fps / max_frames / clip length) and the
  multi-frame -> multi-image expansion, since Inkling has no video tower;
* input processor -- registration, text passthrough, placeholder expansion, and
  the fail-loud placeholder/media count contract.

Small synthetic configs and inputs throughout: no checkpoint, no GPU, no network.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.configs.inkling import InklingConfig
from tensorrt_llm._torch.models.modeling_inkling_multimodal import (
    DEFAULT_AUDIO_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID,
    DecodedVideo,
    InklingAudioModel,
    InklingAudioPreprocessor,
    InklingImagePreprocessor,
    InklingInputProcessor,
    InklingVisionModel,
    fold_timespace_to_depth,
    patch_grid,
    plan_out_scales,
    sample_video_as_images,
    sample_video_frames,
    scaled_image_dimensions,
)

PATCH_SIZE = 40
TEMPORAL = 2
RESCALE_FRAC = 2.0
RESCALE_CAP = 2048

# The checkpoint ``vision_config``, shrunk to ``decoder_dmodel=64`` so the tower
# builds and forwards on CPU in milliseconds. The hMLP geometry (patch/temporal
# size, layer count) is the real one, since that is what the scale plan pins.
VISION_CFG = SimpleNamespace(
    vision_encoder_type="hmlp",
    decoder_dmodel=64,
    patch_size=PATCH_SIZE,
    temporal_patch_size=TEMPORAL,
    n_channels=3,
    n_layers=4,
    use_vision_norm=True,
)

# Resolved four-layer hMLP scale progression for (T=2, P=40, n_layers=4, C=3).
EXPECTED_SCALES = [
    (1, 1, 1, 3),
    (1, 5, 5, 128),
    (1, 10, 10, 320),
    (1, 40, 40, 4800),
    (2, 40, 40, 9600),
]

# Linear weight shapes (out_features, in_features) implied by those scales; the
# last layer projects to ``decoder_dmodel``.
EXPECTED_LINEAR_SHAPES = {
    "layers.linear_0.weight": (128, 75),
    "layers.linear_1.weight": (320, 512),
    "layers.linear_2.weight": (4800, 5120),
    "layers.linear_3.weight": (64, 9600),
}

# The eight checkpoint parameter names (after stripping ``model.visual.``).
EXPECTED_PARAM_NAMES = {
    "layers.linear_0.weight",
    "layers.linear_1.weight",
    "layers.linear_2.weight",
    "layers.linear_3.weight",
    "layers.norm_0.weight",
    "layers.norm_1.weight",
    "layers.norm_2.weight",
    "final_norm.weight",
}

AUDIO_CFG = SimpleNamespace(
    audio_mode="dmel",
    decoder_dmodel=32,
    n_mel_bins=4,
    mel_vocab_size=8,
    use_audio_norm=True,
)


def _image(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _expected_num_patches(h: int, w: int) -> int:
    sw, sh = scaled_image_dimensions(w, h, RESCALE_FRAC, RESCALE_CAP)
    return patch_grid(sh, sw, PATCH_SIZE)[0]


def _waveform(seconds: float = 0.6, sr: int = 16000) -> np.ndarray:
    """A short deterministic multi-tone clip (no file I/O)."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wav = 0.1 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * np.sin(2 * np.pi * 1320.0 * t)
    return wav.astype(np.float32)


def _processor(**config_kwargs) -> InklingInputProcessor:
    config = InklingConfig(vision_config=vars(VISION_CFG), **config_kwargs)
    return InklingInputProcessor(None, config, None)


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------
def test_vision_scale_plan_and_module_tree():
    assert [tuple(int(x) for x in s) for s in plan_out_scales(TEMPORAL, PATCH_SIZE, 4, 3)] == (
        EXPECTED_SCALES
    )
    tower = InklingVisionModel(VISION_CFG)
    assert set(dict(tower.named_parameters())) == EXPECTED_PARAM_NAMES
    state = tower.state_dict()
    for key, shape in EXPECTED_LINEAR_SHAPES.items():
        assert tuple(state[key].shape) == shape, key
    # The last layer projects to the text hidden width; there is no norm_3.
    assert tower.layers["linear_3"].out_features == VISION_CFG.decoder_dmodel
    assert "norm_3" not in tower.layers
    assert tower.final_norm is not None


# Geometries where the planner takes the assignment branch, i.e. there are more
# candidate scales than layers to place (the argmin branch above it is a
# different, pre-existing path that may reuse a scale).
@pytest.mark.parametrize(
    "temporal,patch,n_layers",
    [(2, 40, 4), (2, 40, 3), (2, 40, 5), (1, 16, 2), (2, 64, 5), (4, 32, 3)],
)
def test_scale_plan_is_a_strictly_growing_progression(temporal, patch, n_layers):
    """The planner must hand back an ordered, strictly growing progression.

    Each hMLP layer folds the previous scale into channel depth, so a plan that
    is not monotonic gives a fold that does not divide. This is the property the
    (dependency-free) assignment has to preserve; it replaced a scipy solver
    that could return a crossing assignment of equal cost.
    """
    scales = plan_out_scales(temporal, patch, n_layers, 3)
    assert len(scales) == n_layers + 1
    assert scales[0] == (1, 1, 1, 3)  # first pinned to the raw patch
    assert scales[-1][:3] == (temporal, patch, patch)  # last pinned to the full patch
    sizes = [t * h * w for t, h, w, _ in scales]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    # every step must be an integral fold of its predecessor
    for (t0, h0, w0, _), (t1, h1, w1, _) in zip(scales, scales[1:]):
        assert t1 % t0 == 0 and h1 % h0 == 0 and w1 % w0 == 0


def test_fold_timespace_to_depth_is_value_preserving():
    # (B=1, T=2, H=2, W=2, C=3), fold t=2, hw=1 -> (1, 1, 2, 2, 2*1*1*3=6)
    x = torch.arange(24, dtype=torch.float32).reshape(1, 2, 2, 2, 3)
    y = fold_timespace_to_depth(x, t_fold=2, hw_fold=1)
    assert tuple(y.shape) == (1, 1, 2, 2, 6)
    assert torch.equal(x.reshape(-1).sort().values, y.reshape(-1).sort().values)
    # (B=1, T=1, H=2, W=2, C=1), fold hw=2 -> (1, 1, 1, 1, 1*2*2*1=4)
    x2 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2, 1)
    y2 = fold_timespace_to_depth(x2, t_fold=1, hw_fold=2)
    assert tuple(y2.shape) == (1, 1, 1, 1, 4)
    assert torch.equal(x2.reshape(-1).sort().values, y2.reshape(-1).sort().values)


def test_vision_tower_emits_one_row_per_patch():
    tower = InklingVisionModel(VISION_CFG).to(torch.float32).eval()
    patches = InklingImagePreprocessor(dtype=torch.float32).encode_one(_image(80, 120, seed=1))
    with torch.no_grad():
        out = tower(patches)
    assert out.shape == (patches.shape[0], VISION_CFG.decoder_dmodel)
    assert torch.isfinite(out).all()


def test_vision_tower_load_weights_is_strict():
    tower = InklingVisionModel(VISION_CFG)
    weights = {f"model.visual.{k}": v for k, v in tower.state_dict().items()}
    tower.load_weights(weights)  # full keys
    tower.load_weights(tower.state_dict())  # already-stripped keys
    with pytest.raises(RuntimeError):
        tower.load_weights({k: v for k, v in weights.items() if "final_norm" not in k})


# ---------------------------------------------------------------------------
# Audio tower
# ---------------------------------------------------------------------------
def test_dmel_preprocess_shape_and_range():
    feat = InklingAudioPreprocessor().preprocess(_waveform(0.6))
    bins = feat["dmel_bins"]
    assert bins.dtype == torch.int32
    assert bins.ndim == 2 and bins.shape[1] == 80
    assert bins.shape[0] >= 8, f"expected ~12 frames for 0.6s, got {bins.shape[0]}"
    assert 0 <= int(bins.min()) and int(bins.max()) < 16
    assert feat["num_frames"] == [int(bins.shape[0])]
    assert feat["num_tokens"] == feat["num_frames"]  # one audio token per frame


def test_dmel_multi_clip_concat_and_empty_clip():
    pre = InklingAudioPreprocessor()
    a, b = _waveform(0.5), _waveform(0.3)
    fa = pre.preprocess(a)["num_frames"][0]
    fb = pre.preprocess(b)["num_frames"][0]
    feat = pre.preprocess([a, b])
    assert feat["num_frames"] == [fa, fb]
    assert feat["dmel_bins"].shape[0] == fa + fb
    empty = pre.encode_one(np.zeros(0, dtype=np.float32))
    assert empty.shape == (0, 80) and empty.dtype == torch.int32


def test_audio_tower_matches_codebook_reference():
    torch.manual_seed(0)
    tower = InklingAudioModel(AUDIO_CFG)
    with torch.no_grad():
        tower.encoder.weight.normal_()
        tower.final_norm.weight.normal_()
    frames = torch.randint(
        0, AUDIO_CFG.mel_vocab_size, (5, AUDIO_CFG.n_mel_bins), dtype=torch.int32
    )
    out = tower(frames)
    # Reference: bin m occupies codebook rows [m*V, (m+1)*V); sum over bins; norm.
    offsets = torch.arange(AUDIO_CFG.n_mel_bins) * AUDIO_CFG.mel_vocab_size
    idx = offsets.unsqueeze(0) + frames.long()
    ref = tower.final_norm(
        tower.encoder(idx.reshape(-1))
        .reshape(5, AUDIO_CFG.n_mel_bins, AUDIO_CFG.decoder_dmodel)
        .sum(dim=1)
    )
    assert out.shape == (5, AUDIO_CFG.decoder_dmodel)
    assert torch.allclose(out, ref, atol=1e-5)


def test_audio_tower_optional_norm_and_bin_count_guard():
    no_norm = InklingAudioModel(SimpleNamespace(**{**vars(AUDIO_CFG), "use_audio_norm": False}))
    assert no_norm.final_norm is None
    assert no_norm(torch.zeros((2, AUDIO_CFG.n_mel_bins), dtype=torch.int32)).shape == (2, 32)
    with pytest.raises(ValueError):
        InklingAudioModel(AUDIO_CFG)(torch.zeros((2, AUDIO_CFG.n_mel_bins + 1), dtype=torch.int32))


# ---------------------------------------------------------------------------
# Video: frame sampling onto the image path (no video tower)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "total,avg_fps,desired_fps,max_frames,expected",
    [
        # capped by desired_fps
        (
            100,
            25.0,
            5,
            200,
            [0, 5, 10, 15, 20, 26, 31, 36, 41, 46, 52, 57, 62, 67, 72, 78, 83, 88, 93, 99],
        ),
        (10, 10.0, 100, 5, [0, 2, 4, 6, 9]),  # capped by max_frames
        (50, 25.0, 50, 200, list(range(50))),  # capped by total frames
        (1, 30.0, 0, 0, [0]),  # always at least one frame
    ],
)
def test_sample_video_frames(total, avg_fps, desired_fps, max_frames, expected):
    video = DecodedVideo([_image(8, 8, seed=i) for i in range(total)], avg_fps=avg_fps)
    assert sample_video_frames(video, desired_fps=desired_fps, max_frames=max_frames) == expected


def test_sample_video_as_images_preserves_count_and_order():
    frames = [_image(48, 64, seed=i) for i in range(12)]
    video = DecodedVideo(frames, avg_fps=12.0)
    idxs = sample_video_frames(video, desired_fps=4, max_frames=32)
    picked = sample_video_as_images(video, desired_fps=4, max_frames=32)
    assert idxs == sorted(set(idxs)) and len(picked) == len(idxs) >= 1
    assert all(pf is frames[i] for pf, i in zip(picked, idxs))
    with pytest.raises(ValueError):
        DecodedVideo([], avg_fps=25.0)


def test_video_frames_expand_to_one_image_span_each():
    proc = _processor()
    # Frames of DIFFERENT sizes -> different per-frame patch counts, so a
    # reordering or merge would change span lengths and be detected.
    frames = [_image(48, 64, 1), _image(80, 48, 2), _image(40, 120, 3)]
    per_frame = [_expected_num_patches(*f.shape[:2]) for f in frames]
    ids = [10, proc.image_token_id, 11, proc.image_token_id, 12, proc.image_token_id, 13]
    out_ids, mm = proc.assemble(ids, image_data=frames)

    assert set(mm) == {"image"}
    assert mm["image"]["num_patches"] == per_frame
    assert out_ids.count(proc.image_token_id) == sum(per_frame)
    # non-media tokens survive verbatim, in order
    assert [t for t in out_ids if t != proc.image_token_id] == [10, 11, 12, 13]
    # one contiguous, non-overlapping span per frame
    prev_end = -1
    for (start, end), n_patches in zip(mm["image"]["offsets"], per_frame):
        assert start > prev_end and end - start + 1 == n_patches
        assert all(out_ids[p] == proc.image_token_id for p in range(start, end + 1))
        prev_end = end
    # concatenated rows are the frames' own patches, in encounter order
    patches = mm["image"]["vision_patches_bthwc"]
    assert patches.shape[0] == sum(per_frame)
    assert patches.shape[1:] == (TEMPORAL, PATCH_SIZE, PATCH_SIZE, 3)


# ---------------------------------------------------------------------------
# Input processor: registration, passthrough, expansion, fail-loud
# ---------------------------------------------------------------------------
def test_registration():
    # Importing the model module runs the @register_input_processor decorator.
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration
    from tensorrt_llm.inputs.registry import (
        INPUT_PROCESSOR_REGISTRY,
        MULTIMODAL_PLACEHOLDER_REGISTRY,
    )

    for modality, placeholder in (("image", "<image>"), ("audio", "<audio>")):
        assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("inkling_mm_model", modality)
        assert (
            MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("inkling_mm_model", modality)
            == placeholder
        )
    registry = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type
    assert registry.get(InklingForConditionalGeneration) is InklingInputProcessor
    assert InklingInputProcessor._registered_model_type == "inkling_mm_model"


def test_serving_and_profiler_interface_contract():
    # trtllm-serve calls get_preferred_media_io_kwargs() and the KV-cache encoder
    # profiler calls get_mm_max_tokens_per_item() on any multimodal processor;
    # both live on BaseMultimodalDummyInputsBuilder, so a processor inheriting
    # only BaseMultimodalInputProcessor crashes the server at startup.
    from tensorrt_llm.inputs.registry import (
        BaseMultimodalDummyInputsBuilder,
        BaseMultimodalInputProcessor,
    )

    proc = _processor()
    assert isinstance(proc, BaseMultimodalInputProcessor)
    assert isinstance(proc, BaseMultimodalDummyInputsBuilder)
    assert proc.get_preferred_media_io_kwargs() == {}
    demand = proc.get_mm_max_tokens_per_item()
    # Empty demand -> the profiler's ``total_demand <= 0`` guard returns early.
    assert isinstance(demand, dict) and sum(demand.values()) == 0
    assert callable(getattr(proc, "get_dummy_mm_data", None))


def test_text_only_passthrough():
    proc = _processor()
    ids = [5, 6, 7, 8]
    assert proc.assemble(ids) == (ids, {})
    assert proc.assemble(ids, []) == (ids, {})


def test_image_placeholder_expands_to_num_patches():
    proc = _processor()
    n_patches = _expected_num_patches(80, 120)
    ids = [1, 2, DEFAULT_IMAGE_TOKEN_ID, 3]
    out_ids, mm = proc.assemble(ids, [_image(80, 120, seed=1)])

    assert out_ids.count(DEFAULT_IMAGE_TOKEN_ID) == n_patches
    assert len(out_ids) == len(ids) - 1 + n_patches
    feat = mm["image"]
    assert feat["num_patches"] == [n_patches]
    assert feat["offsets"] == [(2, 2 + n_patches - 1)]
    assert tuple(feat["vision_patches_bthwc"].shape) == (
        n_patches,
        TEMPORAL,
        PATCH_SIZE,
        PATCH_SIZE,
        3,
    )


def test_audio_placeholder_expands_to_num_frames():
    proc = _processor()
    wav = _waveform(0.6)
    n_frames = proc._audio_preprocessor.preprocess(wav)["num_frames"][0]
    ids = [10, 11, DEFAULT_AUDIO_TOKEN_ID, 12]
    out_ids, mm = proc.assemble(ids, audio_data=[wav])

    assert set(mm) == {"audio"}
    assert out_ids.count(DEFAULT_AUDIO_TOKEN_ID) == n_frames
    assert len(out_ids) == len(ids) - 1 + n_frames
    assert mm["audio"]["dmel_bins"].shape[0] == n_frames
    assert mm["audio"]["num_frames"] == [n_frames]
    assert mm["audio"]["offsets"] == [(2, 2 + n_frames - 1)]


def test_image_and_audio_expand_independently():
    proc = _processor()
    img, wav = _image(80, 120, seed=2), _waveform(0.4)
    n_patches = _expected_num_patches(80, 120)
    n_frames = proc._audio_preprocessor.preprocess(wav)["num_frames"][0]
    out_ids, mm = proc.assemble([DEFAULT_IMAGE_TOKEN_ID, 9, DEFAULT_AUDIO_TOKEN_ID], [img], [wav])
    assert set(mm) == {"image", "audio"}
    assert out_ids.count(DEFAULT_IMAGE_TOKEN_ID) == n_patches
    assert out_ids.count(DEFAULT_AUDIO_TOKEN_ID) == n_frames
    assert mm["image"]["offsets"] == [(0, n_patches - 1)]
    assert mm["audio"]["offsets"] == [(n_patches + 1, n_patches + n_frames)]


def test_call_with_token_ids_expands_and_passes_text_through():
    proc = _processor()
    n_patches = _expected_num_patches(80, 120)
    ids = [7, DEFAULT_IMAGE_TOKEN_ID, 8]
    out_ids, extra = proc.call_with_token_ids(
        {"prompt_token_ids": ids, "multi_modal_data": {"image": [_image(80, 120, seed=3)]}}, None
    )
    assert out_ids.count(DEFAULT_IMAGE_TOKEN_ID) == n_patches
    assert extra["multimodal_data"]["image"]["vision_patches_bthwc"].shape[0] == n_patches
    # No media -> plain passthrough, no multimodal payload.
    assert proc.call_with_token_ids(
        {"prompt_token_ids": [1, 2, 3], "multi_modal_data": {}}, None
    ) == ([1, 2, 3], None)


def test_mm_token_ids_cover_both_modalities():
    proc = _processor()
    assert InklingInputProcessor.supports_token_id_mm_expansion is True
    ids = proc.get_mm_token_ids()
    assert ids.dtype == torch.int32
    assert set(ids.tolist()) == {DEFAULT_IMAGE_TOKEN_ID, DEFAULT_AUDIO_TOKEN_ID}


@pytest.mark.parametrize(
    "ids,images",
    [
        ([1, DEFAULT_IMAGE_TOKEN_ID, 2], 2),  # more images than placeholders
        ([DEFAULT_IMAGE_TOKEN_ID, 1, DEFAULT_IMAGE_TOKEN_ID], 1),  # more placeholders
        ([1, 2, 3], 1),  # image without any placeholder
    ],
)
def test_fail_loud_on_placeholder_media_mismatch(ids, images):
    proc = _processor()
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble(ids, [_image(80, 120, seed=4)] * images)


def test_fail_loud_on_audio_placeholder_mismatch():
    proc = _processor()
    wav = _waveform(0.4)
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble([DEFAULT_AUDIO_TOKEN_ID])  # placeholder, no clip
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble([1, 2, 3], audio_data=[wav])  # clip, no placeholder
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble([DEFAULT_AUDIO_TOKEN_ID], audio_data=[wav, wav])  # 2 clips, 1 placeholder


def test_multimodal_hash_prefix_cache_is_refused_per_modality():
    proc = _processor()
    for getter in (
        proc.get_num_tokens_per_image,
        proc.get_num_tokens_per_audio,
        proc.get_num_tokens_per_video,
    ):
        with pytest.raises(NotImplementedError, match="multimodal-hash prefix"):
            getter()
