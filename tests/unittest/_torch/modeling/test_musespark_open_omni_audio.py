# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the MuseSparkOpen wav_ae audio path.

Covers the pieces that are checkable without a GPU or an engine: the token
accounting formula, the placeholder expansion, the tower's shape/dtype
contract, the checkpoint-name mirror, and numeric parity of the tower
against reference-produced embeddings, which needs the checkpoint and the
golden files.

The module under test is loaded by file path so the tests run with any plain
Python that has torch + numpy, without importing (or building) tensorrt_llm.

Data for the golden tests, under LLM_MODELS_ROOT (environment variables override):
    Muse-Spark-Open-816B-A42B-NVFP4 (MUSESPARK_CKPT_DIR)
                            HF checkpoint dir (model.safetensors.index.json +
                            shards) to pull the 72 audio tensors from.
    Muse-Spark-Open-test-data/audio-goldens (MUSESPARK_AUDIO_GOLDENS)
                            Directory containing the speech_<k>.pt and
                            tone_<k>.pt golden files, each a torch.save dict
                            with keys `wav_24k` (float32 [N]), `n_tokens` (int)
                            and `embeddings` (float32 [n_tokens, 8192]),
                            written by the generate_audio_goldens.py next to
                            them.
Either one absent -> the golden tests fail, naming the missing path.
"""

import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from utils.llm_data import get_checkpoint

pytestmark = pytest.mark.cpu_only

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "models"
    / "musespark_open_audio.py"
)


def _load_audio_module():
    spec = importlib.util.spec_from_file_location("musespark_open_audio", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("musespark_open_audio", module)
    spec.loader.exec_module(module)
    return module


audio = _load_audio_module()

SAMPLES_PER_TOKEN = audio.AUDIO_SAMPLES_PER_TOKEN
HIDDEN_SIZE = 8192  # decoder width of the released checkpoints

#: The 72 checkpoint suffixes under
#: modalities.soft_audio_perception.interaction_encoder., transcribed from the
#: shipped safetensors index. The tower's state_dict must mirror them one for one:
#: the weight mapper walks state_dict() and prepends the prefix.
CHECKPOINT_SUFFIXES = sorted(
    ["norm.weight", "o_proj.weight", "o_proj.bias", "dual_norm.weight"]
    + [
        f"stages.{s}.0.{name}"
        for s in range(4)
        for name in ("linear.weight", "linear.bias", "norm.weight")
    ]
    + [
        f"stages.{s}.{b}.{name}"
        for s in range(4)
        for b in (1, 2)
        for name in (
            "conv.conv.weight",
            "conv.conv.bias",
            "norm.weight",
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
        )
    ]
)


# ---------------------------------------------------------------------------
# (a) token accounting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_samples,expected",
    [
        (1, 1),  # rounds up: any non-empty clip is at least one token
        (SAMPLES_PER_TOKEN - 1, 1),
        (SAMPLES_PER_TOKEN, 1),  # exact multiple
        (SAMPLES_PER_TOKEN + 1, 2),
        (2 * SAMPLES_PER_TOKEN, 2),
        (2 * SAMPLES_PER_TOKEN + 1, 3),
        (24_000, 13),  # 1 s @ 24 kHz, the references' dummy
        (10 * 60 * 24_000, 7500),  # 10 min = 7500 tokens (12.5 tok/s)
    ],
)
def test_token_count_formula(num_samples, expected):
    assert audio.num_audio_tokens(num_samples) == expected


def test_token_count_refuses_an_empty_clip():
    with pytest.raises(ValueError, match="empty"):
        audio.num_audio_tokens(0)
    with pytest.raises(ValueError, match="empty"):
        audio.num_audio_tokens(-5)


@pytest.mark.parametrize(
    "num_samples,rate,expected_len",
    [
        (16_000, 16_000, 24_000),  # 1 s @ 16 kHz -> 24000 samples
        (24_000, 24_000, 24_000),  # already on rate: untouched
        (1, 16_000, 2),  # ceil(1 * 24000/16000) = 2, librosa fix=True semantics
        # librosa's ratio-first float order: 44100 * (24000 / 44100) and
        # 88200 * (24000 / 44100) land a hair above 24000 and 48000.
        (44_100, 44_100, 24_001),
        (88_200, 44_100, 48_001),
    ],
)
def test_resampled_length_matches_fix_true_semantics(num_samples, rate, expected_len):
    assert audio.resampled_num_samples(num_samples, rate) == expected_len


@pytest.mark.parametrize(
    "num_samples,rate",
    [
        (44_100, 44_100),
        (88_200, 44_100),
        (16_000, 16_000),
        (12_345, 11_025),
        (77_777, 48_000),
        (1, 22_050),
    ],
)
def test_resampled_length_matches_librosa(num_samples, rate):
    """The count is defined on the waveform `to_mono_24k` actually produces."""
    import numpy as np

    pytest.importorskip("librosa")
    wave = audio.to_mono_24k(np.zeros(num_samples, dtype=np.float32), rate)
    assert audio.resampled_num_samples(num_samples, rate) == wave.shape[0]


def test_to_mono_24k_downmixes_and_counts_post_downmix():
    # (channels, frames) channels-first, already 24 kHz: mean over channels.
    stereo = torch.stack([torch.ones(4000), -torch.ones(4000)]).numpy()
    wav = audio.to_mono_24k(stereo, 24_000)
    assert wav.shape == (4000,)
    assert abs(wav).max() == 0.0  # the two channels cancel
    # (frames, channels) channels-last (soundfile layout) downmixes the same.
    wav2 = audio.to_mono_24k(stereo.T, 24_000)
    assert wav2.shape == (4000,)
    # Token count is defined on the downmixed length: 4000 -> 3 tokens, not 5.
    assert audio.num_audio_tokens(len(wav)) == 3


def test_to_mono_24k_refuses_empty_audio():
    import numpy as np

    with pytest.raises(ValueError, match="empty"):
        audio.to_mono_24k(np.zeros(0, dtype=np.float32), 24_000)


# ---------------------------------------------------------------------------
# (b) placeholder expansion
# ---------------------------------------------------------------------------
def test_placeholder_expansion_ids():
    ids = audio.audio_placeholder_ids(3)
    assert ids == [201463, 200038, 200038, 200038, 201464]
    assert ids[0] == audio.AUDIO_TOKEN_IDS["soft_audio_start"]
    assert ids[-1] == audio.AUDIO_TOKEN_IDS["soft_audio_end"]
    assert set(ids[1:-1]) == {audio.AUDIO_TOKEN_IDS["audio"]}
    # Per-clip prompt growth is count + 2, exactly (verified against a live
    # reference serve: prompt_tokens overhead per clip = count + 2, zero slack).
    assert len(ids) == 3 + 2


def test_placeholder_expansion_refuses_zero_tokens():
    with pytest.raises(ValueError):
        audio.audio_placeholder_ids(0)


def test_token_id_verification_by_vocab_name():
    good = {"<|audio|>": 200038, "<|soft_audio_start|>": 201463, "<|soft_audio_end|>": 201464}
    audio.verify_audio_token_ids(good)  # no raise
    with pytest.raises(ValueError, match="no '<\\|soft_audio_start\\|>'"):
        audio.verify_audio_token_ids({"<|audio|>": 200038, "<|soft_audio_end|>": 201464})
    with pytest.raises(ValueError, match="pinned"):
        audio.verify_audio_token_ids({**good, "<|audio|>": 123})


def test_tokenizer_vocab_unwraps_a_wrapper_whose_get_vocab_is_a_stub():
    """The served path's tokenizer wrapper inherits the PreTrainedTokenizerBase
    get_vocab() stub (raises NotImplementedError); the real vocab lives on the
    wrapped tokenizer at `.tokenizer`. tokenizer_vocab must fall through to
    it, still prefer a working get_vocab on the object itself, and refuse an
    object with no resolvable vocab at all."""
    good = {"<|audio|>": 200038, "<|soft_audio_start|>": 201463, "<|soft_audio_end|>": 201464}

    class Inner:
        def get_vocab(self):
            return good

    class Wrapper:
        tokenizer = Inner()

        def get_vocab(self):
            raise NotImplementedError()

    assert audio.tokenizer_vocab(Inner()) == good  # direct HF-style tokenizer
    assert audio.tokenizer_vocab(Wrapper()) == good  # served-path wrapper

    class NoVocab:
        pass

    with pytest.raises(TypeError, match="cannot resolve a vocab"):
        audio.tokenizer_vocab(NoVocab())


# ---------------------------------------------------------------------------
# (c) tower forward: shapes, dtype, checkpoint-name mirror
# ---------------------------------------------------------------------------
def test_tower_state_dict_mirrors_the_checkpoint_suffixes():
    with torch.device("meta"):
        tower = audio.MuseSparkOpenAudioEncoder(HIDDEN_SIZE)
    assert sorted(tower.state_dict()) == CHECKPOINT_SUFFIXES
    assert len(CHECKPOINT_SUFFIXES) == 72
    pairs = audio.audio_state_dict_to_checkpoint_keys(
        tower, "modalities.soft_audio_perception.interaction_encoder."
    )
    assert len(pairs) == 72
    assert all(
        key == f"modalities.soft_audio_perception.interaction_encoder.{name}" for name, key in pairs
    )


def test_tower_shapes_match_the_checkpoint():
    with torch.device("meta"):
        tower = audio.MuseSparkOpenAudioEncoder(HIDDEN_SIZE)
    state = tower.state_dict()
    # Patchifier linears: product of patchify factors is 1920.
    assert tuple(state["stages.0.0.linear.weight"].shape) == (256, 16)
    assert tuple(state["stages.1.0.linear.weight"].shape) == (512, 1536)
    assert tuple(state["stages.2.0.linear.weight"].shape) == (1024, 2560)
    assert tuple(state["stages.3.0.linear.weight"].shape) == (2048, 4096)
    # Depthwise conv, k=7; 4x MLP; head into decoder width.
    assert tuple(state["stages.3.1.conv.conv.weight"].shape) == (2048, 1, 7)
    assert tuple(state["stages.3.1.linear1.weight"].shape) == (8192, 2048)
    assert tuple(state["o_proj.weight"].shape) == (HIDDEN_SIZE, 2048)
    assert tuple(state["dual_norm.weight"].shape) == (HIDDEN_SIZE,)


def test_tower_builds_under_meta_init():
    """The model loader builds the model on the meta device and falls back to a
    regular init when construction runs an op meta tensors do not support."""
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode  # noqa: PLC0415

    with MetaInitMode():
        tower = audio.MuseSparkOpenAudioEncoder(HIDDEN_SIZE)
    assert tower.o_proj.weight.device.type == "meta"
    assert {p.dtype for p in tower.parameters()} == {torch.bfloat16}


#: Output width of the tower the forward tests run. They check properties that
#: hold at any width (rows per 1920 samples, dtype, clip independence,
#: causality), and a full-width BF16 forward takes minutes on some CPUs, so they
#: keep the shipped patch geometry with narrow stages. The full-width layout is
#: checked by the construction tests above.
NARROW_OUTPUT_SIZE = 16


def _narrow_tower():
    return audio.MuseSparkOpenAudioEncoder(
        NARROW_OUTPUT_SIZE, hidden_sizes=(8, 8, 8, 8), layers_per_stage=(1, 1, 1, 1)
    )


@pytest.mark.parametrize(
    "num_samples",
    [1, SAMPLES_PER_TOKEN, SAMPLES_PER_TOKEN + 1, 3 * SAMPLES_PER_TOKEN, 4_000],
)
def test_tower_forward_row_count_width_dtype(num_samples):
    torch.manual_seed(0)
    tower = _narrow_tower()
    wav = torch.randn(num_samples, dtype=torch.float32) * 0.1
    (out,) = tower([wav])
    assert out.shape == (math.ceil(num_samples / SAMPLES_PER_TOKEN), NARROW_OUTPUT_SIZE)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_tower_forward_multiple_clips_keeps_order_and_per_clip_lengths():
    torch.manual_seed(1)
    tower = _narrow_tower()
    wavs = [torch.randn(n) * 0.1 for n in (SAMPLES_PER_TOKEN, 2 * SAMPLES_PER_TOKEN + 5)]
    outs = tower(wavs)
    assert [o.shape[0] for o in outs] == [1, 3]
    # Each clip is padded independently: encoding a clip alone equals encoding
    # it in a batch of clips (no cross-clip state).
    (alone,) = tower([wavs[1]])
    assert torch.equal(alone, outs[1])


def test_tower_causality():
    """The tower is causal: changing the tail must not change earlier rows."""
    torch.manual_seed(2)
    tower = _narrow_tower()
    base = torch.randn(2 * SAMPLES_PER_TOKEN) * 0.1
    changed = base.clone()
    changed[SAMPLES_PER_TOKEN:] += 1.0
    (out_base,) = tower([base])
    (out_changed,) = tower([changed])
    assert torch.equal(out_base[0], out_changed[0])
    assert not torch.equal(out_base[1], out_changed[1])


# ---------------------------------------------------------------------------
# (d) golden parity against the reference implementation
# ---------------------------------------------------------------------------
# The goldens hold the float32 output of transformers' `MuseSparkAudioModel`
# with the checkpoint's BF16 weights, and the comparison runs this tower in
# float32 too. Two correct implementations then agree to about 1e-5 (8.4e-6 at
# most on the six goldens), so atol=rtol=1e-3 leaves wide headroom for other
# CPUs and torch builds while a real defect, which moves the output by far
# more, still fails.
#
# The BF16 path only gets a loose bound, on one 12-token golden, because some
# CPUs run BF16 far slower than float32. The tower's 13 norms, 13 linears and
# 8 depthwise convs each re-round to BF16, and a one-ulp difference early on
# grows to as much as 1.5 at the output, so two correct BF16 implementations
# differ by that much. Each lands 0.8-12 % from the float32 result in relative
# L2 on the six goldens, this tower and the reference alike, so the bound only
# rejects a broken BF16 path.
GOLDEN_ATOL = 1e-3
GOLDEN_RTOL = 1e-3
GOLDEN_BF16_MAX_REL_L2 = 0.25


#: The golden files, as generate_audio_goldens.py writes them.
GOLDEN_NAMES = ("speech_0", "speech_1", "speech_2", "tone_0", "tone_1", "tone_2")
BF16_GOLDEN = "tone_0"
_AUDIO_CKPT_PREFIX = "modalities.soft_audio_perception.interaction_encoder."


def _checkpoint_dir() -> str:
    return os.environ.get("MUSESPARK_CKPT_DIR") or get_checkpoint("Muse-Spark-Open-816B-A42B-NVFP4")


def _goldens_dir() -> str:
    return os.environ.get("MUSESPARK_AUDIO_GOLDENS") or get_checkpoint(
        os.path.join("Muse-Spark-Open-test-data", "audio-goldens")
    )


def _load_real_tower():
    """Build the tower and load the 72 audio tensors from the checkpoint shards.

    The tower is built on the meta device and materialized empty: every
    parameter comes from the checkpoint, so random initialization is skipped.

    Works on either checkpoint layout. The source export names the tower
    `modalities.soft_audio_perception.interaction_encoder.*`; the open (HF
    re-export) layout names it `model.audio_tower.*`. Filtering on the source
    prefix alone finds ZERO tensors in an open-format checkpoint and the test
    dies at the count assertion before it ever compares an embedding, so the
    layout is resolved through the tier's own mapper rather than a second
    hard-coded name table -- if the mapper is wrong, the parity test should
    fail, not silently test nothing.
    """
    from safetensors import safe_open

    from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_format import (  # noqa: PLC0415
        is_open_format,
        open_format_to_source_key,
    )

    checkpoint = _checkpoint_dir()
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    # key in the file -> key in source spelling (identity on a source checkpoint)
    to_source = (
        {k: open_format_to_source_key(k) for k in weight_map}
        if is_open_format(weight_map)
        else {k: k for k in weight_map}
    )
    audio_keys = {
        k: weight_map[k] for k, src in to_source.items() if src.startswith(_AUDIO_CKPT_PREFIX)
    }
    assert len(audio_keys) == 72, f"expected 72 audio tensors, found {len(audio_keys)}"

    with torch.device("meta"):
        tower = audio.MuseSparkOpenAudioEncoder(HIDDEN_SIZE)
    tower = tower.to_empty(device="cpu")
    assert not list(tower.buffers())
    state = dict(tower.named_parameters())
    by_shard = {}
    for key, shard in audio_keys.items():
        by_shard.setdefault(shard, []).append(key)
    loaded = 0
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(checkpoint, shard), framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                assert tensor.dtype == torch.bfloat16, f"{key} is {tensor.dtype}, not BF16"
                param = state[to_source[key][len(_AUDIO_CKPT_PREFIX) :]]
                assert param.shape == tensor.shape, key
                with torch.no_grad():
                    param.copy_(tensor)
                loaded += 1
    assert loaded == len(state) == 72
    return tower


@pytest.fixture(scope="module")
def real_towers():
    """The tower with the checkpoint's weights in BF16, and a float32 copy."""
    tower = _load_real_tower()
    return tower, copy.deepcopy(tower).float()


def _golden(name):
    golden = torch.load(
        os.path.join(_goldens_dir(), f"{name}.pt"), map_location="cpu", weights_only=True
    )
    wav = golden["wav_24k"].to(torch.float32)
    n_tokens = int(golden["n_tokens"])
    expected = golden["embeddings"]
    # The golden's own accounting must match the formula.
    assert n_tokens == math.ceil(wav.numel() / SAMPLES_PER_TOKEN)
    assert expected.shape == (n_tokens, HIDDEN_SIZE)
    assert expected.dtype == torch.float32
    return wav, expected


def _encode(tower, wav):
    # One thread, as the goldens were generated, so reruns repeat exactly. The
    # setting is restored, so one test cannot reconfigure the rest of the session.
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            (out,) = tower([wav])
    finally:
        torch.set_num_threads(threads_before)
    return out


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_golden_embeddings_parity(name, real_towers):
    wav, expected = _golden(name)
    got = _encode(real_towers[1], wav)
    assert got.shape == expected.shape
    assert got.dtype == torch.float32
    max_abs = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=GOLDEN_ATOL, rtol=GOLDEN_RTOL), (
        f"{name}: float32 max abs diff {max_abs:.4e} exceeds atol={GOLDEN_ATOL}/rtol={GOLDEN_RTOL}"
    )


def test_golden_bf16_output_stays_close(real_towers):
    wav, expected = _golden(BF16_GOLDEN)
    got = _encode(real_towers[0], wav)
    assert got.shape == expected.shape
    assert got.dtype == torch.bfloat16
    rel_l2 = ((got.float() - expected).norm() / expected.norm()).item()
    assert rel_l2 <= GOLDEN_BF16_MAX_REL_L2, (
        f"{BF16_GOLDEN}: the BF16 output is {rel_l2:.3f} from the float32 golden in relative L2, "
        f"above {GOLDEN_BF16_MAX_REL_L2}"
    )
