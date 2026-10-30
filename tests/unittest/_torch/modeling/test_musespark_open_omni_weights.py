# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-key accounting for the MuseSparkOpen perception tower.

The checkpoint carries five subtrees that look alike and mean different things:
the tower itself, a bit-identical duplicate of it that is missing the attention
biases, 50 orphan fused-qkv biases with no matching weight, one projection, and
the audio tensors, which only an audio tower consumes. Loading the wrong
copy produces a tower that runs and answers plausibly and is wrong, so the
partition is pinned here rather than left to a benchmark to notice.

The key inventory is synthesized from the documented checkpoint structure, not
read from a checkpoint: the point is that the classifier splits these names
correctly, which is checkable without touching 500 GiB of payload. That the real
checkpoints actually carry these names is a separate, measured fact.
"""

import logging

import pytest

from tensorrt_llm._torch.models.checkpoints.hf.musespark_open_weight_mapper import (
    AUDIO_PREFIX,
    CANONICAL_VISION_PREFIX,
    DUPLICATE_VISION_PREFIX,
    STALE_FUSED_QKV_PREFIX,
    VISION_PROJECTION_ALIAS,
    VISION_PROJECTION_KEY,
    classify_keys,
    missing_vision_keys,
)
from tensorrt_llm.logger import Logger
from tensorrt_llm.logger import logger as trtllm_logger

pytestmark = pytest.mark.cpu_only

NUM_VISION_LAYERS = 50

# Counts measured on both released checkpoints.
N_CANONICAL = 808
N_DUPLICATE = 658
N_STALE = 50
N_AUDIO = 72
N_BIAS_ONLY_IN_CANONICAL = 150


def canonical_vision_keys():
    """The 808 tensors the tower consumes: 16 per layer plus 8 outside."""
    keys = [
        f"{CANONICAL_VISION_PREFIX}conv1_linear.weight",
        f"{CANONICAL_VISION_PREFIX}positional_embedding_vlm",
    ]
    for norm in ("ln_pre", "ln_post"):
        keys += [
            f"{CANONICAL_VISION_PREFIX}{norm}.weight",
            f"{CANONICAL_VISION_PREFIX}{norm}.bias",
        ]
    keys += [
        f"{CANONICAL_VISION_PREFIX}adapter.c_fc.weight",
        f"{CANONICAL_VISION_PREFIX}adapter.c_proj.weight",
    ]
    for layer_idx in range(NUM_VISION_LAYERS):
        prefix = f"{CANONICAL_VISION_PREFIX}encoder.layers.{layer_idx}"
        for norm in ("ln_1", "ln_2"):
            keys += [f"{prefix}.{norm}.weight", f"{prefix}.{norm}.bias"]
        for name in ("q", "k", "v", "out"):
            keys += [f"{prefix}.attn.{name}_proj.weight", f"{prefix}.attn.{name}_proj.bias"]
        for name in ("c_fc", "c_proj"):
            keys += [f"{prefix}.mlp.{name}.weight", f"{prefix}.mlp.{name}.bias"]
    return keys


def duplicate_vision_keys():
    """The 658 duplicates: the canonical set minus the q/k/v attention biases.

    That gap is the whole reason the canonical copy is the one to consume.
    """
    suffixes = [key[len(CANONICAL_VISION_PREFIX) :] for key in canonical_vision_keys()]
    dropped = tuple(f"attn.{name}_proj.bias" for name in ("q", "k", "v"))
    return [
        f"{DUPLICATE_VISION_PREFIX}{suffix}" for suffix in suffixes if not suffix.endswith(dropped)
    ]


def stale_fused_qkv_keys():
    return [
        f"{STALE_FUSED_QKV_PREFIX}vision_encoder.transformer.resblocks.{i}.attn.wqkv.bias"
        for i in range(NUM_VISION_LAYERS)
    ]


def audio_keys():
    return [
        f"{AUDIO_PREFIX}interaction_encoder.layers.{i // 2}.{'norm' if i % 2 else 'proj'}.weight"
        for i in range(N_AUDIO)
    ]


@pytest.fixture
def trtllm_warnings():
    """Collect TRT-LLM warnings. `caplog` cannot see them.

    `tensorrt_llm.logger.Logger` sets `propagate = False` on its stdlib
    logger and installs its own stdout handler, so pytest's root-logger capture
    never receives a record. It also refuses `set_level` once `TLLM_LOG_LEVEL`
    is set in the environment -- which it is inside the container -- and its
    default level is `error`, so a warning is dropped before it is formatted.
    Attach to the named logger directly and lower both thresholds for the
    duration.
    """
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    stdlib = logging.getLogger(Logger.PREFIX)
    handler = _Collect(level=logging.WARNING)
    previous_level = stdlib.level
    previous_min = trtllm_logger._min_severity
    stdlib.addHandler(handler)
    stdlib.setLevel(logging.WARNING)
    trtllm_logger._min_severity = "warning"
    try:
        yield records
    finally:
        trtllm_logger._min_severity = previous_min
        stdlib.setLevel(previous_level)
        stdlib.removeHandler(handler)


def text_keys(num_hidden_layers=4):
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer_idx in range(num_hidden_layers):
        keys.append(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
    return keys


def all_perception_keys():
    return (
        canonical_vision_keys()
        + duplicate_vision_keys()
        + stale_fused_qkv_keys()
        + audio_keys()
        + [VISION_PROJECTION_KEY]
    )


# --------------------------------------------------------------------------
# The inventory the classifier is asked to split
# --------------------------------------------------------------------------
def test_synthesized_inventory_matches_the_measured_counts():
    """Guard the fixtures themselves against drifting from the checkpoint."""
    canonical = canonical_vision_keys()
    duplicate = duplicate_vision_keys()
    assert len(canonical) == N_CANONICAL
    assert len(duplicate) == N_DUPLICATE
    assert len(stale_fused_qkv_keys()) == N_STALE
    assert len(audio_keys()) == N_AUDIO
    assert len(set(canonical)) == len(canonical), "duplicate names inside the canonical set"

    # 150 tensors exist only in the canonical copy, and all 150 are q/k/v biases.
    canonical_suffixes = {key[len(CANONICAL_VISION_PREFIX) :] for key in canonical}
    duplicate_suffixes = {key[len(DUPLICATE_VISION_PREFIX) :] for key in duplicate}
    only_canonical = canonical_suffixes - duplicate_suffixes
    assert len(only_canonical) == N_BIAS_ONLY_IN_CANONICAL
    assert not duplicate_suffixes - canonical_suffixes
    assert all(
        suffix.endswith(("attn.q_proj.bias", "attn.k_proj.bias", "attn.v_proj.bias"))
        for suffix in only_canonical
    )


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def test_text_path_skips_every_perception_key():
    """Without a tower, all 1589 perception tensors are skipped, none consumed."""
    keys = text_keys() + all_perception_keys()
    accounting = classify_keys(keys, num_hidden_layers=4)

    assert len(accounting.ignored_vision_unused) == N_CANONICAL + 1  # + the projection
    assert len(accounting.ignored_duplicate_vision) == N_DUPLICATE
    assert len(accounting.ignored_stale_fused_qkv) == N_STALE
    assert len(accounting.ignored_audio) == N_AUDIO
    assert len(accounting.ignored_perception) == 1589
    assert accounting.consumed == set()


def test_omni_path_leaves_the_tower_keys_for_the_tower_walk():
    """With a tower, the canonical copy and the projection stop being 'skipped'.

    They are left unbucketed exactly as in-range text keys are, so the loader
    adds them to `consumed` and the zero-unaccounted check still closes.
    """
    keys = text_keys() + all_perception_keys()
    accounting = classify_keys(keys, num_hidden_layers=4, load_vision=True)

    assert accounting.ignored_vision_unused == set()
    assert len(accounting.ignored_duplicate_vision) == N_DUPLICATE
    assert len(accounting.ignored_stale_fused_qkv) == N_STALE
    assert len(accounting.ignored_audio) == N_AUDIO
    assert len(accounting.ignored_perception) == N_DUPLICATE + N_STALE + N_AUDIO == 780


def test_every_perception_key_lands_in_exactly_one_bucket():
    """No key double-counted, none lost between the overlapping prefixes."""
    for load_vision in (False, True):
        accounting = classify_keys(all_perception_keys(), 62, load_vision=load_vision)
        buckets = [
            accounting.ignored_vision_unused,
            accounting.ignored_duplicate_vision,
            accounting.ignored_stale_fused_qkv,
            accounting.ignored_audio,
            accounting.ignored_other,
        ]
        for i, left in enumerate(buckets):
            for right in buckets[i + 1 :]:
                assert not left & right, "a key landed in two buckets"
        classified = set().union(*buckets)
        # Unbucketed means "a walk is going to consume it", so the tower keys
        # are unbucketed exactly when a tower is being loaded.
        expected = set(canonical_vision_keys()) | {VISION_PROJECTION_KEY} if load_vision else set()
        unclassified = set(all_perception_keys()) - classified
        assert unclassified == expected


def test_projection_is_not_swallowed_by_the_duplicate_prefix():
    """`...encoder.vision_projection.weight` shares a prefix with the 658.

    A shortest-prefix-first chain would bucket it as a duplicate and the tower
    would come up with no projection into decoder hidden size.
    """
    accounting = classify_keys([VISION_PROJECTION_KEY], 62)
    assert accounting.ignored_duplicate_vision == set()
    assert accounting.ignored_vision_unused == {VISION_PROJECTION_KEY}


def test_bare_projection_alias_is_recognised():
    """A re-export that drops the wrapper prefix must still be classified."""
    accounting = classify_keys([VISION_PROJECTION_ALIAS], 62)
    assert accounting.ignored_vision_unused == {VISION_PROJECTION_ALIAS}
    assert accounting.ignored_other == set()


def test_unknown_modalities_subtree_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unknown 'modalities.' subtree"):
        classify_keys(["modalities.brand_new_perception.encoder.weight"], 62)


def test_audio_model_prefix_matches_nothing_and_is_not_the_audio_guard():
    """`audio_model.` is in PERCEPTION_KEY_PREFIXES and matches zero tensors.

    The real audio prefix is `modalities.soft_audio_perception.`. Anything
    that did arrive under `audio_model.` is not audio-perception and must not
    be counted as such.
    """
    accounting = classify_keys(["audio_model.something.weight"], 62)
    assert accounting.ignored_audio == set()
    assert accounting.ignored_other == {"audio_model.something.weight"}


# --------------------------------------------------------------------------
# The audio guard
# --------------------------------------------------------------------------
def test_audio_tensors_are_skipped_behind_a_loud_named_guard(trtllm_warnings):
    """A silent filter is not acceptable: the skip has to be announced."""
    accounting = classify_keys(audio_keys() + text_keys(), 4)

    assert len(accounting.ignored_audio) == N_AUDIO
    assert not accounting.ignored_audio & accounting.consumed

    guard = [
        r.getMessage() for r in trtllm_warnings if "audio-perception tensors" in r.getMessage()
    ]
    assert len(guard) == 1, "the guard must fire exactly once per load, not per tensor"
    assert "modalities.soft_audio_perception" in guard[0]
    assert str(N_AUDIO) in guard[0], "the guard reports the count it found"
    assert "NO audio tower" in guard[0]


def test_audio_guard_stays_quiet_when_there_are_no_audio_tensors(trtllm_warnings):
    classify_keys(text_keys(), 4)
    assert not [r for r in trtllm_warnings if "audio-perception tensors" in r.getMessage()]


def test_load_audio_leaves_the_audio_keys_for_the_tower_walk(trtllm_warnings):
    """With an audio tower, the 72 tensors stop being 'skipped' and the guard
    stays quiet: they are left unbucketed for the audio walk to consume."""
    keys = text_keys() + all_perception_keys()
    accounting = classify_keys(keys, num_hidden_layers=4, load_vision=True, load_audio=True)

    assert accounting.ignored_audio == set()
    assert len(accounting.ignored_perception) == N_DUPLICATE + N_STALE
    assert not [r for r in trtllm_warnings if "audio-perception tensors" in r.getMessage()]


# --------------------------------------------------------------------------
# Missing-tensor detection
# --------------------------------------------------------------------------
def test_missing_vision_keys_is_empty_for_a_complete_tower():
    keys = canonical_vision_keys() + [VISION_PROJECTION_KEY]
    assert missing_vision_keys(keys, NUM_VISION_LAYERS) == []


def test_missing_vision_keys_accepts_the_bare_projection_alias():
    keys = canonical_vision_keys() + [VISION_PROJECTION_ALIAS]
    assert missing_vision_keys(keys, NUM_VISION_LAYERS) == []


def test_missing_vision_keys_catches_a_dropped_attention_bias():
    """The failure mode the duplicate copy would cause, caught by name."""
    dropped = f"{CANONICAL_VISION_PREFIX}encoder.layers.7.attn.q_proj.bias"
    keys = [k for k in canonical_vision_keys() if k != dropped] + [VISION_PROJECTION_KEY]
    assert missing_vision_keys(keys, NUM_VISION_LAYERS) == [dropped]


def test_missing_vision_keys_catches_the_whole_duplicate_copy():
    """Loading the duplicate instead of the canonical costs all 150 biases."""
    keys = [k[len(DUPLICATE_VISION_PREFIX) :] for k in duplicate_vision_keys()]
    keys = [f"{CANONICAL_VISION_PREFIX}{k}" for k in keys] + [VISION_PROJECTION_KEY]
    missing = missing_vision_keys(keys, NUM_VISION_LAYERS)
    assert len(missing) == N_BIAS_ONLY_IN_CANONICAL
    assert all(name.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")) for name in missing)


def test_missing_vision_keys_catches_an_absent_projection():
    assert missing_vision_keys(canonical_vision_keys(), NUM_VISION_LAYERS) == [
        VISION_PROJECTION_KEY
    ]


# --------------------------------------------------------------------------
# The mapper/tower seam
#
# Everything above tests names as strings. The checks below build a tower, so
# they catch the loader and the tower agreeing on the checkpoint spelling but
# not on the module spelling. Loading weights into the built tower needs a GPU
# and lives in test_musespark_open_tower_load.py.
# --------------------------------------------------------------------------
def test_mapper_and_tower_agree_on_the_projection_key():
    """One string, two files. They must not drift."""
    from tensorrt_llm._torch.models.musespark_open_vision import VISION_PROJECTION_CKPT_KEY

    assert VISION_PROJECTION_CKPT_KEY == VISION_PROJECTION_KEY


def test_every_checkpoint_name_the_loader_uses_resolves_on_the_tower():
    """The defect this section exists for, stated directly.

    `CKPT_TO_MODULE` is the tower's declaration of where each checkpoint
    suffix lands. Walk a built tower and assert every target actually resolves
    to a module -- `mlp.c_fc` did not.
    """
    from test_modeling_musespark_open_vision import _tiny_model_config  # noqa: PLC0415

    from tensorrt_llm._torch.models.musespark_open_vision import CKPT_TO_MODULE

    tower = _build_tiny_tower(_tiny_model_config())
    layer = tower.encoder.layers[0]
    for ckpt_name, module_path in CKPT_TO_MODULE.items():
        module = layer
        for part in module_path.split("."):
            assert hasattr(module, part), (
                f"{ckpt_name} maps to {module_path!r}, but {part!r} does not exist "
                f"on the built tower -- the loader would raise on a real checkpoint"
            )
            module = getattr(module, part)
        assert hasattr(module, "load_weights") or hasattr(module, "weight")


def _build_tiny_tower(model_config):
    """Construct the tiny tower with its attention kernel stubbed for CPU."""
    from types import SimpleNamespace

    import torch

    from tensorrt_llm._torch.models.musespark_open_vision import (
        MuseSparkOpenNaViTVisionTower,
        MuseSparkOpenVisionAttention,
    )

    original_forward = MuseSparkOpenVisionAttention.forward_impl
    original_prepare = MuseSparkOpenNaViTVisionTower.prepare_attn_metadata

    def _prepare(self, seq_lens, attn_metadata):
        lens = [int(s) for s in seq_lens]
        cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
        torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0, out=cu[1:])
        return SimpleNamespace(cu_q_seqlens=cu, cu_kv_seqlens=cu, seq_lens=lens)

    MuseSparkOpenNaViTVisionTower.prepare_attn_metadata = _prepare
    try:
        return MuseSparkOpenNaViTVisionTower(model_config)
    finally:
        MuseSparkOpenVisionAttention.forward_impl = original_forward
        MuseSparkOpenNaViTVisionTower.prepare_attn_metadata = original_prepare
