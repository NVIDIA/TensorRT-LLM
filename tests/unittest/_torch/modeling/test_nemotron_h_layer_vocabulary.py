# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nemotron-H layer-vocabulary compatibility.

transformers 5.13 renamed the per-layer vocabulary and stopped serializing
``hybrid_override_pattern``, so a checkpoint re-saved under a newer transformers
cannot be parsed by the pinned one. The pattern drives layer counts and the
mamba/attention masks, so it has to be restored rather than the error suppressed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_nemotron_hybrid,
    load_pretrained_config,
    match_nemotron_h_layer_types,
    nemotron_h_legacy_layer_types,
)

_NEW = ["linear_attention", "moe", "linear_attention", "full_attention"]
_LEGACY = ["mamba", "moe", "mamba", "attention"]
_PATTERN = "MEM*"


def _load(**config):
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps({"model_type": "nemotron_h", **config}))
        return load_pretrained_config(d)


class TestLoader(unittest.TestCase):
    """load_pretrained_config's fallback branch."""

    def test_renamed_checkpoint_regains_its_pattern(self):
        config = _load(layers_block_type=_NEW)
        self.assertEqual(config.hybrid_override_pattern, _PATTERN)
        self.assertTrue(is_nemotron_hybrid(config))

    def test_parsable_checkpoint_keeps_its_own_result(self):
        self.assertEqual(_load(layers_block_type=_LEGACY).hybrid_override_pattern, _PATTERN)

    def test_failure_the_rename_cannot_explain_propagates(self):
        with self.assertRaises(Exception):
            _load(layers_block_type=["not_a_layer_type"])


class TestRename(unittest.TestCase):
    """nemotron_h_legacy_layer_types: which configs it claims, and what it edits."""

    def test_declines_configs_it_does_not_own(self):
        for config in (
            {"model_type": "llama", "layers_block_type": _NEW},
            {"model_type": "nemotron_h", "layers_block_type": _LEGACY},
            {"model_type": "nemotron_h"},
        ):
            self.assertIsNone(nemotron_h_legacy_layer_types(config), config)

    def test_covers_every_layer_type_field(self):
        renamed = nemotron_h_legacy_layer_types(
            {
                "model_type": "nemotron_h",
                "layers_block_type": _NEW,
                "mtp_layers_block_type": ["full_attention", "moe"],
            }
        )
        self.assertEqual(renamed["layers_block_type"], _LEGACY)
        self.assertEqual(renamed["mtp_layers_block_type"], ["attention", "moe"])


class TestVocabularyMatch(unittest.TestCase):
    """match_nemotron_h_layer_types, used where assignment skips validation."""

    def test_follows_the_target_vocabulary_from_either_spelling(self):
        for draft in (["full_attention", "moe"], ["attention", "moe"]):
            target = _load(layers_block_type=_NEW)
            target.mtp_layers_block_type = match_nemotron_h_layer_types(target, draft)
            self.assertEqual(target.mtp_hybrid_override_pattern, "*E", draft)

    def test_preserves_layer_types_outside_the_mapping(self):
        self.assertEqual(
            match_nemotron_h_layer_types(_load(layers_block_type=_NEW), ["mlp"]), ["mlp"]
        )


if __name__ == "__main__":
    unittest.main()
