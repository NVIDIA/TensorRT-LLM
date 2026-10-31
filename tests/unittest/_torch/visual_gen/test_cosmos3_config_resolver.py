# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Cosmos3 precedence table, on synthetic files and on the published checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.config_resolver import (
    CHECKPOINT,
    CONFIG,
    MODEL_INDEX,
    SCHEDULER,
    TRANSFORMER,
    VAE,
    Conflict,
    Cosmos3ConfigResolver,
    Default,
    Key,
    Rule,
    fill_pretrained_config,
)

pytestmark = pytest.mark.cpu_only

FIXTURE = Path(__file__).parent / "test_data" / "cosmos3_checkpoint_configs.json"
T_LIST = [1.0, 0.9375, 0.8333333333333334, 0.625]


def _write(checkpoint_dir: Path, files: dict) -> Path:
    for name, doc in files.items():
        path = checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
    return checkpoint_dir


def _resolver(tmp_path: Path, files: dict) -> Cosmos3ConfigResolver:
    _write(tmp_path, files)
    return Cosmos3ConfigResolver(str(tmp_path), files.get(TRANSFORMER, {}))


class TestRule:
    def test_first_present_source_wins(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 1}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Default(7, "x")))
        assert r.resolve_one("a", rule).value == 1
        assert r.resolve_one("a", rule).source == "config.json:a"

    def test_falls_through_missing_and_null_sources(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": None}, MODEL_INDEX: {"a": 2}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Key(CHECKPOINT, "a")))
        assert r.resolve_one("a", rule).value == 2

    def test_default_only_when_nothing_is_declared(self, tmp_path):
        r = _resolver(tmp_path, {})
        rule = Rule((Key(CONFIG, "a"), Default(7, "x")))
        resolved = r.resolve_one("a", rule)
        assert (resolved.value, resolved.source) == (7, "default (x)")

    def test_unresolvable_required_setting_is_an_error(self, tmp_path):
        r = _resolver(tmp_path, {})
        with pytest.raises(ValueError, match="cannot resolve a"):
            r.resolve_one("a", Rule((Key(CONFIG, "a"),)))
        assert r.resolve_one("a", Rule((Key(CONFIG, "a"),), required=False)) is None

    def test_conflict_policies(self, tmp_path):
        r = _resolver(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 2}})
        sources = (Key(CONFIG, "a"), Key(MODEL_INDEX, "a"))
        with pytest.raises(ValueError, match="model_index.json:a=2"):
            r.resolve_one("a", Rule(sources, conflict=Conflict.ERROR))
        warned = r.resolve_one("a", Rule(sources, conflict=Conflict.WARN))
        assert warned.value == 1 and warned.disagreeing == ("model_index.json:a=2",)

    def test_float_lists_compare_within_tolerance(self, tmp_path):
        r = _resolver(
            tmp_path, {CONFIG: {"a": [0.5, 1 / 3]}, MODEL_INDEX: {"a": [0.5, 0.3333333333]}}
        )
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a")), conflict=Conflict.ERROR)
        assert r.resolve_one("a", rule).disagreeing == ()


def _files():
    """A distilled checkpoint whose training config and diffusers index disagree on
    the system prompt."""
    return {
        TRANSFORMER: {"temporal_compression_factor": 4},
        SCHEDULER: {"fixed_step_sampler_config": {"t_list": T_LIST}},
        CONFIG: {"model": {"config": {"vlm_config": {"use_system_prompt": False}}}},
        MODEL_INDEX: {"default_use_system_prompt": True},
        VAE: {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
    }


class TestResolve:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolver(tmp_path, _files()).resolve()
        assert cfg.prompt.use_system_prompt is False
        assert cfg.source_of("prompt.use_system_prompt") == (
            "config.json:model.config.vlm_config.use_system_prompt"
        )
        assert cfg.resolved["prompt.use_system_prompt"].disagreeing == (
            "model_index.json:default_use_system_prompt=True",
        )

    def test_sections(self, tmp_path):
        cfg = _resolver(tmp_path, _files()).resolve()
        assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert cfg.sampling.native_flow_schedule is False
        assert cfg.source_of("sampling.native_flow_schedule").startswith("default (")
        assert cfg.action.policy is None

    def test_vae_compression_copies_must_agree(self, tmp_path):
        files = _files()
        files[TRANSFORMER]["temporal_compression_factor"] = 8
        with pytest.raises(ValueError, match="vae.temporal_compression"):
            _resolver(tmp_path, files).resolve()

    def test_schedule_must_decrease_within_unit_interval(self, tmp_path):
        files = _files()
        files[SCHEDULER]["fixed_step_sampler_config"]["t_list"] = [0.5, 1.0]
        with pytest.raises(ValueError, match="t_list must decrease"):
            _resolver(tmp_path, files).resolve()

    def test_base_checkpoint_has_no_schedule(self, tmp_path):
        files = {k: v for k, v in _files().items() if k != SCHEDULER}
        cfg = _resolver(tmp_path, files).resolve()
        assert cfg.sampling.fixed_sigmas is None

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _files()
        del files[CONFIG]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt"):
            _resolver(tmp_path, files).resolve()

    def test_spatial_compression_must_be_declared(self, tmp_path):
        files = _files()
        del files[VAE]["scale_factor_spatial"]
        with pytest.raises(ValueError, match="cannot resolve vae.spatial_compression"):
            _resolver(tmp_path, files).resolve()


def test_fill_pretrained_config_sets_only_missing_attributes():
    declared = SimpleNamespace(temporal_compression_factor=8)
    fill_pretrained_config(declared)
    assert declared.temporal_compression_factor == 8
    assert declared.position_embedding_type == "unified_3d_mrope"
    assert declared.max_position_embeddings == 262144
    assert declared.temporal_compression_factor_sound == 1


# Disagreements between a checkpoint's own files that the table expects and resolves.
EXPECTED_DISAGREEMENTS = {
    "nvidia/Cosmos3-Nano-Sim-Bimanual": {"prompt.use_system_prompt"},
}

EXPECTED = {
    # repo: (use_system_prompt, native_flow_schedule, distilled)
    "nvidia/Cosmos3-Nano": (False, False, False),
    "nvidia/Cosmos3-Super": (False, False, False),
    "nvidia/Cosmos3-Edge": (False, True, False),
    "nvidia/Cosmos3-Super-Image2Video": (False, False, False),
    "nvidia/Cosmos3-Super-Image2Video-4Step": (False, False, True),
    "nvidia/Cosmos3-Super-Text2Image": (False, False, False),
    "nvidia/Cosmos3-Super-Text2Image-4Step": (False, False, True),
    "nvidia/Cosmos3-Nano-Policy-DROID": (False, False, False),
    "nvidia/Cosmos3-Edge-Policy-DROID": (False, True, False),
    "nvidia/Cosmos3-Nano-Sim-Bimanual": (False, False, True),
}


@pytest.mark.parametrize("repo", sorted(EXPECTED))
def test_published_checkpoint_resolves(tmp_path, repo):
    files = json.loads(FIXTURE.read_text())[repo]
    cfg = _resolver(tmp_path, files).resolve()
    system_prompt, native, distilled = EXPECTED[repo]
    assert cfg.prompt.use_system_prompt is system_prompt
    assert cfg.sampling.native_flow_schedule is native
    assert (cfg.sampling.fixed_sigmas is not None) is distilled
    assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
    assert cfg.architecture.position_embedding_type == "unified_3d_mrope"
    disagreeing = {name for name, r in cfg.resolved.items() if r.disagreeing}
    assert disagreeing == EXPECTED_DISAGREEMENTS.get(repo, set())


def test_fixture_covers_every_expected_checkpoint():
    assert set(json.loads(FIXTURE.read_text())) == set(EXPECTED)
