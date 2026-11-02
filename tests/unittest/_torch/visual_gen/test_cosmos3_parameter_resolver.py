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
"""The Cosmos3 precedence table: the engine on synthetic sources, the checkpoint rules
on synthetic files and on the published checkpoints, and the request rules."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    ACTION_RULES,
    CHECKPOINT,
    CHECKPOINT_RULES,
    CONFIG,
    GENERATION_RULES,
    MODEL_INDEX,
    SCHEDULER,
    TRANSFORMER,
    VAE,
    CheckpointFiles,
    Conflict,
    Default,
    Given,
    Key,
    RequestContext,
    Rule,
    Table,
    check_precedence,
    fill_pretrained_config,
    resolve_action,
    resolve_checkpoint,
    resolve_domain_action_config,
    resolve_generation,
    resolve_setting,
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


def _files_on_disk(tmp_path: Path, files: dict) -> CheckpointFiles:
    _write(tmp_path, files)
    return CheckpointFiles(str(tmp_path), files.get(TRANSFORMER, {}))


def _resolve(tmp_path: Path, files: dict):
    _write(tmp_path, files)
    return resolve_checkpoint(str(tmp_path), files.get(TRANSFORMER, {}))


class TestEngine:
    def test_first_present_source_wins(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 1}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Default(7, "x")))
        assert resolve_setting("a", rule, files).value == 1
        assert resolve_setting("a", rule, files).source == "config.json:a"

    def test_falls_through_missing_and_null_sources(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG: {"a": None}, MODEL_INDEX: {"a": 2}})
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a"), Key(CHECKPOINT, "a")))
        assert resolve_setting("a", rule, files).value == 2

    def test_default_only_when_nothing_is_declared(self, tmp_path):
        files = _files_on_disk(tmp_path, {})
        rule = Rule((Key(CONFIG, "a"), Default(7, "x")))
        resolved = resolve_setting("a", rule, files)
        assert (resolved.value, resolved.source) == (7, "default (x)")

    def test_unresolvable_required_setting_is_an_error(self, tmp_path):
        files = _files_on_disk(tmp_path, {})
        with pytest.raises(ValueError, match="cannot resolve a"):
            resolve_setting("a", Rule((Key(CONFIG, "a"),)), files)
        assert resolve_setting("a", Rule((Key(CONFIG, "a"),), required=False), files) is None

    def test_conflict_policies(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG: {"a": 1}, MODEL_INDEX: {"a": 2}})
        sources = (Key(CONFIG, "a"), Key(MODEL_INDEX, "a"))
        with pytest.raises(ValueError, match="model_index.json:a=2"):
            resolve_setting("a", Rule(sources, conflict=Conflict.ERROR), files)
        warned = resolve_setting("a", Rule(sources, conflict=Conflict.WARN), files)
        assert warned.value == 1 and warned.disagreeing == ("model_index.json:a=2",)
        ignored = resolve_setting("a", Rule(sources), files)
        assert ignored.value == 1 and ignored.disagreeing == ("model_index.json:a=2",)

    def test_float_lists_compare_within_tolerance(self, tmp_path):
        files = _files_on_disk(
            tmp_path, {CONFIG: {"a": [0.5, 1 / 3]}, MODEL_INDEX: {"a": [0.5, 0.3333333333]}}
        )
        rule = Rule((Key(CONFIG, "a"), Key(MODEL_INDEX, "a")), conflict=Conflict.ERROR)
        assert resolve_setting("a", rule, files).disagreeing == ()

    def test_every_rule_keeps_request_over_checkpoint_over_default(self):
        for rules in (CHECKPOINT_RULES, GENERATION_RULES, ACTION_RULES):
            check_precedence(rules)
        with pytest.raises(ValueError, match="breaks the request > checkpoint > default order"):
            check_precedence({"x": Rule((Default(1, "first"), Given()))})
        with pytest.raises(ValueError, match="breaks the request > checkpoint > default order"):
            check_precedence({"x": Rule((Table("x"), Given()))})


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


class TestCheckpoint:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolve(tmp_path, _files())
        assert cfg.prompt.use_system_prompt is False
        assert cfg.source_of("prompt.use_system_prompt") == (
            "config.json:model.config.vlm_config.use_system_prompt"
        )
        assert cfg.resolved["prompt.use_system_prompt"].disagreeing == (
            "model_index.json:default_use_system_prompt=True",
        )

    def test_sections(self, tmp_path):
        cfg = _resolve(tmp_path, _files())
        assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert cfg.sampling.native_flow_schedule is False
        assert cfg.source_of("sampling.native_flow_schedule").startswith("default (")
        assert cfg.action.policy is None

    def test_vae_compression_copies_must_agree(self, tmp_path):
        files = _files()
        files[TRANSFORMER]["temporal_compression_factor"] = 8
        with pytest.raises(ValueError, match="vae.temporal_compression"):
            _resolve(tmp_path, files)

    def test_schedule_must_decrease_within_unit_interval(self, tmp_path):
        files = _files()
        files[SCHEDULER]["fixed_step_sampler_config"]["t_list"] = [0.5, 1.0]
        with pytest.raises(ValueError, match="t_list must decrease"):
            _resolve(tmp_path, files)

    def test_base_checkpoint_has_no_schedule(self, tmp_path):
        files = {k: v for k, v in _files().items() if k != SCHEDULER}
        assert _resolve(tmp_path, files).sampling.fixed_sigmas is None

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _files()
        del files[CONFIG]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt"):
            _resolve(tmp_path, files)

    def test_spatial_compression_must_be_declared(self, tmp_path):
        files = _files()
        del files[VAE]["scale_factor_spatial"]
        with pytest.raises(ValueError, match="cannot resolve vae.spatial_compression"):
            _resolve(tmp_path, files)

    def test_fill_pretrained_config_sets_only_missing_attributes(self):
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
    cfg = _resolve(tmp_path, files)
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


class TestRequest:
    def test_generation_values_name_their_source(self):
        ctx = RequestContext(family="qwen3", mode="image")
        r = resolve_generation(ctx, num_inference_steps=None, num_frames=None, height=20)
        assert r.source_of("num_inference_steps") == "mode table.num_inference_steps"
        assert r["num_inference_steps"] == 50
        assert r.source_of("num_frames") == "video table.num_frames"  # the image table has none
        assert (r["height"], r.source_of("height")) == (20, "request")

    def test_distilled_schedule_beats_the_tables(self):
        ctx = RequestContext(family="qwen3", distilled_steps=4)
        r = resolve_generation(ctx, num_inference_steps=None, guidance_scale=None)
        assert (r["num_inference_steps"], r["guidance_scale"]) == (4, 1.0)
        assert r.source_of("guidance_scale").startswith("checkpoint:fixed_step_sampler_config")

    def test_request_beats_the_distilled_schedule(self):
        """The caller's value wins over the checkpoint, and the checkpoint's value is
        recorded as the disagreeing source."""
        ctx = RequestContext(family="qwen3", distilled_steps=4)
        r = resolve_generation(ctx, num_inference_steps=8)
        assert (r["num_inference_steps"], r.source_of("num_inference_steps")) == (8, "request")
        assert r.values["num_inference_steps"].disagreeing == (
            "checkpoint:fixed_step_sampler_config (steps)=4",
        )

    def test_policy_block_applies_only_to_its_domain(self):
        policy = {"domain_name": "droid_lerobot", "action_chunk_size": 32, "num_inference_steps": 4}
        matched = resolve_domain_action_config(checkpoint_policy_defaults=policy)
        assert matched["domain_name"] == "droid_lerobot"
        assert matched["action_chunk_size"] == 32
        assert matched["sources"]["action_chunk_size"] == "checkpoint.json:policy.action_chunk_size"
        other = resolve_domain_action_config(domain_name="av", checkpoint_policy_defaults=policy)
        assert other["action_chunk_size"] == 60
        assert other["sources"]["action_chunk_size"] == "domain preset.action_chunk_size"
        assert any("overrides checkpoint policy domain_name" in w for w in other["warnings"])

    def test_caller_value_wins_and_the_recommendation_is_reported(self):
        cfg = resolve_domain_action_config(domain_name="av", action_chunk_size=16)
        assert cfg["action_chunk_size"] == 16
        assert cfg["sources"]["action_chunk_size"] == "request"
        assert cfg["warnings"] == [
            "Cosmos3 action_chunk_size=16 differs from recommended 60 for domain 'av'."
        ]

    def test_explicit_none_in_a_mode_table_is_a_value(self):
        """Edge text-to-image switches the guidance window off with an explicit ``None``;
        that must not fall through to the video table."""
        ctx = RequestContext(family="nemotron_dense", mode="image")
        r = resolve_generation(ctx, guidance_interval=None)
        assert r["guidance_interval"] is None
        assert r.source_of("guidance_interval") == "mode table.guidance_interval"

    def test_unresolvable_field_is_an_error(self):
        ctx = RequestContext(family="no-such-family")
        with pytest.raises(ValueError, match="cannot resolve num_frames"):
            resolve_generation(ctx, num_frames=None)

    def test_action_fps_follows_the_frame_rate(self):
        ctx = RequestContext(family="qwen3", action_mode="forward_dynamics", domain_name="av")
        r = resolve_action(
            ctx,
            raw_action_dim=None,
            action_chunk_size=None,
            action_resolution=None,
            frame_rate=None,
            action_fps=None,
            use_state=None,
        )
        assert (r["frame_rate"], r["action_fps"], r["num_frames"]) == (10.0, 10.0, 61)
        assert r.source_of("action_fps") == "frame_rate"
