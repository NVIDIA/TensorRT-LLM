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
"""The Cosmos3 parameter resolver: ``select`` and the checkpoint-file reader on
synthetic inputs, the checkpoint settings on synthetic files and on the published
checkpoints, and the per-request settings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cosmos3_fixtures import bare_resolver

from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    CHECKPOINT,
    CHECKPOINT_FILE,
    CONFIG_FILE,
    DEFAULT,
    MODEL_INDEX_FILE,
    REQUEST,
    SCHEDULER_FILE,
    TRANSFORMER_FILE,
    UNSET,
    VAE_FILE,
    WORKFLOW,
    CheckpointFiles,
    Conflict,
    ParameterResolver,
    RequestParameters,
    fill_pretrained_config,
    resolve_checkpoint,
    select,
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
    return CheckpointFiles(str(tmp_path), files.get(TRANSFORMER_FILE, {}))


def _resolve(tmp_path: Path, files: dict):
    _write(tmp_path, files)
    return resolve_checkpoint(str(tmp_path), files.get(TRANSFORMER_FILE, {}))


class TestSelect:
    def test_request_over_checkpoint_over_default(self):
        assert select(1, 2, 3).origin == REQUEST
        assert select(None, 2, 3).origin == CHECKPOINT
        assert select(None, None, 3).origin == DEFAULT
        assert select(None, None, None) is None

    def test_false_and_zero_are_values(self):
        assert select(False, True).value is False
        assert select(0, 5).value == 0


class TestCheckpointFiles:
    def test_first_file_holding_the_key_wins(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG_FILE: {"a": 1}, MODEL_INDEX_FILE: {"a": 1}})
        r = files.read("a", f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a")
        assert (r.value, r.where, r.disagreeing) == (1, "config.json:a", ())

    def test_missing_and_null_fall_through(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG_FILE: {"a": None}, MODEL_INDEX_FILE: {"a": 2}})
        r = files.read("a", f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a", f"{CHECKPOINT_FILE}:a")
        assert r.value == 2
        assert files.read("b", f"{CONFIG_FILE}:b") is None

    def test_conflict_policies(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG_FILE: {"a": 1}, MODEL_INDEX_FILE: {"a": 2}})
        keys = (f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a")
        with pytest.raises(ValueError, match="model_index.json:a=2"):
            files.read("a", *keys, conflict=Conflict.ERROR)
        warned = files.read("a", *keys, conflict=Conflict.WARN)
        assert warned.value == 1 and warned.disagreeing == ("model_index.json:a=2",)
        ignored = files.read("a", *keys)
        assert ignored.value == 1 and ignored.disagreeing == ("model_index.json:a=2",)

    def test_float_lists_compare_within_tolerance(self, tmp_path):
        files = _files_on_disk(
            tmp_path,
            {CONFIG_FILE: {"a": [0.5, 1 / 3]}, MODEL_INDEX_FILE: {"a": [0.5, 0.3333333333]}},
        )
        r = files.read("a", f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a", conflict=Conflict.ERROR)
        assert r.disagreeing == ()


def _files():
    """A distilled checkpoint whose training config and diffusers index disagree on
    the system prompt."""
    return {
        TRANSFORMER_FILE: {"temporal_compression_factor": 4},
        SCHEDULER_FILE: {"fixed_step_sampler_config": {"t_list": T_LIST}},
        CONFIG_FILE: {"model": {"config": {"vlm_config": {"use_system_prompt": False}}}},
        MODEL_INDEX_FILE: {"default_use_system_prompt": True},
        VAE_FILE: {"scale_factor_temporal": 4, "scale_factor_spatial": 16},
    }


class TestCheckpoint:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolve(tmp_path, _files())
        assert cfg.prompt.use_system_prompt is False
        assert cfg.source_of("prompt.use_system_prompt") == (
            "checkpoint (config.json:model.config.vlm_config.use_system_prompt)"
        )
        assert cfg.resolved["prompt.use_system_prompt"].disagreeing == (
            "model_index.json:default_use_system_prompt=True",
        )

    def test_sections(self, tmp_path):
        cfg = _resolve(tmp_path, _files())
        assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert cfg.sampling.native_flow_schedule is False
        assert cfg.source_of("sampling.native_flow_schedule") == DEFAULT
        assert cfg.action.policy is None

    def test_vae_compression_copies_must_agree(self, tmp_path):
        files = _files()
        files[TRANSFORMER_FILE]["temporal_compression_factor"] = 8
        with pytest.raises(ValueError, match="vae.temporal_compression"):
            _resolve(tmp_path, files)

    def test_schedule_must_decrease_within_unit_interval(self, tmp_path):
        files = _files()
        files[SCHEDULER_FILE]["fixed_step_sampler_config"]["t_list"] = [0.5, 1.0]
        with pytest.raises(ValueError, match="t_list must decrease"):
            _resolve(tmp_path, files)

    def test_base_checkpoint_has_no_schedule(self, tmp_path):
        files = {k: v for k, v in _files().items() if k != SCHEDULER_FILE}
        assert _resolve(tmp_path, files).sampling.fixed_sigmas is None

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _files()
        del files[CONFIG_FILE]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX_FILE]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt"):
            _resolve(tmp_path, files)

    def test_vae_spatial_compression_is_optional(self, tmp_path):
        files = _files()
        del files[VAE_FILE]["scale_factor_spatial"]
        assert _resolve(tmp_path, files).vae.spatial_compression is None

    def test_sigmas_are_coerced_to_floats(self, tmp_path):
        files = _files()
        files[SCHEDULER_FILE]["fixed_step_sampler_config"]["t_list"] = [1, "0.5"]
        assert _resolve(tmp_path, files).sampling.fixed_sigmas == (1.0, 0.5)

    def test_fill_pretrained_config_sets_only_missing_attributes(self):
        declared = SimpleNamespace(temporal_compression_factor=8)
        fill_pretrained_config(declared)
        assert declared.temporal_compression_factor == 8
        assert declared.position_embedding_type == "unified_3d_mrope"
        assert declared.max_position_embeddings == 262144
        assert declared.temporal_compression_factor_sound == 1


# Disagreements between a checkpoint's own files that the resolver expects and resolves.
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


def _resolver(**facts) -> ParameterResolver:
    return bare_resolver(facts.pop("family", "qwen3"), **facts)


class TestRequest:
    def test_generation_values_come_from_the_tables(self):
        r = _resolver().request(RequestParameters(mode="image", height=20))
        assert r.num_inference_steps == 50  # the image table
        assert r.num_frames == 189  # the image table has none: the video table
        assert r.height == 20  # the caller's

    def test_distilled_schedule_beats_the_tables(self):
        r = _resolver(fixed_sigmas=(1.0, 0.75, 0.5, 0.25)).request(RequestParameters())
        assert (r.num_inference_steps, r.guidance_scale) == (4, 1.0)

    def test_request_beats_the_distilled_schedule(self):
        r = _resolver(fixed_sigmas=(1.0, 0.75, 0.5, 0.25)).request(
            RequestParameters(num_inference_steps=8)
        )
        assert r.num_inference_steps == 8

    def test_origins(self):
        _, _, origins = _resolver(fixed_sigmas=(1.0, 0.75, 0.5, 0.25))._resolve(
            RequestParameters(height=20)
        )
        assert (origins["height"], origins["num_inference_steps"], origins["width"]) == (
            REQUEST,
            CHECKPOINT,
            DEFAULT,
        )

    def test_action_fields_stay_none_outside_action_mode(self):
        r = _resolver().request(RequestParameters(mode="video"))
        assert (r.raw_action_dim, r.action_chunk_size, r.action_resolution) == (None, None, None)
        assert (r.action_fps, r.use_state, r.policy_active, r.preset_key) == (
            None,
            None,
            False,
            None,
        )

    def test_policy_block_applies_only_to_its_domain(self):
        policy = {"domain_name": "droid_lerobot", "action_chunk_size": 32, "num_inference_steps": 4}
        resolver = _resolver(policy=policy)
        matched, _, origins = resolver._resolve(RequestParameters(action_mode="policy"))
        assert matched.domain_name == "droid_lerobot"
        assert matched.action_chunk_size == 32
        assert origins["action_chunk_size"] == CHECKPOINT
        other, warnings, origins = resolver._resolve(
            RequestParameters(action_mode="policy", domain_name="av")
        )
        assert other.action_chunk_size == 60
        assert origins["action_chunk_size"] == DEFAULT
        assert any("overrides checkpoint policy domain_name" in w for w in warnings)

    def test_policy_block_needs_policy_mode(self):
        policy = {"domain_name": "droid_lerobot", "action_chunk_size": 32, "num_inference_steps": 4}
        resolver = _resolver(family="nemotron_dense", policy=policy)
        in_policy = resolver.request(RequestParameters(action_mode="policy"))
        assert (
            in_policy.policy_active,
            in_policy.action_chunk_size,
            in_policy.num_inference_steps,
        ) == (True, 32, 4)
        dynamics = resolver.request(RequestParameters(action_mode="forward_dynamics"))
        assert dynamics.policy_active is False
        assert dynamics.num_inference_steps == 30  # the action table, not the block

    def test_caller_value_wins_and_the_recommendation_is_reported(self):
        params, warnings, origins = _resolver()._resolve(
            RequestParameters(action_mode="action", domain_name="av", action_chunk_size=16)
        )
        assert params.action_chunk_size == 16
        assert origins["action_chunk_size"] == REQUEST
        assert warnings == [
            "Cosmos3 action_chunk_size=16 differs from recommended 60 for domain 'av'."
        ]

    def test_table_values_are_not_recommendations(self):
        """A caller's height that differs from the video table is not a warning: only
        the policy block, the domain preset and the width table recommend."""
        _, warnings, _ = _resolver()._resolve(RequestParameters(height=20))
        assert warnings == []

    def test_explicit_none_in_a_mode_table_is_a_value(self):
        """Edge text-to-image switches the guidance window off with an explicit ``None``;
        that must not fall through to the video table."""
        resolver = _resolver(family="nemotron_dense")
        _, _, origins = resolver._resolve(RequestParameters(mode="image"))
        assert resolver.request(RequestParameters(mode="image")).guidance_interval is None
        assert origins["guidance_interval"] == UNSET

    def test_flow_shift_is_the_checkpoint_scheduler_unless_requested(self):
        resolver = _resolver(flow_shift=5.0)
        assert resolver.request(RequestParameters()).flow_shift == 5.0
        assert resolver.request(RequestParameters(mode="image")).flow_shift == 5.0
        assert resolver.request(RequestParameters(flow_shift=3.0)).flow_shift == 3.0
        assert _resolver().request(RequestParameters()).flow_shift is None  # distilled

    def test_system_prompt_rule(self):
        resolver = _resolver(use_system_prompt=True)
        assert resolver.request(RequestParameters()).use_system_prompt is True
        assert resolver.request(RequestParameters(is_transfer=True)).use_system_prompt is False
        assert _resolver().request(RequestParameters(is_v2v=True)).use_system_prompt is True
        assert (
            resolver.request(RequestParameters(use_system_prompt=False)).use_system_prompt is False
        )

    def test_unresolvable_field_is_an_error(self):
        with pytest.raises(ValueError, match="cannot resolve num_inference_steps"):
            _resolver(family="no-such-family").request(RequestParameters())

    def test_action_fps_follows_the_frame_rate(self):
        r = _resolver().request(RequestParameters(action_mode="forward_dynamics", domain_name="av"))
        assert (r.frame_rate, r.action_fps, r.num_frames) == (10.0, 10.0, 61)
        assert r.preset_key == "av"

    def test_workflow_rules_are_tagged(self):
        _, _, origins = _resolver()._resolve(RequestParameters(is_transfer=True))
        assert origins["use_system_prompt"] == WORKFLOW
        _, _, origins = _resolver()._resolve(RequestParameters())
        assert origins["use_system_prompt"] == CHECKPOINT
