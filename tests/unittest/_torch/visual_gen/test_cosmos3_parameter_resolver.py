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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cosmos3_fixtures import VAE_CONFIG, bare_config, bare_resolver, write_checkpoint_files

from tensorrt_llm._torch.visual_gen.models.cosmos3.overrides import ParameterOverride
from tensorrt_llm._torch.visual_gen.models.cosmos3.parameter_resolver import (
    CHECKPOINT,
    CHECKPOINT_FILE,
    CONFIG_FILE,
    DEFAULT,
    MODEL_INDEX_FILE,
    OVERRIDE,
    REQUEST,
    SCHEDULER_FILE,
    SIM_CONFIG_KEY,
    TRANSFORMER_FILE,
    UNSET,
    VAE_FILE,
    CheckpointFiles,
    Conflict,
    Cosmos3CheckpointSettings,
    Cosmos3Embodiment,
    Cosmos3RolloutSettings,
    ParameterResolver,
    RequestParameters,
    fill_pretrained_config,
    resolve_checkpoint,
    select,
)

pytestmark = pytest.mark.cpu_only

FIXTURE = Path(__file__).parent / "test_data" / "cosmos3_checkpoint_configs.json"
T_LIST = [1.0, 0.9375, 0.8333333333333334, 0.625]


def _files_on_disk(tmp_path: Path, files: dict) -> CheckpointFiles:
    write_checkpoint_files(tmp_path, files)
    return CheckpointFiles(str(tmp_path))


def _resolve(tmp_path: Path, files: dict) -> Cosmos3CheckpointSettings:
    write_checkpoint_files(tmp_path, files)
    return resolve_checkpoint(str(tmp_path))


class TestSelect:
    def test_request_over_checkpoint_over_default(self):
        assert select(1, 2, 3).origin == REQUEST
        assert select(None, 2, 3).origin == CHECKPOINT
        assert select(None, None, 3).origin == DEFAULT
        assert select(None, None, None) is None

    def test_an_override_sits_between_the_request_and_the_checkpoint(self):
        always = ParameterOverride(9)
        assert select(1, 2, 3, override=always).origin == REQUEST
        assert select(None, 2, 3, override=always) == select(None, None, None, override=always)
        assert select(None, 2, 3, override=always, where="row").source == "override (row)"
        assert select(None, 2, 3, override=always).value == 9

    def test_an_override_that_names_a_value_replaces_only_that_value(self):
        keyed = ParameterOverride(9, replaces=2)
        assert select(None, 2, 3, override=keyed).origin == OVERRIDE
        assert select(None, 5, 3, override=keyed).origin == CHECKPOINT
        assert select(None, None, 3, override=keyed).origin == DEFAULT
        assert select(None, None, None, override=keyed) is None

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

    def test_nullable_keeps_an_explicit_null(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG_FILE: {"a": None}, MODEL_INDEX_FILE: {"a": 2}})
        keys = (f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a")
        r = files.read("a", *keys, nullable=True)
        assert (r.value, r.where) == (None, "config.json:a")
        assert r.disagreeing == ("model_index.json:a=2",)
        assert files.read("a", *keys).value == 2
        assert files.read("b", f"{CONFIG_FILE}:b", nullable=True) is None

    def test_every_later_key_is_compared_with_the_winner(self, tmp_path):
        files = _files_on_disk(
            tmp_path,
            {
                MODEL_INDEX_FILE: {"a": 1},
                TRANSFORMER_FILE: {"a": 1},
                CHECKPOINT_FILE: {"a": 3},
                VAE_FILE: {"a": 4},
            },
        )
        keys = tuple(
            f"{file}:a"
            for file in (CONFIG_FILE, MODEL_INDEX_FILE, TRANSFORMER_FILE, CHECKPOINT_FILE, VAE_FILE)
        )
        r = files.read("a", *keys)
        assert (r.value, r.where) == (1, "model_index.json:a")
        assert r.disagreeing == ("checkpoint.json:a=3", "vae/config.json:a=4")
        with pytest.raises(ValueError, match=r"checkpoint\.json:a=3; vae/config\.json:a=4"):
            files.read("a", *keys, conflict=Conflict.ERROR)

    def test_conflict_policies(self, tmp_path):
        files = _files_on_disk(tmp_path, {CONFIG_FILE: {"a": 1}, MODEL_INDEX_FILE: {"a": 2}})
        keys = (f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a")
        with pytest.raises(ValueError, match="model_index.json:a=2"):
            files.read("a", *keys, conflict=Conflict.ERROR)
        warned = files.read("a", *keys, conflict=Conflict.WARN)
        assert warned.value == 1 and warned.disagreeing == ("model_index.json:a=2",)
        ignored = files.read("a", *keys)
        assert ignored.value == 1 and ignored.disagreeing == ("model_index.json:a=2",)

    @pytest.mark.parametrize("log", [True, False])
    def test_disagreement_warning_follows_the_log_flag(self, tmp_path, monkeypatch, log):
        from tensorrt_llm._torch.visual_gen.models.cosmos3 import parameter_resolver

        messages = []
        monkeypatch.setattr(parameter_resolver.logger, "warning", messages.append)
        write_checkpoint_files(tmp_path, {CONFIG_FILE: {"a": 1}, MODEL_INDEX_FILE: {"a": 2}})
        files = CheckpointFiles(str(tmp_path), log=log)
        files.read("a", f"{CONFIG_FILE}:a", f"{MODEL_INDEX_FILE}:a", conflict=Conflict.WARN)
        assert len(messages) == (1 if log else 0)

    def test_invalid_json_names_the_file(self, tmp_path):
        (tmp_path / CONFIG_FILE).write_text("{not json")
        with pytest.raises(ValueError, match=r"config\.json is not valid JSON"):
            CheckpointFiles(str(tmp_path)).value(f"{CONFIG_FILE}:a")


def _files() -> dict[str, dict]:
    """A distilled checkpoint whose training config and diffusers index disagree on
    the system prompt."""
    return {
        TRANSFORMER_FILE: {"temporal_compression_factor": 4},
        SCHEDULER_FILE: {"fixed_step_sampler_config": {"t_list": T_LIST}},
        CONFIG_FILE: {"model": {"config": {"vlm_config": {"use_system_prompt": False}}}},
        MODEL_INDEX_FILE: {"default_use_system_prompt": True},
        VAE_FILE: dict(VAE_CONFIG),
    }


class TestCheckpoint:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolve(tmp_path, _files())
        assert cfg.prompt.use_system_prompt is False
        assert cfg.resolved["prompt.use_system_prompt"].source == (
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
        assert cfg.resolved["sampling.native_flow_schedule"].source == DEFAULT
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

    @pytest.mark.parametrize("shift", [0, -1.0, float("inf"), float("nan")])
    def test_scheduler_shift_must_be_positive_and_finite(self, tmp_path, shift):
        files = _files()
        files[SCHEDULER_FILE]["flow_shift"] = shift
        with pytest.raises(ValueError, match="flow_shift must be a positive finite number"):
            _resolve(tmp_path, files)

    def test_base_checkpoint_has_no_schedule(self, tmp_path):
        files = {k: v for k, v in _files().items() if k != SCHEDULER_FILE}
        assert _resolve(tmp_path, files).sampling.fixed_sigmas is None

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _files()
        del files[CONFIG_FILE]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX_FILE]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt") as error:
            _resolve(tmp_path, files)
        assert "config.json:model.config.vlm_config.use_system_prompt" in str(error.value)
        assert "model_index.json:default_use_system_prompt" in str(error.value)

    def test_vae_spatial_compression_is_optional(self, tmp_path):
        files = _files()
        del files[VAE_FILE]["scale_factor_spatial"]
        assert _resolve(tmp_path, files).vae.spatial_compression is None

    @pytest.mark.parametrize("key", ["scale_factor_temporal", "scale_factor_spatial"])
    def test_a_shipped_vae_must_declare_its_compression(self, tmp_path, key):
        files = _files()
        _resolve(tmp_path, files).require_vae_compression(str(tmp_path))
        del files[VAE_FILE][key]
        cfg = _resolve(tmp_path, files)
        with pytest.raises(ValueError, match=f"declares no {key}"):
            cfg.require_vae_compression(str(tmp_path))

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


SIM_CONDITIONING = {
    "action_tokens_per_frame": 4,
    "embodiments": {"agibotworld": {"domain_id": 15, "input_action_dim": 59}},
    "default_embodiment": "agibotworld",
}


def _sim_files(**sim_overrides) -> dict[str, dict]:
    """A checkpoint that generates chunk by chunk: the rollout block in the transformer
    config, with copies of some of its keys in the diffusers index and ``checkpoint.json``."""
    block = {
        "chunk_size": 2,
        "window_frames": None,
        "sink_frames": 0,
        "text_cache_max_len": 1024,
        "video_temporal_causal": True,
        "fixed_step_sampler_config": {"t_list": T_LIST},
        "conditioning": dict(SIM_CONDITIONING),
        **sim_overrides,
    }
    return {
        TRANSFORMER_FILE: {SIM_CONFIG_KEY: block},
        SCHEDULER_FILE: {"fixed_step_sampler_config": {"t_list": T_LIST}},
        CONFIG_FILE: {"model": {"config": {"vlm_config": {"use_system_prompt": False}}}},
        MODEL_INDEX_FILE: {
            "default_use_system_prompt": True,
            f"_{SIM_CONFIG_KEY}": {"text_cache_max_len": 512},
        },
        CHECKPOINT_FILE: {SIM_CONFIG_KEY: {"chunk_partition": [1, 2], "text_cache_max_len": 512}},
        VAE_FILE: dict(VAE_CONFIG),
    }


class TestSim:
    def test_training_config_beats_the_diffusers_index(self, tmp_path):
        cfg = _resolve(tmp_path, _sim_files())
        assert cfg.prompt.use_system_prompt is False
        assert cfg.resolved["prompt.use_system_prompt"].source == (
            "checkpoint (config.json:model.config.vlm_config.use_system_prompt)"
        )
        assert cfg.rollout.text_cache_max_len == 1024
        assert len(cfg.resolved["rollout.text_cache_max_len"].disagreeing) == 2

    def test_rollout_sections(self, tmp_path):
        cfg = _resolve(tmp_path, _sim_files())
        rollout = cfg.rollout
        assert rollout is not None
        assert (rollout.chunk_frames, rollout.chunk_partition) == (2, (1, 2))
        assert (rollout.inference_size, rollout.sink_frames) == (None, 0)
        assert rollout.history_frames is None
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert rollout.action_tokens_per_frame == 4

    def test_partition_must_match_the_reference_rollout(self, tmp_path):
        files = _sim_files()
        files[CHECKPOINT_FILE][SIM_CONFIG_KEY]["chunk_partition"] = [1, 4]
        with pytest.raises(ValueError, match="reference rollout"):
            _resolve(tmp_path, files)

    def test_missing_partition_follows_the_reference_rollout(self, tmp_path):
        files = _sim_files()
        del files[CHECKPOINT_FILE][SIM_CONFIG_KEY]["chunk_partition"]
        cfg = _resolve(tmp_path, files)
        assert cfg.rollout.chunk_partition == (1, 2)
        assert cfg.resolved["rollout.chunk_partition"].source == "default (reference rollout)"

    def test_window_overrides_full_history(self, tmp_path):
        """The reference recipe: 64 frames counting 8 sinks and the frame itself."""
        files = _sim_files()
        files[CONFIG_FILE]["model"]["config"].update(
            kv_cache_inference_size=None, attention_sink_size=0
        )
        cfg = _resolve(tmp_path, files)
        rollout = cfg.rollout.with_window(64, 8)
        assert (rollout.inference_size, rollout.sink_frames) == (64, 8)
        assert rollout.history_frames == 55
        assert rollout == replace(cfg.rollout, inference_size=64, sink_frames=8)
        assert cfg.resolved["rollout.sink_frames"].disagreeing == ()

    def test_unset_window_keeps_the_checkpoint(self, tmp_path):
        files = _sim_files()
        files[CONFIG_FILE]["model"]["config"].update(
            kv_cache_inference_size=None, attention_sink_size=0
        )
        cfg = _resolve(tmp_path, files)
        rollout = cfg.rollout.with_window(None, None)
        assert rollout.inference_size is None and rollout.sink_frames == 0
        assert cfg.resolved["rollout.inference_size"].source == (
            "checkpoint (config.json:model.config.kv_cache_inference_size)"
        )

    def test_sim_block_window_counts_the_chunk_first_frame(self, tmp_path):
        rollout = _resolve(tmp_path, _sim_files(window_frames=96)).rollout
        assert (rollout.inference_size, rollout.history_frames) == (96, 95)

    def test_sim_block_window_counts_the_sinks(self, tmp_path):
        """The block's window is the training config's size under another name: 64
        with 8 sinks leaves 55 recent frames, whichever of the two declares it."""
        cfg = _resolve(tmp_path, _sim_files(window_frames=64, sink_frames=8))
        rollout = cfg.rollout
        assert (rollout.inference_size, rollout.sink_frames) == (64, 8)
        assert rollout.history_frames == 55
        assert rollout == cfg.rollout.with_window(64, 8)

    def test_schedule_copies_must_agree(self, tmp_path):
        files = _sim_files(fixed_step_sampler_config={"t_list": [1.0, 0.5]})
        with pytest.raises(ValueError, match="sampling.fixed_sigmas"):
            _resolve(tmp_path, files)

    def test_schedule_from_the_rollout_block_alone(self, tmp_path):
        files = _sim_files()
        del files[SCHEDULER_FILE]["fixed_step_sampler_config"]
        cfg = _resolve(tmp_path, files)
        assert cfg.sampling.fixed_sigmas == tuple(T_LIST)
        assert cfg.resolved["sampling.fixed_sigmas"].where.startswith(TRANSFORMER_FILE)

    def test_action_rows_must_match_the_vae_temporal_compression(self, tmp_path):
        files = _sim_files(conditioning=dict(SIM_CONDITIONING, action_tokens_per_frame=2))
        with pytest.raises(ValueError, match="action_tokens_per_frame is 2"):
            _resolve(tmp_path, files)

    def test_default_embodiment_must_be_declared(self, tmp_path):
        """A request that names no embodiment takes the checkpoint's default, so a Sim
        checkpoint must declare one, among its embodiments and with a domain id."""
        conditioning = dict(SIM_CONDITIONING, default_embodiment="AgiBotWorld")
        rollout = _resolve(tmp_path, _sim_files(conditioning=conditioning)).rollout
        assert rollout.default_embodiment == "agibotworld"
        conditioning["default_embodiment"] = "droid"
        with pytest.raises(ValueError, match="'droid'.*not one of the checkpoint's embodiments"):
            _resolve(tmp_path, _sim_files(conditioning=conditioning))
        del conditioning["default_embodiment"]
        with pytest.raises(ValueError, match="cannot resolve rollout.default_embodiment"):
            _resolve(tmp_path, _sim_files(conditioning=conditioning))
        no_table = {"action_tokens_per_frame": 4, "default_embodiment": "agibotworld"}
        with pytest.raises(ValueError, match="cannot resolve rollout.embodiments"):
            _resolve(tmp_path, _sim_files(conditioning=no_table))
        no_id = dict(SIM_CONDITIONING, embodiments={"agibotworld": {"input_action_dim": 59}})
        with pytest.raises(ValueError, match="'agibotworld'.*has no domain_id"):
            _resolve(tmp_path, _sim_files(conditioning=no_id))

    def test_resolution_tier_is_read_from_the_training_config(self, tmp_path):
        files = _sim_files()
        assert _resolve(tmp_path, files).rollout.resolution is None
        files[CONFIG_FILE]["model"]["config"]["resolution"] = "480"
        cfg = _resolve(tmp_path, files)
        assert cfg.rollout.resolution == "480"
        assert cfg.resolved["rollout.resolution"].source == (
            "checkpoint (config.json:model.config.resolution)"
        )

    def test_sinks_need_a_window(self, tmp_path):
        files = _sim_files()
        files[CONFIG_FILE]["model"]["config"].update(kv_cache_inference_size=None)
        rollout = _resolve(tmp_path, files).rollout
        with pytest.raises(ValueError, match="need a kv_cache_inference_size"):
            rollout.with_window(None, 8)

    def test_sinks_must_leave_a_recent_frame(self, tmp_path):
        rollout = _resolve(tmp_path, _sim_files()).rollout
        assert rollout.with_window(10, 8).history_frames == 1
        with pytest.raises(ValueError, match="leave no recent frame"):
            rollout.with_window(10, 9)
        with pytest.raises(ValueError, match="leave no recent frame"):
            rollout.with_window(1, None)

    def test_whole_clip_checkpoint_has_no_rollout_sections(self, tmp_path):
        files = {k: v for k, v in _sim_files().items() if k in (CONFIG_FILE, VAE_FILE)}
        files[TRANSFORMER_FILE] = {}
        cfg = _resolve(tmp_path, files)
        assert cfg.rollout is None
        assert not any(name.startswith("rollout.") for name in cfg.resolved)

    def test_system_prompt_must_be_declared(self, tmp_path):
        files = _sim_files()
        del files[CONFIG_FILE]["model"]["config"]["vlm_config"]["use_system_prompt"]
        del files[MODEL_INDEX_FILE]["default_use_system_prompt"]
        with pytest.raises(ValueError, match="cannot resolve prompt.use_system_prompt"):
            _resolve(tmp_path, files)


# Disagreements between a checkpoint's own files that the resolver expects and resolves.
EXPECTED_DISAGREEMENTS = {
    "nvidia/Cosmos3-Nano-Sim-Bimanual": {"prompt.use_system_prompt", "rollout.text_cache_max_len"},
}

EXPECTED = {
    # repo: (use_system_prompt, native_flow_schedule, distilled, scheduler flow_shift)
    "nvidia/Cosmos3-Nano": (False, False, False, 1.0),
    "nvidia/Cosmos3-Super": (False, False, False, 1.0),
    "nvidia/Cosmos3-Edge": (False, True, False, 1.0),
    "nvidia/Cosmos3-Super-Image2Video": (False, False, False, 1.0),
    "nvidia/Cosmos3-Super-Image2Video-4Step": (False, False, True, None),
    "nvidia/Cosmos3-Super-Text2Image": (False, False, False, 3.0),
    "nvidia/Cosmos3-Super-Text2Image-4Step": (False, False, True, None),
    "nvidia/Cosmos3-Nano-Policy-DROID": (False, False, False, 1.0),
    "nvidia/Cosmos3-Edge-Policy-DROID": (False, True, False, 5.0),
    "nvidia/Cosmos3-Nano-Sim-Bimanual": (False, False, True, None),
}

# The checkpoints that generate chunk by chunk, with the rollout they declare; the
# embodiment table is checked on its own.
EXPECTED_ROLLOUT = {
    "nvidia/Cosmos3-Nano-Sim-Bimanual": Cosmos3RolloutSettings(
        chunk_frames=2,
        chunk_partition=(1, 2),
        inference_size=None,
        sink_frames=0,
        text_cache_max_len=1024,
        action_tokens_per_frame=4,
        embodiments={},
        default_embodiment="agibotworld",
        resolution="480",
    ),
}


@pytest.mark.parametrize("repo", sorted(EXPECTED))
def test_published_checkpoint_resolves(tmp_path, repo):
    files = json.loads(FIXTURE.read_text())[repo]
    cfg = _resolve(tmp_path, files)
    system_prompt, native, distilled, flow_shift = EXPECTED[repo]
    assert cfg.prompt.use_system_prompt is system_prompt
    assert cfg.sampling.native_flow_schedule is native
    assert (cfg.sampling.fixed_sigmas is not None) is distilled
    assert cfg.sampling.flow_shift == flow_shift
    assert (cfg.vae.temporal_compression, cfg.vae.spatial_compression) == (4, 16)
    disagreeing = {name for name, r in cfg.resolved.items() if r.disagreeing}
    assert disagreeing == EXPECTED_DISAGREEMENTS.get(repo, set())
    rollout = EXPECTED_ROLLOUT.get(repo)
    if rollout is None:
        assert cfg.rollout is None
    else:
        assert replace(cfg.rollout, embodiments={}) == rollout
        assert cfg.rollout.history_frames is None


def test_published_sim_checkpoint_declares_its_embodiments(tmp_path):
    files = json.loads(FIXTURE.read_text())["nvidia/Cosmos3-Nano-Sim-Bimanual"]
    embodiments = _resolve(tmp_path, files).rollout.embodiments
    assert embodiments["agibotworld"] == Cosmos3Embodiment(domain_id=15, action_dim=59)
    assert {name: e.domain_id for name, e in embodiments.items()} == {
        "abc_yam": 16,
        "agibot_gear_gripper": 15,
        "agibot_gear_gripper_ext": 15,
        "agibotworld": 15,
        "behavior1k_lerobot": 22,
        "camera_pose": 2,
        "hand_pose": 3,
        "molmoact2_yam": 16,
        "xdof_yam": 16,
    }
    assert {e.action_dim for e in embodiments.values()} == {59}


def test_fixture_covers_every_expected_checkpoint():
    assert set(json.loads(FIXTURE.read_text())) == set(EXPECTED)


def _resolver(**facts: Any) -> ParameterResolver:
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
            RequestParameters(
                action_mode="forward_dynamics", domain_name="av", action_chunk_size=16
            )
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
        assert resolver.request(RequestParameters(is_v2v=True)).flow_shift == 5.0
        assert resolver.request(RequestParameters(is_transfer=True)).flow_shift == 5.0
        assert resolver.request(RequestParameters(flow_shift=3.0)).flow_shift == 3.0
        assert _resolver().request(RequestParameters()).flow_shift is None  # no checkpoint shift

    def test_flow_shift_overrides_replace_the_unread_scheduler_shift(self):
        """The scheduler configs that ship a Karras or native grid carry flow_shift
        1.0; the requests that sample a linear grid run at the override instead."""
        nano = _resolver(flow_shift=1.0)
        assert nano.request(RequestParameters()).flow_shift == 1.0
        assert nano.request(RequestParameters(mode="image")).flow_shift == 3.0
        assert nano.request(RequestParameters(is_v2v=True)).flow_shift == 10.0
        assert nano.request(RequestParameters(is_transfer=True)).flow_shift == 10.0
        assert nano.request(RequestParameters(is_v2v=True, flow_shift=4.0)).flow_shift == 4.0
        edge = _resolver(family="nemotron_dense", flow_shift=1.0)
        assert edge.request(RequestParameters()).flow_shift == 3.0
        assert edge.request(RequestParameters(mode="image")).flow_shift == 3.0
        assert edge.request(RequestParameters(is_v2v=True)).flow_shift == 10.0
        _, _, origins = edge._resolve(RequestParameters())
        assert origins["flow_shift"] == OVERRIDE

    def test_only_the_kinds_of_request_pin_the_sigma_grid(self):
        resolver = _resolver(flow_shift=1.0)
        assert resolver.use_karras_sigmas() is None
        assert resolver.use_karras_sigmas(mode="image") is None
        assert resolver.use_karras_sigmas(is_v2v=True) is False
        assert resolver.use_karras_sigmas(is_transfer=True) is False

    def test_an_override_the_checkpoint_moved_off_is_reported_once(self, monkeypatch):
        from tensorrt_llm._torch.visual_gen.models.cosmos3 import parameter_resolver

        messages = []
        monkeypatch.setattr(parameter_resolver.logger, "warning", messages.append)
        resolver = ParameterResolver("qwen3", bare_config(flow_shift=5.0))
        for _ in range(2):
            assert resolver.flow_shift(is_v2v=True).value == 5.0
        assert len(messages) == 1 and "replaces checkpoint value 1.0" in messages[0]

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

    def test_overrides_are_tagged(self):
        _, _, origins = _resolver()._resolve(RequestParameters(is_transfer=True))
        assert origins["use_system_prompt"] == OVERRIDE
        _, _, origins = _resolver()._resolve(RequestParameters())
        assert origins["use_system_prompt"] == CHECKPOINT

    def test_checkpoint_embodiment_table_declares_the_action_width(self, tmp_path):
        """The generic table has AgiBot at 29 values per action row; a checkpoint that
        lists the embodiment says how wide its own rows are."""
        config = _resolve(tmp_path, _sim_files())
        resolver = ParameterResolver("qwen3", config, log=False)
        request = RequestParameters(action_mode="forward_dynamics", domain_name="agibotworld")

        params, warnings, origins = resolver._resolve(request)
        assert (params.raw_action_dim, origins["raw_action_dim"]) == (59, CHECKPOINT)
        assert warnings == []
        params, warnings, origins = resolver._resolve(replace(request, raw_action_dim=59))
        assert (params.raw_action_dim, origins["raw_action_dim"]) == (59, REQUEST)
        assert warnings == []
        by_id, _, origins = resolver._resolve(replace(request, domain_name=None, domain_id=15))
        assert (by_id.raw_action_dim, origins["raw_action_dim"]) == (59, CHECKPOINT)

        # An embodiment the checkpoint does not list keeps the generic width, and a
        # checkpoint without the table recommends the generic one.
        other, _, origins = resolver._resolve(replace(request, domain_name="av"))
        assert (other.raw_action_dim, origins["raw_action_dim"]) == (9, DEFAULT)
        _, warnings, _ = _resolver()._resolve(replace(request, raw_action_dim=59))
        assert warnings == [
            "Cosmos3 raw_action_dim=59 differs from recommended 29 for domain 'agibotworld'."
        ]
