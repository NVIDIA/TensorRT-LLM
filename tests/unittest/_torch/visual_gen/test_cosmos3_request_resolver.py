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
"""The request rules: precedence between the caller, the checkpoint's facts and the
tables, and the source each value reports."""

import pytest

from tensorrt_llm._torch.visual_gen.models.cosmos3.request_resolver import (
    RequestContext,
    resolve_action,
    resolve_domain_action_config,
    resolve_generation,
)

pytestmark = pytest.mark.cpu_only


class TestSources:
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

    def test_disagreeing_recommendation_names_its_source(self):
        cfg = resolve_domain_action_config(domain_name="av", action_chunk_size=16)
        assert cfg["action_chunk_size"] == 16
        assert cfg["sources"]["action_chunk_size"] == "request"

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
