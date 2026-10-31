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
"""The request-level defaults every Cosmos3 family, mode and domain resolves to when
the caller sets nothing.

``record()`` evaluates the grid through the request resolver;
``python cosmos3_request_golden.py > test_data/cosmos3_request_defaults.json`` stores
it. ``test_cosmos3_request_resolver.py`` compares the live grid with the stored one,
so a change of a table, a rule or a precedence is a visible diff of that file that
has to be regenerated on purpose.
"""

import json
import sys

from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from tensorrt_llm._torch.visual_gen.models.cosmos3.action import EMBODIMENT_TO_DOMAIN_ID
from tensorrt_llm._torch.visual_gen.models.cosmos3.defaults import (
    COSMOS3_DOMAIN_PRESETS,
    resolve_checkpoint_policy_defaults,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniMoTPipeline,
    _resolve_use_system_prompt,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.request_resolver import (
    resolve_domain_action_config,
)
from tensorrt_llm._torch.visual_gen.models.cosmos3.sampling import Cosmos3SamplingPolicy

FAMILIES = ("qwen3", "nemotron_dense")
MODES = ("video", "image")
GENERATION_FIELDS = (
    "height",
    "width",
    "num_frames",
    "frame_rate",
    "num_inference_steps",
    "guidance_scale",
    "guidance_interval",
    "max_sequence_length",
)
DROID_POLICY = {"domain_name": "droid_lerobot", "action_chunk_size": 32, "conditioning_fps": 15.0}

UNIPC = {
    "_class_name": "UniPCMultistepScheduler",
    "num_train_timesteps": 1000,
    "flow_shift": 1.0,
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "use_karras_sigmas": True,
    "solver_order": 2,
}
DISTILLED = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "stochastic_sampling": True,
    "fixed_step_sampler_config": {"sample_type": "sde", "t_list": [1.0, 0.9375, 0.8333, 0.625]},
}


def _policy(distilled: bool) -> Cosmos3SamplingPolicy:
    if distilled:
        return Cosmos3SamplingPolicy.from_scheduler(
            FlowMatchEulerDiscreteScheduler.from_config(DISTILLED)
        )
    return Cosmos3SamplingPolicy.from_scheduler(UniPCMultistepScheduler.from_config(UNIPC))


def _pipeline(family: str, distilled: bool, policy_block=None) -> Cosmos3OmniMoTPipeline:
    pipe = object.__new__(Cosmos3OmniMoTPipeline)
    pipe.family = family
    pipe.sampling = _policy(distilled)
    pipe.checkpoint_policy_defaults = resolve_checkpoint_policy_defaults(policy_block)
    return pipe


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def record() -> dict:
    out = {"generation": {}, "action": {}, "system_prompt": {}}
    for family in FAMILIES:
        for distilled in (False, True):
            if family == "nemotron_dense" and distilled:
                continue  # no such checkpoint; the pipeline refuses the combination
            pipe = _pipeline(family, distilled)
            for mode in MODES:
                values = pipe._resolve_generation_params(
                    mode, **{f: None for f in GENERATION_FIELDS}
                )
                key = f"{family}/{mode}/{'distilled' if distilled else 'base'}"
                out["generation"][key] = {k: _jsonable(v) for k, v in values.items()}
            # Action sampling fields with and without the DROID policy block.
            for label, block in (("", None), ("/droid-policy", DROID_POLICY)):
                pipe = _pipeline(family, distilled, block)
                for action_mode in ("forward_dynamics", "policy"):
                    values = pipe._resolve_generation_params(
                        "video",
                        action_mode=action_mode,
                        num_inference_steps=None,
                        guidance_scale=None,
                        guidance_interval=None,
                        max_sequence_length=None,
                    )
                    key = f"{family}/action:{action_mode}/{'distilled' if distilled else 'base'}{label}"
                    out["generation"][key] = {k: _jsonable(v) for k, v in values.items()}
    domains = sorted(set(EMBODIMENT_TO_DOMAIN_ID) | set(COSMOS3_DOMAIN_PRESETS)) + [None]
    for policy_label, block in (("", None), ("/droid-policy", DROID_POLICY)):
        policy_defaults = resolve_checkpoint_policy_defaults(block)
        for domain in domains:
            try:
                resolved = resolve_domain_action_config(
                    domain_name=domain, checkpoint_policy_defaults=policy_defaults
                )
                cfg = {
                    k: _jsonable(v) for k, v in resolved.items() if k not in ("warnings", "sources")
                }
                cfg["num_warnings"] = len(resolved["warnings"])
            except ValueError as exc:
                cfg = {"error": str(exc)}
            out["action"][f"{domain}{policy_label}"] = cfg
    for checkpoint_default in (False, True):
        for is_v2v in (False, True):
            for is_transfer in (False, True):
                key = f"checkpoint={checkpoint_default}/v2v={is_v2v}/transfer={is_transfer}"
                out["system_prompt"][key] = _resolve_use_system_prompt(
                    None,
                    checkpoint_default=checkpoint_default,
                    is_v2v=is_v2v,
                    is_transfer=is_transfer,
                )
    return out


if __name__ == "__main__":
    json.dump(record(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
