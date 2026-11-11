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
"""Actual matching-main ModelConfig default serialization, allocated CPU only.

Uses real current-main ModelConfig, LocalityDomainPolicy and HF Qwen3Config.
Only the engine/native-loader handles are explicit dependency doubles. No model
weights, model execution, checkpoint authenticity or GPU behavior is claimed.
"""

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from pydantic import BaseModel, ConfigDict
from transformers import Qwen3Config

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor import retained_identity as native
from tensorrt_llm.llmapi.llm_args import LoadFormat

pytestmark = pytest.mark.cpu_only


class ActualModelConfigIdentity(unittest.TestCase):
    def test_actual_pydantic_nested_dtype_remains_typed_until_canonicalization(self):
        class Leaf(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            dtype: object
            metadata: dict = {}

        class Parent(BaseModel):
            leaf: Leaf

        value = Parent(leaf=Leaf(dtype=torch.bfloat16, metadata={0: "zero", "0": "text"}))
        before = native._canonical(value)
        self.assertEqual("torch.bfloat16", before["leaf"]["dtype"])
        self.assertEqual(before, native._canonical(value))
        self.assertNotEqual(native._canonical({0: "zero"}), native._canonical({"0": "zero"}))
        value.leaf.dtype = object()
        with self.assertRaisesRegex(ValueError, "unqualified effective model configuration type"):
            native._canonical(value)

    def test_actual_qwen_defaults_are_complete_stable_and_unknown_state_fails_closed(self):
        config = ModelConfig(
            pretrained_config=Qwen3Config(architectures=["Qwen3ForCausalLM"], dtype="bfloat16")
        )
        # No model/weights are built; these doubles isolate the metadata path.
        Model = type(
            "NativeModelFixture", (torch.nn.Module,), {"__module__": "tensorrt_llm.testing"}
        )
        Loader = type(
            "ModelLoader", (), {"__module__": "tensorrt_llm._torch.pyexecutor.model_loader"}
        )
        Checkpoint = type(
            "HfCheckpointLoader",
            (),
            {"__module__": "tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader"},
        )
        Weights = type(
            "HfWeightLoader",
            (),
            {"__module__": "tensorrt_llm._torch.models.checkpoints.hf.weight_loader"},
        )
        model, loader, checkpoint = Model(), Loader(), Checkpoint()
        model.model_config = config
        checkpoint.weight_loader = Weights()
        loader._checkpoint_loader = checkpoint
        with tempfile.TemporaryDirectory() as directory:
            executor = NS(
                model_engine=NS(model=model, model_loader=loader),
                llm_args=NS(
                    model=str(Path(directory)),
                    load_format=LoadFormat.AUTO,
                    checkpoint_loader=None,
                    lora_config=None,
                    speculative_config=None,
                    custom_tokenizer=None,
                    tokenizer=None,
                    tokenizer_mode="auto",
                    model_kwargs=None,
                ),
            )
            before = native.snapshot_effective_config(executor)
            expected_fields = {field.name for field in fields(config)} - native._CAPACITY_FIELDS
            self.assertEqual(expected_fields, set(before["model_config"]))
            self.assertEqual(before, native.snapshot_effective_config(executor))
            policy = before["model_config"]["locality_domain_policy"]
            self.assertEqual("builtins.frozenset", policy["allowed_ops"]["set_type"])
            self.assertEqual(
                sorted(config.locality_domain_policy.allowed_ops), policy["allowed_ops"]["items"]
            )
            self.assertEqual(
                policy, before["model_config"]["extra_attrs"]["locality_domain_policy"]
            )
            # Native HF dataclasses also carry effective attributes added by
            # model configuration normalization; official to_dict retains them.
            config.pretrained_config.effective_rope_probe = {"theta": 12345}
            expanded = native.snapshot_effective_config(executor)
            self.assertEqual(
                {"theta": 12345},
                expanded["model_config"]["pretrained_config"]["effective_rope_probe"],
            )
            self.assertNotEqual(before, expanded)
            # An actual semantic field is retained, even when its default is false.
            config.mnnvl_allreduce_fp8_payload = not config.mnnvl_allreduce_fp8_payload
            self.assertNotEqual(before, native.snapshot_effective_config(executor))
            config.extra_attrs["unqualified_effective_state"] = object()
            with self.assertRaisesRegex(
                ValueError, "unqualified effective model configuration type"
            ):
                native.snapshot_effective_config(executor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
