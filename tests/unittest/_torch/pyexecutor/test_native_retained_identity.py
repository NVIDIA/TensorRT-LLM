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
"""Allocated CPU unit tests of manifest verification; no engine/model proof.

Fixture engine configuration and read-only mount flags are explicit dependency
doubles. Actual file inventories, SHA256, index parsing and confinement execute.
Run on SLURM: python3 implementation-tests/test_retained_identity.py
"""

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.pyexecutor import retained_identity as native

pytestmark = pytest.mark.cpu_only


class HFConfig:
    def __init__(self):
        self.values = {"rope_theta": 1000000, "hidden_size": 8, "_name_or_path": "operational"}

    def to_dict(self):
        return dict(self.values)


@dataclass(frozen=True)
class SourceShapedLocalityPolicy:
    # Matches the actual current-main default field types, not a runtime model.
    enabled: bool = False
    num_partitions: int = 2
    allowed_ops: frozenset = field(
        default_factory=lambda: frozenset(
            {"nvfp4_linear", "bf16_linear", "bf16_bmm", "nvfp4_moe", "bf16_moe"}
        )
    )
    allowed_backends: tuple = ("cutlass", "cublaslt", "cuda_core")


@dataclass
class LoadedConfig:
    pretrained_config: object = field(default_factory=HFConfig)
    quant_config: dict = field(default_factory=lambda: {"quant_algo": "NVFP4", "group_size": 16})
    lora_config: object = None
    spec_config: object = None
    modeling_v2_target: object = None
    mapping: object = None
    max_num_tokens: int = 1024
    torch_dtype: str = "torch.bfloat16"
    sparse_attention_config: dict = field(default_factory=lambda: {"mode": "qualified-test-double"})
    locality_domain_policy: object = None


class Identity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "checkpoint"
        self.root.mkdir()
        contents = {
            "config.json": b'{"model_type":"fixture"}',
            "tokenizer.json": b'{"model":{"vocab":{"hello":0}}}',
            "tokenizer_config.json": b'{"chat_template":"fixture"}',
            "model.safetensors.index.json": (
                b'{"weight_map":{"layer0":"model-1.safetensors","layer1":"model-2.safetensors"}}'
            ),
            "model-1.safetensors": b"actual-file-byte-fixture-1",
            "model-2.safetensors": b"actual-file-byte-fixture-2",
        }
        for name, content in contents.items():
            (self.root / name).write_bytes(content)
        self.model_config = LoadedConfig()
        Model = type("NativeModelFixture", (), {"__module__": "tensorrt_llm.testing"})
        model = Model()
        model.model_config = self.model_config
        WeightLoader = type(
            "HfWeightLoader",
            (),
            {"__module__": "tensorrt_llm._torch.models.checkpoints.hf.weight_loader"},
        )
        CheckpointLoader = type(
            "HfCheckpointLoader",
            (),
            {"__module__": "tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader"},
        )
        Loader = type(
            "ModelLoader", (), {"__module__": "tensorrt_llm._torch.pyexecutor.model_loader"}
        )
        loader, checkpoint_loader = Loader(), CheckpointLoader()
        checkpoint_loader.weight_loader = WeightLoader()
        loader._checkpoint_loader = checkpoint_loader
        self.executor = NS(
            llm_args=NS(
                model=str(self.root),
                tokenizer=None,
                tokenizer_mode="auto",
                custom_tokenizer=None,
                lora_config=None,
                speculative_config=None,
                model_kwargs=None,
                load_format=NS(name="AUTO"),
                checkpoint_loader=None,
            ),
            model_engine=NS(model=model, model_loader=loader),
        )
        self.mount = patch.object(native.os, "statvfs", return_value=NS(f_flag=native.os.ST_RDONLY))
        self.mount_mock = self.mount.start()
        self.addCleanup(self.mount.stop)
        self.manifest = self.base / "manifest.json"
        self.identity = native.build_retained_model_manifest(self.executor, self.manifest)
        self.provider_config = NS(model_manifest=str(self.manifest), model_identity=self.identity)

    def replace_manifest(self, manifest):
        raw = json.dumps(manifest).encode()
        self.manifest.write_bytes(raw)
        self.provider_config.model_identity = hashlib.sha256(raw).hexdigest()

    def verify(self):
        return native.verify_retained_model_identity(self.executor, self.provider_config)

    def test_current_main_frozen_locality_policy_is_stable_and_semantically_bound(self):
        self.model_config.locality_domain_policy = SourceShapedLocalityPolicy()
        before = native.snapshot_effective_config(self.executor)
        self.model_config.locality_domain_policy = SourceShapedLocalityPolicy(
            allowed_ops=frozenset(
                reversed(sorted(self.model_config.locality_domain_policy.allowed_ops))
            )
        )
        self.assertEqual(before, native.snapshot_effective_config(self.executor))
        prepared = self.base / "actual-policy-types-manifest.json"
        identity = native.build_retained_model_manifest(self.executor, prepared)
        self.provider_config.model_manifest = str(prepared)
        self.provider_config.model_identity = identity
        self.assertEqual(identity, self.verify().model_identity)
        encoded = before["model_config"]["locality_domain_policy"]["allowed_ops"]
        self.assertEqual("builtins.frozenset", encoded["set_type"])
        self.assertEqual(sorted(SourceShapedLocalityPolicy().allowed_ops), encoded["items"])
        self.model_config.locality_domain_policy = SourceShapedLocalityPolicy(
            allowed_ops=frozenset({"bf16_linear"})
        )
        with self.assertRaisesRegex(ValueError, "effective"):
            self.verify()
        self.assertNotEqual(native._canonical({"a", "b"}), native._canonical(["a", "b"]))
        self.assertNotEqual(native._canonical({"a", "b"}), native._canonical(frozenset({"a", "b"})))

    def test_native_hf_integer_label_keys_remain_distinct_from_string_keys(self):
        self.model_config.pretrained_config.values["id2label"] = {0: "LABEL_0", 1: "LABEL_1"}
        first = native.snapshot_effective_config(self.executor)
        mapping = first["model_config"]["pretrained_config"]["id2label"]
        self.assertEqual([0, 1], [item["key"] for item in mapping["typed_mapping"]])
        self.assertEqual({"builtins.int"}, {item["key_type"] for item in mapping["typed_mapping"]})
        self.model_config.pretrained_config.values["id2label"] = {1: "LABEL_1", 0: "LABEL_0"}
        self.assertEqual(first, native.snapshot_effective_config(self.executor))
        self.model_config.pretrained_config.values["id2label"] = {"0": "LABEL_0", "1": "LABEL_1"}
        self.assertNotEqual(first, native.snapshot_effective_config(self.executor))
        for invalid in ({False: "label"}, {object(): "label"}):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "dictionary key types"),
            ):
                native._canonical(invalid)

    def test_complete_actual_inventory_and_immutable_receipt(self):
        receipt = self.verify()
        self.assertEqual(self.identity, receipt.model_identity)
        self.assertEqual(6, len(receipt.members))
        with self.assertRaises(AttributeError):
            receipt.model_identity = "forged"
        with self.assertRaises(FileExistsError):
            native.build_retained_model_manifest(self.executor, self.manifest)

    def test_weight_and_tokenizer_corruption_fail_even_with_same_sizes(self):
        for name in ("model-1.safetensors", "tokenizer.json"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(b"X" + original[1:])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "does not match"):
                self.verify()
            path.write_bytes(original)

    def test_rope_quantization_and_actual_model_kwargs_are_bound(self):
        self.model_config.pretrained_config.values["rope_theta"] = 500000
        with self.assertRaisesRegex(ValueError, "effective model"):
            self.verify()
        self.model_config.pretrained_config.values["rope_theta"] = 1000000
        self.model_config.quant_config["group_size"] = 32
        with self.assertRaisesRegex(ValueError, "effective model"):
            self.verify()
        self.model_config.quant_config["group_size"] = 16
        self.executor.llm_args.model_kwargs = {"rope_theta": 500000}
        with self.assertRaisesRegex(ValueError, "effective model"):
            self.verify()

    def test_capacity_topology_and_source_path_do_not_change_semantic_identity(self):
        self.model_config.max_num_tokens = 4096
        self.model_config.mapping = object()
        self.model_config.pretrained_config.values["_name_or_path"] = "/other/snapshot"
        self.verify()

    def test_unlisted_member_and_missing_index_shard_fail(self):
        (self.root / "unlisted.safetensors").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "every checkpoint file"):
            self.verify()
        (self.root / "unlisted.safetensors").unlink()
        doc = json.loads(self.manifest.read_bytes())
        doc["files"] = [entry for entry in doc["files"] if entry["path"] != "model-2.safetensors"]
        self.replace_manifest(doc)
        with self.assertRaisesRegex(ValueError, "every checkpoint file"):
            self.verify()

    def test_path_escape_external_symlink_and_duplicate_members_fail(self):
        original = json.loads(self.manifest.read_bytes())
        bad = json.loads(self.manifest.read_bytes())
        bad["files"][0]["path"] = "../manifest.json"
        self.replace_manifest(bad)
        with self.assertRaisesRegex(ValueError, "relative path"):
            self.verify()
        self.replace_manifest(original)
        (self.base / "outside.bin").write_bytes(b"outside")
        (self.root / "outside-link").symlink_to(self.base / "outside.bin")
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.verify()
        (self.root / "outside-link").unlink()
        original["files"].append(original["files"][0])
        self.replace_manifest(original)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()

    def test_identity_label_unknown_schema_and_writable_mount_fail(self):
        self.provider_config.model_identity = "0" * 64
        with self.assertRaisesRegex(ValueError, "exact manifest bytes"):
            self.verify()
        self.provider_config.model_identity = self.identity
        doc = json.loads(self.manifest.read_bytes())
        doc["untrusted_extra"] = True
        self.replace_manifest(doc)
        with self.assertRaisesRegex(ValueError, "schema"):
            self.verify()
        self.mount_mock.return_value = NS(f_flag=0)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.verify()

    def test_external_tokenizer_lora_and_nonserializable_effective_state_fail(self):
        self.executor.llm_args.tokenizer = str(self.base)
        with self.assertRaisesRegex(ValueError, "external tokenizer"):
            self.verify()
        self.executor.llm_args.tokenizer = None
        self.executor.llm_args.lora_config = {"adapter": "unqualified"}
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self.verify()
        self.executor.llm_args.lora_config = None
        self.model_config.quant_config["unknown_runtime_object"] = object()
        with self.assertRaisesRegex(ValueError, "unqualified effective"):
            self.verify()

    def test_dummy_and_untracked_loaded_weights_are_rejected(self):
        self.executor.llm_args.load_format = NS(name="DUMMY")
        with self.assertRaisesRegex(ValueError, "dummy/GMS/custom"):
            self.verify()
        self.executor.llm_args.load_format = NS(name="AUTO")
        self.executor.model_engine.model_loader = None
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
