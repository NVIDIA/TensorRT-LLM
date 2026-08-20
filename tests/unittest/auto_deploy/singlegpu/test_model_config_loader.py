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

from pathlib import Path

import pytest
import yaml

import tensorrt_llm._torch.auto_deploy.model_config_loader as mcl


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A temp two-layer registry: internal (package) + user (examples) dirs."""
    internal_dir = tmp_path / "internal_configs"
    user_dir = tmp_path / "user_configs"
    internal_dir.mkdir()
    user_dir.mkdir()
    models_yaml = tmp_path / "models.yaml"

    _write_yaml(
        internal_dir / "llama_ad.yaml", {"transforms": {"match_swiglu_pattern": {"enabled": False}}}
    )
    _write_yaml(user_dir / "dashboard_default.yaml", {"max_seq_len": 512, "attn_backend": "trtllm"})
    _write_yaml(user_dir / "llama.yaml", {"max_batch_size": 1024, "max_seq_len": 4096})
    # A model whose user filename differs from its config_id (dotted/dashed names).
    _write_yaml(user_dir / "glm-4.7.yaml", {"max_batch_size": 64})
    models_yaml.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "name": "meta-llama/Llama-3.3-70B-Instruct",
                        "config_id": "llama",
                        "yaml_extra": ["dashboard_default.yaml", "llama_ad.yaml", "llama.yaml"],
                    },
                    {
                        "name": "zai-org/GLM-4.7",
                        "config_id": "glm_4_7",
                        "yaml_extra": [
                            "dashboard_default.yaml",
                            "world_size_4.yaml",
                            "glm-4.7.yaml",
                        ],
                    },
                    {"name": "dup/model", "config_id": "a", "yaml_extra": ["llama.yaml"]},
                    {"name": "dup/model", "config_id": "b", "yaml_extra": ["llama.yaml"]},
                ]
            }
        )
    )

    monkeypatch.setattr(mcl, "MODELS_YAML", models_yaml)
    monkeypatch.setattr(mcl, "INTERNAL_CONFIGS_DIR", internal_dir)
    monkeypatch.setattr(mcl, "USER_CONFIGS_DIR", user_dir)
    return {"internal": internal_dir, "user": user_dir, "models": models_yaml}


def test_find_registry_entry_found(registry):
    entry = mcl.find_registry_entry("meta-llama/Llama-3.3-70B-Instruct")
    assert entry["config_id"] == "llama"


def test_find_registry_entry_miss_returns_none(registry):
    assert mcl.find_registry_entry("unknown/model") is None
    assert mcl.find_registry_entry("") is None


def test_find_registry_entry_ambiguous_raises(registry):
    with pytest.raises(KeyError):
        mcl.find_registry_entry("dup/model")
    assert mcl.find_registry_entry("dup/model", "b")["config_id"] == "b"


def test_resolve_yaml_extra_spans_both_dirs(registry):
    paths = mcl.resolve_registry_yaml_extra("meta-llama/Llama-3.3-70B-Instruct")
    names = [Path(p).name for p in paths]
    assert names == ["dashboard_default.yaml", "llama_ad.yaml", "llama.yaml"]
    # internal file resolves from the package dir, user files from the user dir.
    assert Path(paths[1]).parent == registry["internal"]
    assert Path(paths[0]).parent == registry["user"]


def test_resolve_yaml_extra_internal_only(registry):
    paths = mcl.resolve_registry_yaml_extra("meta-llama/Llama-3.3-70B-Instruct", internal_only=True)
    assert [Path(p).name for p in paths] == ["llama_ad.yaml"]


def test_resolve_yaml_extra_miss(registry):
    assert mcl.resolve_registry_yaml_extra("unknown/model") == []
    with pytest.raises(KeyError):
        mcl.resolve_registry_yaml_extra("unknown/model", required=True)


def test_resolve_yaml_extra_user_dirs_override(registry, tmp_path):
    override_dir = tmp_path / "override_configs"
    _write_yaml(override_dir / "llama.yaml", {"max_batch_size": 8})
    paths = mcl.resolve_registry_yaml_extra(
        "meta-llama/Llama-3.3-70B-Instruct", user_dirs=[override_dir]
    )
    # internal file still resolves from the package dir; user files only from override_dir.
    assert [Path(p).name for p in paths] == ["llama_ad.yaml", "llama.yaml"]
    assert Path(paths[1]).parent == override_dir


@pytest.mark.skipif(
    not mcl.USER_CONFIGS_DIR.is_dir(), reason="source-tree examples/ dir not available"
)
def test_resolve_yaml_extra_real_registry_two_layers():
    """Two-layer loading against the real registry: internal _ad.yaml + user-facing files."""
    # User-facing files only.
    paths = mcl.resolve_registry_yaml_extra("meta-llama/Llama-3.1-8B-Instruct", required=True)
    assert [Path(p).name for p in paths] == ["dashboard_default.yaml", "world_size_1.yaml"]
    assert all(Path(p).parent == mcl.USER_CONFIGS_DIR for p in paths)

    # Internal _ad.yaml layered with user-facing files.
    paths = mcl.resolve_registry_yaml_extra("nvidia/Llama-3.1-8B-Instruct-FP8", required=True)
    by_name = {Path(p).name: Path(p).parent for p in paths}
    assert by_name["llama3_1_8b_ad.yaml"] == mcl.INTERNAL_CONFIGS_DIR
    assert by_name["llama3_1_8b.yaml"] == mcl.USER_CONFIGS_DIR
    assert by_name["dashboard_default.yaml"] == mcl.USER_CONFIGS_DIR


def test_translate_parallel_fields():
    args = {"tensor_parallel_size": 4, "model": "m"}
    mcl.translate_parallel_fields(args)
    assert args == {"world_size": 4, "model": "m"}


def test_translate_parallel_fields_keeps_explicit_world_size():
    args = {"tensor_parallel_size": 4, "world_size": 2}
    mcl.translate_parallel_fields(args)
    assert args == {"world_size": 2}


def test_inject_autodeploy_registry_defaults(registry):
    args = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "tensor_parallel_size": 2,
        "yaml_extra": ["/user/override.yaml"],
    }
    mcl.inject_autodeploy_registry_defaults(args)
    # internal _ad config prepended (lowest priority), user override kept last.
    assert [Path(p).name for p in args["yaml_extra"][:-1]] == ["llama_ad.yaml"]
    assert args["yaml_extra"][-1] == "/user/override.yaml"
    assert args["world_size"] == 2
    assert "tensor_parallel_size" not in args


def test_inject_autodeploy_registry_defaults_miss_is_noop(registry):
    args = {"model": "unknown/model"}
    mcl.inject_autodeploy_registry_defaults(args)
    assert args == {"model": "unknown/model"}
