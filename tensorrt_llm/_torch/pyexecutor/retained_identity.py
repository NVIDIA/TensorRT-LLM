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
"""Strict startup identity for the opt-in retained-source provider.

Hash full checkpoint members only on allocated serving/SLURM resources. The
checkpoint must remain immutable while it is loaded and served; verification
of disk bytes after a model was loaded cannot prove the history of earlier
reads. This verifier does not implement a checkpoint loader or mutable-adapter
identity. Deploy a materialized immutable snapshot, not mutable Hub symlinks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import weakref
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

_CAPACITY_FIELDS = frozenset(
    {
        "mapping",
        "max_num_tokens",
        "moe_max_num_tokens",
        "_moe_max_num_tokens_is_default",
        "max_seq_len",
        "cuda_graph_batch_sizes",
        "use_cuda_graph",
        "_frozen",
    }
)
_REQUIRED_FILES = frozenset({"config.json", "tokenizer.json", "tokenizer_config.json"})


@dataclass(frozen=True)
class VerifiedRetainedIdentity:
    model_identity: str
    model_root: str
    effective_config_sha256: str
    members: tuple[tuple[str, int, str], ...]


def _strict_json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _canonical(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("effective model configuration contains a non-finite value")
        return value
    if isinstance(value, Enum):
        return {
            "enum": type(value).__module__ + "." + type(value).__qualname__,
            "value": _canonical(value.value),
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # Actual ModelConfig.LocalityDomainPolicy stores allowed_ops as a
        # frozenset. Preserve membership/type without iteration-order hashes.
        items = [_canonical(item) for item in value]
        return {
            "set_type": type(value).__module__ + "." + type(value).__qualname__,
            "items": sorted(items, key=_json_bytes),
        }
    if isinstance(value, dict):
        if all(type(key) is str for key in value):
            return {key: _canonical(item) for key, item in value.items()}
        # Native HF Qwen3Config.id2label has integer keys. Preserve their
        # type rather than coercing 0 and "0" into the same JSON object key.
        if not all(type(key) in (str, int) for key in value):
            types = sorted({type(key).__module__ + "." + type(key).__qualname__ for key in value})
            raise ValueError(f"unqualified effective model dictionary key types: {types}")
        items = [
            {
                "key_type": type(key).__module__ + "." + type(key).__qualname__,
                "key": key,
                "value": _canonical(item),
            }
            for key, item in value.items()
        ]
        return {
            "typed_mapping": sorted(
                items, key=lambda item: (item["key_type"], _json_bytes(item["key"]))
            )
        }
    # Current HF configs are dataclasses too. Their official serialization
    # includes effective attributes added after construction; fields() alone
    # would silently omit those semantic overrides.
    if callable(getattr(value, "to_dict", None)):
        return _canonical(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    # The actual Pydantic model configuration surfaces.
    if callable(getattr(value, "model_dump", None)):
        return _canonical(value.model_dump(mode="python"))
    raise ValueError(f"unqualified effective model configuration type: {type(value)!r}")


def _json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _snapshot_extra_attrs(executor) -> dict:
    """Separate native borrowed-resource registries from semantic configuration.

    Attention registers live modules (attention.py:550); ModelCaller overwrites
    attention_metadata before every model call (model_call.py:64). RopeParams
    caches derived tensors through weakrefs (interface.py:910-985). Their cache
    population and addresses are operational; the actual bound RopeParams are
    included in each layer's semantic descriptor. Unknown state stays strict.
    """
    model = executor.model_engine.model
    attrs = model.model_config.extra_attrs
    if type(attrs) is not dict:
        raise ValueError("native extra_attrs must be an exact dictionary")
    import torch

    from tensorrt_llm._torch.attention.attention import Attention
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata, RopeParams

    registry = attrs.get("attn_layers", {})
    if type(registry) is not dict:
        raise ValueError("native attention registry must be an exact dictionary")
    modules = {}
    expected_attention = set()
    for name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, Attention):
            continue
        if id(module) in modules:
            raise ValueError("aliased native attention module has ambiguous registry ownership")
        modules[id(module)] = (name, module)
        expected_attention.add(id(module))
    layers = {}
    bound_rope = {}
    registered_attention = set()
    for key, reference in registry.items():
        if type(key) is not str or type(reference) is not weakref.ReferenceType:
            raise ValueError("native attention registry requires string keys and exact weakrefs")
        module = reference()
        if (
            module is None
            or not isinstance(module, Attention)
            or id(module) not in modules
            or modules[id(module)][1] is not module
            or type(module.layer_idx) is not int
            or module.layer_idx < 0
            or module.layer_idx_str != key
            or str(module.layer_idx) != key
            or id(module) in registered_attention
        ):
            raise ValueError("dead, foreign, or misidentified native attention registry entry")
        registered_attention.add(id(module))
        descriptor = {
            "module": modules[id(module)][0],
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "global_layer": _canonical(module.layer_idx),
        }
        positional = getattr(module, "pos_embd_params", None)
        rope = getattr(positional, "rope", None)
        if rope is not None:
            if type(rope) is not RopeParams:
                raise ValueError("unqualified native attention RoPE parameters")
            descriptor["rope"] = _canonical(rope)
            bound_rope[id(rope)] = rope
        layers[key] = descriptor
    if registered_attention != expected_attention:
        raise ValueError(
            "native attention registry does not cover every owned attention module exactly"
        )

    if "attention_metadata" in attrs:
        reference = attrs["attention_metadata"]
        if type(reference) is not weakref.ReferenceType:
            raise ValueError("attention metadata must be the native borrowed weakref")
        metadata = reference()
        # An expired borrow cannot affect a subsequent model call: the actual
        # native ModelCaller replaces this slot before invoking model.forward.
        if metadata is not None and (
            not isinstance(metadata, AttentionMetadata)
            or metadata is not executor.model_engine.attn_metadata
        ):
            raise ValueError("attention metadata does not belong to the actual engine")
    if "spec_metadata" in attrs and attrs["spec_metadata"] is not None:
        raise ValueError("retained identity does not qualify speculative metadata")
    if "rope_const_params" in attrs:
        cache = attrs["rope_const_params"]
        if type(cache) is not dict:
            raise ValueError("RoPE constant cache must be an exact dictionary")
        for key, constants in cache.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or type(key[0]) is not RopeParams
                or type(key[1]) is not bool
                or bound_rope.get(id(key[0])) is not key[0]
            ):
                raise ValueError("foreign or malformed native RoPE cache key")
            if (
                not isinstance(constants, tuple)
                or type(constants).__name__ != "RopeConstParams"
                or type(constants).__module__ != "tensorrt_llm._torch.attention.backends.interface"
                or getattr(constants, "_fields", None) != ("inv_freq", "cos_sin")
            ):
                raise ValueError("unqualified native RoPE cache value")
            for index, reference in enumerate(constants):
                if reference is None and index == 0:
                    continue
                if type(reference) is not weakref.ReferenceType:
                    raise ValueError("native RoPE cache requires borrowed tensor weakrefs")
                tensor = reference()
                # Expired cache entries are a normal cache miss in the native
                # create_rope_const_params implementation, never identity state.
                if tensor is not None and (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_cuda
                    or tensor.dtype != torch.float32
                    or tensor.device.index != executor.device_id
                ):
                    raise ValueError("unqualified live native RoPE constant tensor")
    runtime = {"attn_layers", "attention_metadata", "spec_metadata", "rope_const_params"}
    result = {key: _canonical(value) for key, value in attrs.items() if key not in runtime}
    if registry:
        result["attn_layers"] = layers
    return result


def snapshot_effective_config(executor) -> dict:
    """Read the loaded engine's actual model configuration, not caller labels.

    Source contract: ModelEngine.model.model_config is current-main's ModelConfig
    dataclass; its pretrained_config is the resolved HF configuration including
    effective RoPE. Unknown/nonserializable effective state fails closed.
    Backend/quantization/sparse/compression settings are retained. Only explicit
    capacity/topology/CUDA-graph scheduling fields above are excluded.
    """
    args = executor.llm_args
    engine = executor.model_engine
    config = engine.model.model_config
    if args.load_format.name != "AUTO" or args.checkpoint_loader is not None:
        raise ValueError(
            "retained identity requires native checkpoint loading, not dummy/GMS/custom weights"
        )
    loader = engine.model_loader
    if (
        loader is None
        or type(loader).__module__ != "tensorrt_llm._torch.pyexecutor.model_loader"
        or type(loader).__name__ != "ModelLoader"
    ):
        raise ValueError("retained identity cannot qualify an injected or untracked loaded model")
    checkpoint_loader = loader._checkpoint_loader
    if (
        checkpoint_loader is None
        or type(checkpoint_loader).__module__
        != "tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader"
        or type(checkpoint_loader).__name__ != "HfCheckpointLoader"
        or type(checkpoint_loader.weight_loader).__module__
        != "tensorrt_llm._torch.models.checkpoints.hf.weight_loader"
        or type(checkpoint_loader.weight_loader).__name__ != "HfWeightLoader"
    ):
        raise ValueError("retained identity requires the actual native HF checkpoint/weight loader")
    if not is_dataclass(config) or isinstance(config, type):
        raise ValueError("retained identity requires the actual ModelConfig dataclass")
    if args.lora_config is not None or config.lora_config is not None:
        raise ValueError("retained identity does not qualify LoRA/adapters")
    if args.speculative_config is not None or config.spec_config is not None:
        raise ValueError("retained identity does not qualify speculative/draft model state")
    if config.modeling_v2_target is not None:
        raise ValueError("retained identity does not qualify custom modeling implementations")
    if args.custom_tokenizer is not None:
        raise ValueError("retained identity requires the checkpoint's native tokenizer")
    if not isinstance(
        engine.model.__class__.__module__, str
    ) or not engine.model.__class__.__module__.startswith("tensorrt_llm."):
        raise ValueError("retained identity requires a framework-native loaded model class")
    if config.pretrained_config is None:
        raise ValueError("retained identity requires resolved pretrained configuration")
    if config.pretrained_config.__class__.__module__.startswith("transformers_modules."):
        raise ValueError(
            "retained identity does not qualify externally cached remote configuration code"
        )
    root = Path(args.model).resolve(strict=True)
    if args.tokenizer is not None:
        if (
            not isinstance(args.tokenizer, (str, Path))
            or Path(args.tokenizer).resolve(strict=True) != root
        ):
            raise ValueError("retained identity rejects external tokenizer state")
    snapshot = {}
    for field in fields(config):
        if field.name not in _CAPACITY_FIELDS:
            snapshot[field.name] = (
                _snapshot_extra_attrs(executor)
                if field.name == "extra_attrs"
                else _canonical(getattr(config, field.name))
            )
    # HF stores an operational source path in this otherwise semantic dict.
    if isinstance(snapshot.get("pretrained_config"), dict):
        snapshot["pretrained_config"].pop("_name_or_path", None)
    return {
        "model_config": snapshot,
        "effective_torch_dtype": str(config.torch_dtype),
        "tokenizer_mode": args.tokenizer_mode,
        "model_kwargs": _canonical(args.model_kwargs),
    }


def _member(root: Path, name: str) -> Path:
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or ":" in name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("manifest member must be a normalized relative path")
    candidate = root / name
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("manifest member escapes checkpoint root or is not a file")
    return resolved


def _hash_member(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("checkpoint changed while its member was verified")
    return after.st_size, digest.hexdigest()


def _checkpoint_root(executor) -> Path:
    root = Path(executor.llm_args.model).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("retained identity requires an existing local checkpoint directory")
    if not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise ValueError("retained identity requires a read-only checkpoint filesystem/mount")
    return root


def _validate_weight_inventory(root: Path, actual: set[str]) -> None:
    if not _REQUIRED_FILES <= actual:
        raise ValueError("manifest requires config.json and complete native tokenizer files")
    shards = {name for name in actual if name.endswith(".safetensors")}
    index_name = "model.safetensors.index.json"
    if index_name in actual:
        index = _strict_json((root / index_name).read_bytes())
        if (
            type(index) is not dict
            or type(index.get("weight_map")) is not dict
            or not index["weight_map"]
        ):
            raise ValueError("checkpoint weight index is malformed")
        names = list(index["weight_map"].values())
        if (
            not all(type(name) is str and name.endswith(".safetensors") for name in names)
            or set(names) != shards
        ):
            raise ValueError("manifest/index must cover every referenced safetensor shard exactly")
    elif shards != {"model.safetensors"}:
        raise ValueError(
            "checkpoint requires a complete safetensors weight index or one model.safetensors"
        )


def build_retained_model_manifest(executor, output_path) -> str:
    """Explicit preparation step on a loaded stock engine; never a trust fallback.

    This can hash a multi-terabyte checkpoint. Run only in a deliberate allocated
    startup/preparation job. Output must not exist and must be outside the
    read-only checkpoint root. Return the SHA256 to configure on later workers.
    """
    root = _checkpoint_root(executor)
    output = Path(output_path).resolve()
    if output.is_relative_to(root):
        raise ValueError("manifest output must be outside checkpoint inventory")
    paths = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError("checkpoint directory symlinks are not qualified")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            paths[name] = _member(root, name)
    _validate_weight_inventory(root, set(paths))
    members = []
    for name, path in sorted(paths.items()):
        size, digest = _hash_member(path)
        members.append({"path": name, "size": size, "sha256": digest})
    manifest = {
        "schema_version": 1,
        "files": members,
        "effective_config": snapshot_effective_config(executor),
    }
    raw = _json_bytes(manifest) + b"\n"
    identity = hashlib.sha256(raw).hexdigest()
    # Exclusive target creation prevents replacing a previously trusted manifest.
    with output.open("xb") as stream:
        stream.write(raw)
    return identity


def verify_retained_model_identity(executor, config) -> VerifiedRetainedIdentity:
    """Verify required config.model_manifest/model_identity against actual state.

    Manifest v1 has exactly {schema_version:1, files:[{path,size,sha256}],
    effective_config:<snapshot_effective_config(executor)>}. The manifest lives
    outside the model root; model_identity is SHA256 of its exact UTF-8 bytes.
    Every checkpoint file is inventoried; required tokenizer/config members and
    either model.safetensors or the complete weight-index shard set are required.
    """
    root = _checkpoint_root(executor)
    manifest_path = Path(config.model_manifest).resolve(strict=True)
    if manifest_path.is_relative_to(root):
        raise ValueError("manifest must live outside the checkpoint inventory")
    raw = manifest_path.read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    if identity != config.model_identity:
        raise ValueError("retained model identity does not match exact manifest bytes")
    manifest = _strict_json(raw)
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "files", "effective_config"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or type(manifest["files"]) is not list
    ):
        raise ValueError("unsupported or malformed full model manifest schema")
    listed = {}
    for item in manifest["files"]:
        if (
            type(item) is not dict
            or set(item) != {"path", "size", "sha256"}
            or type(item["size"]) is not int
            or item["size"] < 0
            or type(item["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            raise ValueError("malformed model manifest member")
        name = item["path"]
        path = _member(root, name)
        if name in listed:
            raise ValueError("duplicate model manifest member")
        listed[name] = (path, item["size"], item["sha256"])
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink() and path.is_dir():
            raise ValueError("checkpoint directory symlinks are not qualified")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            _member(root, name)
            actual.add(name)
    if actual != set(listed) or not _REQUIRED_FILES <= actual:
        raise ValueError(
            "manifest must cover every checkpoint file and required config/tokenizer files"
        )
    for name, (path, size, digest) in listed.items():
        if _hash_member(path) != (size, digest):
            raise ValueError(f"checkpoint member does not match manifest: {name}")
    _validate_weight_inventory(root, actual)
    effective = snapshot_effective_config(executor)
    if _json_bytes(effective) != _json_bytes(manifest["effective_config"]):
        raise ValueError(
            "manifest effective model/RoPE/tokenizer configuration differs from loaded engine"
        )
    return VerifiedRetainedIdentity(
        identity,
        str(root),
        hashlib.sha256(_json_bytes(effective)).hexdigest(),
        tuple((name, size, digest) for name, (_, size, digest) in sorted(listed.items())),
    )
