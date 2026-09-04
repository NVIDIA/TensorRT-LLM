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

"""Pre-weight-loading pipeline cache for AutoDeploy.

This cache is an AutoDeploy transform. Place ``pipeline_cache`` at the boundary
where the graph should be snapshotted, at or before the sharding stage and
before ``load_weights``. The transform saves the incoming module on a miss and
the optimizer asks the same transform for a restore before running the prefix so
a hit skips the transforms before the cache point.

The main artifact is a ``torch.save`` structural FX payload for the
``GraphModule`` or GraphModule-bearing wrapper. Load hooks are never part of the
artifact contract: recognized hooks are scrubbed before save and rebuilt from
``hooks.json``.
"""

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import Field, model_validator

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import named_graphmodules
from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .common import (
    atomic_publish_rank_dir,
    fsync_dir,
    hash_payload,
    read_json,
    sha256_file,
    write_json_atomic,
)
from .hooks import (
    collect_hook_specs,
    reattach_hooks,
    restore_load_hooks,
    snapshot_and_clear_load_hooks,
)
from .structural import load_module_structural, save_module_structural, validate_pre_weight_snapshot

MANIFEST_FILE_NAME = "manifest.json"
MODULE_FILE_NAME = "module.pt"
HOOKS_FILE_NAME = "hooks.json"


def _validate_no_forward_hooks(model: nn.Module) -> None:
    modules_with_hooks = [
        name or "root"
        for name, module in model.named_modules()
        if getattr(module, "_forward_pre_hooks", None) or getattr(module, "_forward_hooks", None)
    ]
    if modules_with_hooks:
        raise ValueError(
            "pipeline_cache does not support caching modules with forward hooks; "
            f"modules with forward hooks: {modules_with_hooks}"
        )


def _restore_managed_graph_inputs(model: nn.Module, cm: CachedSequenceInterface) -> None:
    """Activate interface-managed inputs required by a restored graph."""
    active_args = set(cm.info.named_args)
    available_args = cm.info.available_args

    for _, graph_module in named_graphmodules(model):
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            name = str(node.target)
            if name in available_args and name not in active_args:
                cm.info.activate_arg(name)
                active_args.add(name)


def _default_pipeline_cache_root() -> str:
    return str(Path.home() / ".cache" / "tensorrt_llm" / "auto_deploy" / "pipeline_cache")


class PipelineCacheConfig(TransformConfig):
    """Configuration for the torch-save pipeline cache transform."""

    model_config = {
        "extra": "forbid",
    }

    enabled: bool = Field(
        default=False,
        description="Whether to enable the torch-save pipeline cache transform.",
    )
    run_per_gm: ClassVar[bool] = False
    run_graph_cleanup: ClassVar[bool] = False
    requires_clean_graph: ClassVar[bool] = False
    run_shape_prop: ClassVar[bool] = False
    requires_shape_prop: ClassVar[bool] = False
    skip_on_error: ClassVar[bool] = True
    debug_visualize_dir: ClassVar[str | None] = None
    expect_mem_change: ClassVar[bool] = False
    root: str | None = Field(
        default=None,
        description=(
            "Cache root. Defaults to ~/.cache/tensorrt_llm/auto_deploy/pipeline_cache "
            "when the transform is enabled."
        ),
    )

    @model_validator(mode="after")
    def validate_enabled_cache(self) -> "PipelineCacheConfig":
        if not self.enabled:
            return self
        if self.root in (None, ""):
            self.root = _default_pipeline_cache_root()
        if self.stage > Stages.SHARDING:
            raise ValueError(
                "pipeline_cache must be placed at or before the sharding stage so restore can "
                "resume before weight loading."
            )
        return self


def _dist_config_payload(shared_config: SharedConfig) -> dict[str, Any]:
    dist_config = shared_config.dist_config
    if dist_config is not None:
        payload = dist_config.to_dict()
        payload.pop("rank", None)
        return payload
    return {"world_size": shared_config.world_size}


def _cache_transform_index(
    items: Sequence[tuple[str, TransformConfig]], transform_name: str
) -> int:
    for idx, (name, _) in enumerate(items):
        if name == transform_name:
            return idx
    raise ValueError(f"{transform_name} is missing from the optimizer config.")


def _collective_bool_and(local_value: bool) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return local_value
    backend = dist.get_backend()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    agreed = torch.tensor(1 if local_value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    return bool(agreed.item())


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@TransformRegistry.register("pipeline_cache")
class PipelineCache(BaseTransform):
    """Transform that snapshots/restores the model at its configured pipeline position."""

    config: PipelineCacheConfig
    _cache_key_config: StrictInferenceOptimizerConfig | None

    @classmethod
    def get_config_class(cls) -> type[TransformConfig]:
        return PipelineCacheConfig

    def _post_init(self):
        self._cache_key_config = None

    def set_cache_key_config(self, cache_key_config: StrictInferenceOptimizerConfig) -> None:
        self._cache_key_config = cache_key_config

    def maybe_restore(
        self,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
        transform_index: int,
    ) -> nn.Module | None:
        """Return a cached module for this transform point, or ``None`` on a miss."""
        if not self.config.enabled:
            return None
        items = self._cache_key_items()
        context = self._build_context(factory, shared_config, items[:transform_index])
        if not _collective_bool_and(self._validate_manifest(context)):
            return None

        local_success = False
        module: nn.Module | None = None
        try:
            module = self._load_module(context)
            _restore_managed_graph_inputs(module, cm)
            local_success = True
        except Exception as exc:
            ad_logger.warning(f"Failed to restore AutoDeploy pipeline cache: {exc}")
            module = None

        if not _collective_bool_and(local_success):
            return None
        assert module is not None
        ad_logger.info(f"Restored AutoDeploy pipeline cache from {self._rank_dir(context)}")
        return module

    def _apply_to_full_model(
        self,
        model: nn.Module,
        _cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> tuple[nn.Module, TransformInfo]:
        items = self._cache_key_items()
        transform_index = _cache_transform_index(items, self.get_transform_key())
        context = self._build_context(factory, shared_config, items[:transform_index])
        saved = self._save_module(context, model)
        info = TransformInfo(
            skipped=not saved,
            num_matches=1 if saved else 0,
            is_clean=True,
            has_valid_shapes=True,
        )
        return model, info

    def _build_context(
        self,
        factory: ModelFactory,
        shared_config: SharedConfig,
        prefix_items: Sequence[tuple[str, TransformConfig]],
    ) -> dict[str, Any]:
        root = Path(str(self.config.root)).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        transform_prefix = [
            {"name": name, "config": config.model_dump(mode="python")}
            for name, config in prefix_items
        ]
        transform_prefix_hash = hash_payload({"transforms": transform_prefix})
        dist_config_payload = (
            _dist_config_payload(shared_config)
            if any(config.stage >= Stages.SHARDING for _, config in prefix_items)
            else None
        )
        identity = {
            "model_identifier": factory.get_pipeline_cache_model_identifier(),
            "checkpoint_fingerprint": factory.get_pipeline_cache_checkpoint_fingerprint(),
            "transform_prefix_hash": transform_prefix_hash,
            "dist_config": dist_config_payload,
        }
        return {
            "root": root,
            "cache_key": hash_payload(identity),
            "shared_config": shared_config,
        }

    def _cache_key_items(self) -> list[tuple[str, TransformConfig]]:
        if self._cache_key_config is None:
            raise ValueError("pipeline_cache requires the cache-key transform config.")
        return list(self._cache_key_config.items())

    def _save_module(
        self,
        context: Mapping[str, Any],
        model: nn.Module,
    ) -> bool:
        _barrier()
        rank_dir = self._rank_dir(context)
        tmp_rank_dir = self._tmp_rank_dir(context)
        cache_entry_dir = self._cache_entry_dir(context)
        try:
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)
            tmp_rank_dir.mkdir(parents=True, exist_ok=True)

            local_save_success = False
            try:
                validate_pre_weight_snapshot(model)
                hook_specs, has_unknown = collect_hook_specs(model)
                if has_unknown:
                    raise ValueError("graph contains unrecognized pipeline cache load hooks")
                _validate_no_forward_hooks(model)

                hook_records = snapshot_and_clear_load_hooks(model)
                try:
                    with open(tmp_rank_dir / MODULE_FILE_NAME, "wb") as module_file:
                        save_module_structural(model, module_file)
                        module_file.flush()
                        os.fsync(module_file.fileno())
                finally:
                    restore_load_hooks(hook_records)

                write_json_atomic(tmp_rank_dir / HOOKS_FILE_NAME, hook_specs)
                checksums = {
                    file_name: sha256_file(tmp_rank_dir / file_name)
                    for file_name in (
                        MODULE_FILE_NAME,
                        HOOKS_FILE_NAME,
                    )
                }
                manifest = self._build_manifest(
                    context,
                    file_checksums=checksums,
                )
                write_json_atomic(tmp_rank_dir / MANIFEST_FILE_NAME, manifest)
                fsync_dir(tmp_rank_dir)
                local_save_success = True
            except Exception as exc:
                ad_logger.warning(f"Skipping AutoDeploy pipeline cache save: {exc}")

            if not _collective_bool_and(local_save_success):
                shutil.rmtree(rank_dir, ignore_errors=True)
                _barrier()
                return False

            local_publish_success = False
            try:
                cache_entry_dir.mkdir(parents=True, exist_ok=True)
                fsync_dir(cache_entry_dir)
                atomic_publish_rank_dir(tmp_rank_dir, rank_dir)
                local_publish_success = True
            except Exception as exc:
                ad_logger.warning(f"Skipping AutoDeploy pipeline cache save: {exc}")

            if not _collective_bool_and(local_publish_success):
                shutil.rmtree(rank_dir, ignore_errors=True)
                _barrier()
                return False

            _barrier()
            ad_logger.info(f"Saved AutoDeploy pipeline cache to {rank_dir}")
            return True
        finally:
            shutil.rmtree(tmp_rank_dir, ignore_errors=True)

    def _build_manifest(
        self,
        context: Mapping[str, Any],
        file_checksums: Mapping[str, str],
    ) -> dict[str, Any]:
        shared_config = context["shared_config"]
        return {
            "cache_key": context["cache_key"],
            "file_checksums": dict(file_checksums),
            "rank": shared_config.local_rank,
        }

    def _validate_manifest(self, context: Mapping[str, Any]) -> bool:
        if not self._has_complete_snapshot(context):
            return False
        manifest_path = self._rank_dir(context) / MANIFEST_FILE_NAME
        try:
            manifest = read_json(manifest_path)
        except (json.JSONDecodeError, OSError) as exc:
            ad_logger.warning(f"Ignoring invalid pipeline cache manifest {manifest_path}: {exc}")
            return False

        expected = {
            "cache_key": context["cache_key"],
            "rank": context["shared_config"].local_rank,
        }
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                ad_logger.info(
                    f"Pipeline cache manifest mismatch: {key}={manifest.get(key)!r} "
                    f"!= {expected_value!r}"
                )
                return False
        try:
            self._verify_file_checksums(context, manifest)
        except ValueError as exc:
            ad_logger.warning(str(exc))
            return False
        return True

    def _load_module(
        self,
        context: Mapping[str, Any],
    ) -> nn.Module:
        rank_dir = self._rank_dir(context)
        module = load_module_structural(rank_dir / MODULE_FILE_NAME)

        hook_specs = read_json(rank_dir / HOOKS_FILE_NAME)
        reattach_hooks(module, hook_specs)
        return module

    def _verify_file_checksums(
        self, context: Mapping[str, Any], manifest: Mapping[str, Any]
    ) -> None:
        rank_dir = self._rank_dir(context)
        checksums = manifest.get("file_checksums", {}) or {}
        required_files = (
            MODULE_FILE_NAME,
            HOOKS_FILE_NAME,
        )
        for file_name in required_files:
            expected = checksums.get(file_name)
            if not expected:
                raise ValueError(f"Pipeline cache manifest is missing checksum for {file_name}.")
            path = rank_dir / file_name
            if not path.exists():
                raise ValueError(f"Pipeline cache file is missing: {path}")
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(
                    f"Pipeline cache checksum mismatch for {path}: {actual} != {expected}"
                )

    def _has_complete_snapshot(self, context: Mapping[str, Any]) -> bool:
        cache_entry_dir = self._cache_entry_dir(context)
        for rank in range(context["shared_config"].world_size):
            rank_dir = cache_entry_dir / f"rank_{rank}"
            for file_name in (
                MANIFEST_FILE_NAME,
                MODULE_FILE_NAME,
                HOOKS_FILE_NAME,
            ):
                if not (rank_dir / file_name).exists():
                    return False
        return True

    def _cache_entry_dir(self, context: Mapping[str, Any]) -> Path:
        return context["root"] / context["cache_key"]

    def _rank_dir(self, context: Mapping[str, Any]) -> Path:
        return self._cache_entry_dir(context) / f"rank_{context['shared_config'].local_rank}"

    def _tmp_rank_dir(self, context: Mapping[str, Any]) -> Path:
        shared_config = context["shared_config"]
        return context["root"] / (
            f".{context['cache_key']}.rank_{shared_config.local_rank}.tmp.{os.getpid()}"
        )
