# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Cache for intermediate FX graph modules produced by the AutoDeploy transform pipeline.

Stores serialized meta-device graphs (before weight loading) so that subsequent runs with
identical configuration can skip the export, pattern matching, and sharding stages.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..utils.logger import ad_logger

_DEFAULT_CACHE_DIR = "/tmp/.auto_deploy_cache"
_DEFAULT_MAX_ENTRIES = 10
_DEFAULT_MAX_SIZE_GB = 50

_GRAPH_FILENAME = "graph.pt"
_METADATA_FILENAME = "metadata.json"


class GraphCache:
    """Disk-backed LRU cache for intermediate FX graph modules.

    The cache is keyed by a SHA-256 hash of the transform configuration, model
    identifier, parallelism settings, and sequence dimensions.  Cached graphs are
    stored as ``torch.save`` artifacts alongside a JSON metadata sidecar that
    tracks access times for LRU eviction.

    The cache is completely opt-in and is only active when the ``AD_ENABLE_CACHING``
    environment variable is set to ``"1"``.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: Optional[int] = None,
        max_size_gb: Optional[float] = None,
    ):
        self._cache_dir = Path(cache_dir or os.environ.get("AD_CACHE_DIR", _DEFAULT_CACHE_DIR))
        self._max_entries = (
            max_entries
            if max_entries is not None
            else int(os.environ.get("AD_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES))
        )
        self._max_size_bytes = (
            max_size_gb * (1024**3)
            if max_size_gb is not None
            else float(os.environ.get("AD_CACHE_MAX_SIZE_GB", _DEFAULT_MAX_SIZE_GB)) * (1024**3)
        )

    @staticmethod
    def is_enabled() -> bool:
        return os.environ.get("AD_ENABLE_CACHING", "0") == "1"

    # -- public API ----------------------------------------------------------------

    @staticmethod
    def compute_cache_key(
        pre_weight_config: Dict[str, Any],
        model_id: Optional[str],
        model_kwargs: Dict[str, Any],
        world_size: int,
        rank: int,
        max_seq_len: int,
        max_batch_size: int,
    ) -> str:
        """Compute a deterministic SHA-256 cache key from pipeline parameters."""
        import tensorrt_llm

        config_for_key = _serializable_config(pre_weight_config)
        blob = {
            "model_id": str(model_id) if model_id else "",
            "model_kwargs": model_kwargs,
            "pre_weight_config": config_for_key,
            "world_size": world_size,
            "rank": rank,
            "max_seq_len": max_seq_len,
            "max_batch_size": max_batch_size,
            "trtllm_version": tensorrt_llm.__version__,
        }
        canonical = json.dumps(blob, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load(self, cache_key: str) -> Optional[nn.Module]:
        """Attempt to load a cached graph module.

        Returns ``None`` on cache miss or if the cached artifact is corrupt.
        """
        entry = self._entry_path(cache_key)
        graph_path = entry / _GRAPH_FILENAME
        meta_path = entry / _METADATA_FILENAME

        if not graph_path.exists():
            return None

        try:
            mod = torch.load(graph_path, map_location="meta", weights_only=False)
        except Exception as e:
            ad_logger.warning(f"[cache] Corrupt cache entry {cache_key[:12]}…, rebuilding: {e}")
            self._remove_entry(entry)
            return None

        self._touch_metadata(meta_path)
        ad_logger.info(f"[cache] Loaded graph from {graph_path}")
        return mod

    def save(self, cache_key: str, mod: nn.Module) -> None:
        """Persist a graph module to the cache and run eviction if needed."""
        entry = self._entry_path(cache_key)
        try:
            entry.mkdir(parents=True, exist_ok=True)
            graph_path = entry / _GRAPH_FILENAME
            meta_path = entry / _METADATA_FILENAME

            # Atomic write: save to a temp file then rename
            tmp_fd, tmp_path = tempfile.mkstemp(dir=entry, suffix=".pt.tmp")
            os.close(tmp_fd)
            try:
                torch.save(mod, tmp_path)
                os.replace(tmp_path, str(graph_path))
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            self._write_metadata(meta_path, cache_key, graph_path.stat().st_size)
            ad_logger.info(f"[cache] Saved graph to {graph_path}")
            self._evict()
        except OSError as e:
            ad_logger.warning(f"[cache] Failed to write cache entry: {e}")

    # -- private helpers -----------------------------------------------------------

    def _entry_path(self, cache_key: str) -> Path:
        return self._cache_dir / cache_key

    @staticmethod
    def _remove_entry(entry: Path) -> None:
        try:
            shutil.rmtree(entry)
        except OSError:
            pass

    @staticmethod
    def _write_metadata(meta_path: Path, cache_key: str, size_bytes: int) -> None:
        meta = {
            "cache_key": cache_key,
            "created": time.time(),
            "last_accessed": time.time(),
            "size_bytes": size_bytes,
        }
        tmp_meta = str(meta_path) + ".tmp"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f)
        os.replace(tmp_meta, str(meta_path))

    @staticmethod
    def _touch_metadata(meta_path: Path) -> None:
        """Update ``last_accessed`` in the metadata sidecar."""
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["last_accessed"] = time.time()
            tmp_meta = str(meta_path) + ".tmp"
            with open(tmp_meta, "w") as f:
                json.dump(meta, f)
            os.replace(tmp_meta, str(meta_path))
        except (OSError, json.JSONDecodeError):
            pass

    @staticmethod
    def _read_metadata(meta_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with open(meta_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _list_entries(self):
        """Yield ``(entry_path, metadata_dict)`` for every valid cache entry."""
        if not self._cache_dir.exists():
            return
        for child in self._cache_dir.iterdir():
            if not child.is_dir():
                continue
            meta_path = child / _METADATA_FILENAME
            meta = self._read_metadata(meta_path)
            if meta is not None:
                yield child, meta

    def _evict(self) -> None:
        """Remove least-recently-accessed entries until within budget."""
        entries = sorted(self._list_entries(), key=lambda e: e[1].get("last_accessed", 0))

        total_size = sum(m.get("size_bytes", 0) for _, m in entries)
        count = len(entries)

        while entries and (count > self._max_entries or total_size > self._max_size_bytes):
            path, meta = entries.pop(0)
            ad_logger.info(f"[cache] Evicting {path.name[:12]}…")
            total_size -= meta.get("size_bytes", 0)
            count -= 1
            self._remove_entry(path)


def _serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ``StrictInferenceOptimizerConfig`` dict to a JSON-safe dict."""
    result = {}
    for key, value in config.items():
        if hasattr(value, "model_dump"):
            result[key] = value.model_dump()
        elif isinstance(value, dict):
            result[key] = value
        else:
            result[key] = str(value)
    return result
