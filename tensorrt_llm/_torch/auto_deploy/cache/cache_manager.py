"""Cache manager for compilation cache.

Provides the main interface for managing compilation caches including
saving, loading, and cache lifecycle management.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.fx import GraphModule

from ..utils.logger import ad_logger
from .cache_key import CacheKey
from .graph_serializer import GraphSerializer


class CompilationCacheConfig:
    """Configuration for the compilation cache."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enabled: bool = True,
        save_cache: bool = True,
        load_cache: bool = True,
        max_cache_size_gb: float = 50.0,
    ):
        """Initialize cache configuration.

        Args:
            cache_dir: Directory to store compilation cache.
                      Supports environment variable expansion.
                      Defaults to AUTODEPLOY_CACHE_DIR env var or ~/.cache/autodeploy/compilation.
            enabled: Master switch to enable/disable caching.
            save_cache: Whether to save compilation results to cache.
            load_cache: Whether to load from cache if available.
            max_cache_size_gb: Maximum total cache size in GB before evicting old entries.
        """
        self.enabled = enabled
        self.save_cache = save_cache
        self.load_cache = load_cache
        self.max_cache_size_gb = max_cache_size_gb

        # Resolve cache directory
        if cache_dir:
            self.cache_dir = Path(os.path.expandvars(os.path.expanduser(cache_dir)))
        elif os.environ.get("AUTODEPLOY_CACHE_DIR"):
            self.cache_dir = Path(os.environ["AUTODEPLOY_CACHE_DIR"]) / "compilation"
        elif os.environ.get("TRTLLM_CACHE_DIR"):
            self.cache_dir = Path(os.environ["TRTLLM_CACHE_DIR"]) / "autodeploy" / "compilation"
        else:
            self.cache_dir = Path.home() / ".cache" / "autodeploy" / "compilation"


class CompilationCacheManager:
    """Manages the compilation cache for AutoDeploy.

    This class handles:
    - Cache key management
    - Checking for valid cached graphs
    - Loading cached graphs
    - Saving graphs to cache
    - Cache cleanup and eviction
    """

    VERSION = "1.0"

    def __init__(self, config: CompilationCacheConfig):
        """Initialize the cache manager.

        Args:
            config: Cache configuration.
        """
        self.config = config
        self._current_key: Optional[CacheKey] = None

        # Ensure cache directory exists
        if self.config.enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def set_cache_key(self, key: CacheKey) -> None:
        """Set the current cache key for this compilation.

        Args:
            key: The cache key for the current compilation.
        """
        self._current_key = key
        ad_logger.debug(
            f"Cache key set: model={key.model_id}, hash={key.transforms_config_hash}, "
            f"world_size={key.world_size}, rank={key.local_rank}"
        )

    def get_cache_path(self) -> Optional[Path]:
        """Get the cache path for the current key.

        Returns:
            Path to cache directory, or None if no key is set.
        """
        if self._current_key is None:
            return None
        return self._current_key.to_cache_path(self.config.cache_dir)

    def has_valid_cache(self) -> bool:
        """Check if a valid cache exists for the current configuration.

        Returns:
            True if a valid cache exists and can be loaded.
        """
        if not self.config.enabled or not self.config.load_cache:
            return False

        cache_path = self.get_cache_path()
        if cache_path is None:
            return False

        if not GraphSerializer.is_valid_cache(cache_path):
            return False

        # Verify cache version
        try:
            metadata_file = cache_path / GraphSerializer.METADATA_FILE
            if metadata_file.exists():
                import json

                with open(metadata_file) as f:
                    metadata = json.load(f)
                if metadata.get("cache_version") != self.VERSION:
                    ad_logger.warning(
                        f"Cache version mismatch: {metadata.get('cache_version')} != {self.VERSION}"
                    )
                    return False
            return True
        except Exception as e:
            ad_logger.warning(f"Cache validation failed: {e}")
            return False

    def load_cached_graph(self) -> Optional[Tuple[GraphModule, Dict[str, Any]]]:
        """Load the cached graph if available.

        Returns:
            Tuple of (GraphModule, metadata dict), or None if no valid cache.
        """
        if not self.has_valid_cache():
            return None

        cache_path = self.get_cache_path()
        try:
            gm, metadata = GraphSerializer.load(cache_path)
            ad_logger.info(
                f"Loaded compilation cache from {cache_path} "
                f"(cached stage: {metadata.get('cached_stage', 'unknown')})"
            )
            return gm, metadata
        except Exception as e:
            ad_logger.warning(f"Failed to load cache: {e}")
            return None

    def save_graph_to_cache(
        self,
        gm: GraphModule,
        transform_history: Dict[str, Any],
        cached_stage: str,
        exported_program: Optional[Any] = None,
    ) -> None:
        """Save the current graph to cache using torch.export.save().

        Args:
            gm: The GraphModule to cache.
            transform_history: History of transforms that were applied.
            cached_stage: Name of the stage after which we're caching.
            exported_program: The ExportedProgram to save. Required for torch.export.save().
        """
        if not self.config.enabled or not self.config.save_cache:
            return

        cache_path = self.get_cache_path()
        if cache_path is None:
            ad_logger.warning("No cache key set, cannot save cache")
            return

        try:
            # Prepare metadata
            metadata = {
                "cache_version": self.VERSION,
                "cached_stage": cached_stage,
                "transform_history": transform_history,
                "created_at": datetime.now().isoformat(),
                "cache_key": self._current_key.to_dict() if self._current_key else {},
            }

            GraphSerializer.save(gm, cache_path, metadata, exported_program=exported_program)

            # Cleanup old caches if needed
            self._cleanup_old_caches()

        except Exception as e:
            ad_logger.warning(f"Failed to save cache: {e}")

    def invalidate_cache(self) -> None:
        """Invalidate (delete) the cache for the current configuration."""
        cache_path = self.get_cache_path()
        if cache_path and cache_path.exists():
            shutil.rmtree(cache_path)
            ad_logger.info(f"Invalidated cache at {cache_path}")

    def clear_all_caches(self) -> None:
        """Clear all cached compilations."""
        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            ad_logger.info("Cleared all compilation caches")

    def list_caches(self) -> List[Dict[str, Any]]:
        """List all cached compilations with metadata.

        Returns:
            List of cache info dictionaries.
        """
        caches = []
        if not self.config.cache_dir.exists():
            return caches

        for cache_dir in self.config.cache_dir.iterdir():
            if cache_dir.is_dir() and GraphSerializer.is_valid_cache(cache_dir):
                try:
                    import json

                    metadata_file = cache_dir / GraphSerializer.METADATA_FILE
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    # Calculate size
                    size_bytes = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())

                    caches.append(
                        {
                            "path": str(cache_dir),
                            "size_mb": size_bytes / (1024 * 1024),
                            **metadata,
                        }
                    )
                except Exception as e:
                    ad_logger.debug(f"Error reading cache {cache_dir}: {e}")

        return caches

    def _cleanup_old_caches(self) -> None:
        """Remove old caches if total size exceeds limit."""
        caches = self.list_caches()
        total_size_gb = sum(c.get("size_mb", 0) for c in caches) / 1024

        if total_size_gb <= self.config.max_cache_size_gb:
            return

        # Sort by creation time, oldest first
        caches.sort(key=lambda c: c.get("created_at", ""))

        # Remove oldest caches until we're under 80% of limit
        target_size = self.config.max_cache_size_gb * 0.8
        while total_size_gb > target_size and caches:
            oldest = caches.pop(0)
            try:
                shutil.rmtree(oldest["path"])
                total_size_gb -= oldest.get("size_mb", 0) / 1024
                ad_logger.info(f"Evicted old cache: {oldest['path']}")
            except Exception as e:
                ad_logger.warning(f"Failed to evict cache {oldest['path']}: {e}")
