"""Cache key generation for compilation cache.

The cache key uniquely identifies a compilation configuration based on:
- Model identity (path/name)
- Transforms that ran and their configuration
- World size and local rank (for sharding)
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CacheKey:
    """A unique identifier for a cached compilation.

    Attributes:
        model_id: Model path or HuggingFace hub name.
        transforms_config_hash: Hash of the transforms that ran and their configs.
        world_size: Number of GPUs (affects sharding).
        local_rank: Rank-specific graph variations.
        enabled_transforms: List of enabled transform names (for debugging/logging).
    """

    model_id: str
    transforms_config_hash: str
    world_size: int
    local_rank: int
    enabled_transforms: List[str] = field(default_factory=list)

    def to_cache_path(self, cache_dir: Path) -> Path:
        """Generate a unique cache directory path.

        Args:
            cache_dir: Base cache directory.

        Returns:
            Path to the cache directory for this configuration.
        """
        # Create a readable prefix from model name
        model_name = Path(self.model_id).name if "/" in self.model_id else self.model_id
        # Sanitize model name for filesystem
        model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)

        # Create unique hash from all key components
        key_components = (
            f"{self.model_id}|{self.transforms_config_hash}|ws{self.world_size}|r{self.local_rank}"
        )
        key_hash = hashlib.sha256(key_components.encode()).hexdigest()[:16]

        return cache_dir / f"{model_name}_{key_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "transforms_config_hash": self.transforms_config_hash,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "enabled_transforms": self.enabled_transforms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheKey":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            transforms_config_hash=data["transforms_config_hash"],
            world_size=data["world_size"],
            local_rank=data["local_rank"],
            enabled_transforms=data.get("enabled_transforms", []),
        )

    @classmethod
    def from_config(
        cls,
        model: str,
        transforms_config: Dict[str, Any],
        world_size: int = 1,
        local_rank: int = 0,
    ) -> "CacheKey":
        """Create a cache key from configuration.

        Args:
            model: Model path or HuggingFace hub name.
            transforms_config: Dictionary of transform configurations.
            world_size: Number of GPUs.
            local_rank: Local rank of this process.

        Returns:
            CacheKey instance.
        """
        # Extract enabled transforms and their graph-affecting configs
        enabled_transforms = []
        config_for_hash = {}

        for t_name, t_config in transforms_config.items():
            # Check if transform is enabled (default True if not specified)
            if isinstance(t_config, dict):
                enabled = t_config.get("enabled", True)
            else:
                # TransformConfig object
                enabled = getattr(t_config, "enabled", True)
                t_config = (
                    t_config.model_dump() if hasattr(t_config, "model_dump") else vars(t_config)
                )

            if enabled:
                enabled_transforms.append(t_name)
                # Store the full config for this transform
                config_for_hash[t_name] = _extract_graph_affecting_config(t_config)

        enabled_transforms.sort()

        # Create deterministic hash of the config
        config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
        transforms_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        return cls(
            model_id=model,
            transforms_config_hash=transforms_hash,
            world_size=world_size,
            local_rank=local_rank,
            enabled_transforms=enabled_transforms,
        )


def _extract_graph_affecting_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the config values that affect graph structure.

    This filters out debug/logging settings that don't change the output graph.

    Args:
        config: Full transform configuration.

    Returns:
        Filtered configuration with only graph-affecting keys.
    """
    # Keys that DON'T affect the graph output
    NON_GRAPH_AFFECTING_KEYS = {
        "skip_on_error",
        "run_graph_cleanup",
        "run_shape_prop",
        "requires_clean_graph",
        "requires_shape_prop",
    }

    return {k: v for k, v in config.items() if k not in NON_GRAPH_AFFECTING_KEYS}
