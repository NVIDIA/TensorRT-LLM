"""Cache key generation for export cache.

The cache key uniquely identifies an exported graph based on:
- Model identity (path/name)
- Model configuration hash
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class CacheKey:
    """A unique identifier for a cached export.

    Attributes:
        model_id: Model path or HuggingFace hub name.
        config_hash: Hash of the model configuration.
    """

    model_id: str
    config_hash: str

    def to_cache_path(self, cache_dir: Path) -> Path:
        """Generate a unique cache directory path."""
        # Create a readable prefix from model name
        model_name = Path(self.model_id).name if "/" in self.model_id else self.model_id
        # Sanitize model name for filesystem
        model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)

        return cache_dir / f"{model_name}_{self.config_hash}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_model_config(cls, model: str, model_config: Dict[str, Any]) -> "CacheKey":
        """Create a cache key from model and its configuration.

        Args:
            model: Model path or HuggingFace hub name.
            model_config: Model configuration dictionary.

        Returns:
            CacheKey instance.
        """
        # Create deterministic hash of the config
        config_str = json.dumps(model_config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return cls(model_id=model, config_hash=config_hash)
