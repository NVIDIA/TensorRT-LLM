"""Graph serialization for export cache.

Uses torch.export.save/load to serialize ExportedPrograms.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch.export as te
from torch.fx import GraphModule

from ..utils.logger import ad_logger


class GraphSerializer:
    """Serialize and deserialize exported graphs using torch.export."""

    EXPORTED_PROGRAM_FILE = "exported_program.pt2"
    METADATA_FILE = "metadata.json"

    @classmethod
    def save(
        cls,
        exported_program: te.ExportedProgram,
        cache_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save an ExportedProgram to disk.

        Args:
            exported_program: The ExportedProgram from torch.export.
            cache_path: Directory to save the cache.
            metadata: Optional metadata to save alongside.
        """
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save the ExportedProgram
        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE
        te.save(exported_program, ep_path)

        # Save metadata
        if metadata:
            metadata_path = cache_path / cls.METADATA_FILE
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        ad_logger.info(f"Saved export cache to {cache_path}")

    @classmethod
    def load(cls, cache_path: Path) -> Tuple[GraphModule, Dict[str, Any]]:
        """Load a cached export from disk.

        Args:
            cache_path: Directory containing the cached export.

        Returns:
            Tuple of (GraphModule, metadata dict)
        """
        cache_path = Path(cache_path)
        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE

        # Load the ExportedProgram
        ep = te.load(ep_path)
        gm = ep.module()

        # Store reference to ExportedProgram
        gm._exported_program = ep

        # Ensure gm has meta dict for transforms
        if not hasattr(gm, "meta"):
            gm.meta = {}
        gm.meta["_autodeploy"] = {}

        # Load metadata
        metadata = {}
        metadata_path = cache_path / cls.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        ad_logger.info(f"Loaded export cache from {cache_path}")
        return gm, metadata

    @classmethod
    def is_valid_cache(cls, cache_path: Path) -> bool:
        """Check if a cache directory contains valid cache files."""
        cache_path = Path(cache_path)
        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE
        return ep_path.exists()
