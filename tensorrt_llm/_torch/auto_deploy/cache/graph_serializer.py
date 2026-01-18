"""Graph serialization for compilation cache.

Uses torch.export.save/load to properly serialize ExportedPrograms.
This avoids pickling issues with C++ ops and preserves graph structure.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch.export as te
from torch.fx import GraphModule

from ..utils.logger import ad_logger


class GraphSerializer:
    """Serialize and deserialize FX GraphModules using torch.export."""

    EXPORTED_PROGRAM_FILE = "exported_program.pt2"
    METADATA_FILE = "metadata.json"

    @classmethod
    def save(
        cls,
        gm: GraphModule,
        cache_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        exported_program: Optional[te.ExportedProgram] = None,
    ) -> None:
        """Save an FX GraphModule to disk using torch.export.save.

        Args:
            gm: The GraphModule to save (used for fallback if no ExportedProgram).
            cache_path: Directory to save the cache.
            metadata: Optional metadata to save alongside (transform history, etc.)
            exported_program: The ExportedProgram to save. If provided, this is saved
                directly. Otherwise, we attempt to create one from the GraphModule.
        """
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE

        # Try to save the ExportedProgram
        saved = False

        if exported_program is not None:
            # Use the provided ExportedProgram directly
            try:
                te.save(exported_program, ep_path)
                saved = True
                ad_logger.info(f"Saved ExportedProgram cache to {cache_path}")
            except Exception as e:
                ad_logger.warning(f"Failed to save ExportedProgram: {e}")

        if not saved:
            # Try to get ExportedProgram from GraphModule's meta if available
            ep = getattr(gm, "_exported_program", None)
            if ep is not None:
                try:
                    te.save(ep, ep_path)
                    saved = True
                    ad_logger.info(f"Saved ExportedProgram cache to {cache_path}")
                except Exception as e:
                    ad_logger.warning(f"Failed to save ExportedProgram from meta: {e}")

        if not saved:
            ad_logger.warning(
                "No ExportedProgram available for caching. Cache will only contain metadata."
            )

        # Save metadata as JSON
        if metadata:
            metadata_path = cache_path / cls.METADATA_FILE
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(
        cls,
        cache_path: Path,
        device: str = "meta",
    ) -> Tuple[GraphModule, Dict[str, Any]]:
        """Load an FX GraphModule from disk using torch.export.load.

        Args:
            cache_path: Directory containing the cached graph.
            device: Device to load weights on (default: "meta" for placeholder weights).

        Returns:
            Tuple of (GraphModule with placeholder weights, metadata dict)
        """
        cache_path = Path(cache_path)
        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE

        # Load the ExportedProgram
        ep = te.load(ep_path)

        # Get the GraphModule from the ExportedProgram
        gm = ep.module()

        # Store reference to ExportedProgram for potential re-saving
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

        ad_logger.info(f"Loaded ExportedProgram cache from {cache_path}")
        return gm, metadata

    @classmethod
    def is_valid_cache(cls, cache_path: Path) -> bool:
        """Check if a cache directory contains valid cache files.

        Args:
            cache_path: Directory to check.

        Returns:
            True if the cache appears valid.
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return False

        ep_path = cache_path / cls.EXPORTED_PROGRAM_FILE
        if not ep_path.exists():
            return False

        # Quick validation - check metadata exists and is valid JSON
        try:
            metadata_file = cache_path / cls.METADATA_FILE
            if metadata_file.exists():
                with open(metadata_file) as f:
                    json.load(f)
            return True
        except Exception:
            return False
