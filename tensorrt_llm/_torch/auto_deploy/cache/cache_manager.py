"""Cache manager for export cache.

Simple interface for saving and loading cached exports.
"""

import os
from pathlib import Path
from typing import Optional

from ..utils.logger import ad_logger


class ExportCacheConfig:
    """Configuration for the export cache."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enabled: bool = True,
    ):
        """Initialize cache configuration.

        Args:
            cache_dir: Directory to store cache.
                      Supports environment variable expansion.
                      Defaults to TRTLLM_CACHE_DIR/autodeploy/export or ~/.cache/autodeploy/export.
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled

        # Resolve cache directory
        if cache_dir:
            self.cache_dir = Path(os.path.expandvars(os.path.expanduser(cache_dir)))
        elif os.environ.get("TRTLLM_CACHE_DIR"):
            self.cache_dir = Path(os.environ["TRTLLM_CACHE_DIR"]) / "autodeploy" / "export"
        else:
            self.cache_dir = Path.home() / ".cache" / "autodeploy" / "export"

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            ad_logger.debug(f"Export cache directory: {self.cache_dir}")
