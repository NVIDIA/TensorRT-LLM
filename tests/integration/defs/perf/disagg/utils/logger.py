"""Unified logging module for disagg benchmark.

Provides structured logging with both console and file output.
Logger is automatically initialized and ready to use upon import.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class DisaggLogger:
    """Unified logger that writes to both console and file.

    Features:
    - Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Simultaneous output to stdout and file
    - Automatic log file creation in output directory
    - Clean formatting without emoji icons

    Note: Use get_logger() function to get the singleton instance.
    """

    def __init__(self):
        """Initialize logger.

        Note: This should only be called once by get_logger().
        Multiple calls are safe due to handler clearing.
        """
        self.logger = logging.getLogger("disagg_benchmark")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # Clear any existing handlers to avoid duplicates

        # Console handler - outputs to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler will be added when output path is available
        self.file_handler: Optional[logging.FileHandler] = None

    def setup_file_logging(self, output_path: str) -> None:
        """Setup file logging to output directory.

        Args:
            output_path: Output directory path from EnvManager.get_output_path()
        """
        if self.file_handler:
            # File handler already exists
            return

        try:
            os.makedirs(output_path, exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(output_path, f"disagg_benchmark_{timestamp}.log")

            self.file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            self.file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)

            self.info(f"Log file created: {log_file}")
        except Exception as e:
            self.warning(f"Failed to setup file logging: {e}")

    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log critical message."""
        self.logger.critical(msg)

    def success(self, msg: str) -> None:
        """Log success message (as INFO level)."""
        self.logger.info(f"[SUCCESS] {msg}")

    def failure(self, msg: str) -> None:
        """Log failure message (as ERROR level)."""
        self.logger.error(f"[FAILED] {msg}")


# Global logger instance - lazy initialization via proxy class
_logger_instance: Optional[DisaggLogger] = None


class LazyLogger:
    """Lazy logger proxy that initializes on first access."""

    def _ensure_logger(self) -> DisaggLogger:
        """Ensure logger is initialized and return it."""
        global _logger_instance
        if _logger_instance is None:
            _logger_instance = DisaggLogger()
            # Auto-setup file logging on first access
            try:
                from .common import EnvManager

                output_path = EnvManager.get_output_path()
                if output_path and not output_path.startswith("<"):
                    _logger_instance.setup_file_logging(output_path)
                else:
                    _logger_instance.warning(
                        f"OUTPUT_PATH not configured (current: '{output_path}'). Logging to console only."
                    )
                    _logger_instance.info(
                        "Set OUTPUT_PATH environment variable to enable file logging."
                    )
            except Exception as e:
                _logger_instance.warning(
                    f"Failed to setup file logging: {e}. Logging to console only."
                )
        return _logger_instance

    def __getattr__(self, name):
        """Delegate all attribute access to the real logger."""
        return getattr(self._ensure_logger(), name)


# Export lazy logger instance - only initializes when actually used
logger = LazyLogger()
