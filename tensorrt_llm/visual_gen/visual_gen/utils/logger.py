# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


def get_dist_info():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), dist.get_backend()
    else:
        return 0, 1, "cpu"


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        rank, _, _ = get_dist_info()
        # Add rank info for distributed inference
        record.rank = f"[Rank {rank}]"
        return super().format(record)


class ditLogger:
    """centralized logging system."""

    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False

    @classmethod
    def _auto_setup_if_needed(cls) -> None:
        """Automatically setup logging if not already initialized."""
        if not cls._initialized:
            cls._setup_from_env()

    @classmethod
    def _setup_from_env(cls) -> None:
        """Setup logging from environment variables."""
        log_level = os.environ.get("DIT_LOG_LEVEL", "INFO")
        log_dir = os.environ.get("DIT_LOG_DIR", None)
        file_output = False if log_dir is None else True
        distributed = True if "WORLD_SIZE" in os.environ else False

        cls.setup_logging(
            log_level=log_level, log_dir=log_dir, console_output=True, file_output=file_output, distributed=distributed
        )

    @classmethod
    def setup_logging(
        cls,
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        distributed: bool = False,
    ) -> None:
        """Setup the global logging configuration.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to save log files
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            distributed: Whether running in distributed mode
        """
        if cls._initialized:
            return

        # logging for distributed inference, disable all ranks except rank 0 by setting level to CRITICAL
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))

        # Set log level
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create log directory if needed
        if file_output and log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        if world_size > 1 and rank != 0 and numeric_level == logging.INFO:
            root_logger.setLevel(logging.CRITICAL)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            if world_size > 1 and rank != 0 and numeric_level == logging.INFO:
                console_handler.setLevel(logging.CRITICAL)
            # Use colored formatter for console
            console_format = "%(rank)s%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            console_formatter = ColoredFormatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if file_output and log_dir:
            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rank_suffix = f"_rank{rank}" if world_size > 1 else ""
            log_file = Path(log_dir) / f"{timestamp}{rank_suffix}.log"

            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            if world_size > 1 and rank != 0 and numeric_level == logging.INFO:
                file_handler.setLevel(logging.CRITICAL)

            # Use detailed formatter for file
            file_format = (
                "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
            )
            file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

            # Log the log file location
            root_logger.info(f"Logging to file: {log_file}")

        # Set third-party loggers to WARNING to reduce noise
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for the given name.

        Args:
            name: Logger name (usually __name__)

        Returns:
            Logger instance
        """
        # Auto-setup logging if not already done
        cls._auto_setup_if_needed()

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]

    @classmethod
    def log_system_info(cls) -> None:
        """Log system and environment information."""
        logger = cls.get_logger("visual_gen.system")

        logger.info("=" * 60)
        logger.info("System Information")
        logger.info("=" * 60)

        # Python and system info
        import platform

        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Distributed inference info
        rank, world_size, backend = get_dist_info()
        if world_size > 1:
            logger.info("Distributed inference: Enabled")
            logger.info(f"World size: {world_size}")
            logger.info(f"Current rank: {rank}")
            logger.info(f"Backend: {backend}")
        else:
            logger.info("Distributed inference: Disabled")

        # Environment variables
        important_env_vars = ["CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]

        logger.info("Environment variables:")
        for var in important_env_vars:
            value = os.environ.get(var, "Not set")
            logger.info(f"  {var}: {value}")

        logger.info("=" * 60)

    @classmethod
    def log_model_info(cls, model: Any, model_name: str = "Model") -> None:
        """Log model information including parameters and memory usage.

        Args:
            model: The model to log information about
            model_name: Name of the model for logging
        """
        logger = cls.get_logger("visual_gen.model")

        logger.info(f"{model_name} Information:")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")

        # Model size in MB
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        logger.info(f"  Model size: {model_size_mb:.2f} MB")

        # Device information
        if hasattr(model, "device"):
            logger.info(f"  Device: {model.device}")

        # Memory usage if on CUDA
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            device = next(model.parameters()).device
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(f"  GPU memory allocated: {memory_allocated:.2f} GB")
            logger.info(f"  GPU memory reserved: {memory_reserved:.2f} GB")


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger.

    Logging is automatically configured from environment variables
    on the first call to this function.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return ditLogger.get_logger(name)


# Performance monitoring decorator
def log_execution_time(logger_name: Optional[str] = None):
    """Decorator to log function execution time.

    Args:
        logger_name: Name of the logger to use
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            logger = get_logger(logger_name or func.__module__)

            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"Completed {func.__name__} in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {execution_time:.4f}s: {e}")
                raise

        return wrapper

    return decorator


# Memory monitoring context manager
class MemoryMonitor:
    """Context manager to monitor GPU memory usage."""

    def __init__(self, logger_name: str, operation_name: str):
        self.logger = get_logger(logger_name)
        self.operation_name = operation_name
        self.start_memory = None

    def __enter__(self):
        if self.logger.level < logging.INFO:
            return self

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated()
            self.logger.debug(f"Starting {self.operation_name} - GPU memory: {self.start_memory / 1024**3:.2f} GB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger.level < logging.INFO:
            return

        if torch.cuda.is_available() and self.start_memory is not None:
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_diff = end_memory - self.start_memory
            self.logger.debug(
                f"Completed {self.operation_name} - GPU memory: {end_memory / 1024**3:.2f} GB "
                f"(diff: {memory_diff / 1024**3:+.2f} GB)"
            )
