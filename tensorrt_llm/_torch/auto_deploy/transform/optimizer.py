"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
import logging
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from .interface import (
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)

logger = logging.getLogger(__name__)


def _format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.2f} PB"


def _log_memory_status(label: str, device: Optional[int] = None) -> dict:
    """Log detailed GPU memory status.

    Args:
        label: A label to identify when this status was captured.
        device: The CUDA device index. If None, uses current device.

    Returns:
        A dict containing memory statistics for comparison.
    """
    if not torch.cuda.is_available():
        logger.info(f"[{label}] CUDA not available")
        return {}

    if device is None:
        device = torch.cuda.current_device()

    # Basic memory stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    free_memory = total_memory - reserved

    # Calculate percentages
    alloc_pct = (allocated / total_memory) * 100 if total_memory > 0 else 0
    reserved_pct = (reserved / total_memory) * 100 if total_memory > 0 else 0

    logger.info(f"\n{'=' * 70}")
    logger.info(f"[{label}] GPU Memory Status (Device {device})")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Allocated:       {_format_bytes(allocated):>15} ({alloc_pct:.1f}% of total)")
    logger.info(f"  Reserved:        {_format_bytes(reserved):>15} ({reserved_pct:.1f}% of total)")
    logger.info(f"  Free (unreserved): {_format_bytes(free_memory):>13}")
    logger.info(f"  Total GPU Memory: {_format_bytes(total_memory):>14}")
    logger.info(f"  Peak Allocated:  {_format_bytes(max_allocated):>15}")
    logger.info(f"  Peak Reserved:   {_format_bytes(max_reserved):>15}")

    # Memory allocator stats (more detailed)
    try:
        stats = torch.cuda.memory_stats(device)
        active_blocks = stats.get("active.all.current", 0)
        active_bytes = stats.get("active_bytes.all.current", 0)
        inactive_bytes = stats.get("inactive_split_bytes.all.current", 0)
        num_alloc_retries = stats.get("num_alloc_retries", 0)
        num_ooms = stats.get("num_ooms", 0)

        logger.info(f"  Active Blocks:   {active_blocks:>15,}")
        logger.info(f"  Active Bytes:    {_format_bytes(active_bytes):>15}")
        logger.info(f"  Inactive Split:  {_format_bytes(inactive_bytes):>15}")
        logger.info(f"  Alloc Retries:   {num_alloc_retries:>15,}")
        logger.info(f"  OOM Events:      {num_ooms:>15,}")
    except Exception as e:
        logger.debug(f"  Could not get detailed memory stats: {e}")

    logger.info(f"{'=' * 70}\n")

    return {
        "allocated": allocated,
        "reserved": reserved,
        "max_allocated": max_allocated,
        "free": free_memory,
        "total": total_memory,
    }


def _log_memory_diff(before: dict, after: dict, transform_name: str) -> None:
    """Log the difference in memory between before and after a transform."""
    if not before or not after:
        return

    alloc_diff = after["allocated"] - before["allocated"]
    reserved_diff = after["reserved"] - before["reserved"]

    sign_alloc = "+" if alloc_diff >= 0 else ""
    sign_reserved = "+" if reserved_diff >= 0 else ""

    logger.info(f"[{transform_name}] Memory Change Summary:")
    logger.info(f"  Allocated: {sign_alloc}{_format_bytes(alloc_diff)}")
    logger.info(f"  Reserved:  {sign_reserved}{_format_bytes(reserved_diff)}")
    logger.info("")


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config: InferenceOptimizerConfig):
        self.factory = factory
        self.config = self._clean_config(config)
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(local_rank=local_rank, world_size=world_size)

    def _clean_config(self, config: InferenceOptimizerConfig) -> StrictInferenceOptimizerConfig:
        """Get a typed checked ("strict") config with sorted keys according to stages."""
        # convert to nested kwargs, no TransformConfig objects allowed
        nested_kwargs = {
            k: v.model_dump() if isinstance(v, TransformConfig) else v for k, v in config.items()
        }
        # sort by stage
        keys_sorted = sorted(nested_kwargs.keys(), key=lambda k: Stages(nested_kwargs[k]["stage"]))
        # create strict config with correct config classes and correct order
        strict_config: StrictInferenceOptimizerConfig = {
            k: TransformRegistry.get_config_class(k)(**nested_kwargs[k]) for k in keys_sorted
        }
        # return strict config
        return strict_config

    def __call__(self, cm: CachedSequenceInterface, mod: Optional[nn.Module] = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        # Log initial memory status
        logger.info("\n" + "#" * 70)
        logger.info("# INFERENCE OPTIMIZER - Starting Transformations")
        logger.info(f"# Number of transforms: {len(self.config)}")
        logger.info(f"# Transforms: {list(self.config.keys())}")
        logger.info("#" * 70)
        initial_mem = _log_memory_status("INITIAL (before all transforms)")

        # Reset peak memory stats for tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)

            # Log memory status before transform
            logger.info(f"\n>>> Starting transform: {t_name}")
            mem_before = _log_memory_status(f"BEFORE {t_name}")

            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            # Synchronize CUDA and collect garbage for accurate memory measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log memory status after transform
            mem_after = _log_memory_status(f"AFTER {t_name}")
            _log_memory_diff(mem_before, mem_after, t_name)
            logger.info(f"<<< Completed transform: {t_name}\n")

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()

        # Log final memory status and summary
        final_mem = _log_memory_status("FINAL (after all transforms)")

        logger.info("\n" + "#" * 70)
        logger.info("# INFERENCE OPTIMIZER - Transformation Complete")
        logger.info("#" * 70)
        if initial_mem and final_mem:
            total_alloc_change = final_mem["allocated"] - initial_mem["allocated"]
            total_reserved_change = final_mem["reserved"] - initial_mem["reserved"]
            peak_allocated = final_mem.get("max_allocated", 0)

            sign_alloc = "+" if total_alloc_change >= 0 else ""
            sign_reserved = "+" if total_reserved_change >= 0 else ""

            logger.info("# Total Memory Change Summary:")
            logger.info(f"#   Allocated Change: {sign_alloc}{_format_bytes(total_alloc_change)}")
            logger.info(
                f"#   Reserved Change:  {sign_reserved}{_format_bytes(total_reserved_change)}"
            )
            logger.info(f"#   Peak Allocated:   {_format_bytes(peak_allocated)}")
            logger.info(f"#   Final Allocated:  {_format_bytes(final_mem['allocated'])}")
            logger.info(f"#   Final Free:       {_format_bytes(final_mem['free'])}")
        logger.info("#" * 70 + "\n")

        return mod
