"""Utilities for distributed parallelism setup in diffusion models."""

from typing import Optional, Tuple

import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


def setup_sequence_parallelism(
    model_config: DiffusionModelConfig,
    num_attention_heads: int,
) -> Tuple[bool, int, Optional[dist.ProcessGroup], int]:
    """
    Setup sequence parallelism (currently Ulysses only) with CFG support.

    Creates nested process groups where each CFG group has its own Ulysses group.
    Example with cfg_size=2, ulysses_size=2, world_size=4:
        GPU 0-1: CFG group 0, Ulysses group 0
        GPU 2-3: CFG group 1, Ulysses group 1

    Args:
        model_config: Model configuration containing parallel settings
        num_attention_heads: Number of attention heads in the model

    Returns:
        Tuple of (use_parallelism, parallelism_size, parallelism_pg, parallelism_rank):
            - use_parallelism: Whether sequence parallelism is enabled
            - parallelism_size: The sequence parallelism degree
            - parallelism_pg: The process group for this rank (or None)
            - parallelism_rank: This rank's position within its parallelism group

    Raises:
        RuntimeError: If torch.distributed is not initialized
        ValueError: If configuration is invalid (incompatible sizes, head count not divisible, etc.)
        NotImplementedError: If Ring attention is requested (not yet implemented)

    Side Effects:
        - Sets model_config.ulysses_process_group to the created process group

    Note:
        Both num_attention_heads and sequence length must be divisible by ulysses_size.
        Head count is validated here; sequence length is validated at runtime during forward pass.
    """
    ulysses_size = model_config.parallel.dit_ulysses_size
    ring_size = model_config.parallel.dit_ring_size
    cfg_size = model_config.parallel.dit_cfg_size

    # Check for ring attention (not yet implemented)
    if ring_size > 1:
        raise NotImplementedError("Ring attention parallelism is not yet implemented")

    # Early exit if not using sequence parallelism
    if ulysses_size <= 1:
        model_config.ulysses_process_group = None
        return False, 1, None, 0

    # Validate distributed initialization
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed.init_process_group() must be called before "
            "setting up sequence parallelism"
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Validate total parallelism capacity
    total_parallel = cfg_size * ulysses_size
    if total_parallel > world_size:
        raise ValueError(
            f"cfg_size ({cfg_size}) * ulysses_size ({ulysses_size}) = "
            f"{total_parallel} exceeds world_size ({world_size})"
        )

    # Validate head count divisibility
    if num_attention_heads % ulysses_size != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"ulysses_size ({ulysses_size})"
        )

    # Create nested process groups
    # Each CFG group has its own Ulysses group
    ulysses_pg = None
    ulysses_rank = 0

    for cfg_id in range(cfg_size):
        ulysses_ranks = list(range(cfg_id * ulysses_size, (cfg_id + 1) * ulysses_size))
        pg = dist.new_group(ulysses_ranks, use_local_synchronization=True)

        # Store if this rank belongs to this group
        if rank in ulysses_ranks:
            ulysses_pg = pg
            ulysses_rank = rank - cfg_id * ulysses_size

    # Store in config for Attention modules
    model_config.ulysses_process_group = ulysses_pg

    return True, ulysses_size, ulysses_pg, ulysses_rank
