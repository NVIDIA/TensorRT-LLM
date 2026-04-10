"""Utilities for distributed parallelism setup in diffusion models."""

from typing import Optional, Tuple

import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


def setup_sequence_parallelism(
    model_config: DiffusionModelConfig,
    num_attention_heads: int,
) -> Tuple[bool, int, Optional[dist.ProcessGroup], int]:
    """
    Setup sequence parallelism (Ulysses or Attention2D) with CFG support.

    Creates nested process groups where each CFG group has its own Ulysses group.
    Example with cfg_size=2, ulysses_size=2, world_size=4:
        GPU 0-1: CFG group 0, Ulysses group 0
        GPU 2-3: CFG group 1, Ulysses group 1

    For Attention2D, a 2D mesh is created within each CFG group.
    Example with cfg_size=1, attn2d_row_size=2, attn2d_col_size=2, world_size=4:
        Mesh layout (local index = r * col_size + c):
            (r=0,c=0)=GPU 0   (r=0,c=1)=GPU 1
            (r=1,c=0)=GPU 2   (r=1,c=1)=GPU 3
        row groups: {0,1}, {2,3}   (Q all-gather + output reduce-scatter)
        col groups: {0,2}, {1,3}   (K/V all-gather)
        mesh group: {0,1,2,3}

    Args:
        model_config: Model configuration containing parallel settings
        num_attention_heads: Number of attention heads in the model

    Returns:
        Tuple of (use_parallelism, parallelism_size, parallelism_pg, parallelism_rank):
            - use_parallelism: Whether sequence parallelism is enabled
            - parallelism_size: The sequence parallelism degree
            - parallelism_pg: The process group for this rank (or None)
              For Attention2D this is the full mesh group (all P ranks per CFG
              group), used for model-level hidden-state scatter/gather
            - parallelism_rank: This rank's position within its parallelism group

    Raises:
        RuntimeError: If torch.distributed is not initialized
        ValueError: If configuration is invalid (incompatible sizes, head count not divisible, etc.)
        NotImplementedError: If Ring attention is requested (not yet implemented)

    Side Effects:
        - Attention2D: sets model_config.attn2d_row_process_group,
          model_config.attn2d_col_process_group, and model_config.attn2d_mesh_process_group

    Note:
        Ulysses: both num_attention_heads and sequence length must be divisible by ulysses_size.
        Head count is validated here; sequence length is validated at runtime during forward pass.
        Attention2D: sequence length must be divisible by attn2d_row_size (for Q) and by
        attn2d_col_size (for K/V). Both are validated at runtime during the forward pass.
    """
    ulysses_size = model_config.parallel.dit_ulysses_size
    ring_size = model_config.parallel.dit_ring_size
    cfg_size = model_config.parallel.dit_cfg_size
    attn2d_row_size = model_config.parallel.dit_attn2d_row_size
    attn2d_col_size = model_config.parallel.dit_attn2d_col_size
    attn2d_mesh_size = attn2d_row_size * attn2d_col_size

    use_attn2d = attn2d_mesh_size > 1
    use_ulysses = ulysses_size > 1

    # Check for ring attention (not yet implemented)
    if ring_size > 1:
        raise NotImplementedError("Ring attention parallelism is not yet implemented")

    if use_attn2d and use_ulysses:
        raise ValueError(
            "Ulysses (dit_ulysses_size > 1) and Attention2D "
            "(dit_attn2d_row_size * dit_attn2d_col_size > 1) are mutually exclusive."
        )

    # Clear all Attention2D process groups before setting the active strategy
    model_config.attn2d_row_process_group = None
    model_config.attn2d_col_process_group = None
    model_config.attn2d_mesh_process_group = None

    # Early exit if not using sequence parallelism
    if not use_attn2d and not use_ulysses:
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
    seq_size = ulysses_size if use_ulysses else attn2d_mesh_size
    total_parallel = cfg_size * seq_size
    if total_parallel > world_size:
        raise ValueError(
            f"cfg_size ({cfg_size}) * seq_parallel_size ({seq_size}) = "
            f"{total_parallel} exceeds world_size ({world_size})"
        )

    if use_ulysses:
        # Validate head count divisibility
        if num_attention_heads % ulysses_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"ulysses_size ({ulysses_size})"
            )
        return _setup_ulysses(model_config, cfg_size, ulysses_size, rank)
    else:
        return _setup_attn2d(model_config, cfg_size, attn2d_row_size, attn2d_col_size, rank)


def _setup_ulysses(
    model_config: DiffusionModelConfig,
    cfg_size: int,
    ulysses_size: int,
    rank: int,
) -> Tuple[bool, int, Optional[dist.ProcessGroup], int]:
    """Create Ulysses process groups (one per CFG group) and update model_config."""
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

    return True, ulysses_size, ulysses_pg, ulysses_rank


def _setup_attn2d(
    model_config: DiffusionModelConfig,
    cfg_size: int,
    row_size: int,
    col_size: int,
    rank: int,
) -> Tuple[bool, int, Optional[dist.ProcessGroup], int]:
    """
    Create Attention2D process groups (row, col, full mesh per CFG group)
    and update model_config.

    Within each CFG group, rank (r, c) has local index r * col_size + c:
      - row_process_group: all ranks with same r (Q all-gather + output reduce-scatter)
      - col_process_group: all ranks with same c (K/V all-gather)
      - mesh_process_group: all P ranks (model-level hidden-state scatter/gather)
    """
    mesh_size = row_size * col_size
    row_pg = None
    col_pg = None
    mesh_pg = None
    mesh_rank = 0

    for cfg_id in range(cfg_size):
        base = cfg_id * mesh_size

        # Full mesh group (all P ranks in this CFG group)
        all_ranks = list(range(base, base + mesh_size))
        pg = dist.new_group(all_ranks, use_local_synchronization=True)
        if rank in all_ranks:
            mesh_pg = pg
            mesh_rank = rank - base

        # Row groups: same r, varying c
        for r in range(row_size):
            ranks = [base + r * col_size + c for c in range(col_size)]
            pg = dist.new_group(ranks, use_local_synchronization=True)
            if rank in ranks:
                row_pg = pg

        # Col groups: varying r, same c
        for c in range(col_size):
            ranks = [base + r * col_size + c for r in range(row_size)]
            pg = dist.new_group(ranks, use_local_synchronization=True)
            if rank in ranks:
                col_pg = pg

    # Store in config for Attention modules and transformer forward
    model_config.attn2d_row_process_group = row_pg
    model_config.attn2d_col_process_group = col_pg
    model_config.attn2d_mesh_process_group = mesh_pg
    return True, mesh_size, mesh_pg, mesh_rank
