import torch
import torch.distributed as dist


def split(color: int, key: int,
          pg_boxed: torch.ScriptObject) -> torch.ScriptObject:
    """Create a subgroup ProcessGroup.

    This gathers (color, key) from all ranks, selects members with matching color,
    sorts them by key to determine rank ordering, and creates a new ProcessGroup
    for those ranks using torch.distributed.new_group, inheriting backend from
    the global ProcessGroup.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    try:
        pg = torch.distributed.ProcessGroup.unbox(pg_boxed)
    except Exception as e:
        raise ValueError(f"Error unboxing ProcessGroup: {e}") from e

    group_size = dist.get_world_size(group=pg)
    # gather (color, key, global_rank) within the provided pg
    payload = (int(color), int(key), int(dist.get_rank(group=pg)))
    gathered = [None] * group_size
    dist.all_gather_object(gathered, payload, group=pg)

    members = []
    for c, k, global_rank in gathered:
        if c == color:
            members.append((int(k), int(global_rank)))

    members.sort()
    ranks = [r for _, r in members]
    if not ranks:
        raise ValueError(f"Split by color {color} produced empty subgroup")
    if (current_rank := dist.get_rank()) not in ranks:
        raise ValueError(
            f"Current rank {current_rank} not in color {color} subgroup")

    # Create subgroup under the provided pg; ranks are global ranks
    sub_pg = dist.new_group(ranks=ranks, use_local_synchronization=True)
    # Return TorchScript boxed ProcessGroup so C++ can unwrap it
    return sub_pg.boxed()
