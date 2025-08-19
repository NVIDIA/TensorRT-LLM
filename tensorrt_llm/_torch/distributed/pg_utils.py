import torch
import torch.distributed as dist


def split(color: int, key: int, pg_boxed: torch.ScriptObject):
    """Create a subgroup ProcessGroup.

    This gathers (color, key) from all ranks, selects members with matching color,
    sorts them by key to determine rank ordering, and creates a new ProcessGroup
    for those ranks using torch.distributed.new_group with Gloo available.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    try:
        pg = torch.distributed.ProcessGroup.unbox(pg_boxed)
    except Exception as e:
        print(f"Error unboxing ProcessGroup: {e}")
        raise e

    group_size = dist.get_world_size(group=pg)
    # gather (color, key, global_rank) within the provided pg
    payload = (int(color), int(key), int(dist.get_rank()))
    gathered = [None] * group_size
    dist.all_gather_object(gathered, payload, group=pg)

    members = []  # list of (key, global_rank)
    for c, k, global_rank in gathered:
        if c == color:
            members.append((int(k), int(global_rank)))

    members.sort()  # sort by key
    ranks = [r for _, r in members]  # global ranks in subgroup
    assert len(ranks) > 0, "split produced empty subgroup"
    assert dist.get_rank() in ranks, "current rank not in the requested color subgroup"

    # Create subgroup under the provided pg; ranks are global ranks
    sub_pg = dist.new_group(ranks=ranks, use_local_synchronization=True, backend='gloo')
    # Return TorchScript boxed ProcessGroup so C++ can unwrap it
    return sub_pg.boxed()


