# Tentative script to clean up Ray cluster resources during dev.
# This script will be removed.

import ray
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID
from ray.util.placement_group import (PlacementGroup, placement_group_table,
                                      remove_placement_group)

ray.init()

for placement_group_info in placement_group_table().values():
    # https://github.com/ray-project/ray/blob/ray-2.7.0/python/ray/util/placement_group.py#L291
    pg = PlacementGroup(
        PlacementGroupID(
            hex_to_binary(placement_group_info["placement_group_id"])))
    print(f"removing {placement_group_info['placement_group_id']}")
    remove_placement_group(pg)

actors = ray.util.list_named_actors(all_namespaces=True)
for actor_name, actor_info in actors:
    try:
        print(f"Killing actor: {actor_name}")
        ray.kill(ray.get_actor(actor_name, namespace="trtllm"))
    except Exception as e:
        print(f"Failed to kill {actor_name}: {e}")
