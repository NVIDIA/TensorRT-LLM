from dataclasses import dataclass
from typing import List, Optional, Tuple

AgentHierarchy = List[Tuple[str, int]]


@dataclass(slots=True, kw_only=True)
class SchedulingParams:
    """Schedule parameters.

    Args:
        attention_dp_rank (int): The rank of target attention dp
        attention_dp_relax (bool): Whether to allow the request to be scheduled to other attention dp for better throughput
        agent_hierarchy (AgentHierarchy): Path of (agent_type, node_id) tuples
            identifying this request's position in an agent execution tree.
            Used by the batch scheduler for hierarchy-aware scheduling.
    """

    attention_dp_rank: Optional[int] = None
    attention_dp_relax: Optional[bool] = None
    agent_hierarchy: Optional[AgentHierarchy] = None
