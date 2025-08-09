from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, kw_only=True)
class SchedulingParams:
    """Schedule parameters.

    Args:
        attention_dp_rank (int): The rank of target attention dp
        attention_dp_relax (bool): Whether to allow the request to be scheduled to other attention dp for better throughput
    """

    attention_dp_rank: Optional[int] = None
    attention_dp_relax: Optional[bool] = None
