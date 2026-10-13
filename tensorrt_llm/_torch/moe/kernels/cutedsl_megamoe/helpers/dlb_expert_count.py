"""Resolve logical experts and helper capacity at the host construction boundary."""

from ..api import ImplDesc, ProblemDesc


def resolve_dlb_expert_counts(
    problem_desc: ProblemDesc, impl_desc: ImplDesc, *, world_size: int
) -> tuple[ProblemDesc, ImplDesc]:
    """Normalize the new logical-count API while retaining legacy physical counts."""
    if "total_helper_slots" not in impl_desc:
        return problem_desc, impl_desc
    if "helper_expert_count" in impl_desc:
        raise ValueError("Use total_helper_slots or legacy helper_expert_count, not both.")

    logical_experts = problem_desc["expert_count"]
    total_helpers = impl_desc["total_helper_slots"]
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive exact int.")
    if type(logical_experts) is not int or logical_experts <= 0:
        raise ValueError("Logical expert_count must be a positive exact int.")
    if type(total_helpers) is not int or total_helpers < 0:
        raise ValueError("total_helper_slots must be a non-negative exact int.")
    if logical_experts % world_size or total_helpers % world_size:
        raise ValueError("Logical expert_count and total_helper_slots must be divisible by world_size.")

    physical_problem = ProblemDesc({**problem_desc, "expert_count": logical_experts + total_helpers})
    physical_impl = dict(impl_desc)
    del physical_impl["total_helper_slots"]
    physical_impl["helper_expert_count"] = total_helpers // world_size
    return physical_problem, ImplDesc(physical_impl)
