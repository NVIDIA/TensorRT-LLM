"""SM89 kernel recipe."""

from .mainloop import SolAttnForwardSm89


def make_kernel(
    *,
    route_sum_order: int = 3,
    debug_route_trace: bool = False,
    debug_index_trace: bool = False,
    debug_route_group_limit: int = 0,
    debug_score_trace: bool = False,
    debug_probability_trace: bool = False,
):
    return SolAttnForwardSm89(
        route_sum_order=route_sum_order,
        debug_route_trace=debug_route_trace,
        debug_index_trace=debug_index_trace,
        debug_route_group_limit=debug_route_group_limit,
        debug_score_trace=debug_score_trace,
        debug_probability_trace=debug_probability_trace,
    )


__all__ = ["make_kernel"]
