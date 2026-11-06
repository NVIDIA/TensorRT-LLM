"""Exact single-item or single-exclusion selection in a small crossing bin."""
import cutlass
import cutlass.cute as cute

@cute.jit
def select_extreme(ck, ci, staged, m, need, above):
    # One complete warp participates; arbitrary equal-score indices are valid.
    lane=cute.arch.lane_idx()
    keep_one=need==1
    selected=cutlass.Uint32(0xFFFFFFFF) if keep_one else cutlass.Uint32(0)
    for j in cutlass.range_constexpr(4):
        i=lane+32*j
        if i<m:
            key=cutlass.Uint32(ck[i])
            selected=min(selected,key) if keep_one else max(selected,key)
    if keep_one:selected=cute.arch.warp_redux_sync(selected,'umin')
    else:selected=cute.arch.warp_redux_sync(selected,'umax')
    ordinal=cutlass.Uint32(0xFFFFFFFF)
    for j in cutlass.range_constexpr(4):
        i=lane+32*j
        if i<m:
            if cutlass.Uint32(ck[i])==selected:ordinal=min(ordinal,cutlass.Uint32(i))
    ordinal=cute.arch.warp_redux_sync(ordinal,'umin')
    if keep_one:
        if lane==0:staged[above]=ci[cutlass.Int32(ordinal)]
    else:
        for j in cutlass.range_constexpr(4):
            i=lane+32*j
            if (i<m)&(i!=cutlass.Int32(ordinal)):
                staged[above+i-cutlass.Int32(i>cutlass.Int32(ordinal))]=ci[i]
