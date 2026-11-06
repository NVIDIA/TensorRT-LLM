"""Public-shape K1 margin variant; failure detection and exact K4 unchanged."""
import cutlass
import cutlass.cute as cute
@cute.jit
def shape_margin_rank(width,samples,k:cutlass.Constexpr,sigma:cutlass.Constexpr):
    mu=cutlass.Float32(k*samples)/cutlass.Float32(max(1,width))
    rank=cutlass.Int32(mu+cutlass.Float32(sigma)*cute.math.sqrt(mu))+1
    return min(samples,max(cutlass.Int32(1),rank))
