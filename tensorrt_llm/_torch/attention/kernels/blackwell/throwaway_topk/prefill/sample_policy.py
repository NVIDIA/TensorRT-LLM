import cutlass,cutlass.cute as cute
@cute.jit
def proposal_rank(width,samples,k:cutlass.Constexpr,sigma:cutlass.Constexpr):
 # A larger descending sample rank lowers the threshold. This is a generic
 # safety margin, not a probability guarantee for correlated stratified samples.
 # Keep exact recovery for rows whose retained candidates still fall below K.
 mu=cutlass.Float32(k*samples)/cutlass.Float32(max(1,width))
 margin=cutlass.Float32(max(sigma,4.5))
 rank=cutlass.Int32(mu+margin*cute.math.sqrt(mu))+1
 return min(samples,max(cutlass.Int32(1),rank))
