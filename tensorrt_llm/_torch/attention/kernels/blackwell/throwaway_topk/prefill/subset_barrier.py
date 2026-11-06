"""Synchronize a subset of warp-specialized CTA participants at distinct PCs.

Use barrier.cta.sync; aligned bar.sync requires uniform participation."""
import cutlass
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm

@dsl_user_op
def subset_barrier(*,barrier_id,number_of_threads,loc=None,ip=None):
    bid=cutlass.Int32(barrier_id).ir_value(loc=loc,ip=ip)
    count=cutlass.Int32(number_of_threads).ir_value(loc=loc,ip=ip)
    nvvm.barrier_cta_sync(bid,thread_count=count,aligned=False,loc=loc,ip=ip)
