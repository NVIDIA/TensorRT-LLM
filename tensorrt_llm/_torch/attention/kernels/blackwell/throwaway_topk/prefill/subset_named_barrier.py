import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import dsl_user_op
from .subset_barrier import subset_barrier

class SubsetNamedBarrier(pipeline.NamedBarrier):
    @dsl_user_op
    def arrive_and_wait(self,*,loc=None,ip=None):
        subset_barrier(barrier_id=self.barrier_id,number_of_threads=self.num_threads,loc=loc,ip=ip)

    @dsl_user_op
    def arrive(self,*,loc=None,ip=None):
        cute.arch.barrier_arrive(barrier_id=self.barrier_id,number_of_threads=self.num_threads,aligned=False,loc=loc,ip=ip)

    @dsl_user_op
    def wait_unaligned(self,*,loc=None,ip=None):
        subset_barrier(barrier_id=self.barrier_id,number_of_threads=self.num_threads,loc=loc,ip=ip)
