#!/usr/bin/env python
from mpi4py.futures import MPICommExecutor

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.hlapi.mpi_session import MpiCommSession, MPINodeState


class MpiTask:

    def __init__(self):
        self.executor = MpiCommSession(n_workers=4)

    @staticmethod
    def task():
        if MPINodeState.state is None:
            MPINodeState.state = 0

        MPINodeState.state += 1
        print(f"rank: {mpi_rank()}, state: {MPINodeState.state}")

        return (mpi_rank(), MPINodeState.state)

    def run(self):
        results = self.executor.submit_sync(MpiTask.task)
        results = sorted(results, key=lambda x: x[0])
        assert results == [(i, 1) for i in range(mpi_world_size())], results


def main():
    # The root node
    if mpi_rank() == 0:
        the_task = MpiTask()
        the_task.run()

    else:  # The worker node
        with MPICommExecutor(mpi_comm()) as executor:
            pass


if __name__ == '__main__':
    main()
