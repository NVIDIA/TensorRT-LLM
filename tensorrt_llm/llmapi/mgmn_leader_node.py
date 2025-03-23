'''
This script is used to start the MPICommSession in the rank0 and wait for the
MPI Proxy process to connect and get the MPI task to run.
'''
from tensorrt_llm._utils import mpi_world_size
from tensorrt_llm.executor.utils import get_spawn_proxy_process_ipc_addr_env
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer
from tensorrt_llm.llmapi.utils import print_colored_debug


def launch_server_main(sub_comm=None):
    num_ranks = sub_comm.Get_size() if sub_comm is not None else mpi_world_size(
    )
    print_colored_debug(f"Starting MPI Comm Server with {num_ranks} workers\n",
                        "yellow")
    server = RemoteMpiCommSessionServer(
        comm=sub_comm,
        n_workers=num_ranks,
        addr=get_spawn_proxy_process_ipc_addr_env(),
        is_comm=True)
    print_colored_debug(
        f"MPI Comm Server started at {get_spawn_proxy_process_ipc_addr_env()}")
    server.serve()

    print_colored_debug("MPI Comm Server stopped")


if __name__ == '__main__':
    launch_server_main()
