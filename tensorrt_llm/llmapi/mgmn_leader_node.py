# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
'''
This script is used to start the MPICommSession in the rank0 and wait for the
MPI Proxy process to connect and get the MPI task to run.
'''
import sys
import time
from typing import Literal

import click
import zmq

from tensorrt_llm._utils import global_mpi_rank, mpi_world_size
from tensorrt_llm.executor.ipc import ZeroMqQueue
from tensorrt_llm.executor.utils import (
    get_spawn_proxy_process_ipc_addr_env,
    get_spawn_proxy_process_ipc_hmac_key_env)
from tensorrt_llm.llmapi.mpi_session import RemoteMpiCommSessionServer
from tensorrt_llm.llmapi.utils import logger_debug
from tensorrt_llm.logger import logger


def launch_server_main(sub_comm=None):
    num_ranks = sub_comm.Get_size() if sub_comm is not None else mpi_world_size(
    )
    logger_debug(f"Starting MPI Comm Server with {num_ranks} workers\n",
                 "yellow")
    server = RemoteMpiCommSessionServer(
        comm=sub_comm,
        n_workers=num_ranks,
        addr=get_spawn_proxy_process_ipc_addr_env(),
        hmac_key=get_spawn_proxy_process_ipc_hmac_key_env(),
        is_comm=True)
    logger_debug(
        f"MPI Comm Server started at {get_spawn_proxy_process_ipc_addr_env()}")
    server.serve()

    logger_debug("RemoteMpiCommSessionServer stopped\n", "yellow")


# How long stop_server_main waits for the server to acknowledge one shutdown
# request before it resends on a new connection, and how many times it sends.
STOP_ACK_TIMEOUT_SECONDS = 1.0
STOP_MAX_ATTEMPTS = 300


def stop_server_main(ack_timeout: float = STOP_ACK_TIMEOUT_SECONDS,
                     max_attempts: int = STOP_MAX_ATTEMPTS) -> bool:
    """Ask the RemoteMpiCommSessionServer to shut down, until it acknowledges.

    The server's PAIR socket talks to one peer at a time and only accepts this
    client once it has noticed that the client that submitted the last task
    (the engine, which has just exited) is gone. A request sent before then is
    dropped, so each attempt uses a new connection and the request is resent
    until the server replies with ``RemoteMpiCommSessionServer.SHUTDOWN_ACK``.
    The server also sends worker results and errors to whichever client is
    attached, so an attempt reads past those messages until the
    acknowledgment arrives or ``ack_timeout`` runs out.

    Returns:
        True if the server acknowledged, False if every attempt timed out.
    """
    hmac_key = get_spawn_proxy_process_ipc_hmac_key_env()
    addr = get_spawn_proxy_process_ipc_addr_env()
    for attempt in range(1, max_attempts + 1):
        queue = ZeroMqQueue((addr, hmac_key),
                            use_hmac_encryption=True,
                            is_server=False,
                            socket_type=zmq.PAIR)
        try:
            logger_debug(
                f"RemoteMpiCommSessionClient [rank{global_mpi_rank()}] send shutdown signal to server (attempt {attempt})\n",
                "green")
            queue.put(None)  # ask RemoteMpiCommSessionServer to shutdown
            deadline = time.monotonic() + ack_timeout
            while ((remaining := deadline - time.monotonic()) > 0
                   and queue.poll(remaining)):
                if queue.get() == RemoteMpiCommSessionServer.SHUTDOWN_ACK:
                    return True
        except zmq.error.ZMQError as e:
            logger_debug(
                f"Error during RemoteMpiCommSessionClient shutdown: {e}\n",
                "red")
        finally:
            # Drop a request the server never accepted: with the default
            # linger, closing the queue would wait for it forever.
            if queue.socket is not None:
                queue.socket.setsockopt(zmq.LINGER, 0)
            queue.close()
    logger.warning(
        f"MPI Comm Server did not acknowledge the shutdown request after "
        f"{max_attempts} attempts")
    return False


@click.command()
@click.option("--action", type=click.Choice(["start", "stop"]), default="start")
def main(action: Literal["start", "stop"] = "start"):
    '''
    Arguments:
        action: The action to perform.
            start: Start the MPI Comm Server.
            stop: Stop the MPI Comm Server.
    '''
    if action == "start":
        launch_server_main()
    elif action == "stop":
        if not stop_server_main():
            sys.exit(1)
    else:
        raise ValueError(f"Invalid action: {action}")


if __name__ == '__main__':
    main()
