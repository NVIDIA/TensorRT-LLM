# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import time
from contextlib import contextmanager
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Optional, Union

from zmq import POLLIN, PULL, Context

from tensorrt_llm import logger

_ITERATION_WRITER_JOIN_TIMEOUT_SEC = 5.0
_ITERATION_WRITER_POLL_MS = 100
# Allow queued tail messages to bridge transient empty polls after shutdown.
_ITERATION_WRITER_IDLE_TIMEOUT_SEC = 1.0


# The IterationWriter class implements a multi-process logging system that captures and writes
# iteration data to a specified file using ZeroMQ (ZMQ) for inter-process communication.
# It uses a producer-consumer pattern where the main process produces messages and a separate
# logging process consumes and writes them to a file.
class IterationWriter:
    """Manages the logging of iteration data to a specified file using inter-process communication.

    This class sets up a separate process for logging data to avoid I/O operations blocking the
    main process. It uses ZeroMQ's PULL socket pattern for reliable message passing between processes.

    Attributes:
        address (str): The network address for ZMQ inter-process communication (e.g., "localhost").
        port (int): The network port for ZMQ communication.
        log_path (Optional[Path]): The filesystem path where iteration data will be logged.
                                 If None, logging is disabled.

    Usage:
        writer = IterationWriter(Path("iterations.log"))
        with writer.capture():
            # Any iteration data sent during this context will be logged
            # Send data using ZMQ PUSH socket to writer.full_address
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        """Initialize the IterationWriter with network communication parameters.

        Sets up the basic configuration for the logging system. The actual logging process
        is not started until the capture() context manager is used.

        Args:
            address (str): The network address for ZMQ communication (e.g., "localhost").
            port (int): The network port number for ZMQ communication.
            log_path (Optional[Path]): Path where iteration data will be logged. If None,
                                     logging is disabled and capture() will be a no-op.
        """
        self.log_path = log_path
        self._socket_path = None
        if log_path is not None:
            fd, socket_path = tempfile.mkstemp()
            os.close(fd)
            self._socket_path = Path(socket_path)

    @property
    def full_address(self) -> Union[str, None]:
        """Construct the complete ZMQ IPC address string.

        Combines the address and port into a ZMQ-compatible IPC URL format.
        This address is used by both the logging process (PULL socket) and
        any processes that want to send data to be logged (PUSH socket).

        Returns:
            Union[str, None]: A ZMQ IPC URL (e.g., "ipc://localhost:5555") if log_path
                            is provided, otherwise None to indicate logging is disabled.
        """
        if self._socket_path is not None:
            return f"ipc://{self._socket_path}"
        else:
            return None

    @contextmanager
    def capture(self) -> contextmanager:
        """Create a context for capturing and logging iteration data.

        This context manager handles the lifecycle of the logging process:
        1. If logging is enabled (log_path is set):
           - Creates a new process for handling log writes
           - Sets up an event for coordinating process shutdown
           - Starts the logging process
        2. If logging is disabled:
           - Acts as a no-op context manager
        3. On context exit:
           - Signals the logging process to stop
           - Waits for the process to finish
        Yields:
            None: The context manager doesn't provide any values to the caller.

        Example:
            writer = IterationWriter(log_path=Path("log.txt"))
            with writer.capture():
                # Send data to writer.full_address using ZMQ PUSH socket
                # Data will be logged in a separate process
        """
        if self._socket_path is None:
            logger.info("No log path provided, skipping logging.")
            yield
        else:
            logger.info(f"Logging iterations to {self.log_path}...")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Surface file errors in the parent before starting the benchmark.
            with self.log_path.open("a"):
                pass
            stop = Event()
            process = Process(name="IterationWriter",
                              target=self.run,
                              args=(self.full_address, self.log_path, stop))
            process.start()
            try:
                yield
            finally:
                stop.set()
                process.join(timeout=_ITERATION_WRITER_JOIN_TIMEOUT_SEC)
                if process.is_alive():
                    logger.warning("Iteration writer timed out; the iteration "
                                   "log may be incomplete.")
                    process.kill()
                    process.join()
                elif process.exitcode != 0:
                    logger.warning("Iteration writer failed; the iteration "
                                   "log may be incomplete.")

    def __del__(self) -> None:
        if self._socket_path is not None:
            os.remove(f"{self._socket_path}")

    @staticmethod
    def run(address: str, log_path: Path, stop_event: MpEvent) -> None:
        """Execute the logging process that receives and writes iteration data.

        This method runs in a separate process and:
        1. Sets up a ZMQ PULL socket to receive messages
        2. Opens the log file for writing
        3. Continuously receives messages and writes them to the log file
        4. Handles graceful shutdown on keyboard interrupt
        5. Cleans up ZMQ resources on exit

        The process continues running until either:
        - The stop_event is set (normal shutdown)
        - An "end" message is received
        - A KeyboardInterrupt occurs

        Args:
            address (str): The ZMQ IPC address to bind to for receiving messages.
            log_path (Path): The file path where received messages will be written.
            stop_event (MpEvent): Multiprocessing event used to signal process shutdown.
        """
        context = None
        socket = None

        try:
            with open(log_path, "w") as f:
                logger.info(f"Iteration logging: Opened log file {log_path}...")
                context = Context(io_threads=1)
                socket = context.socket(PULL)
                socket.bind(address)
                drain_deadline = None
                while True:
                    if stop_event.is_set() and drain_deadline is None:
                        drain_deadline = (time.monotonic() +
                                          _ITERATION_WRITER_IDLE_TIMEOUT_SEC)
                    if not socket.poll(_ITERATION_WRITER_POLL_MS, POLLIN):
                        if (drain_deadline is not None
                                and time.monotonic() >= drain_deadline):
                            logger.warning(
                                "Iteration writer stopped without receiving the "
                                "end marker; the iteration log may be incomplete."
                            )
                            break
                        continue
                    message = socket.recv_json()
                    if "end" in message:
                        break
                    f.write(f"{message}\n")
                    if stop_event.is_set():
                        # Keep draining while queued tail messages are arriving.
                        drain_deadline = (time.monotonic() +
                                          _ITERATION_WRITER_IDLE_TIMEOUT_SEC)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting iteration logging...")
        finally:
            # Finalize the logging process by closing the socket and terminating
            # the context
            logger.info("Finalizing iteration logging...")
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()
        logger.debug("Iteration logging exiting.")
