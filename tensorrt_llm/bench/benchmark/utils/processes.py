import os
import tempfile
from contextlib import contextmanager
from multiprocessing import Event, Process
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Optional, Union

from zmq import PULL, Context

from tensorrt_llm import logger


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
        self._socket_path = Path(
            tempfile.mkstemp()[1]) if log_path is not None else None

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
            stop = Event()
            process = Process(name="IterationWriter",
                              target=self.run,
                              args=(self.full_address, self.log_path, stop))
            process.start()
            try:
                yield
            finally:
                stop.set()
                process.join()

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
            # Create a ZeroMQ context and socket for inter-process communication
            logger.debug(f"Iteration logging: Binding to {address}...")
            context = Context(io_threads=1)
            socket = context.socket(PULL)
            socket.bind(address)

            # Open the log file for writing and start listening for messages
            logger.debug(
                f"Iteration logging: Listening for messages on {address}...")
            with open(log_path, "w") as f:
                logger.info(f"Iteration logging: Opened log file {log_path}...")
                # Receive the first message from the socket
                message = socket.recv_json()
                logger.debug(f"Iteration logging: Received initial message")
                # Continue receiving messages until the stop event is set or an
                # "end" message is received
                while not stop_event.is_set() and "end" not in message:
                    f.write(f"{message}\n")
                    message = socket.recv_json()
                logger.debug(f"Iteration logging: Received end message")
        except KeyboardInterrupt:
            # Handle keyboard interrupt by continuing to receive
            # messages until "None" is received. LlmManager will
            # send "None" when it is finished.
            logger.info("Keyboard interrupt, exiting iteration logging...")
            while message != b"None":
                message = socket.recv_json()
        finally:
            # Finalize the logging process by closing the socket and terminating
            # the context
            logger.info("Finalizing iteration logging...")
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()
        logger.debug("Iteration logging exiting.")
