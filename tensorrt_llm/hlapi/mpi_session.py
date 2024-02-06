import pickle  # nosec B403
import socket
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, List, Optional

from mpi4py.futures import MPIPoolExecutor


class NodeSession:
    ''' NodeSession Act as a central global state shares between tasks on MPI node.

    An example:
        def task():
            if NodeSession.state is None:
                NodeSession.state = 0
            NodeSession.state += 1
            return NodeSession.state

        n_workers = 4
        with MPIPoolExecutor(max_workers=n_workers) as executor:
            for i in range(2):
                futures = [executor.submit(task) for i in range(n_workers)]

        This should produce the following output:
        - [1, 1, 1, 1]
        - [2, 2, 2, 2]
    '''

    state = None

    @staticmethod
    def is_initialized() -> bool:
        return NodeSession.state is not None


class MpiSession:

    def __init__(self,
                 n_workers: int,
                 async_callback: Callable[[Any], None] = None):
        self.n_workers = n_workers
        self.mpi_pool: Optional[MPIPoolExecutor] = None
        self.async_callback = async_callback
        self._start_mpi_pool()

        if self.async_callback:
            self._socket_listener = SocketListener(callback=async_callback)

    def submit(self, task: (...), *args) -> List[Future]:
        return [
            self.mpi_pool.submit(task, *args) for i in range(self.n_workers)
        ]

    def submit_sync(self, task: (...), *args) -> List[Any]:
        futures = [
            self.mpi_pool.submit(task, *args) for i in range(self.n_workers)
        ]
        return [future.result() for future in futures]

    def shutdown(self):
        if self.mpi_pool is not None:
            self.mpi_pool.shutdown()
            self.mpi_pool = None

        if self.async_callback is not None and self._socket_listener is not None:
            self._socket_listener.shutdown()
            self._socket_listener = None

    def _start(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers)

    @property
    def async_enabled(self) -> bool:
        return hasattr(self, '_socket_listener')

    def get_socket_client(self) -> "SocketClient":
        return self._socket_listener.get_client()

    def _start_mpi_pool(self):
        assert not self.mpi_pool, 'MPI session already started'

        self.mpi_pool = MPIPoolExecutor(max_workers=self.n_workers)

    def __del__(self):
        self.shutdown()

    def __reduce__(self):
        raise TypeError('cannot pickle MPI session')


class SocketClient:

    def __init__(self, port):
        self.port = port

    def send(self, data: Any):
        # TODO[chunweiy]: reuse socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SocketListener.IP, self.port))
        client_socket.send(pickle.dumps(data))
        client_socket.close()


class SocketListener:
    IP = 'localhost'

    def __init__(self,
                 callback: Optional[Callable[[Any], Any]],
                 buf_size: int = 4096):
        self.buf_size = buf_size
        self.callback = callback
        self.port = -1
        self.server_socket = None

        self._start_service()

    def _start_service(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = find_free_port()
        self.server_socket.bind((SocketListener.IP, self.port))

        def loop():
            self.server_socket.listen(5)
            try:
                while True:
                    client_socket, address = self.server_socket.accept()
                    received_data = client_socket.recv(self.buf_size)
                    real_data = pickle.loads(received_data)  # nosec B301
                    if real_data is None:
                        # get the quit signal
                        break

                    self.callback(real_data)

            finally:
                self.server_socket.close()

        self.thread = threading.Thread(target=loop)
        self.thread.start()

    def get_client(self) -> SocketClient:
        return SocketClient(self.port)

    def shutdown(self):
        if self.server_socket is not None:
            client = self.get_client()
            client.send(None)
            time.sleep(0.1)
            self.server_socket = None

    def __del__(self):
        self.shutdown()

        self.thread.join()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
