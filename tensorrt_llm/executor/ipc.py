import io
import time
import traceback
from multiprocessing.shared_memory import SharedMemory
from queue import Queue
from typing import Any, Optional

import torch
import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from ..llmapi.utils import (ManagedThread, enable_llm_debug, nvtx_mark,
                            nvtx_range, print_colored, print_colored_debug)
from .utils import ExecutorResponse, ExecutorResponseTensors


class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
    }

    def __init__(self,
                 address: Optional[str] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None):
        '''
        Parameters:
            address (Tuple[str, str], optional): The address (tcp-ip_port, authkey) for the IPC. Defaults to None.
            is_server (bool): Whether the current process is the server or the client.
        '''

        self.socket_type = socket_type
        self.address = address or "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket_type = socket_type

        self.socket = self.context.socket(socket_type)

        if (socket_type == zmq.PAIR
                and self.is_server) or socket_type == zmq.PULL:
            self.socket.bind(
                self.address
            )  # Binds to the address and occupy a port immediately
            self.address = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            print_colored_debug(
                f"Server [{name}] bound to {self.address} in {self.socket_type_str[socket_type]}\n",
                "green")

    def setup_lazily(self):
        if self._setup_done:
            return
        self._setup_done = True

        if not self.is_server:
            print_colored_debug(
                f"Client [{self.name}] connecting to {self.address} in {self.socket_type_str[self.socket_type]}\n",
                "green")
            self.socket.connect(self.address)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def put(self, obj: Any):
        self.setup_lazily()

        if isinstance(obj, ExecutorResponse):
            tensors = self._store_tensors_in_shmm(obj.tensors)
            obj = ExecutorResponse(client_id=obj.client_id,
                                   sequence_index=obj.sequence_index,
                                   tensors=tensors,
                                   finish_reasons=obj.finish_reasons,
                                   is_final=obj.is_final,
                                   error=obj.error,
                                   timestamp=obj.timestamp)

        with nvtx_range("send", color="blue", category="IPC"):
            self.socket.send_pyobj(obj)

    async def put_async(self, obj: Any):
        self.setup_lazily()
        if isinstance(obj, ExecutorResponse):
            tensors = self._store_tensors_in_shmm(obj.tensors)
            obj = ExecutorResponse(client_id=obj.client_id,
                                   tensors=tensors,
                                   finish_reasons=obj.finish_reasons,
                                   is_final=obj.is_final,
                                   error=obj.error,
                                   timestamp=obj.timestamp)

        try:
            await self.socket.send_pyobj(obj)
        except TypeError as e:
            logger.error(f"Cannot pickle {obj}")
            raise e
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

        nvtx_mark("ipc.send", color="blue", category="IPC")

    def get(self) -> Any:
        self.setup_lazily()

        obj = self.socket.recv_pyobj()
        nvtx_mark("ipc.get", color="orange", category="IPC")

        if isinstance(obj, ExecutorResponse):
            tensors = self._load_tensors_from_shmm(obj.tensors)
            obj = ExecutorResponse(client_id=obj.client_id,
                                   tensors=tensors,
                                   finish_reasons=obj.finish_reasons,
                                   is_final=obj.is_final,
                                   error=obj.error,
                                   timestamp=obj.timestamp)
        return obj

    async def get_async(self) -> Any:
        self.setup_lazily()

        obj = await self.socket.recv_pyobj()
        nvtx_mark("ipc.get", color="orange", category="IPC")

        if isinstance(obj, ExecutorResponse):
            tensors = self._load_tensors_from_shmm(obj.tensors)
            obj = ExecutorResponse(client_id=obj.client_id,
                                   tensors=tensors,
                                   sequence_index=obj.sequence_index,
                                   finish_reasons=obj.finish_reasons,
                                   is_final=obj.is_final,
                                   error=obj.error,
                                   timestamp=obj.timestamp)
        return obj

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _store_tensors_in_shmm(
        self, tensors: Optional["ExecutorResponseTensors"]
    ) -> Optional["ExecutorResponseTensors"]:
        if tensors is None:
            return tensors

        # The tensors are huge and cannot be transferred through socket directly. We need to store them in shared memory,
        # and replace the tensors with the shared memory path.
        def store_tensor(tensor: Optional[torch.Tensor]) -> Optional[str]:
            if tensor is None:
                return None
            # NOTE: We create random shmm here rather than two specific shmm for context and generation logit, since the
            # shmm may not be read timely by the IpcQueue.get() in the other side, so there might be multiple alive shmm
            # for logits.
            # A known issue: the shmm instance may leak if the IpcQueue.get() thread is stopped before the IpcQueue.put()
            # thread. This is not a big issue since the shmm will be automatically cleaned up when the process exits.
            shm = SharedMemory(create=True, size=tensor.nbytes + 2048)
            torch.save(tensor, shm._mmap)
            shm.close()
            return shm.name

        return ExecutorResponseTensors(
            output_token_ids=tensors.output_token_ids,
            context_logits=store_tensor(tensors.context_logits),
            generation_logits=store_tensor(tensors.generation_logits),
            log_probs=tensors.log_probs,
            cum_log_probs=tensors.cum_log_probs,
        )

    def _load_tensors_from_shmm(
        self, tensors: Optional["ExecutorResponseTensors"]
    ) -> Optional["ExecutorResponseTensors"]:
        if tensors is None:
            return tensors

        def load_tensor(tensor: Optional[str]) -> Optional[torch.Tensor]:
            if tensor is None or isinstance(tensor, torch.Tensor):
                return tensor

            shm = SharedMemory(name=tensor, create=False)
            tensor = torch.load(io.BytesIO(shm.buf))
            shm.close()
            shm.unlink()
            return tensor

        return ExecutorResponseTensors(
            output_token_ids=tensors.output_token_ids,
            context_logits=load_tensor(tensors.context_logits),
            generation_logits=load_tensor(tensors.generation_logits),
            log_probs=tensors.log_probs,
            cum_log_probs=tensors.cum_log_probs,
        )

    def __del__(self):
        self.close()


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[str] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(self._prepare_message(obj))
        else:
            batch = obj if isinstance(obj, list) else [obj]
            batch = [self._prepare_message(x) for x in batch]
            self.queue.put(batch)

    def get(self) -> Any:
        obj = self.queue.get()
        if isinstance(obj, list):
            return [self._process_message(o) for o in obj]
        return self._process_message(obj)

    def _prepare_message(self, obj: Any) -> Any:
        if isinstance(obj, ExecutorResponse):
            tensors = self.queue._store_tensors_in_shmm(obj.tensors)
            return ExecutorResponse(client_id=obj.client_id,
                                    tensors=tensors,
                                    finish_reasons=obj.finish_reasons,
                                    is_final=obj.is_final,
                                    sequence_index=obj.sequence_index,
                                    error=obj.error)
        return obj

    def _process_message(self, obj: Any) -> Any:
        if isinstance(obj, ExecutorResponse):
            tensors = self.queue._load_tensors_from_shmm(obj.tensors)
            return ExecutorResponse(client_id=obj.client_id,
                                    tensors=tensors,
                                    finish_reasons=obj.finish_reasons,
                                    is_final=obj.is_final,
                                    sequence_index=obj.sequence_index,
                                    error=obj.error)
        return obj

    @property
    def address(self) -> str:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
