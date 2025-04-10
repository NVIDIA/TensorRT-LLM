import io
import time
import traceback
import hmac
import os
import pickle
import hashlib
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
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 use_hmac_encryption: bool = True):
        '''
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address (tcp-ip_port, hmac_auth_key) for the IPC. Defaults to None. If hmac_auth_key is None and use_hmac_encryption is False, the queue will not use HMAC encryption.
            is_server (bool): Whether the current process is the server or the client.
            use_hmac_encryption (bool): Whether to use HMAC encryption for pickled data. Defaults to True.
        '''

        self.socket_type = socket_type
        self.address_endpoint = address[0] if address is not None else "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket = self.context.socket(socket_type)

        self.hmac_key = address[1] if address is not None else None
        self.use_hmac_encryption = use_hmac_encryption


        # Check HMAC key
        if self.use_hmac_encryption and self.is_server and self.hmac_key is not None:
            raise ValueError("Server should not receive HMAC key when encryption is enabled")
        elif self.use_hmac_encryption and not self.is_server and self.hmac_key is None:
            raise ValueError("Client must receive HMAC key when encryption is enabled") 
        elif not self.use_hmac_encryption and self.hmac_key is not None:
            raise ValueError("Server and client should not receive HMAC key when encryption is disabled")

        if (socket_type == zmq.PAIR
                and self.is_server) or socket_type == zmq.PULL:
            self.socket.bind(
                self.address_endpoint
            )  # Binds to the address and occupy a port immediately
            self.address_endpoint = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode()
            print_colored_debug(
                f"Server [{name}] bound to {self.address_endpoint} in {self.socket_type_str[socket_type]}\n",
                "green")
            
            if self.use_hmac_encryption:
                # Initialize HMAC key for pickle encryption
                logger.info(f"Generating a new HMAC key for server {self.name}")
                self.hmac_key = os.urandom(32)
            
            self.address = (self.address_endpoint, self.hmac_key)
        
        self.verbose = False

        if self.verbose: # for debugging
            logger.debug(f"In ZeroMqQueue init, HMAC key: {self.hmac_key}")
            logger.debug(f"In ZeroMqQueue init, self.name: {self.name}")

    def _verify_hmac(self, data: bytes, actual_hmac: bytes) -> bool:
        """Verify the HMAC of received pickle data."""
        expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).digest()
        if self.verbose: # for debugging
            logger.debug("in _verify_hmac")
            logger.debug(f"Data: {data}")
            logger.debug(f"HMAC key: {self.hmac_key}")
            logger.debug(f"Expected HMAC: {expected_hmac}")
            logger.debug(f"Actual HMAC: {actual_hmac}")
        return hmac.compare_digest(expected_hmac, actual_hmac)

    def _sign_data(self, data_before_encoding: bytes) -> bytes:
        """Generate HMAC for data."""
        if self.verbose: # for debugging
            logger.debug("in _sign_data")
            logger.debug(f"Signing data: {data_before_encoding}")
            logger.debug(f"HMAC key: {self.hmac_key}")
        hmac_signature = hmac.new(self.hmac_key, data_before_encoding, hashlib.sha256).digest()
        return data_before_encoding + hmac_signature

    def setup_lazily(self):
        if self._setup_done:
            return
        self._setup_done = True

        if not self.is_server:
            print_colored_debug(
                f"Client [{self.name}] connecting to {self.address_endpoint} in {self.socket_type_str[self.socket_type]}\n",
                "green")
            self.socket.connect(self.address_endpoint)

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
            obj = ExecutorResponse(
                client_id=obj.client_id,
                sequence_index=obj.sequence_index,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)

        with nvtx_range("send", color="blue", category="IPC"):
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = pickle.dumps(obj)
                signed_data = self._sign_data(data)
                self.socket.send(signed_data)
            else:
                # Send data without HMAC
                self.socket.send_pyobj(obj)

    async def put_async(self, obj: Any):
        self.setup_lazily()
        if isinstance(obj, ExecutorResponse):
            tensors = self._store_tensors_in_shmm(obj.tensors)
            obj = ExecutorResponse(
                client_id=obj.client_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)

        try:
            if self.use_hmac_encryption:
                # Send pickled data with HMAC appended
                data = pickle.dumps(obj)
                signed_data = self._sign_data(data)
                await self.socket.send(signed_data)
            else:
                # Send data without HMAC
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
        
        if self.use_hmac_encryption:
            # Receive signed data with HMAC
            signed_data = self.socket.recv()

            # Split data and HMAC
            data = signed_data[:-32]
            actual_hmac = signed_data[-32:]
            
            # Verify HMAC
            if not self._verify_hmac(data, actual_hmac):
                raise ValueError("HMAC verification failed")
                
            obj = pickle.loads(data)
        else:
            # Receive data without HMAC
            obj = self.socket.recv_pyobj()

        nvtx_mark("ipc.get", color="orange", category="IPC")

        if isinstance(obj, ExecutorResponse):
            tensors = self._load_tensors_from_shmm(obj.tensors)
            obj = ExecutorResponse(
                client_id=obj.client_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)
        return obj

    async def get_async(self) -> Any:
        self.setup_lazily()

        if self.use_hmac_encryption:
            # Receive signed data with HMAC
            signed_data = await self.socket.recv()

            # Split data and HMAC
            data = signed_data[:-32]
            actual_hmac = signed_data[-32:]
            
            # Verify HMAC
            if not self._verify_hmac(data, actual_hmac):
                raise ValueError("HMAC verification failed")
            
            obj = pickle.loads(data)
        else:
            # Receive data without HMAC
            obj = await self.socket.recv_pyobj()

        nvtx_mark("ipc.get", color="orange", category="IPC")

        if isinstance(obj, ExecutorResponse):
            tensors = self._load_tensors_from_shmm(obj.tensors)
            obj = ExecutorResponse(
                client_id=obj.client_id,
                tensors=tensors,
                sequence_index=obj.sequence_index,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)
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
            return ExecutorResponse(
                client_id=obj.client_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                sequence_index=obj.sequence_index,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)
        return obj

    def _process_message(self, obj: Any) -> Any:
        if isinstance(obj, ExecutorResponse):
            tensors = self.queue._load_tensors_from_shmm(obj.tensors)
            return ExecutorResponse(
                client_id=obj.client_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                sequence_index=obj.sequence_index,
                error=obj.error,
                timestamp=obj.timestamp,
                disaggregated_params=obj.disaggregated_params)
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
