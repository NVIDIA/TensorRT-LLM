import multiprocessing
import pickle
import time
from contextlib import contextmanager

import pytest
import zmq

from tensorrt_llm.executor.ipc import ZeroMqQueue


@contextmanager
def router_dealer_pair(use_hmac_encryption=False,
                       server_name="test_router",
                       client_name="test_dealer"):
    """Context manager to create and manage a ROUTER-DEALER queue pair."""
    server_queue = ZeroMqQueue(socket_type=zmq.ROUTER,
                               is_server=True,
                               name=server_name,
                               use_hmac_encryption=use_hmac_encryption)

    client_queue = ZeroMqQueue(address=server_queue.address,
                               socket_type=zmq.DEALER,
                               is_server=False,
                               name=client_name,
                               use_hmac_encryption=use_hmac_encryption)

    try:
        yield server_queue, client_queue
    finally:
        server_queue.close()
        client_queue.close()


def basic_communication_helper(server_queue, client_queue, test_message,
                               expected_reply):
    """Helper function to test basic bidirectional communication."""
    # Send message from client to server
    client_queue.put(test_message)

    # Server should receive the message
    assert server_queue.poll(2), "Server should receive message"
    received_message = server_queue.get()
    assert received_message == test_message

    # Send reply from server to client
    server_queue.put(expected_reply)

    # Client should receive the reply
    assert client_queue.poll(2), "Client should receive reply"
    received_reply = client_queue.get()
    assert received_reply == expected_reply

    return received_message, received_reply


@contextmanager
def multiprocess_runner():
    """Context manager to handle multiprocess execution and cleanup."""
    processes = []
    queues = []

    def add_process(target, args, daemon=True):
        proc = multiprocessing.Process(target=target, args=args, daemon=daemon)
        processes.append(proc)
        return proc

    def add_queue():
        queue = multiprocessing.Queue()
        queues.append(queue)
        return queue

    try:
        yield add_process, add_queue

        # Start all processes
        for proc in processes:
            proc.start()

        # Wait for all processes to complete
        for proc in processes:
            proc.join(timeout=10)

        # Check if any process is still alive and terminate if needed
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

        # Verify all processes finished successfully
        for i, proc in enumerate(processes):
            assert not proc.is_alive(), f"Process {i} did not finish in time"
            if proc.exitcode != 0:
                pytest.fail(
                    f"Process {i} failed with exit code: {proc.exitcode}")

    except Exception as e:
        # Emergency cleanup
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)
        raise e


def collect_process_results(result_queues, timeout=2):
    """Collect results from multiple process result queues."""
    results = []
    for queue in result_queues:
        try:
            result = queue.get(timeout=timeout)
            results.append(result)
        except Exception as e:
            results.append({"error": f"Failed to get result: {e}"})
    return results


def verify_worker_proxy_results(worker_result,
                                proxy_result,
                                expected_signal=b"READY"):
    """
    Universal verification function for worker-proxy communication results.
    Works identically for both encrypted and non-encrypted communication.

    Args:
        worker_result: Worker process result dictionary
        proxy_result: Proxy process result dictionary
        expected_signal: Expected signal in the message
    """
    # Basic result verification
    assert worker_result[
        "error"] is None, f"Worker error: {worker_result['error']}"
    assert worker_result["success"], "Worker should have succeeded"
    assert proxy_result[
        "error"] is None, f"Proxy error: {proxy_result['error']}"
    assert proxy_result[
        "message"] is not None, "Proxy should have received message"

    # Verify message content - same format regardless of encryption
    ready_signal, error_trace = proxy_result["message"]
    assert ready_signal == expected_signal
    assert error_trace is None


def run_worker_proxy_multiprocess_test(use_hmac_encryption=False,
                                       ready_signal=b"READY"):
    """
    Generic multiprocess test for worker-proxy communication.

    Args:
        use_hmac_encryption: Whether to use HMAC encryption
        ready_signal: Signal to send from worker to proxy
    """

    def proxy_process(address_q, result_q):
        """Generic proxy process."""
        try:
            proxy_queue = ZeroMqQueue(socket_type=zmq.ROUTER,
                                      is_server=True,
                                      name="proxy_multiproc",
                                      use_hmac_encryption=use_hmac_encryption)

            address_q.put(proxy_queue.address)

            try:
                if proxy_queue.poll(5):
                    message = proxy_queue.get()
                    result_q.put({"message": message, "error": None})
                    proxy_queue.put("ACK")
                else:
                    result_q.put({
                        "message": None,
                        "error": "Timeout waiting for worker message"
                    })
            finally:
                proxy_queue.close()
        except Exception as e:
            result_q.put({"message": None, "error": str(e)})

    def worker_process(address_q, result_q):
        """Generic worker process."""
        try:
            proxy_addr = address_q.get(timeout=5)
            worker_queue = ZeroMqQueue(socket_type=zmq.DEALER,
                                       is_server=False,
                                       address=proxy_addr,
                                       name="worker_multiproc",
                                       use_hmac_encryption=use_hmac_encryption)

            try:
                time.sleep(0.1)  # Simulate initialization time
                ready_message = (ready_signal, None)
                worker_queue.put(ready_message)

                if worker_queue.poll(3):
                    ack = worker_queue.get()
                    success = ack == "ACK"
                    result_q.put({
                        "success": success,
                        "error": None if success else f"Invalid ACK: {ack}",
                        "ack": ack if success else None
                    })
                else:
                    result_q.put({
                        "success": False,
                        "error": "Timeout waiting for ACK"
                    })
            finally:
                worker_queue.close()
        except Exception as e:
            result_q.put({"success": False, "error": str(e)})

    # Run the multiprocess test
    with multiprocess_runner() as (add_process, add_queue):
        address_queue = add_queue()
        proxy_result_queue = add_queue()
        worker_result_queue = add_queue()

        add_process(proxy_process, (address_queue, proxy_result_queue))
        add_process(worker_process, (address_queue, worker_result_queue))

    # Collect and verify results
    worker_result, proxy_result = collect_process_results(
        [worker_result_queue, proxy_result_queue])
    verify_worker_proxy_results(worker_result,
                                proxy_result,
                                expected_signal=ready_signal)

    return worker_result, proxy_result


class TestRouterDealerIPC:
    """Test suite for ROUTER/DEALER socket communication patterns."""

    @pytest.fixture
    def zmq_context(self):
        """Create a ZMQ context for testing."""
        context = zmq.Context()
        yield context
        context.term()

    @pytest.fixture
    def router_socket(self, zmq_context):
        """Create a ROUTER socket for testing."""
        socket = zmq_context.socket(zmq.ROUTER)
        socket.bind("tcp://127.0.0.1:*")
        endpoint = socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        yield socket, endpoint
        socket.close()

    @pytest.fixture
    def dealer_socket(self, zmq_context):
        """Create a DEALER socket for testing."""
        socket = zmq_context.socket(zmq.DEALER)
        yield socket
        socket.close()

    def test_basic_router_dealer_communication(self, router_socket,
                                               dealer_socket):
        """Test basic ROUTER-DEALER communication with multipart messages."""
        router, endpoint = router_socket
        dealer = dealer_socket

        # Connect dealer to router
        dealer.connect(endpoint)
        time.sleep(0.1)  # Allow connection to establish

        # Send message from dealer to router
        test_message = b"Hello Router"
        dealer.send(test_message)

        # Receive multipart message at router (identity + message)
        identity, message = router.recv_multipart()
        assert message == test_message
        assert len(identity) > 0  # Identity should be present

        # Send reply from router to dealer
        reply_message = b"Hello Dealer"
        router.send_multipart([identity, reply_message])

        # Receive reply at dealer
        received_reply = dealer.recv()
        assert received_reply == reply_message

    def test_multipart_message_handling(self, router_socket, dealer_socket):
        """Test multipart message handling with multiple frames."""
        router, endpoint = router_socket
        dealer = dealer_socket

        dealer.connect(endpoint)
        time.sleep(0.1)

        # Send multipart message from dealer
        message_parts = [b"frame1", b"frame2", b"frame3"]
        dealer.send_multipart(message_parts)

        # Receive at router (identity + all message frames)
        frames = router.recv_multipart()
        identity = frames[0]
        received_parts = frames[1:]

        assert received_parts == message_parts
        assert len(identity) > 0

        # Send multipart reply from router
        reply_parts = [b"reply1", b"reply2"]
        router.send_multipart([identity] + reply_parts)

        # Receive multipart reply at dealer
        received_reply = dealer.recv_multipart()
        assert received_reply == reply_parts

    def test_pickle_message_serialization(self, router_socket, dealer_socket):
        """Test sending pickled Python objects through ROUTER-DEALER."""
        router, endpoint = router_socket
        dealer = dealer_socket

        dealer.connect(endpoint)
        time.sleep(0.1)

        # Send pickled object from dealer
        test_obj = {"signal": "READY", "data": [1, 2, 3], "error": None}
        pickled_data = pickle.dumps(test_obj)
        dealer.send(pickled_data)

        # Receive at router and unpickle
        identity, message = router.recv_multipart()
        received_obj = pickle.loads(message)
        assert received_obj == test_obj

        # Send pickled reply from router
        reply_obj = {"status": "ACK", "timestamp": 1234567890}
        reply_data = pickle.dumps(reply_obj)
        router.send_multipart([identity, reply_data])

        # Receive and unpickle reply at dealer
        reply_raw = dealer.recv()
        received_reply = pickle.loads(reply_raw)
        assert received_reply == reply_obj

    @pytest.mark.parametrize("use_hmac", [False, True])
    def test_router_dealer_communication(self, use_hmac):
        """Test ROUTER-DEALER communication with and without HMAC encryption."""
        # Same message content regardless of encryption
        test_message = ("READY", "initialization_complete")
        reply_message = "ACK"

        with router_dealer_pair(use_hmac_encryption=use_hmac) as (server_queue,
                                                                  client_queue):
            basic_communication_helper(server_queue, client_queue, test_message,
                                       reply_message)

    @pytest.mark.parametrize("use_hmac", [False, True])
    def test_router_dealer_basic(self, use_hmac):
        """Test router_dealer with and without HMAC encryption."""
        # Same message content regardless of encryption
        ready_message = (b"READY", None)
        ack_message = "ACK"

        with router_dealer_pair(use_hmac_encryption=use_hmac) as (proxy_queue,
                                                                  worker_queue):
            received_message, received_ack = basic_communication_helper(
                proxy_queue, worker_queue, ready_message, ack_message)

            # Encryption is transparent - same verification for both
            assert received_ack == "ACK"

    @pytest.mark.parametrize("use_hmac", [False, True])
    def test_router_dealer_multiprocess(self, use_hmac):
        """Test router_dealer using separate processes, with and without HMAC encryption."""
        # Same signal regardless of encryption
        run_worker_proxy_multiprocess_test(use_hmac_encryption=use_hmac,
                                           ready_signal=b"READY")
