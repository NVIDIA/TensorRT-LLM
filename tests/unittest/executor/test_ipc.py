import asyncio
import time
from threading import Thread

import pytest
import zmq

from tensorrt_llm.executor.ipc import ZeroMqQueue


class TestIpcBasics:
    """Test basic synchronous IPC operations."""

    def test_pair_socket_with_hmac(self):
        """Test PAIR socket with HMAC encryption."""
        # Create server
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_server",
            use_hmac_encryption=True,
        )

        # Create client with server's address
        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="test_client",
            use_hmac_encryption=True,
        )

        try:
            # Test basic send/receive
            test_data = {"message": "hello", "value": 42}
            client.put(test_data)
            received = server.get()
            assert received == test_data

            # Test reverse direction
            response = {"status": "ok", "result": 100}
            server.put(response)
            received = client.get()
            assert received == response
        finally:
            client.close()
            server.close()

    def test_pair_socket_without_hmac(self):
        """Test PAIR socket without HMAC encryption."""
        # Create server without HMAC
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_server_no_hmac",
            use_hmac_encryption=False,
        )

        # Create client
        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="test_client_no_hmac",
            use_hmac_encryption=False,
        )

        try:
            # Test send/receive
            test_data = {"message": "hello without encryption", "numbers": [1, 2, 3]}
            client.put(test_data)
            received = server.get()
            assert received == test_data
        finally:
            client.close()
            server.close()

    def test_poll_timeout(self):
        """Test poll timeout behavior."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_poll_server",
            use_hmac_encryption=False,
        )

        try:
            # Poll should timeout when no data available
            start = time.time()
            result = server.poll(timeout=1)
            elapsed = time.time() - start
            assert result is False
            assert elapsed >= 1.0
            assert elapsed < 1.5  # Allow some margin
        finally:
            server.close()

    def test_poll_with_data(self):
        """Test poll returns True when data is available."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_poll_data_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="test_poll_data_client",
            use_hmac_encryption=False,
        )

        try:
            # Send data in background
            def send_data():
                time.sleep(0.1)  # Small delay
                client.put({"data": "test"})

            thread = Thread(target=send_data)
            thread.start()

            # Poll should return True
            result = server.poll(timeout=2)
            assert result is True

            # Verify data
            received = server.get()
            assert received == {"data": "test"}

            thread.join()
        finally:
            client.close()
            server.close()

    def test_router_socket_with_hmac(self):
        """Test ROUTER socket with HMAC encryption and identity tracking."""
        # Create ROUTER server
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=False,
            name="test_router_server",
            use_hmac_encryption=True,
        )

        # Create DEALER client
        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=False,
            name="test_dealer_client",
            use_hmac_encryption=True,
        )

        try:
            # Client sends request
            request = {"action": "process", "data": [1, 2, 3]}
            client.put(request)

            # Server receives and tracks identity
            received = server.get()
            assert received == request

            # Server sends response (using stored identity)
            response = {"status": "done", "result": 6}
            server.put(response)

            # Client receives response
            received = client.get()
            assert received == response
        finally:
            client.close()
            server.close()

    def test_dealer_notify_with_retry(self):
        """Test DEALER socket notify_with_retry mechanism."""
        # Create ROUTER server
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=False,
            name="test_router_ack_server",
            use_hmac_encryption=False,
        )

        # Create DEALER client
        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=False,
            name="test_dealer_ack_client",
            use_hmac_encryption=False,
        )

        try:
            # Server thread that acknowledges messages
            def server_ack():
                msg = server.get()
                assert msg == {"notify": "test"}
                # Send ACK
                server.put({"ack": True})

            thread = Thread(target=server_ack)
            thread.start()

            # Client sends with retry
            result = client.notify_with_retry({"notify": "test"}, max_retries=3, timeout=1)
            assert result is True

            thread.join()
        finally:
            client.close()
            server.close()

    def test_dealer_notify_with_retry_timeout(self):
        """Test DEALER socket notify_with_retry timeout behavior."""
        # Create ROUTER server (but don't respond)
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=False,
            name="test_router_no_ack_server",
            use_hmac_encryption=False,
        )

        # Create DEALER client
        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=False,
            name="test_dealer_no_ack_client",
            use_hmac_encryption=False,
        )

        try:
            # Client sends but server doesn't acknowledge
            result = client.notify_with_retry({"notify": "test"}, max_retries=2, timeout=0.5)
            assert result is False
        finally:
            client.close()
            server.close()

    def test_hmac_key_generation(self):
        """Test that server generates HMAC key when encryption is enabled."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_hmac_gen",
            use_hmac_encryption=True,
        )

        try:
            # Server should have generated an HMAC key
            assert server.hmac_key is not None
            assert len(server.hmac_key) == 32
        finally:
            server.close()

    def test_hmac_validation_error_client_no_key(self):
        """Test that client without HMAC key raises ValueError when encryption enabled."""
        with pytest.raises(ValueError, match="Client must receive HMAC key"):
            ZeroMqQueue(
                address=("tcp://127.0.0.1:5555", None),  # No HMAC key
                socket_type=zmq.PAIR,
                is_server=False,
                is_async=False,
                name="test_client_no_key",
                use_hmac_encryption=True,  # But encryption enabled
            )

    def test_hmac_validation_error_key_when_disabled(self):
        """Test that providing HMAC key when encryption disabled raises ValueError."""
        with pytest.raises(ValueError, match="should not receive HMAC key"):
            ZeroMqQueue(
                address=("tcp://127.0.0.1:5555", b"some_key"),  # Has key
                socket_type=zmq.PAIR,
                is_server=False,
                is_async=False,
                name="test_client_key_disabled",
                use_hmac_encryption=False,  # But encryption disabled
            )

    def test_put_noblock_retry(self):
        """Test put_noblock with retry mechanism."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="test_noblock_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="test_noblock_client",
            use_hmac_encryption=False,
        )

        try:
            # Send with put_noblock
            test_data = {"nonblocking": True, "value": 123}
            client.put_noblock(test_data, retry=3, wait_time=0.001)

            # Should be able to receive
            received = server.get()
            assert received == test_data
        finally:
            client.close()
            server.close()


class TestIpcAsyncBasics:
    """Test asynchronous IPC operations."""

    @pytest.mark.asyncio
    async def test_async_pair_with_hmac(self):
        """Test async PAIR socket with HMAC encryption."""
        # Create async server
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="async_server",
            use_hmac_encryption=True,
        )

        # Create async client
        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=True,
            name="async_client",
            use_hmac_encryption=True,
        )

        try:
            # Test async send/receive
            test_data = {"async": True, "value": 999}
            await client.put_async(test_data)
            received = await server.get_async()
            assert received == test_data

            # Test reverse direction
            response = {"status": "async_ok"}
            await server.put_async(response)
            received = await client.get_async()
            assert received == response
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_async_pair_without_hmac(self):
        """Test async PAIR socket without HMAC encryption."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="async_server_no_hmac",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=True,
            name="async_client_no_hmac",
            use_hmac_encryption=False,
        )

        try:
            # Test async operations
            test_data = {"no_encryption": True, "items": [1, 2, 3, 4, 5]}
            await client.put_async(test_data)
            received = await server.get_async()
            assert received == test_data
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_async_router_with_identity(self):
        """Test async ROUTER socket with identity handling."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=True,
            name="async_router_server",
            use_hmac_encryption=True,
        )

        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=True,
            name="async_dealer_client",
            use_hmac_encryption=True,
        )

        try:
            # Client sends async request
            request = {"async_request": "process"}
            await client.put_async(request)

            # Server receives with identity
            received = await server.get_async()
            assert received == request

            # Server replies
            response = {"async_response": "completed"}
            await server.put_async(response)

            # Client receives
            received = await client.get_async()
            assert received == response
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_get_async_noblock_timeout(self):
        """Test get_async_noblock timeout expiration."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="async_timeout_server",
            use_hmac_encryption=False,
        )

        try:
            # Should timeout when no data available
            with pytest.raises(asyncio.TimeoutError):
                await server.get_async_noblock(timeout=0.5)
        finally:
            server.close()

    @pytest.mark.asyncio
    async def test_get_async_noblock_success(self):
        """Test get_async_noblock successful receive before timeout."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="async_noblock_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=True,
            name="async_noblock_client",
            use_hmac_encryption=False,
        )

        try:
            # Send data in background
            async def send_delayed():
                await asyncio.sleep(0.1)
                await client.put_async({"delayed": True})

            send_task = asyncio.create_task(send_delayed())

            # Should receive before timeout
            received = await server.get_async_noblock(timeout=2.0)
            assert received == {"delayed": True}

            await send_task
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_put_async_noblock(self):
        """Test put_async_noblock with NOBLOCK flag."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="async_put_noblock_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=True,
            name="async_put_noblock_client",
            use_hmac_encryption=False,
        )

        try:
            # Send with noblock
            test_data = {"noblock_async": True}
            await client.put_async_noblock(test_data)

            # Should be able to receive
            received = await server.get_async()
            assert received == test_data
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_async_router_without_hmac(self):
        """Test async ROUTER socket without HMAC encryption."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=True,
            name="async_router_server_no_hmac",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=True,
            name="async_dealer_client_no_hmac",
            use_hmac_encryption=False,
        )

        try:
            # Client sends async request
            request = {"async_request": "process_no_hmac"}
            await client.put_async(request)

            # Server receives with identity
            received = await server.get_async()
            assert received == request

            # Server replies
            response = {"async_response": "completed_no_hmac"}
            await server.put_async(response)

            # Client receives
            received = await client.get_async()
            assert received == response
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_async_router_get_noblock(self):
        """Test get_async_noblock on ROUTER socket (handling multipart)."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=True,
            name="async_router_noblock_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=True,
            name="async_dealer_noblock_client",
            use_hmac_encryption=False,
        )

        try:
            # Client sends async request
            request = {"noblock_request": "test"}

            # Send with small delay to ensure we test the polling/waiting
            async def send_delayed():
                await asyncio.sleep(0.1)
                await client.put_async(request)

            send_task = asyncio.create_task(send_delayed())

            # Server receives using get_async_noblock
            # This exercises the ROUTER specific recv_multipart path
            received = await server.get_async_noblock(timeout=2.0)
            assert received == request

            # Ensure identity was captured so we can reply
            assert server._last_identity is not None

            # Server replies
            response = {"noblock_response": "done"}
            await server.put_async(response)

            # Client receives
            received = await client.get_async()
            assert received == response

            await send_task
        finally:
            client.close()
            server.close()


class TestIpcPressureTest:
    """Test performance and load handling."""

    def test_high_frequency_small_messages(self):
        """Test sending many small messages rapidly."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="pressure_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="pressure_client",
            use_hmac_encryption=False,
        )

        num_messages = 10000

        try:
            # Send many small messages
            def sender():
                for i in range(num_messages):
                    client.put({"id": i, "data": f"msg_{i}"})

            # Receive in parallel
            def receiver():
                received_count = 0
                for i in range(num_messages):
                    msg = server.get()
                    assert msg["id"] == i
                    assert msg["data"] == f"msg_{i}"
                    received_count += 1
                return received_count

            send_thread = Thread(target=sender)
            start_time = time.time()

            send_thread.start()
            count = receiver()
            send_thread.join()

            elapsed = time.time() - start_time

            # Verify all messages received
            assert count == num_messages
            print(
                f"\nHigh frequency test: {num_messages} messages in {elapsed:.2f}s "
                f"({num_messages / elapsed:.0f} msg/s)"
            )
        finally:
            client.close()
            server.close()

    def test_large_message_size(self):
        """Test sending large messages with HMAC encryption."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=False,
            name="large_msg_server",
            use_hmac_encryption=True,
        )

        client = ZeroMqQueue(
            address=server.address,
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=False,
            name="large_msg_client",
            use_hmac_encryption=True,
        )

        num_messages = 100
        message_size = 1024 * 1024  # 1 MB

        try:
            start_time = time.time()

            for i in range(num_messages):
                # Create large message (1 MB of data)
                large_data = {"id": i, "payload": "x" * message_size}
                client.put(large_data)

                received = server.get()
                assert received["id"] == i
                assert len(received["payload"]) == message_size

            elapsed = time.time() - start_time
            total_mb = (num_messages * message_size) / (1024 * 1024)

            print(
                f"\nLarge message test: {num_messages} x 1MB messages in {elapsed:.2f}s "
                f"({total_mb / elapsed:.1f} MB/s)"
            )
        finally:
            client.close()
            server.close()

    @pytest.mark.asyncio
    async def test_concurrent_async_access(self):
        """Test multiple async coroutines sending/receiving simultaneously."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.PAIR,
            is_server=True,
            is_async=True,
            name="concurrent_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.PAIR,
            is_server=False,
            is_async=True,
            name="concurrent_client",
            use_hmac_encryption=False,
        )

        num_messages = 1000

        try:
            # Sender coroutine
            async def sender():
                for i in range(num_messages):
                    await client.put_async({"id": i, "data": f"concurrent_{i}"})
                    if i % 100 == 0:
                        await asyncio.sleep(0.001)  # Small yield

            # Receiver coroutine
            async def receiver():
                received_ids = set()
                for _ in range(num_messages):
                    msg = await server.get_async()
                    received_ids.add(msg["id"])
                return received_ids

            # Run concurrently
            start_time = time.time()
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            received_ids = await receiver_task
            await sender_task
            elapsed = time.time() - start_time

            # Verify all messages received
            assert len(received_ids) == num_messages
            assert received_ids == set(range(num_messages))

            print(f"\nConcurrent async test: {num_messages} messages in {elapsed:.2f}s")
        finally:
            client.close()
            server.close()

    def test_router_socket_multiple_requests(self):
        """Test ROUTER socket handling multiple sequential requests."""
        server = ZeroMqQueue(
            address=None,
            socket_type=zmq.ROUTER,
            is_server=True,
            is_async=False,
            name="router_load_server",
            use_hmac_encryption=False,
        )

        client = ZeroMqQueue(
            address=(server.address[0], None),
            socket_type=zmq.DEALER,
            is_server=False,
            is_async=False,
            name="dealer_load_client",
            use_hmac_encryption=False,
        )

        num_requests = 1000

        try:
            start_time = time.time()

            for i in range(num_requests):
                # Client sends request
                client.put({"request_id": i, "action": "process"})

                # Server receives
                request = server.get()
                assert request["request_id"] == i

                # Server responds
                server.put({"request_id": i, "result": i * 2})

                # Client receives response
                response = client.get()
                assert response["request_id"] == i
                assert response["result"] == i * 2

            elapsed = time.time() - start_time

            print(
                f"\nROUTER socket test: {num_requests} round-trips in {elapsed:.2f}s "
                f"({num_requests / elapsed:.0f} req/s)"
            )
        finally:
            client.close()
            server.close()
