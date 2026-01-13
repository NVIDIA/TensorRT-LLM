import asyncio
import concurrent.futures
import threading
import time

import pytest

from tensorrt_llm.executor.rpc import (RPCCancelled, RPCClient, RPCError,
                                       RPCServer, RPCStreamingError, RPCTimeout)
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr


class RpcServerWrapper(RPCServer):
    """ A helper class to wrap the RPCServer and manage its lifecycle. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addr = get_unique_ipc_addr()

    def __enter__(self):
        self.bind(self.addr)
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class TestRpcBasics:
    """ Test the basic functionality of the RPC server and client. """

    def test_rpc_server_basics(self):

        class App:

            def hello(self):
                print("hello")

        with RpcServerWrapper(App()) as server:
            pass

    def test_remote_call_without_arg(self):

        class App:

            def hello(self):
                print("hello")
                return "world"

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                ret = client.hello().remote()  # sync call
                assert ret == "world"

    def test_remote_call_with_args(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                ret = client.hello("app", "Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_remote_call_with_kwargs(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                ret = client.hello(name="app", location="Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_remote_call_with_args_and_kwargs(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                ret = client.hello(name="app", location="Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_rpc_server_address(self):

        class App:
            pass

        with RpcServerWrapper(App()) as server:
            assert server.address == server.addr

    def test_rpc_with_error(self):

        class App:

            def hello(self):
                raise ValueError("hello")

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                with pytest.raises(RPCError):
                    client.hello().remote()

    def test_rpc_without_wait_response(self):

        class App:

            def __init__(self):
                self.task_submitted = False

            def send_task(self) -> None:
                # Just submit the task and return immediately
                # The result is not important
                self.task_submitted = True
                return None

            def get_task_submitted(self) -> bool:
                return self.task_submitted

        with RpcServerWrapper(App()) as server:
            with RPCClient(server.addr) as client:
                client.send_task().remote(need_response=False)
                time.sleep(
                    0.1
                )  # wait for some time to make sure the task is submitted
                assert client.get_task_submitted().remote()


class TestRpcCorrectness:
    """ Test the correctness of the RPC framework with various large tasks. """

    class App:

        def incremental_task(self, v: int):
            return v + 1

        async def incremental_task_async(self, v: int):
            return v + 1

        async def streaming_task(self, n: int):
            for i in range(n):
                yield i

    def test_incremental_task(self, num_tasks: int = 10000):
        with RpcServerWrapper(TestRpcCorrectness.App()) as server:
            with RPCClient(server.addr) as client:
                for i in range(num_tasks):  # a large number of tasks
                    result = client.incremental_task(i).remote()
                    if i % 1000 == 0:
                        print(f"incremental_task {i} done")
                    assert result == i + 1, f"result {result} != {i + 1}"

    def test_incremental_task_async(self, num_tasks: int = 10000):
        with RpcServerWrapper(TestRpcCorrectness.App()) as server:
            with RPCClient(server.addr) as client:

                async def test_incremental_task_async():
                    for i in range(num_tasks):  # a large number of tasks
                        result = await client.incremental_task_async(
                            i).remote_async()
                        if i % 1000 == 0:
                            print(f"incremental_task_async {i} done")
                        assert result == i + 1, f"result {result} != {i + 1}"

                asyncio.run(test_incremental_task_async())

    @pytest.mark.skip(reason="This test is flaky, need to fix it")
    def test_incremental_task_future(self):
        with RpcServerWrapper(TestRpcCorrectness.App()) as server:
            # Create client with more workers to handle concurrent futures
            with RPCClient(server.addr, num_workers=16) as client:
                # Process in smaller batches to avoid overwhelming the system
                batch_size = 50
                total_tasks = 1000  # Reduced from 10000 for stability

                for batch_start in range(0, total_tasks, batch_size):
                    batch_end = min(batch_start + batch_size, total_tasks)
                    futures = []

                    # Create futures for this batch
                    for i in range(batch_start, batch_end):
                        futures.append(
                            client.incremental_task(i).remote_future())

                    # Wait for all futures in this batch to complete
                    for idx, future in enumerate(futures):
                        no = batch_start + idx
                        if no % 100 == 0:
                            print(f"incremental_task_future {no} done")
                        assert future.result(
                        ) == no + 1, f"result {future.result()} != {no + 1}"

    def test_incremental_task_streaming(self):
        with RpcServerWrapper(TestRpcCorrectness.App(),
                              async_run_task=True) as server:

            with RPCClient(server.addr) as client:

                async def test_streaming_task():
                    results = []
                    no = 0
                    async for result in client.streaming_task(
                            10000).remote_streaming():
                        results.append(result)
                        if no % 1000 == 0:
                            print(f"streaming_task {no} done")
                        no += 1
                    assert results == [
                        i for i in range(10000)
                    ], f"results {results} != {[i for i in range(10000)]}"

                asyncio.run(test_streaming_task())

    def test_multi_client_to_single_server(self):
        """Test that multiple RPC clients can concurrently connect to a single RPC server and execute tasks."""

        class App:

            def echo(self, msg: str) -> str:
                return msg

        with RpcServerWrapper(App()) as server:
            # Create multiple clients
            num_clients = 10
            clients = [RPCClient(server.addr) for _ in range(num_clients)]

            try:
                # Perform requests from all clients
                for i, client in enumerate(clients):
                    msg = f"hello from client {i}"
                    ret = client.echo(msg).remote()
                    assert ret == msg, f"Client {i} failed: expected '{msg}', got '{ret}'"
            finally:
                # Clean up clients
                for client in clients:
                    client.close()


class TestRpcError:

    class CustomError(Exception):
        pass

    def test_task_error(self):
        """Test that server-side exceptions are properly wrapped in RPCError with details."""

        class App:

            def hello(self):
                raise ValueError("Test error message")

            def divide_by_zero(self):
                return 1 / 0

            def custom_exception(self):
                raise TestRpcError.CustomError("Custom error occurred")

        addr = get_unique_ipc_addr()
        with RPCServer(App()) as server:
            server.bind(addr)
            server.start()
            time.sleep(0.1)
            with RPCClient(addr) as client:
                # Test ValueError handling
                with pytest.raises(RPCError) as exc_info:
                    client.hello().remote()

                error = exc_info.value
                assert "Test error message" in str(error)
                assert error.cause is not None
                assert isinstance(error.cause, ValueError)
                assert error.traceback is not None
                assert "ValueError: Test error message" in error.traceback

                # Test ZeroDivisionError handling
                with pytest.raises(RPCError) as exc_info:
                    client.divide_by_zero().remote()

                error = exc_info.value
                assert "division by zero" in str(error)
                assert error.cause is not None
                assert isinstance(error.cause, ZeroDivisionError)
                assert error.traceback is not None

                # Test custom exception handling
                with pytest.raises(RPCError) as exc_info:
                    client.custom_exception().remote()

                error = exc_info.value
                assert "Custom error occurred" in str(error)
                assert error.cause is not None
                assert error.traceback is not None

    def test_shutdown_cancelled_error(self):
        """Test that pending requests are cancelled with RPCCancelled when server shuts down."""

        class App:

            def task(self):
                time.sleep(10)
                return True

        addr = get_unique_ipc_addr()

        server = RPCServer(
            App(),
            # only one worker to make it easier to pend requests
            num_workers=1)
        server.bind(addr)
        server.start()
        time.sleep(0.1)

        client = RPCClient(addr)
        try:
            client.shutdown_server()
            pending_futures = [client.task().remote_future() for _ in range(10)]

            for future in pending_futures:
                with pytest.raises(RPCCancelled):
                    future.result()
        finally:
            # Ensure proper cleanup
            client.close()
            # Wait for background threads to exit
            time.sleep(1.0)

    @pytest.mark.skip(reason="This test is flaky, need to fix it")
    def test_timeout_error(self):
        """Test that requests that exceed timeout are handled with proper error."""

        class App:

            def slow_method(self):
                # Sleep longer than the timeout
                time.sleep(2.0)
                return "completed"

        with RpcServerWrapper(App()) as server:
            time.sleep(0.1)

            # Create client with short timeout
            with RPCClient(server.addr, timeout=0.5) as client:
                with pytest.raises(RPCError) as exc_info:
                    client.slow_method().remote(timeout=0.5)

                error = exc_info.value
                # Should be either a timeout error or RPC error indicating timeout
                assert "timed out" in str(error).lower() or "timeout" in str(
                    error).lower()

    def test_method_not_found_error(self):
        """Test that calling non-existent methods returns proper error."""

        class App:

            def existing_method(self):
                return "exists"

        with RpcServerWrapper(App()) as server:
            time.sleep(0.1)

            with RPCClient(server.addr) as client:
                with pytest.raises(RPCError) as exc_info:
                    client.non_existent_method().remote()

                error = exc_info.value
                assert "not found" in str(error)
                assert error.traceback is not None


@pytest.mark.skip(reason="This test is flaky, need to fix it")
def test_rpc_shutdown_server():

    class App:

        def hello(self):
            return "world"

    addr = get_unique_ipc_addr()
    server = RPCServer(App())
    server.bind(addr)
    server.start()
    time.sleep(0.1)
    try:
        with RPCClient(addr) as client:
            ret = client.hello().remote()
            assert ret == "world"

            client.shutdown_server()
    finally:
        # Wait for the server dispatcher thread to quit
        time.sleep(1.0)


@pytest.mark.skip(reason="This test is flaky, need to fix it")
def test_rpc_without_response_performance():
    # At any circumstances, the RPC call without response should be faster than the one with response
    class App:

        def __init__(self):
            self.task_submitted = False

        def send_task(self) -> None:
            # Just submit the task and return immediately
            # The result is not important
            time.sleep(0.001)
            return None

    addr = get_unique_ipc_addr()
    with RPCServer(App(), num_workers=10) as server:
        server.bind(addr)
        server.start()
        time.sleep(0.1)
        with RPCClient(addr) as client:
            time_start = time.time()
            for i in range(100):
                client.send_task().remote(need_response=False)
            time_end = time.time()

            no_wait_time = time_end - time_start

            time_start = time.time()
            for i in range(100):
                client.send_task().remote(need_response=True)
            time_end = time.time()
            wait_time = time_end - time_start

            assert no_wait_time < wait_time, f"{no_wait_time} > {wait_time}"


@pytest.mark.parametrize("async_run_task", [True, False])
@pytest.mark.parametrize("use_ipc_addr", [True, False])
def test_rpc_benchmark(async_run_task: bool, use_ipc_addr: bool):

    class App:

        def cal(self, n: int):
            return n * 2

    with RPCServer(App(), async_run_task=async_run_task) as server:
        address = get_unique_ipc_addr() if use_ipc_addr else "tcp://127.0.0.1:*"

        server.bind(address)
        server.start()
        time.sleep(0.1)

        with RPCClient(server.address) as client:

            time_start = time.time()
            for i in range(100):
                ret = client.cal(i).remote(timeout=10)  # sync call
                assert ret == i * 2, f"{ret} != {i * 2}"
            time_end = time.time()
            print(
                f"Time taken: {time_end - time_start} seconds, {10000 / (time_end - time_start)} calls/second"
            )


class TestRpcTimeout:
    """Test RPC timeout functionality for both sync and async calls, sharing server/client."""

    class App:

        def slow_operation(self, delay: float):
            """A method that takes a long time to complete."""
            time.sleep(delay)
            return "completed"

    def setup_method(self, method):
        """Setup RPC server and client for timeout tests."""
        # Use unique address to avoid socket conflicts
        self.address = get_unique_ipc_addr()
        self.server = RPCServer(self.App())
        self.server.bind(self.address)
        self.server.start()
        time.sleep(0.1)
        self.client = RPCClient(self.address)

    def teardown_method(self):
        """Shutdown server and close client."""
        # Shutdown server first to stop accepting new requests
        if hasattr(self, 'server') and self.server:
            self.server.shutdown()
        # Then close client to clean up connections
        if hasattr(self, 'client') and self.client:
            self.client.close()
        # Wait longer to ensure all background threads exit completely
        time.sleep(1.0)

    def run_sync_timeout_test(self):
        with pytest.raises(RPCTimeout) as exc_info:
            self.client.slow_operation(2.0).remote(timeout=0.1)
        assert "timed out" in str(
            exc_info.value), f"Timeout message not found: {exc_info.value}"

    def run_async_timeout_test(self):
        import asyncio

        async def async_timeout():
            with pytest.raises(RPCTimeout) as exc_info:
                await self.client.slow_operation(2.0).remote_async(timeout=0.1)
            assert "timed out" in str(
                exc_info.value), f"Timeout message not found: {exc_info.value}"

        asyncio.run(async_timeout())

    def run_sync_success_test(self):
        result = self.client.slow_operation(0.1).remote(timeout=10.0)
        assert result == "completed"
        print(f"final result: {result}")

    def run_async_success_test(self):
        import asyncio

        async def async_success():
            result = await self.client.slow_operation(0.1).remote_async(
                timeout=10.0)
            assert result == "completed"
            print(f"final result: {result}")
            return result

        return asyncio.run(async_success())

    @pytest.mark.parametrize("use_async", [True, False])
    def test_rpc_timeout(self, use_async):
        if use_async:
            self.run_async_timeout_test()
            self.run_async_success_test()
        else:
            self.run_sync_timeout_test()
            self.run_sync_success_test()


class TestRpcShutdown:

    def test_duplicate_shutdown(self):

        class App:

            def quick_task(self, task_id: int):
                return f"quick_task_{task_id}"

        with RpcServerWrapper(App()) as server:
            time.sleep(0.1)
            with RPCClient(server.addr) as client:
                client.quick_task(1).remote()

                # repeated shutdown should not raise an error
                for i in range(10):
                    server.shutdown()

    @pytest.mark.skip(reason="This test is flaky, need to fix it")
    def test_submit_request_after_server_shutdown(self):

        class App:

            def foo(self, delay: int):
                time.sleep(delay)
                return "foo"

        addr = get_unique_ipc_addr()
        server = RPCServer(App())
        server.bind(addr)
        server.start()

        time.sleep(0.1)
        with RPCClient(addr) as client:
            # This task should be cancelled when server shuts down
            res = client.foo(10).remote_future(timeout=12)

            # The shutdown will now immediately cancel pending requests
            server.shutdown()

            # Verify the request was cancelled
            with pytest.raises(RPCCancelled):
                res.result()


class TestApp:
    """Test application with various method types."""

    def __init__(self):
        self.call_count = 0

    def sync_add(self, a: int, b: int) -> int:
        """Sync method."""
        self.call_count += 1
        return a + b

    async def async_multiply(self, x: int, y: int) -> int:
        """Async method."""
        self.call_count += 1
        return x * y

    async def streaming_range(self, n: int):
        """Streaming generator."""
        for i in range(n):
            yield i

    async def streaming_error(self, n: int):
        """Streaming generator that raises error."""
        for i in range(n):
            if i == 2:
                raise ValueError("Test error at i=2")
            yield i

    async def streaming_timeout(self, delay: float):
        """Streaming generator with configurable delay for timeout testing."""
        for i in range(10):
            await asyncio.sleep(delay)
            yield i

    async def streaming_forever(self):
        """Streaming generator that never ends, used for cancellation testing."""
        i = 0
        while True:
            await asyncio.sleep(0.1)
            yield i
            i += 1


@pytest.mark.asyncio
async def test_streaming_task_cancelled():
    # Test the streaming task cancelled when the server is shutdown
    # This emulates the RpcWorker.fetch_responses_loop_async behavior
    app = TestApp()
    with RpcServerWrapper(app, num_workers=2, async_run_task=True) as server:
        with RPCClient(server.address) as client:
            iter = client.streaming_forever().remote_streaming()
            # Only get the first 3 values
            for i in range(3):
                v = await iter.__anext__()
                print(f"value {i}: {v}")

            # The server should be shutdown while the task is not finished


class TestRpcAsync:
    # Use setup_method/teardown_method for pytest class-based setup/teardown
    def setup_method(self):
        """Setup RPC server and client for tests."""
        self.app = TestApp()
        self.server = RPCServer(self.app, num_workers=2, async_run_task=True)
        self.server.bind("tcp://127.0.0.1:0")  # Use random port
        self.server.start()
        # Get actual address after binding
        address = f"tcp://127.0.0.1:{self.server.address.split(':')[-1]}"
        self.client = RPCClient(address)

    def teardown_method(self):
        self.server.shutdown()
        self.client.close()

    @pytest.mark.asyncio
    async def test_sync_method(self):
        """Test traditional sync method still works."""
        app, client, server = self.app, self.client, self.server

        # Test sync call
        result = client.sync_add(5, 3).remote()
        assert result == 8
        assert app.call_count == 1

    @pytest.mark.asyncio
    async def test_async_method(self):
        """Test async method execution."""
        app, client, server = self.app, self.client, self.server

        # Test async call
        result = await client.async_multiply(4, 7).remote_async()
        assert result == 28
        assert app.call_count == 1

    @pytest.mark.asyncio
    async def test_streaming_basic(self):
        """Test basic streaming functionality."""
        app, client, server = self.app, self.client, self.server

        results = []
        async for value in client.streaming_range(5).remote_streaming():
            results.append(value)

        assert results == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_streaming_concurrent(self):
        """Test concurrent streaming calls."""
        app, client, server = self.app, self.client, self.server

        async def collect_stream(n):
            results = []
            async for value in client.streaming_range(n).remote_streaming():
                results.append(value)
            return results

        # Run 3 concurrent streams
        results = await asyncio.gather(collect_stream(3), collect_stream(4),
                                       collect_stream(5))

        assert results[0] == [0, 1, 2]
        assert results[1] == [0, 1, 2, 3]
        assert results[2] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """Test error handling in streaming."""
        app, client, server = self.app, self.client, self.server

        results = []
        with pytest.raises(RPCStreamingError, match="Test error at i=2"):
            async for value in client.streaming_error(5).remote_streaming():
                results.append(value)

        # Should have received values before error
        assert results == [0, 1]

    @pytest.mark.asyncio
    async def test_streaming_timeout(self):
        """Test timeout handling in streaming."""
        app, client, server = self.app, self.client, self.server

        # Set short timeout
        with pytest.raises(RPCTimeout):
            async for value in client.streaming_timeout(
                    delay=2.0).remote_streaming(timeout=0.5):
                pass  # Should timeout before first yield

    @pytest.mark.asyncio
    async def test_mixed_calls(self):
        """Test mixing different call types."""
        app, client, server = self.app, self.client, self.server

        # Run sync, async, and streaming calls together
        sync_result = client.sync_add(1, 2).remote()
        async_future = client.async_multiply(3, 4).remote_future()

        streaming_results = []
        async for value in client.streaming_range(3).remote_streaming():
            streaming_results.append(value)

        async_result = async_future.result()

        assert sync_result == 3
        assert async_result == 12
        assert streaming_results == [0, 1, 2]
        assert app.call_count == 2  # sync + async (streaming doesn't increment)

    @pytest.mark.asyncio
    async def test_invalid_streaming_call(self):
        """Test calling non-streaming method with streaming."""
        app, client, server = self.app, self.client, self.server

        # This should fail because sync_add is not an async generator
        with pytest.raises(RPCStreamingError):
            async for value in client.sync_add(1, 2).remote_streaming():
                pass


class TestResponsePickleError:
    """ The pickle error will break the whole server, test the error handling. """

    class App:

        def unpickleable_return(self):
            # Functions defined locally are not pickleable
            def nested_function():
                pass

            return nested_function

        async def unpickleable_streaming_return(self):
            # Functions defined locally are not pickleable
            def nested_function():
                pass

            yield nested_function

    def test_unpickleable_error(self):
        with RpcServerWrapper(self.App()) as server:
            with RPCClient(server.addr) as client:
                with pytest.raises(RPCError) as exc_info:
                    client.unpickleable_return().remote()

                assert "Failed to pickle response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unpickleable_streaming_error(self):
        with RpcServerWrapper(self.App(), async_run_task=True) as server:
            with RPCClient(server.addr) as client:
                with pytest.raises(RPCStreamingError) as exc_info:
                    async for _ in client.unpickleable_streaming_return(
                    ).remote_streaming():
                        pass

                assert "Failed to pickle response" in str(exc_info.value)


class TestRpcRobustness:

    class App:
        LARGE_RESPONSE_SIZE = 1024 * 1024 * 10  # 10MB

        def remote_with_large_response(self):
            return b"a" * self.LARGE_RESPONSE_SIZE

        async def streaming_with_large_response(self):
            for i in range(1000):
                yield b"a" * self.LARGE_RESPONSE_SIZE

        async def get_streaming(self):
            for i in range(1000):
                yield i

    def test_remote_with_large_response(self):
        with RpcServerWrapper(self.App()) as server:
            with RPCClient(server.addr) as client:
                for i in range(100):
                    result = client.remote_with_large_response().remote()
                    assert result == b"a" * self.App.LARGE_RESPONSE_SIZE

    @pytest.mark.asyncio
    async def test_streaming_with_large_response(self):
        with RpcServerWrapper(self.App()) as server:
            with RPCClient(server.addr) as client:
                async for result in client.streaming_with_large_response(
                ).remote_streaming():
                    assert result == b"a" * self.App.LARGE_RESPONSE_SIZE

    def test_threaded_streaming(self):
        """Test that get_streaming can be safely called from multiple threads."""
        # All the async remote calls will be submitted to the RPCClient._loop, let
        # it handle the concurrent requests.  Once the response arrives, it will
        # be processed by the RPCClient._loop, and dispatch to the corresponding
        # task via the dedicated AsyncQueue.
        num_threads = 100
        items_per_stream = 100

        # Use shorter stream for faster test
        class TestApp:

            async def get_streaming(self):
                for i in range(items_per_stream):
                    yield i

        with RpcServerWrapper(TestApp(), async_run_task=True) as server:
            errors = []
            results = [None] * num_threads

            def stream_consumer(thread_id: int):
                """Function to be executed in each thread."""
                print(f"Thread {thread_id} started")
                try:
                    # Each thread creates its own client connection
                    with RPCClient(server.addr) as client:
                        collected = []

                        async def consume_stream():
                            async for value in client.get_streaming(
                            ).remote_streaming():
                                collected.append(value)

                        # Run the async streaming call in this thread
                        asyncio.run(consume_stream())

                        # Verify we got all expected values
                        expected = list(range(items_per_stream))
                        if collected != expected:
                            errors.append(
                                f"Thread {thread_id}: Expected {expected}, got {collected}"
                            )
                        else:
                            results[thread_id] = collected

                except Exception as e:
                    errors.append(
                        f"Thread {thread_id}: {type(e).__name__}: {str(e)}")

            # Create and start multiple threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=stream_consumer, args=(i, ))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout per thread

            # Check for any errors
            if errors:
                error_msg = "\n".join(errors)
                pytest.fail(
                    f"Thread safety test failed with errors:\n{error_msg}")

            # Verify all threads completed successfully
            for i, result in enumerate(results):
                assert result is not None, f"Thread {i} did not complete successfully"
                assert len(
                    result
                ) == items_per_stream, f"Thread {i} got {len(result)} items, expected {items_per_stream}"

    def test_threaded_remote_call(self):
        """Test that regular remote calls can be safely made from multiple threads."""
        # Each thread will make multiple synchronous remote calls
        # This tests if RPCClient can handle concurrent requests from different threads
        num_threads = 100
        calls_per_thread = 100

        class TestApp:

            def __init__(self):
                self.call_count = 0
                self.lock = threading.Lock()

            def increment(self, v):
                with self.lock:
                    self.call_count += 1
                threading.get_ident()
                return v + 1

        app = TestApp()
        with RpcServerWrapper(app) as server:
            errors = []
            results = [None] * num_threads

            client = RPCClient(server.addr)

            def remote_caller(thread_id: int):
                """Function to be executed in each thread."""
                print(f"Thread {thread_id} started")
                try:
                    thread_results = []

                    for i in range(calls_per_thread):
                        result = client.increment(i).remote()
                        expected = i + 1

                        if result != expected:
                            errors.append(
                                f"Thread {thread_id}, call {i}: Expected {expected}, got {result}"
                            )
                        thread_results.append(result)

                    results[thread_id] = thread_results

                except Exception as e:
                    errors.append(
                        f"Thread {thread_id}: {type(e).__name__}: {str(e)}")
                finally:
                    print(f"Thread {thread_id} completed")

            # Create and start multiple threads
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=remote_caller,
                                          args=(i, ),
                                          daemon=True)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout per thread

            client.close()

            # Check for any errors
            if errors:
                error_msg = "\n".join(errors)
                pytest.fail(
                    f"Thread safety test failed with errors:\n{error_msg}")

            # Verify all threads completed successfully
            for i, result in enumerate(results):
                assert result is not None, f"Thread {i} did not complete successfully"
                assert len(
                    result
                ) == calls_per_thread, f"Thread {i} made {len(result)} calls, expected {calls_per_thread}"

            # Verify total call count
            expected_total_calls = num_threads * calls_per_thread
            assert app.call_count == expected_total_calls, \
                f"Expected {expected_total_calls} total calls, but got {app.call_count}"

    def test_repeated_creation_and_destruction(self, num_calls: int = 100):
        """Test robustness of repeated RPCServer/RPCClient creation and destruction.

        This test ensures there are no resource leaks, socket exhaustion, or other
        issues when repeatedly creating and destroying server/client pairs.
        """

        class TestApp:

            def __init__(self):
                self.counter = 0

            def increment(self, value: int) -> int:
                self.counter += 1
                return value + 1

            def get_counter(self) -> int:
                return self.counter

        for i in range(num_calls):
            # Create app, server, and client
            # RpcServerWrapper automatically generates unique addresses
            app = TestApp()

            with RpcServerWrapper(app) as server:
                with RPCClient(server.addr) as client:
                    # Perform a few remote calls to verify functionality
                    result1 = client.increment(10).remote()
                    assert result1 == 11, f"Iteration {i}: Expected 11, got {result1}"

                    result2 = client.increment(20).remote()
                    assert result2 == 21, f"Iteration {i}: Expected 21, got {result2}"

                    counter = client.get_counter().remote()
                    assert counter == 2, f"Iteration {i}: Expected counter=2, got {counter}"

                    if i % 10 == 0:
                        print(
                            f"Iteration {i}/{num_calls} completed successfully")

        print(f"All {num_calls} iterations completed successfully")

    @pytest.mark.parametrize("concurrency", [10, 50, 100])
    def test_many_client_to_single_server(self, concurrency):
        """
        Pressure test where many clients connect to a single server.
        Controls concurrency via parameter and ensures each client performs multiple operations.
        """

        class App:

            def echo(self, msg: str) -> str:
                return msg

        total_clients = max(200, concurrency * 2)
        requests_per_client = 100

        with RpcServerWrapper(App(), async_run_task=True) as server:
            errors = []

            def run_client_session(client_id):
                try:
                    with RPCClient(server.addr) as client:
                        for i in range(requests_per_client):
                            msg = f"c{client_id}-req{i}"
                            ret = client.echo(msg).remote()
                            assert ret == msg
                except Exception as e:
                    errors.append(f"Client {client_id} error: {e}")
                    raise

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency) as executor:
                futures = [
                    executor.submit(run_client_session, i)
                    for i in range(total_clients)
                ]
                concurrent.futures.wait(futures)

                # Check for exceptions in futures
                for f in futures:
                    if f.exception():
                        errors.append(str(f.exception()))

            assert not errors, f"Encountered errors: {errors[:5]}..."

    @pytest.mark.parametrize("concurrency", [10, 50, 100])
    def test_many_client_to_single_server_threaded(self, concurrency):
        """
        Pressure test where clients are created and used in different threads.
        """
        import concurrent.futures

        class App:

            def echo(self, msg: str) -> str:
                return msg

        # Scale total clients to be more than concurrency to force queueing/reuse
        total_clients = max(200, concurrency * 2)
        requests_per_client = 100

        with RpcServerWrapper(App(), async_run_task=True) as server:
            errors = []

            def run_client_session(client_id):
                try:
                    # Client creation and usage happens strictly within this thread
                    with RPCClient(server.addr) as client:
                        for i in range(requests_per_client):
                            msg = f"c{client_id}-req{i}"
                            ret = client.echo(msg).remote()
                            assert ret == msg
                except Exception as e:
                    errors.append(f"Client {client_id} error: {e}")
                    raise

            # Use ThreadPoolExecutor to simulate concurrent threads
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency) as executor:
                futures = [
                    executor.submit(run_client_session, i)
                    for i in range(total_clients)
                ]
                concurrent.futures.wait(futures)

                for f in futures:
                    if f.exception():
                        errors.append(str(f.exception()))

            assert not errors, f"Encountered errors: {errors[:5]}..."
