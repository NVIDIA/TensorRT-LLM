import asyncio
import time

import pytest

from tensorrt_llm.executor.rpc import (RPCCancelled, RPCClient, RPCError,
                                       RPCServer, RPCStreamingError, RPCTimeout)
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr


class RpcServerWrapper(RPCServer):

    def __init__(self, *args, addr: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.addr = addr

    def __enter__(self):
        self.bind(self.addr)
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


class TestRpcBasics:

    def test_rpc_server_basics(self):

        class App:

            def hello(self):
                print("hello")

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            pass

    def test_remote_call_without_arg(self):

        class App:

            def hello(self):
                print("hello")
                return "world"

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
                ret = client.hello().remote()  # sync call
                assert ret == "world"

    def test_remote_call_with_args(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
                ret = client.hello("app", "Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_remote_call_with_kwargs(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
                ret = client.hello(name="app", location="Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_remote_call_with_args_and_kwargs(self):

        class App:

            def hello(self, name: str, location: str):
                print("hello")
                return f"hello {name} from {location}"

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
                ret = client.hello(name="app", location="Marvel").remote()
                assert ret == "hello app from Marvel"

    def test_rpc_server_address(self):

        class App:
            pass

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            assert server.address == addr

    def test_rpc_with_error(self):

        class App:

            def hello(self):
                raise ValueError("hello")

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
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

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            with RPCClient(addr) as client:
                client.send_task().remote(need_response=False)
                time.sleep(
                    0.1
                )  # wait for some time to make sure the task is submitted
                assert client.get_task_submitted().remote()


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

        with RPCClient(addr) as client:
            client.shutdown_server()
            pending_futures = [client.task().remote_future() for _ in range(10)]

            for future in pending_futures:
                with pytest.raises(RPCCancelled):
                    future.result()

        time.sleep(5)

        client.close()

    def test_timeout_error(self):
        """Test that requests that exceed timeout are handled with proper error."""

        class App:

            def slow_method(self):
                # Sleep longer than the timeout
                time.sleep(2.0)
                return "completed"

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            time.sleep(0.1)

            # Create client with short timeout
            with RPCClient(addr, timeout=0.5) as client:
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

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            time.sleep(0.1)

            with RPCClient(addr) as client:
                with pytest.raises(RPCError) as exc_info:
                    client.non_existent_method().remote()

                error = exc_info.value
                assert "not found" in str(error)
                assert error.traceback is not None


def test_rpc_shutdown_server():

    class App:

        def hello(self):
            return "world"

    addr = get_unique_ipc_addr()
    with RPCServer(App()) as server:
        server.bind(addr)
        server.start()
        time.sleep(0.1)
        with RPCClient(addr) as client:
            ret = client.hello().remote()
            assert ret == "world"

            client.shutdown_server()

    time.sleep(5)  # the server dispatcher thread need some time to quit


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
        # Use unique address based on the test parameter to avoid socket conflicts
        test_name = method.__name__
        self.address = f"ipc:///tmp/rpc_test_timeout_{test_name}_{id(self)}"
        self.server = RPCServer(self.App())
        self.server.bind(self.address)
        self.server.start()
        time.sleep(0.1)
        self.client = RPCClient(self.address)

    def teardown_method(self):
        """Shutdown server and close client."""
        self.client.close()
        self.server.shutdown()
        # Add a small delay to ensure the socket is fully released before the next test
        time.sleep(0.5)

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

        addr = get_unique_ipc_addr()
        with RpcServerWrapper(App(), addr=addr) as server:
            time.sleep(0.1)
            with RPCClient(addr) as client:
                client.quick_task(1).remote()

                # repeated shutdown should not raise an error
                for i in range(10):
                    server.shutdown()

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
            # This task should be continued after server shutdown
            res = client.foo(10).remote_future(timeout=12)

            # The shutdown will block until all pending requests are finished
            server.shutdown()

            assert res.result() == "foo"


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
        await asyncio.sleep(0.01)
        self.call_count += 1
        return x * y

    async def streaming_range(self, n: int):
        """Streaming generator."""
        for i in range(n):
            await asyncio.sleep(0.01)
            yield i

    async def streaming_error(self, n: int):
        """Streaming generator that raises error."""
        for i in range(n):
            if i == 2:
                raise ValueError("Test error at i=2")
            yield i

    async def streaming_timeout(self, delay: float):
        """Streaming generator with configurable delay."""
        for i in range(10):
            await asyncio.sleep(delay)
            yield i


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
        addr = get_unique_ipc_addr()
        with RpcServerWrapper(self.App(), addr=addr) as server:
            with RPCClient(addr) as client:
                with pytest.raises(RPCError) as exc_info:
                    client.unpickleable_return().remote()

                assert "Failed to pickle response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unpickleable_streaming_error(self):
        addr = get_unique_ipc_addr()
        with RpcServerWrapper(self.App(), addr=addr,
                              async_run_task=True) as server:
            with RPCClient(addr) as client:
                with pytest.raises(RPCStreamingError) as exc_info:
                    async for _ in client.unpickleable_streaming_return(
                    ).remote_streaming():
                        pass

                assert "Failed to pickle response" in str(exc_info.value)
