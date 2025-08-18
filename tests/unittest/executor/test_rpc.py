import threading
import time

import pytest

from tensorrt_llm.executor.rpc import (RPCCancelled, RPCClient, RPCError,
                                       RPCServer, RPCTimeout)


def test_rpc_server_basics():

    class App:

        def hello(self):
            print("hello")

    server = RPCServer(App())
    print("bind")
    server.bind("ipc:///tmp/rpc_test")
    print("start")
    server.start()
    print("sleep")

    time.sleep(1)
    print("shutdown")
    server.shutdown()


def test_rpc_client_context_manager():

    class App:

        def hello(self):
            print("hello")

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test")
        server.start()
        time.sleep(1)


def test_rpc_hello_without_arg():

    class App:

        def hello(self):
            print("hello")
            return "world"

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test")
        ret = client.hello()  # sync call
        assert ret == "world"


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

        with RPCServer(App()) as server:
            server.bind("ipc:///tmp/rpc_test_error")
            server.start()
            time.sleep(0.1)
            client = RPCClient("ipc:///tmp/rpc_test_error")

            # Test ValueError handling
            with pytest.raises(RPCError) as exc_info:
                client.hello()

            error = exc_info.value
            assert "Test error message" in str(error)
            assert error.cause is not None
            assert isinstance(error.cause, ValueError)
            assert error.traceback is not None
            assert "ValueError: Test error message" in error.traceback

            # Test ZeroDivisionError handling
            with pytest.raises(RPCError) as exc_info:
                client.divide_by_zero()

            error = exc_info.value
            assert "division by zero" in str(error)
            assert error.cause is not None
            assert isinstance(error.cause, ZeroDivisionError)
            assert error.traceback is not None

            # Test custom exception handling
            with pytest.raises(RPCError) as exc_info:
                client.custom_exception()

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

        addr = "ipc:///tmp/rpc_test_cancelled"

        server = RPCServer(
            App(),
            # only one worker to make it easier to pend requests
            num_workers=1)
        server.bind(addr)
        server.start()
        time.sleep(0.1)

        client = RPCClient(addr)
        client.shutdown_server()
        pending_futures = [client.task(__rpc_mode="future") for _ in range(10)]

        for future in pending_futures:
            with pytest.raises(RPCCancelled):
                future.result()

        time.sleep(1)

        client.close()

    def test_timeout_error(self):
        """Test that requests that exceed timeout are handled with proper error."""

        class App:

            def slow_method(self):
                # Sleep longer than the timeout
                time.sleep(2.0)
                return "completed"

        with RPCServer(App()) as server:
            server.bind("ipc:///tmp/rpc_test_timeout")
            server.start()
            time.sleep(0.1)

            # Create client with short timeout
            client = RPCClient("ipc:///tmp/rpc_test_timeout", timeout=0.5)

            with pytest.raises(RPCError) as exc_info:
                client.slow_method(__rpc_timeout=0.5)

            error = exc_info.value
            # Should be either a timeout error or RPC error indicating timeout
            assert "timed out" in str(error).lower() or "timeout" in str(
                error).lower()

    def test_method_not_found_error(self):
        """Test that calling non-existent methods returns proper error."""

        class App:

            def existing_method(self):
                return "exists"

        with RPCServer(App()) as server:
            server.bind("ipc:///tmp/rpc_test_not_found")
            server.start()
            time.sleep(0.1)

            client = RPCClient("ipc:///tmp/rpc_test_not_found")

            with pytest.raises(RPCError) as exc_info:
                client.non_existent_method()

            error = exc_info.value
            assert "not found" in str(error)
            assert error.traceback is not None


def test_rpc_shutdown_server():

    class App:

        def hello(self):
            return "world"

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test_shutdown")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test_shutdown")
        ret = client.hello()
        assert ret == "world"

        client.shutdown_server()

    time.sleep(5)  # the server dispatcher thread need some time to quit


def test_rpc_hello_with_arg():

    class App:

        def hello(self, name: str, location: str):
            print("hello")
            return f"hello {name} from {location}"

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test")
        ret = client.hello("app", location="Marvel")  # sync call
        assert ret == "hello app from Marvel"


def test_rpc_server_address():

    class App:

        def hello(self):
            print("hello")
            return "world"

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test")
        server.start()
        time.sleep(0.1)
        assert server.address == "ipc:///tmp/rpc_test"


def test_rpc_with_error():

    class App:

        def hello(self):
            raise ValueError("hello")

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test_error")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test_error")
        with pytest.raises(RPCError):
            client.hello()


def test_rpc_without_wait_response():

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

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test_no_wait")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test_no_wait")
        client.send_task(__rpc_need_response=False)
        time.sleep(0.1)  # wait for some time to make sure the task is submitted
        assert client.get_task_submitted()


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

    with RPCServer(App(), num_workers=10) as server:
        server.bind("ipc:///tmp/rpc_test_no_wait")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test_no_wait")

        time_start = time.time()
        for i in range(100):
            client.send_task(__rpc_need_response=False)
        time_end = time.time()

        no_wait_time = time_end - time_start

        time_start = time.time()
        for i in range(100):
            client.send_task(__rpc_need_response=True)
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
        address = "ipc:///tmp/rpc_test" if use_ipc_addr else "tcp://127.0.0.1:*"

        server.bind(address)
        server.start()
        time.sleep(0.1)

        client = RPCClient(server.address)

        time_start = time.time()
        for i in range(10000):
            ret = client.cal(i, __rpc_timeout=10)  # sync call
            assert ret == i * 2, f"{ret} != {i * 2}"
        time_end = time.time()
        print(
            f"Time taken: {time_end - time_start} seconds, {10000 / (time_end - time_start)} calls/second"
        )


@pytest.mark.parametrize("use_async", [True, False])
def test_rpc_timeout(use_async: bool):
    """Test RPC timeout functionality.

    Args:
        use_async: Whether to test async RPC calls or sync RPC calls
    """

    class App:

        def slow_operation(self, delay: float):
            """A method that takes a long time to complete."""
            time.sleep(delay)
            return "completed"

    with RPCServer(App()) as server:
        server.bind("ipc:///tmp/rpc_test_timeout")
        server.start()
        time.sleep(0.1)
        client = RPCClient("ipc:///tmp/rpc_test_timeout")

        # Test that a short timeout causes RPCTimeout exception
        with pytest.raises(RPCTimeout) as exc_info:
            if use_async:
                # Test async call with timeout
                import asyncio

                async def test_async_timeout():
                    return await client.call_async('slow_operation',
                                                   2.0,
                                                   __rpc_timeout=0.1)

                asyncio.run(test_async_timeout())
            else:
                client.slow_operation(2.0, __rpc_timeout=0.1)  # small timeout

            assert "timed out" in str(
                exc_info.value), f"Timeout message not found: {exc_info.value}"

        # Test that a long timeout allows the operation to complete
        if use_async:
            import asyncio

            async def test_async_success():
                return await client.call_async('slow_operation',
                                               0.1,
                                               __rpc_timeout=10.0)

            result = asyncio.run(test_async_success())
        else:
            result = client.slow_operation(0.1, __rpc_timeout=10.0)

        assert result == "completed"

        client.close()


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


class TestRpcShutdown:

    def test_duplicate_shutdown(self):

        class App:
            pass

        with RpcServerWrapper(App(),
                              addr="ipc:///tmp/rpc_test_shutdown") as server:
            time.sleep(0.1)
            client = RPCClient("ipc:///tmp/rpc_test_shutdown")
            client.quick_task(1)

            # repeated shutdown should not raise an error
            for i in range(10):
                server.shutdown()

    def test_submit_request_after_server_shutdown(self):

        class App:

            def foo(self):
                return "foo"

        with RpcServerWrapper(App(),
                              addr="ipc:///tmp/rpc_test_shutdown") as server:
            client = RPCClient("ipc:///tmp/rpc_test_shutdown")
            assert client.foo() == "foo"

            server.shutdown()
            time.sleep(2)  # wait for the server to shutdown

            with pytest.raises(RPCServerShuttingDown):
                client.foo()

    def test_submit_request_after_shutdown(self):

        class App:

            def __init__(self):
                self.completed_requests = []
                self.lock = threading.Lock()

            def quick_task(self, task_id):
                with self.lock:
                    self.completed_requests.append(task_id)
                return f"quick_task_{task_id}_completed"

        app = App()
        with RPCServer(app) as server:
            server.bind("ipc:///tmp/rpc_test_shutdown")
            server.start()
            time.sleep(0.1)
            client = RPCClient("ipc:///tmp/rpc_test_shutdown")
            client.quick_task(1)
            server.shutdown()
            time.sleep(2)  # wait for the server to shutdown
            with pytest.raises(RPCServerShuttingDown):
                client.quick_task(2)


def test_enhanced_shutdown_mechanism():
    """Test enhanced shutdown mechanism that refuses new requests and waits for pending ones."""

    class App:

        def __init__(self):
            self.completed_requests = []
            self.lock = threading.Lock()

        def quick_task(self, task_id):
            """A quick task for testing."""
            with self.lock:
                self.completed_requests.append(task_id)
            return f"quick_task_{task_id}_completed"

        def slow_task(self, task_id, duration=1.0):
            """A slow task that takes some time to complete."""
            time.sleep(duration)
            with self.lock:
                self.completed_requests.append(task_id)
            return f"slow_task_{task_id}_completed"

    app = App()

    with RPCServer(app) as server:
        server.bind("ipc:///tmp/rpc_test_enhanced_shutdown")
        server.start()
        time.sleep(0.1)

        client = RPCClient("ipc:///tmp/rpc_test_enhanced_shutdown")

        # Submit some quick requests that should complete
        quick_future1 = client.quick_task.call_future(1)
        quick_future2 = client.quick_task.call_future(2)

        # Submit a slow request that should still complete during shutdown
        slow_future = client.slow_task.call_future(10, duration=0.5)

        # Wait a bit to ensure requests are queued
        time.sleep(0.1)

        # Start shutdown in a separate thread to test concurrent behavior
        def shutdown_server():
            time.sleep(0.2)  # Let pending requests start processing
            server.shutdown()

        shutdown_thread = threading.Thread(target=shutdown_server)
        shutdown_thread.start()

        # Try to submit a new request after shutdown has started
        # This should eventually fail with RPCServerShuttingDown
        time.sleep(0.3)  # Wait for shutdown to start

        try:
            client.quick_task(999)
            pytest.fail("Expected RPCServerShuttingDown exception")
        except RPCServerShuttingDown as e:
            assert "Server is shutting down" in str(e)
            print(f"✓ New request properly refused during shutdown: {e}")
        except Exception as e:
            # Might get other connection errors which is also acceptable
            print(f"✓ New request failed as expected during shutdown: {e}")

        # Wait for shutdown to complete
        shutdown_thread.join(timeout=5.0)

        # Check that all pending requests completed
        assert quick_future1.result() == "quick_task_1_completed"
        assert quick_future2.result() == "quick_task_2_completed"
        assert slow_future.result() == "slow_task_10_completed"

        # Verify all tasks were completed
        with app.lock:
            assert 1 in app.completed_requests
            assert 2 in app.completed_requests
            assert 10 in app.completed_requests

        print("✓ Enhanced shutdown mechanism test passed")


if __name__ == "__main__":
    #TestRpcError().test_shutdown_cancelled_error()
    #test_rpc_shutdown_server()
    #TestRpcShutdown().test_submit_request_after_server_shutdown()
    test_rpc_timeout(True)
