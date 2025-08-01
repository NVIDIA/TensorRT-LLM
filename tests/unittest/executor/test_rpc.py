import time

import pytest

from tensorrt_llm.executor.rpc import RPCClient, RPCError, RPCServer


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
        with pytest.raises(RPCError) as exc_info:
            if use_async:
                # Test async call with timeout
                import asyncio

                async def test_async_timeout():
                    return await client.call_async('slow_operation',
                                                   2.0,
                                                   __rpc_timeout=0.1)

                asyncio.run(test_async_timeout())
            else:
                # Test sync call with timeout
                client.slow_operation(2.0, __rpc_timeout=0.1)

            assert "timed out" in str(
                exc_info.value), f"Timeout message not found: {exc_info.value}"

        # Test that a long timeout allows the operation to complete
        if use_async:
            # Test async call with sufficient timeout
            import asyncio

            async def test_async_success():
                return await client.call_async('slow_operation',
                                               0.1,
                                               __rpc_timeout=1.0)

            result = asyncio.run(test_async_success())
        else:
            result = client.slow_operation(0.1, __rpc_timeout=1.0)

        assert result == "completed"
