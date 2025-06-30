import time

import pytest

from tensorrt_llm.executor.rpc import RPCClient, RPCServer


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
            ret = client.cal(i)  # sync call
            assert ret == i * 2, f"{ret} != {i * 2}"
        time_end = time.time()
        print(
            f"Time taken: {time_end - time_start} seconds, {10000 / (time_end - time_start)} calls/second"
        )
