from dataclasses import dataclass
from typing import List

from tensorrt_llm.hlapi.mpi_session import SocketListener


@dataclass
class ComplexData:
    a: str
    b: int
    c: List[int]


def test_SocketServer():

    messages = [
        "hello",  # str
        123,  # int
        ComplexData("hello", 123, [1, 2, 3])  # complex
    ]

    offset = 0

    def callback(data):
        nonlocal offset
        print('get data', data)
        assert data == messages[offset]
        offset += 1

    server = SocketListener(callback=callback)

    client = server.get_client()

    for data in messages:
        client.send(data)

    server.shutdown()


if __name__ == '__main__':
    test_SocketServer()
