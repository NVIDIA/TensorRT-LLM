import subprocess  # nosec B404
import time

from tensorrt_llm.executor.utils import create_mpi_comm_session
from tensorrt_llm.llmapi.utils import print_colored


def main():
    tasks = [0, 1, 2, 3, 4]
    client = create_mpi_comm_session(n_workers=4)
    for task in tasks:
        client.submit(print_colored, f"{task}\n", "green")
    time.sleep(10)
    client.shutdown()


if __name__ == "__main__":
    main()
