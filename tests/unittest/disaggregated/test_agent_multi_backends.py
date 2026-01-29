import os
import subprocess

import pytest


@pytest.mark.parametrize("use_py_nixl", ["0", "1"])
def test_run_with_different_env(use_py_nixl):
    os.environ["TRTLLM_USE_PY_NIXL_KVCACHE"] = use_py_nixl
    print(f"Running tests with TRTLLM_USE_PY_NIXL_KVCACHE={use_py_nixl}")

    test_file_path = os.path.join(os.path.dirname(__file__), "test_agent.py")
    print(f"Running tests in: {test_file_path}")

    result = subprocess.run(
        ["pytest", "--capture=no", test_file_path],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print("Test failed. stderr output:")
        print(result.stderr)

    assert result.returncode == 0, f"Tests failed with TRTLLM_USE_PY_NIXL_KVCACHE={use_py_nixl}"


if __name__ == "__main__":
    pytest.main()
