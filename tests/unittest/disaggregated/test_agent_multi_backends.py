import os
import subprocess

import pytest


def test_load_agent_missing_module():
    """_load_agent returns (None, ImportError) for a non-existent module.

    Regression test: previously a missing nixl package caused an AssertionError
    at module import time, making pytest exit with code 2 (collection failure)
    instead of a clear ImportError with a descriptive message.
    """
    from tensorrt_llm._torch.disaggregation.nixl.agent import _load_agent

    agent, err = _load_agent("_trtllm_nonexistent_module_xyz_", ["SomeClass"])
    assert agent is None
    assert isinstance(err, ImportError), f"Expected ImportError, got {type(err)}: {err}"
    assert "No module named" in str(err) or "_trtllm_nonexistent_module_xyz_" in str(err)


def test_load_agent_missing_attributes():
    """_load_agent returns (None, ImportError) and logs a warning when attributes are missing."""
    from tensorrt_llm._torch.disaggregation.nixl.agent import _load_agent

    # 'os' exists but has no NixlTransferAgent attribute
    agent, err = _load_agent("os", ["NixlTransferAgent"])
    assert agent is None
    assert isinstance(err, ImportError), f"Expected ImportError, got {type(err)}: {err}"
    assert "NixlTransferAgent" in str(err)


def test_load_agent_success():
    """_load_agent returns (module, None) on success."""
    from tensorrt_llm._torch.disaggregation.nixl.agent import _load_agent

    agent, err = _load_agent("os", ["path", "getcwd"])
    assert agent is not None
    assert err is None
    assert hasattr(agent, "path")
    assert hasattr(agent, "getcwd")


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
