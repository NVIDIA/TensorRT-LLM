import pytest

from tensorrt_llm._utils import mpi_disabled

if not mpi_disabled():
    pytest.skip("Only tested in ray stage", allow_module_level=True)
