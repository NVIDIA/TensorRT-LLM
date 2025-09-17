import pytest
import torch
import torch._inductor


@pytest.fixture(autouse=True, scope='function')
def set_torchinductor_compile_threads():
    original_value = torch._inductor.config.compile_threads
    torch._inductor.config.compile_threads = 1
    yield
    torch._inductor.config.compile_threads = original_value
