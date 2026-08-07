import pytest

pytestmark = pytest.mark.threadleak(enabled=False)
