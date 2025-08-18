import pytest

# Mark all tests in this directory as Ray tests so they can be selected with `-m ray`
# and are enabled when running with `--run-ray`.
pytestmark = pytest.mark.ray
