import pytest

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", params=["pytorch"])
def backend(request):
    return request.param
