import pytest


@pytest.fixture
def stub_requests_head(monkeypatch):
    class R:
        status_code = 200

    def _head(url, timeout=5):
        return R()

    import requests

    monkeypatch.setattr(requests, "head", _head)
