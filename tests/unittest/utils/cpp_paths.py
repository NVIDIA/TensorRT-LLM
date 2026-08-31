import os as _os
import pathlib as _pl
import sys as _sys

import pytest

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', '..', '..'))


@pytest.fixture(scope="module")
def llm_root() -> _pl.Path:
    environ_root = _os.environ.get("LLM_ROOT", None)
    return _pl.Path(environ_root) if environ_root is not None else _pl.Path(
        __file__).resolve().parent.parent.parent.parent
