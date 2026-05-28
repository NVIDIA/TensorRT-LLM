from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console

from .console import _THEME

_LOG_DIR = Path("logs")

_logger: "Logger | None" = None


class Logger:

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = path.open("w", encoding="utf-8")
        self.console = Console(
            file=self._file,
            theme=_THEME,
            highlight=False,
            force_terminal=False,
            width=120,
        )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def get_logger() -> Logger:
    global _logger
    if _logger is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _logger = Logger(_LOG_DIR / f"session-{timestamp}.log")
        atexit.register(_logger.close)
    return _logger
