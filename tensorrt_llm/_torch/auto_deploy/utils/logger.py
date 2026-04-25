import logging
import os


class Singleton(type):
    """Metaclass that ensures only one instance of a class exists."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ADLogger(metaclass=Singleton):
    """Logger for auto_deploy using Python's standard logging module.

    Provides the same API surface as TRT-LLM's Logger class so auto_deploy code
    works identically in both standalone and TRT-LLM-integrated modes.

    Uses the Singleton metaclass to ensure a single logger instance.
    """

    ENV_VARIABLE = "AUTO_DEPLOY_LOG_LEVEL"
    PREFIX = "AUTO-DEPLOY"
    DEFAULT_LEVEL = "info"

    # Severity constants matching TRT-LLM's Logger
    INTERNAL_ERROR = "internal_error"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    VERBOSE = "verbose"
    DEBUG = "debug"

    _SEVERITY_TO_LEVEL = {
        "internal_error": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "verbose": logging.DEBUG,
        "debug": logging.DEBUG,
    }

    def __init__(self):
        self._logger = logging.getLogger("auto_deploy")
        level_str = os.environ.get(self.ENV_VARIABLE, self.DEFAULT_LEVEL).upper()
        self._logger.setLevel(getattr(logging, level_str, logging.INFO))
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(f"[{self.PREFIX}] [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
        self.rank = None
        self._appeared_keys = set()

    def set_rank(self, rank: int):
        self.rank = rank

    def log(self, severity, *msg):
        level = self._SEVERITY_TO_LEVEL.get(severity, logging.INFO)
        parts = [f"[{self.PREFIX}]"]
        if self.rank is not None:
            parts.append(f"[RANK {self.rank}]")
        parts.append(severity)
        parts.extend(map(str, msg))
        self._logger.log(level, " ".join(parts))

    def log_once(self, severity, *msg, key):
        if key not in self._appeared_keys:
            self._appeared_keys.add(key)
            self.log(severity, *msg)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def warning_once(self, *msg, key):
        self.log_once(self.WARNING, *msg, key=key)

    def debug_once(self, *msg, key):
        self.log_once(self.DEBUG, *msg, key=key)

    def set_level(self, level):
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)


ad_logger = ADLogger()
