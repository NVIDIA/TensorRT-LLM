"""Compatibility wrapper for optional in-kernel event tracing support."""

try:
    from cutlass.cute.experimental import iket
except (ImportError, NotImplementedError):
    try:
        from cutlass.cute import iket  # type: ignore
    except (ImportError, NotImplementedError):
        class _IketShim:
            """No-op IKET interface for toolchains without the dialect."""

            @staticmethod
            def range_push(_name, *_args, **_kwargs):
                return None

            @staticmethod
            def range_pop(*_args, **_kwargs):
                return None

            @staticmethod
            def range_start(_name, *_args, **_kwargs):
                return None

            @staticmethod
            def range_end(_token=None, *_args, **_kwargs):
                return None

            @staticmethod
            def mark(_name, *_args, **_kwargs):
                return None

        iket = _IketShim()  # type: ignore


__all__ = ["iket"]
