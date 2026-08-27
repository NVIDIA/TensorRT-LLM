"""Ada SM89 backend."""


def make_kernel(*args, **kwargs):
    """Construct the SM89 kernel without eager architecture imports."""

    from .kernel import make_kernel as _make_kernel

    return _make_kernel(*args, **kwargs)


__all__ = ["make_kernel"]
